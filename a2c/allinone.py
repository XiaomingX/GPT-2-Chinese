import time
import numpy as np
import tensorflow as tf
import gym
from collections import deque
from gym.vector import DummyVecEnv  # 简化的并行环境（单环境也支持）


# -------------------------- 1. 核心辅助函数 --------------------------
def safemean(xs):
    """避免空列表求均值报错"""
    return np.nan if len(xs) == 0 else np.mean(xs)

def discount_with_dones(rewards, dones, gamma):
    """带终止状态的折扣回报计算（A2C核心）"""
    discounted = []
    running_reward = 0  # 反向累积回报
    for reward, done in zip(rewards[::-1], dones[::-1]):
        if done:
            running_reward = 0  # 回合结束，重置累积值
        running_reward = reward + gamma * running_reward
        discounted.append(running_reward)
    return discounted[::-1]  # 反转回原顺序

def find_trainable_variables(scope):
    """获取指定作用域下的可训练参数"""
    return tf.trainable_variables(scope=scope)

def linear_schedule(initial_value, final_value, total_steps, current_step):
    """线性学习率衰减"""
    frac = min(current_step / total_steps, 1.0)
    return initial_value * (1 - frac) + final_value


# -------------------------- 2. 经验收集器（Runner） --------------------------
class Runner:
    """与环境交互，收集n步轨迹数据用于训练"""
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env  # 环境（支持并行环境）
        self.model = model  # A2C模型
        self.nsteps = nsteps  # 每轮收集的步数
        self.gamma = gamma  # 折扣因子
        
        # 初始化环境状态和终止标志
        self.obs = self.env.reset()
        self.dones = np.zeros((self.env.num_envs,), dtype=bool)

    def run(self):
        """收集n步经验，返回训练用批次数据"""
        # 存储经验的列表
        mb_obs, mb_actions, mb_values, mb_rewards, mb_dones = [], [], [], [], []
        epinfos = []  # 存储回合信息（总回报、长度）

        for _ in range(self.nsteps):
            # 1. 模型预测：动作、价值
            actions, values = self.model.step(self.obs)
            
            # 2. 存储当前步经验
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # 3. 与环境交互执行动作
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            mb_rewards.append(rewards)

            # 4. 记录回合完成信息（如总回报、长度）
            for info in infos:
                if 'episode' in info:
                    epinfos.append(info['episode'])

        # 5. 处理数据形状：[nsteps, nenvs, ...] → [nenvs, nsteps, ...]
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=bool).swapaxes(0, 1)

        # 6. 计算n步折扣回报（用最后一步的价值函数做Bootstrapping）
        last_values = self.model.value(self.obs)
        for i in range(self.env.num_envs):
            mb_rewards[i] = discount_with_dones(
                rewards=mb_rewards[i].tolist() + [last_values[i]],
                dones=mb_dones[i].tolist() + [self.dones[i]],
                gamma=self.gamma
            )[:-1]  # 去掉最后添加的bootstrap值

        # 7. 展平数据适配批训练：[nenvs, nsteps, ...] → [nenvs*nsteps, ...]
        mb_obs = mb_obs.reshape(-1, *self.obs.shape[1:])
        mb_actions = mb_actions.reshape(-1)
        mb_values = mb_values.flatten()
        mb_rewards = mb_rewards.flatten()

        return mb_obs, mb_actions, mb_values, mb_rewards, epinfos


# -------------------------- 3. A2C核心模型（策略+价值网络） --------------------------
class A2CModel:
    """
    A2C模型：共享特征提取层，分别输出策略（动作分布）和价值（状态价值）
    适配离散动作空间（如CartPole-v1）
    """
    def __init__(self, obs_space, action_space, hidden_sizes=(64, 64), lr=7e-4):
        self.obs_dim = obs_space.shape[0]  # 状态维度
        self.action_dim = action_space.n   # 动作数量（离散）
        self.lr = lr                       # 初始学习率

        # 1. 定义输入占位符
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim], name="obs")
        self.actions_ph = tf.placeholder(tf.int32, [None], name="actions")
        self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards")
        self.old_values_ph = tf.placeholder(tf.float32, [None], name="old_values")

        # 2. 共享特征提取（2层全连接网络）
        with tf.variable_scope("a2c_net"):
            x = tf.layers.dense(self.obs_ph, hidden_sizes[0], activation=tf.nn.tanh)
            x = tf.layers.dense(x, hidden_sizes[1], activation=tf.nn.tanh)

            # 3. 策略头（输出动作概率）
            logits = tf.layers.dense(x, self.action_dim, activation=None)
            self.action_probs = tf.nn.softmax(logits)  # 动作概率分布

            # 4. 价值头（输出状态价值）
            self.value = tf.layers.dense(x, 1, activation=None)[:, 0]  # 标量价值

        # 5. 计算损失（A2C核心公式）
        # 优势值 A(s,a) = 折扣回报 - 旧价值（减少方差）
        adv = self.rewards_ph - self.old_values_ph

        # 策略损失：带优势加权的负对数似然（梯度上升→转化为损失下降）
        action_onehot = tf.one_hot(self.actions_ph, self.action_dim)
        neg_log_prob = -tf.reduce_sum(action_onehot * tf.log(self.action_probs + 1e-8), axis=1)
        policy_loss = tf.reduce_mean(adv * neg_log_prob)

        # 价值损失：MSE（价值函数拟合折扣回报）
        value_loss = tf.reduce_mean(tf.square(self.rewards_ph - self.value))

        # 熵奖励：鼓励探索（系数越小，探索越弱）
        entropy = -tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 1e-8), axis=1)
        entropy_loss = -tf.reduce_mean(entropy)  # 负号→转化为损失

        # 总损失（价值损失权重0.5，熵损失权重0.01为经验值）
        self.total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        # 6. 优化器（RMSProp为A2C常用优化器）
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99, epsilon=1e-5)
        self.train_op = self.optimizer.minimize(self.total_loss)

        # 7. 初始化会话
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs):
        """预测动作和价值（用于与环境交互）"""
        action_probs, value = self.sess.run(
            [self.action_probs, self.value],
            feed_dict={self.obs_ph: obs}
        )
        # 从概率分布中采样动作
        actions = [np.random.choice(self.action_dim, p=p) for p in action_probs]
        return np.array(actions), value

    def value(self, obs):
        """单独预测价值（用于Bootstrapping）"""
        return self.sess.run(self.value, feed_dict={self.obs_ph: obs})

    def train(self, obs, actions, old_values, rewards, current_lr):
        """训练模型（更新策略和价值网络）"""
        # 动态设置学习率
        self.optimizer._set_hyper("learning_rate", current_lr)
        
        loss, _ = self.sess.run(
            [self.total_loss, self.train_op],
            feed_dict={
                self.obs_ph: obs,
                self.actions_ph: actions,
                self.old_values_ph: old_values,
                self.rewards_ph: rewards
            }
        )
        return loss

    def save(self, path):
        """保存模型参数"""
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        """加载模型参数"""
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


# -------------------------- 4. 端到端训练流程 --------------------------
def train_a2c(env_id="CartPole-v1", total_timesteps=1e4, nsteps=5, gamma=0.99, save_path="./a2c_model"):
    # 1. 创建环境（用DummyVecEnv包装，支持并行，单环境也可）
    def make_env():
        env = gym.make(env_id)
        env = gym.wrappers.Monitor(env, "./logs", force=True)  # 记录回合信息
        return env
    env = DummyVecEnv([make_env])  # 1个并行环境

    # 2. 初始化模型和经验收集器
    model = A2CModel(
        obs_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=(64, 64),  # 2层64维隐藏层
        lr=7e-4
    )
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma)

    # 3. 训练配置
    nupdates = int(total_timesteps // (env.num_envs * nsteps))
    epinfobuf = deque(maxlen=100)  # 存储最近100个回合的信息
    tstart = time.time()

    # 4. 训练循环
    for update in range(1, nupdates + 1):
        # a. 收集n步经验
        obs, actions, old_values, rewards, epinfos = runner.run()
        epinfobuf.extend(epinfos)

        # b. 计算当前学习率（线性衰减）
        current_lr = linear_schedule(initial_value=7e-4, final_value=1e-5, total_steps=nupdates, current_step=update)

        # c. 训练模型
        loss = model.train(obs, actions, old_values, rewards, current_lr)

        # d. 打印训练日志（每10轮更新打印一次）
        if update % 10 == 0 or update == 1:
            total_steps = update * env.num_envs * nsteps
            fps = int(total_steps / (time.time() - tstart))
            print(f"更新轮数: {update}/{nupdates} | "
                  f"总步数: {total_steps} | "
                  f"FPS: {fps} | "
                  f"损失: {loss:.4f} | "
                  f"平均回合回报: {safemean([e['r'] for e in epinfobuf]):.2f} | "
                  f"平均回合长度: {safemean([e['l'] for e in epinfobuf]):.1f}")

    # 5. 保存模型
    model.save(save_path)
    print(f"\n训练完成！模型已保存至: {save_path}")
    env.close()
    return model


# -------------------------- 5. 模型推理测试流程 --------------------------
def test_a2c(env_id="CartPole-v1", model_path="./a2c_model", num_episodes=5):
    # 1. 创建测试环境
    env = gym.make(env_id)
    env = gym.wrappers.Monitor(env, "./test_logs", force=True, video_callable=lambda ep: True)  # 保存测试视频

    # 2. 加载模型
    model = A2CModel(
        obs_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=(64, 64)
    )
    model.load(model_path)
    print(f"模型加载完成！开始测试{num_episodes}个回合...")

    # 3. 测试循环
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            env.render()  # 可视化（关闭可加速）
            # 模型预测动作（测试时可用贪心策略，取概率最大的动作）
            action_probs, _ = model.sess.run(
                [model.action_probs, model.value],
                feed_dict={model.obs_ph: [obs]}
            )
            action = np.argmax(action_probs[0])  # 贪心动作

            # 执行动作
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            ep_length += 1

        print(f"回合 {ep}: 回报 = {ep_reward}, 长度 = {ep_length}")

    env.close()
    print("测试完成！视频已保存至./test_logs")


# -------------------------- 6. 主程序入口 --------------------------
if __name__ == "__main__":
    # 第一步：训练模型（约1分钟，CartPole-v1目标回报≥500）
    trained_model = train_a2c(total_timesteps=2e4)  # 2万步训练

    # 第二步：测试模型（加载训练好的模型，可视化执行）
    test_a2c(model_path="./a2c_model")