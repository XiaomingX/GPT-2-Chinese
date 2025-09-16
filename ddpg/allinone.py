import time
import numpy as np
import tensorflow as tf
import gym
from collections import deque


# -------------------------- 1. 核心辅助函数（简化版） --------------------------
class RunningMeanStd:
    """简化的均值标准差统计（用于状态归一化）"""
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
        self.count = 1e-4  # 避免除以零

    def update(self, x):
        """更新均值和标准差"""
        batch_mean = np.mean(x, axis=0)
        batch_std = np.std(x, axis=0)
        batch_count = x.shape[0]
        
        # 滑动平均更新
        self.mean = (self.count * self.mean + batch_count * batch_mean) / (self.count + batch_count)
        self.std = np.sqrt((self.count * self.std**2 + batch_count * batch_std**2) / (self.count + batch_count))
        self.count += batch_count


def normalize(x, stats):
    """状态归一化"""
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)  # 加小值避免除零


def get_target_updates(source_vars, target_vars, tau):
    """目标网络软更新（核心：缓慢跟进主网络参数）"""
    soft_updates = []
    init_updates = []  # 初始化：目标网络=主网络
    for var, target_var in zip(source_vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1 - tau) * target_var + tau * var))
    return tf.group(*init_updates), tf.group(*soft_updates)


# -------------------------- 2. 经验回放池（简化版） --------------------------
class Memory:
    """存储交互经验，用于随机采样打破数据相关性"""
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []  # 存储格式：(obs0, action, reward, obs1, done)
        self.pos = 0

    def append(self, obs0, action, reward, obs1, done):
        """添加一条经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs0, action, reward, obs1, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样一批经验（返回字典格式）"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs0, action, reward, obs1, done = zip(*[self.buffer[i] for i in batch])
        return {
            "obs0": np.array(obs0, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "reward": np.array(reward, dtype=np.float32)[:, None],  # 扩展为列向量
            "obs1": np.array(obs1, dtype=np.float32),
            "done": np.array(done, dtype=np.float32)[:, None]  # done=1表示回合结束
        }

    @property
    def size(self):
        return len(self.buffer)


# -------------------------- 3. 动作噪声（DDPG必备，鼓励探索） --------------------------
class OrnsteinUhlenbeckNoise:
    """OU噪声：连续动作空间常用的探索噪声（有时间相关性）"""
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.x = np.ones(self.action_dim) * self.mu

    def __call__(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.action_dim)
        self.x += dx
        return self.x


# -------------------------- 4. 核心网络（Actor + Critic） --------------------------
class Actor:
    """策略网络（Actor）：输入状态，输出确定性动作（连续值）"""
    def __init__(self, obs_dim, action_dim, action_bound, name="actor"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound  # 动作边界（如[-1,1]）
        self.name = name

        # 构建网络
        with tf.variable_scope(self.name):
            self.obs = tf.placeholder(tf.float32, [None, obs_dim], name="obs")  # 输入：状态
            # 全连接网络：64→64→动作维度
            x = tf.layers.dense(self.obs, 64, activation=tf.nn.relu)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            # 输出动作（用tanh缩放到[-1,1]，再乘动作边界）
            self.action = tf.layers.dense(x, action_dim, activation=tf.nn.tanh) * action_bound

        # 可训练参数
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Critic:
    """价值网络（Critic）：输入状态+动作，输出Q值（状态-动作价值）"""
    def __init__(self, obs_dim, action_dim, name="critic"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.name = name

        # 构建网络
        with tf.variable_scope(self.name):
            self.obs = tf.placeholder(tf.float32, [None, obs_dim], name="obs")  # 输入1：状态
            self.action = tf.placeholder(tf.float32, [None, action_dim], name="action")  # 输入2：动作
            # 拼接状态和动作，再过全连接网络
            x = tf.concat([self.obs, self.action], axis=1)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            self.q_value = tf.layers.dense(x, 1, activation=None)  # 输出：Q值（标量）

        # 可训练参数
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


# -------------------------- 5. DDPG核心逻辑（整合Actor-Critic + 目标网络） --------------------------
class DDPG:
    def __init__(self, obs_dim, action_dim, action_bound, gamma=0.99, tau=0.001, lr_actor=1e-4, lr_critic=1e-3):
        # 1. 超参数
        self.gamma = gamma  # 折扣因子（未来奖励的权重）
        self.tau = tau      # 目标网络软更新系数（越小越稳定）

        # 2. 构建主网络和目标网络
        self.actor = Actor(obs_dim, action_dim, action_bound, name="actor")
        self.critic = Critic(obs_dim, action_dim, name="critic")
        self.target_actor = Actor(obs_dim, action_dim, action_bound, name="target_actor")  # 目标Actor
        self.target_critic = Critic(obs_dim, action_dim, name="target_critic")  # 目标Critic

        # 3. 目标网络更新操作（初始化+软更新）
        self.init_target_ops, self.update_target_ops = get_target_updates(
            source_vars=self.actor.trainable_vars + self.critic.trainable_vars,
            target_vars=self.target_actor.trainable_vars + self.target_critic.trainable_vars,
            tau=self.tau
        )

        # 4.  Critic损失（拟合贝尔曼方程）
        self.q_target = tf.placeholder(tf.float32, [None, 1], name="q_target")  # 目标Q值
        self.critic_loss = tf.reduce_mean(tf.square(self.q_target - self.critic.q_value))  # MSE损失

        # 5.  Actor损失（最大化Q值）
        self.actor_loss = -tf.reduce_mean(self.critic.q_value)  # 负号：梯度上升→损失下降

        # 6. 优化器
        self.opt_actor = tf.train.AdamOptimizer(lr_actor).minimize(self.actor_loss, var_list=self.actor.trainable_vars)
        self.opt_critic = tf.train.AdamOptimizer(lr_critic).minimize(self.critic_loss, var_list=self.critic.trainable_vars)

        # 7. 状态归一化（可选，加速训练）
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))

        # 8. TensorFlow会话
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.init_target_ops)  # 初始化：目标网络=主网络

        # 9. 模型保存器
        self.saver = tf.train.Saver()

    def select_action(self, obs, add_noise=True, noise=None):
        """选择动作（训练时加噪声探索，测试时不加）"""
        # 状态归一化
        obs_normalized = normalize(obs[None, :], self.obs_rms)  # 扩展为batch维度
        # 主Actor输出动作
        action = self.sess.run(self.actor.action, feed_dict={self.actor.obs: obs_normalized})[0]  # 去掉batch维度
        # 加噪声探索
        if add_noise and noise is not None:
            action += noise()
            # 裁剪动作到合法范围
            action = np.clip(action, -self.actor.action_bound, self.actor.action_bound)
        return action

    def update(self, memory, batch_size=64):
        """用经验回放更新网络"""
        # 1. 采样一批经验
        batch = memory.sample(batch_size)
        obs0, action, reward, obs1, done = batch["obs0"], batch["action"], batch["reward"], batch["obs1"], batch["done"]

        # 2. 归一化状态
        obs0_normalized = normalize(obs0, self.obs_rms)
        obs1_normalized = normalize(obs1, self.obs_rms)

        # 3. 计算目标Q值（贝尔曼方程）
        # 目标Actor输出obs1的动作 → 目标Critic计算Q值
        action1_target = self.sess.run(self.target_actor.action, feed_dict={self.target_actor.obs: obs1_normalized})
        q1_target = self.sess.run(self.target_critic.q_value, feed_dict={
            self.target_critic.obs: obs1_normalized,
            self.target_critic.action: action1_target
        })
        # 目标Q值 = 即时奖励 + 折扣*未来Q值（回合结束时未来Q值为0）
        q_target = reward + self.gamma * (1 - done) * q1_target

        # 4. 更新Critic
        _, critic_loss = self.sess.run([self.opt_critic, self.critic_loss], feed_dict={
            self.critic.obs: obs0_normalized,
            self.critic.action: action,
            self.q_target: q_target
        })

        # 5. 更新Actor（固定Critic，最大化Q值）
        _, actor_loss = self.sess.run([self.opt_actor, self.actor_loss], feed_dict={
            self.actor.obs: obs0_normalized,
            self.critic.obs: obs0_normalized,  # Critic需要相同的状态
            self.critic.action: self.sess.run(self.actor.action, feed_dict={self.actor.obs: obs0_normalized})  # Actor的动作
        })

        # 6. 软更新目标网络
        self.sess.run(self.update_target_ops)

        return critic_loss, actor_loss

    def save_model(self, path="./ddpg_model"):
        """保存模型"""
        self.saver.save(self.sess, path)
        print(f"模型已保存至: {path}")

    def load_model(self, path="./ddpg_model"):
        """加载模型"""
        self.saver.restore(self.sess, path)
        print(f"模型已从: {path} 加载")


# -------------------------- 6. 端到端训练流程 --------------------------
def train_ddpg(env_id="Pendulum-v1", total_episodes=200, batch_size=64, save_path="./ddpg_model"):
    # 1. 创建环境（Pendulum-v1是连续动作空间，适合DDPG）
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作边界（Pendulum的动作范围是[-2,2]）

    # 2. 初始化DDPG、经验池、动作噪声
    agent = DDPG(obs_dim, action_dim, action_bound)
    memory = Memory(capacity=100000)
    noise = OrnsteinUhlenbeckNoise(action_dim)

    # 3. 记录训练信息
    episode_rewards = deque(maxlen=10)  # 记录最近10个回合的奖励
    start_time = time.time()

    # 4. 训练循环
    for episode in range(1, total_episodes + 1):
        obs = env.reset()
        noise.reset()  # 每个回合重置噪声
        ep_reward = 0.0
        ep_step = 0

        while True:
            # a. 选择动作（加噪声探索）
            action = agent.select_action(obs, add_noise=True, noise=noise)
            # b. 执行动作
            next_obs, reward, done, _ = env.step(action)
            # c. 存储经验
            memory.append(obs, action, reward, next_obs, done)
            # d. 更新状态和累计奖励
            obs = next_obs
            ep_reward += reward
            ep_step += 1

            # e. 经验池满了之后才开始训练
            if memory.size > 1000:  # 先存1000条经验再训练
                critic_loss, actor_loss = agent.update(memory, batch_size)

            # f. 回合结束
            if done:
                episode_rewards.append(ep_reward)
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
                # 打印进度（每10回合详细打印）
                if episode % 10 == 0:
                    print(f"回合: {episode:3d} | 步数: {ep_step:3d} | 奖励: {ep_reward:6.1f} | "
                          f"平均奖励: {avg_reward:6.1f} | 耗时: {time.time()-start_time:.1f}s")
                break

    # 5. 训练完成，保存模型
    agent.save_model(save_path)
    env.close()
    return agent


# -------------------------- 7. 模型测试流程（加载+运行） --------------------------
def test_ddpg(env_id="Pendulum-v1", model_path="./ddpg_model", num_episodes=5):
    # 1. 创建测试环境
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # 2. 初始化并加载模型
    agent = DDPG(obs_dim, action_dim, action_bound)
    agent.load_model(model_path)

    # 3. 测试循环
    print(f"\n开始测试{num_episodes}个回合（可视化开启）...")
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_step = 0

        while True:
            env.render()  # 可视化动作
            # 选择动作（不加噪声，用纯策略）
            action = agent.select_action(obs, add_noise=False)
            # 执行动作
            next_obs, reward, done, _ = env.step(action)
            # 更新状态和奖励
            obs = next_obs
            ep_reward += reward
            ep_step += 1

            if done:
                print(f"测试回合 {episode}: 奖励 = {ep_reward:6.1f}, 步数 = {ep_step}")
                break

    env.close()
    print("测试完成！")


# -------------------------- 8. 主程序入口（端到端执行） --------------------------
if __name__ == "__main__":
    # 第一步：训练模型（约5分钟，Pendulum-v1目标：平均奖励≥-2000）
    trained_agent = train_ddpg(total_episodes=200)

    # 第二步：测试模型（加载训练好的模型，可视化运行）
    test_ddpg(model_path="./ddpg_model", num_episodes=5)