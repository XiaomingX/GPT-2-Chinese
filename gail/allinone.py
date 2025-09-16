import time
import numpy as np
import tensorflow as tf
import gym
from collections import deque


# -------------------------- 1. 核心辅助功能 --------------------------
def load_expert_data(expert_path):
    """加载专家数据（npz格式，包含obs和acs）"""
    data = np.load(expert_path)
    return {
        "obs": data["obs"].astype(np.float32),
        "acs": data["acs"].astype(np.float32)
    }


def split_train_val(data, val_ratio=0.1):
    """划分训练/验证集"""
    n = len(data["obs"])
    val_idx = int(n * val_ratio)
    train_data = {"obs": data["obs"][val_idx:], "acs": data["acs"][val_idx:]}
    val_data = {"obs": data["obs"][:val_idx], "acs": data["acs"][:val_idx]}
    return train_data, val_data


# -------------------------- 2. 策略网络（模仿者的"大脑"） --------------------------
class MlpPolicy:
    """多层感知机策略网络：输入状态，输出连续动作（高斯分布）"""
    def __init__(self, obs_dim, action_dim, hidden_size=64, name="policy"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.name = name

        # 构建网络
        with tf.variable_scope(self.name):
            # 输入：状态（[批量大小, 状态维度]）
            self.obs = tf.placeholder(tf.float32, [None, obs_dim], name="obs")
            
            # 特征提取：2层全连接网络
            x = tf.layers.dense(self.obs, hidden_size, activation=tf.nn.tanh)
            x = tf.layers.dense(x, hidden_size, activation=tf.nn.tanh)
            
            # 策略头：输出动作均值（高斯分布的均值）
            self.action_mean = tf.layers.dense(x, action_dim, activation=None)
            # 策略头：动作标准差（固定值，简化实现）
            self.action_logstd = tf.get_variable(
                "logstd", [action_dim], initializer=tf.zeros_initializer()
            )
            self.action_std = tf.exp(self.action_logstd)

        # 可训练参数
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        # 模型保存器
        self.saver = tf.train.Saver(self.trainable_vars)

    def sample_action(self, obs, stochastic=True):
        """采样动作（训练时随机，测试时取均值）"""
        mean, std = tf.get_default_session().run(
            [self.action_mean, self.action_std],
            feed_dict={self.obs: obs[None, :]}  # 扩展批量维度
        )
        if stochastic:
            return np.random.normal(mean, std).flatten()  # 加噪声探索
        else:
            return mean.flatten()  # 确定性动作

    def save(self, path):
        """保存模型参数"""
        self.saver.save(tf.get_default_session(), path)

    def load(self, path):
        """加载模型参数"""
        self.saver.restore(tf.get_default_session(), path)


# -------------------------- 3. 行为克隆（BC）：直接模仿专家动作 --------------------------
class BehaviorCloning:
    """行为克隆：监督学习方式，让策略网络拟合专家的"状态→动作"映射"""
    def __init__(self, policy, lr=3e-4):
        self.policy = policy

        # 输入占位符
        self.acs_target = tf.placeholder(tf.float32, [None, policy.action_dim], name="acs_target")

        # 损失：均方误差（策略输出动作 与 专家动作 的差距）
        self.loss = tf.reduce_mean(tf.square(self.policy.action_mean - self.acs_target))

        # 优化器
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=policy.trainable_vars)

    def train(self, train_data, val_data, batch_size=128, max_iter=10000, log_interval=1000):
        """训练BC模型"""
        sess = tf.get_default_session()
        n_train = len(train_data["obs"])
        train_losses = deque(maxlen=100)

        print("开始行为克隆（BC）训练...")
        start_time = time.time()

        for iter in range(1, max_iter + 1):
            # 随机采样批量数据
            idx = np.random.choice(n_train, batch_size, replace=False)
            obs_batch = train_data["obs"][idx]
            acs_batch = train_data["acs"][idx]

            # 计算损失并更新
            loss, _ = sess.run(
                [self.loss, self.optimizer],
                feed_dict={
                    self.policy.obs: obs_batch,
                    self.acs_target: acs_batch
                }
            )
            train_losses.append(loss)

            # 打印日志
            if iter % log_interval == 0:
                # 验证集损失
                val_loss = sess.run(
                    self.loss,
                    feed_dict={
                        self.policy.obs: val_data["obs"],
                        self.acs_target: val_data["acs"]
                    }
                )
                print(f"BC迭代 {iter:5d} | 训练损失: {np.mean(train_losses):.4f} | "
                      f"验证损失: {val_loss:.4f} | 耗时: {time.time()-start_time:.1f}s")

        print("BC训练完成！")


# -------------------------- 4. GAIL判别器：区分专家与模仿者的行为 --------------------------
class GAILDiscriminator:
    """判别器：二分类网络，判断轨迹是"专家生成"还是"模仿者生成"，并输出奖励"""
    def __init__(self, obs_dim, action_dim, hidden_size=64, lr=3e-4):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 输入占位符：专家数据和模仿者数据
        self.expert_obs = tf.placeholder(tf.float32, [None, obs_dim], name="expert_obs")
        self.expert_acs = tf.placeholder(tf.float32, [None, action_dim], name="expert_acs")
        self.imitator_obs = tf.placeholder(tf.float32, [None, obs_dim], name="imitator_obs")
        self.imitator_acs = tf.placeholder(tf.float32, [None, action_dim], name="imitator_acs")

        # 构建共享网络（输入：状态+动作）
        def build_network(obs, acs, reuse):
            with tf.variable_scope("discriminator", reuse=reuse):
                x = tf.concat([obs, acs], axis=1)  # 拼接状态和动作
                x = tf.layers.dense(x, hidden_size, activation=tf.nn.tanh)
                x = tf.layers.dense(x, hidden_size, activation=tf.nn.tanh)
                logits = tf.layers.dense(x, 1, activation=None)  # 输出未激活的logits
                return logits

        # 专家和模仿者的logits
        expert_logits = build_network(self.expert_obs, self.expert_acs, reuse=False)
        imitator_logits = build_network(self.imitator_obs, self.imitator_acs, reuse=True)

        # 损失：二分类交叉熵（专家标为1，模仿者标为0）
        expert_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=expert_logits, labels=tf.ones_like(expert_logits)
        ))
        imitator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=imitator_logits, labels=tf.zeros_like(imitator_logits)
        ))
        self.total_loss = expert_loss + imitator_loss

        # 优化器
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.total_loss)

        # 模仿者的奖励：-log(1 - sigmoid(logits))（越像专家奖励越高）
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(imitator_logits) + 1e-8)

    def train_step(self, expert_obs, expert_acs, imitator_obs, imitator_acs):
        """训练判别器（一次迭代）"""
        sess = tf.get_default_session()
        loss, _ = sess.run(
            [self.total_loss, self.optimizer],
            feed_dict={
                self.expert_obs: expert_obs,
                self.expert_acs: expert_acs,
                self.imitator_obs: imitator_obs,
                self.imitator_acs: imitator_acs
            }
        )
        return loss

    def get_reward(self, obs, acs):
        """计算模仿者的奖励"""
        sess = tf.get_default_session()
        return sess.run(
            self.reward_op,
            feed_dict={self.imitator_obs: obs, self.imitator_acs: acs}
        ).flatten()


# -------------------------- 5. GAIL核心训练：对抗式模仿 --------------------------
def train_gail(env, policy, discriminator, expert_data, batch_size=128, 
               max_iter=1000, g_step=3, d_step=1, log_interval=10):
    """
    GAIL训练流程：交替更新模仿者（生成器G）和判别器（D）
    g_step: 每轮更新G的次数
    d_step: 每轮更新D的次数
    """
    sess = tf.get_default_session()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    n_expert = len(expert_data["obs"])

    # 模仿者损失：最大化判别器给出的奖励（用策略梯度）
    action = tf.placeholder(tf.float32, [None, action_dim], name="action")
    reward = tf.placeholder(tf.float32, [None], name="reward")
    
    # 策略梯度：用奖励加权的负对数概率（鼓励高奖励动作）
    log_prob = -0.5 * tf.reduce_sum(
        tf.square((action - policy.action_mean) / policy.action_std) + 
        2 * policy.action_logstd + np.log(2 * np.pi),
        axis=1
    )
    policy_loss = -tf.reduce_mean(reward * log_prob)
    policy_optimizer = tf.train.AdamOptimizer(1e-4).minimize(policy_loss, var_list=policy.trainable_vars)

    # 存储模仿者轨迹的缓冲区
    imitator_buffer = deque(maxlen=10000)
    print("开始GAIL训练...")
    start_time = time.time()

    for iter in range(1, max_iter + 1):
        # -------------------------- 1. 收集模仿者轨迹（用于训练D和G） --------------------------
        obs = env.reset()
        ep_reward = 0.0
        ep_trajectory = []

        while True:
            action = policy.sample_action(obs, stochastic=True)
            next_obs, _, done, _ = env.step(action)
            ep_trajectory.append((obs, action))
            obs = next_obs
            ep_reward += discriminator.get_reward(obs[None, :], action[None, :])[0]

            if done:
                imitator_buffer.extend(ep_trajectory)
                break

        # -------------------------- 2. 更新判别器D --------------------------
        d_losses = []
        for _ in range(d_step):
            # 采样模仿者数据
            if len(imitator_buffer) < batch_size:
                continue
            imitator_idx = np.random.choice(len(imitator_buffer), batch_size, replace=False)
            imitator_obs = np.array([imitator_buffer[i][0] for i in imitator_idx])
            imitator_acs = np.array([imitator_buffer[i][1] for i in imitator_idx])

            # 采样专家数据
            expert_idx = np.random.choice(n_expert, batch_size, replace=False)
            expert_obs = expert_data["obs"][expert_idx]
            expert_acs = expert_data["acs"][expert_idx]

            # 训练D
            d_loss = discriminator.train_step(expert_obs, expert_acs, imitator_obs, imitator_acs)
            d_losses.append(d_loss)

        # -------------------------- 3. 更新模仿者G --------------------------
        g_losses = []
        for _ in range(g_step):
            if len(imitator_buffer) < batch_size:
                continue
            # 采样模仿者数据和对应的奖励
            imitator_idx = np.random.choice(len(imitator_buffer), batch_size, replace=False)
            imitator_obs = np.array([imitator_buffer[i][0] for i in imitator_idx])
            imitator_acs = np.array([imitator_buffer[i][1] for i in imitator_idx])
            imitator_rewards = discriminator.get_reward(imitator_obs, imitator_acs)

            # 训练G
            g_loss, _ = sess.run(
                [policy_loss, policy_optimizer],
                feed_dict={
                    policy.obs: imitator_obs,
                    action: imitator_acs,
                    reward: imitator_rewards
                }
            )
            g_losses.append(g_loss)

        # -------------------------- 4. 打印日志 --------------------------
        if iter % log_interval == 0:
            avg_d_loss = np.mean(d_losses) if d_losses else 0.0
            avg_g_loss = np.mean(g_losses) if g_losses else 0.0
            print(f"GAIL迭代 {iter:4d} | D损失: {avg_d_loss:.4f} | G损失: {avg_g_loss:.4f} | "
                  f"单回合奖励: {ep_reward:.2f} | 耗时: {time.time()-start_time:.1f}s")

    print("GAIL训练完成！")


# -------------------------- 6. 模型测试：验证模仿效果 --------------------------
def test_policy(env, policy, num_episodes=5, render=True):
    """测试训练好的策略"""
    print(f"\n开始测试{num_episodes}个回合...")
    total_rewards = []
    total_steps = []

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_step = 0

        while True:
            if render:
                env.render()  # 可视化
            action = policy.sample_action(obs, stochastic=False)  # 确定性动作
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            ep_step += 1

            if done:
                total_rewards.append(ep_reward)
                total_steps.append(ep_step)
                print(f"回合 {ep}: 奖励 = {ep_reward:.1f}, 步数 = {ep_step}")
                break

    env.close()
    print(f"\n测试总结：平均奖励 = {np.mean(total_rewards):.1f}, 平均步数 = {np.mean(total_steps):.1f}")
    return np.mean(total_rewards)


# -------------------------- 7. 端到端主流程 --------------------------
def main():
    # 配置（根据需要修改）
    env_id = "Hopper-v2"  #  mujoco环境（需安装mujoco-py），简单替代可用"Pendulum-v1"
    expert_data_path = "expert_data.npz"  # 专家数据路径（格式：包含obs和acs数组）
    bc_model_path = "./bc_policy"
    gail_model_path = "./gail_policy"
    bc_max_iter = 10000
    gail_max_iter = 500

    # 1. 准备环境和专家数据
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 加载专家数据（需提前准备，格式参考：https://github.com/openai/imitation/tree/master/data）
    # 若没有专家数据，可先运行BC/GAIL的简化版，或用"Pendulum-v1"手动收集
    try:
        expert_data = load_expert_data(expert_data_path)
        train_data, val_data = split_train_val(expert_data)
        print(f"加载专家数据成功：共{len(expert_data['obs'])}条轨迹")
    except FileNotFoundError:
        print(f"警告：未找到专家数据{expert_data_path}，请先准备数据！")
        return

    # 2. 初始化TensorFlow会话
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # -------------------------- 选项1：行为克隆（BC）训练与测试 --------------------------
    print("\n=== 行为克隆（BC）流程 ===")
    bc_policy = MlpPolicy(obs_dim, action_dim, hidden_size=64, name="bc_policy")
    bc_trainer = BehaviorCloning(bc_policy, lr=3e-4)
    bc_trainer.train(train_data, val_data, max_iter=bc_max_iter)
    bc_policy.save(bc_model_path)

    # 测试BC模型
    print("\n=== 测试BC模型 ===")
    bc_policy.load(bc_model_path)
    test_policy(env, bc_policy, num_episodes=5)

    # -------------------------- 选项2：GAIL训练与测试（可基于BC预训练） --------------------------
    print("\n=== GAIL流程 ===")
    # 初始化GAIL策略（可加载BC预训练权重加速）
    gail_policy = MlpPolicy(obs_dim, action_dim, hidden_size=64, name="gail_policy")
    # gail_policy.load(bc_model_path)  # 可选：加载BC预训练权重

    # 初始化判别器
    discriminator = GAILDiscriminator(obs_dim, action_dim, hidden_size=64)
    sess.run(tf.variables_initializer(discriminator.optimizer.variables()))  # 单独初始化判别器参数

    # 训练GAIL
    train_gail(env, gail_policy, discriminator, expert_data, max_iter=gail_max_iter)
    gail_policy.save(gail_model_path)

    # 测试GAIL模型
    print("\n=== 测试GAIL模型 ===")
    gail_policy.load(gail_model_path)
    test_policy(env, gail_policy, num_episodes=5)

    sess.close()


if __name__ == "__main__":
    # 注意：需安装依赖：pip install tensorflow==1.15 gym[classic_control] mujoco-py（可选）
    # 专家数据获取：参考https://github.com/openai/imitation/tree/master/data，或用代码生成
    main()