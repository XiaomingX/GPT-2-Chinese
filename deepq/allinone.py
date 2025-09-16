import time
import random
import numpy as np
import tensorflow as tf
import gym
from collections import deque


# -------------------------- 1. 经验回放池（核心组件） --------------------------
class ReplayBuffer:
    """存储交互经验，打破时序相关性，稳定训练"""
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []  # 存储格式：(obs0, action, reward, obs1, done)
        self.pos = 0

    def add(self, obs0, action, reward, obs1, done):
        """添加一条经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs0, action, reward, obs1, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        obs0, action, reward, obs1, done = zip(*batch)
        return (
            np.array(obs0, dtype=np.float32),
            np.array(action, dtype=np.int32),
            np.array(reward, dtype=np.float32)[:, None],  # 扩展为列向量
            np.array(obs1, dtype=np.float32),
            np.array(done, dtype=np.float32)[:, None]     # done=1表示回合结束
        )

    @property
    def size(self):
        return len(self.buffer)


# -------------------------- 2. DQN核心类（整合Q网络+训练逻辑） --------------------------
class DQN:
    def __init__(self, obs_dim, action_dim, lr=5e-4, gamma=0.99, epsilon_init=1.0, epsilon_min=0.02, epsilon_decay=0.995):
        # 超参数
        self.obs_dim = obs_dim          # 状态维度
        self.action_dim = action_dim    # 动作数量（离散）
        self.lr = lr                    # 学习率
        self.gamma = gamma              # 折扣因子（未来奖励权重）
        self.epsilon = epsilon_init     # ε-贪心初始值（探索概率）
        self.epsilon_min = epsilon_min  # ε最小阈值
        self.epsilon_decay = epsilon_decay  # ε衰减系数

        # 1. 构建Q网络（主网络）和目标网络
        self.q_network = self.build_network("q_network")
        self.target_q_network = self.build_network("target_q_network")

        # 2. 目标网络参数更新操作
        self.update_target_op = self.get_target_update_op()

        # 3. 损失函数与优化器（拟合贝尔曼方程）
        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim], name="obs")
        self.action_ph = tf.placeholder(tf.int32, [None], name="action")
        self.reward_ph = tf.placeholder(tf.float32, [None, 1], name="reward")
        self.next_obs_ph = tf.placeholder(tf.float32, [None, obs_dim], name="next_obs")
        self.done_ph = tf.placeholder(tf.float32, [None, 1], name="done")

        # 主网络当前Q值（只取选中动作的Q值）
        q_values = self.q_network(self.obs_ph)
        action_onehot = tf.one_hot(self.action_ph, action_dim)
        self.current_q = tf.reduce_sum(q_values * action_onehot, axis=1, keepdims=True)

        # 目标网络下一状态的最大Q值（done时为0，因为没有未来奖励）
        next_q_values = self.target_q_network(self.next_obs_ph)
        self.next_max_q = tf.reduce_max(next_q_values, axis=1, keepdims=True)
        self.target_q = self.reward_ph + self.gamma * (1 - self.done_ph) * self.next_max_q

        # 均方误差损失（MSE）
        self.loss = tf.reduce_mean(tf.square(self.current_q - tf.stop_gradient(self.target_q)))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # 4. 初始化会话与模型保存器
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_network(self, scope):
        """构建Q网络（输入状态，输出每个动作的Q值）"""
        def network(obs):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                # 简单全连接网络：64→64→动作数
                x = tf.layers.dense(obs, 64, activation=tf.nn.relu)
                x = tf.layers.dense(x, 64, activation=tf.nn.relu)
                q_values = tf.layers.dense(x, self.action_dim, activation=None)  # 输出每个动作的Q值
            return q_values
        return network

    def get_target_update_op(self):
        """目标网络硬更新（直接复制主网络参数）"""
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
        target_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_q_network")
        return [tf.assign(t_var, q_var) for t_var, q_var in zip(target_q_vars, q_vars)]

    def select_action(self, obs, is_training=True):
        """ε-贪心选动作（训练时探索，测试时贪心）"""
        if is_training and random.random() < self.epsilon:
            # 探索：随机选动作
            return random.randint(0, self.action_dim - 1)
        else:
            # 贪心：选Q值最大的动作
            q_values = self.sess.run(self.q_network(self.obs_ph), feed_dict={self.obs_ph: [obs]})
            return np.argmax(q_values[0])

    def decay_epsilon(self):
        """ε衰减（训练越久，探索越少）"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, obs0, action, reward, obs1, done):
        """用经验回放数据训练Q网络"""
        loss, _ = self.sess.run(
            [self.loss, self.optimizer],
            feed_dict={
                self.obs_ph: obs0,
                self.action_ph: action,
                self.reward_ph: reward,
                self.next_obs_ph: obs1,
                self.done_ph: done
            }
        )
        return loss

    def update_target_network(self):
        """更新目标网络（定期同步主网络参数）"""
        self.sess.run(self.update_target_op)

    def save_model(self, path="./dqn_model"):
        """保存模型参数"""
        self.saver.save(self.sess, path)
        print(f"模型已保存至: {path}")

    def load_model(self, path="./dqn_model"):
        """加载模型参数"""
        self.saver.restore(self.sess, path)
        print(f"模型已从: {path} 加载")


# -------------------------- 3. 端到端训练流程 --------------------------
def train_dqn(env_id="CartPole-v1", total_episodes=200, batch_size=32, target_update_freq=20, save_path="./dqn_model"):
    # 1. 创建环境（CartPole是离散动作空间，适合DQN）
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 2. 初始化DQN智能体和经验回放池
    agent = DQN(obs_dim, action_dim)
    replay_buffer = ReplayBuffer(capacity=100000)

    # 3. 训练日志记录
    episode_rewards = deque(maxlen=10)  # 最近10回合奖励
    start_time = time.time()

    # 4. 训练循环（按回合迭代）
    for episode in range(1, total_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_step = 0

        while True:
            # a. 选动作（训练时带ε-贪心探索）
            action = agent.select_action(obs, is_training=True)
            # b. 执行动作，与环境交互
            next_obs, reward, done, _ = env.step(action)
            # c. 存储经验到回放池
            replay_buffer.add(obs, action, reward, next_obs, done)
            # d. 更新状态和累计奖励
            obs = next_obs
            ep_reward += reward
            ep_step += 1

            # e. 回放池满1000条后开始训练
            if replay_buffer.size > 1000:
                # 采样一批经验
                obs0, action_batch, reward_batch, obs1, done_batch = replay_buffer.sample(batch_size)
                # 训练主网络
                loss = agent.train(obs0, action_batch, reward_batch, obs1, done_batch)
                # ε衰减
                agent.decay_epsilon()

            # f. 回合结束
            if done:
                episode_rewards.append(ep_reward)
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
                # 每10回合打印进度
                if episode % 10 == 0:
                    print(f"回合: {episode:3d} | 步数: {ep_step:3d} | 奖励: {ep_reward:6.1f} | "
                          f"平均奖励: {avg_reward:6.1f} | ε: {agent.epsilon:.3f} | 耗时: {time.time()-start_time:.1f}s")
                break

        # g. 定期更新目标网络（每20回合一次）
        if episode % target_update_freq == 0:
            agent.update_target_network()

    # 5. 训练完成，保存模型
    agent.save_model(save_path)
    env.close()
    return agent


# -------------------------- 4. 模型测试流程（加载+可视化） --------------------------
def test_dqn(env_id="CartPole-v1", model_path="./dqn_model", num_episodes=5):
    # 1. 创建测试环境
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 2. 初始化并加载模型
    agent = DQN(obs_dim, action_dim)
    agent.load_model(model_path)
    agent.epsilon = 0.0  # 测试时禁用探索，纯贪心选动作

    # 3. 测试循环
    print(f"\n开始测试{num_episodes}个回合（可视化开启）...")
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_step = 0

        while True:
            env.render()  # 可视化动作（关闭可加速）
            # 选动作（纯贪心）
            action = agent.select_action(obs, is_training=False)
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


# -------------------------- 5. 主程序入口（训练→测试） --------------------------
if __name__ == "__main__":
    # 第一步：训练模型（CartPole目标：平均奖励≥500）
    trained_agent = train_dqn(total_episodes=200, target_update_freq=20)

    # 第二步：测试模型（加载训练好的参数，可视化运行）
    test_dqn(model_path="./dqn_model", num_episodes=5)