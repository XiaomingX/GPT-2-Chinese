import time
import numpy as np
import tensorflow as tf
import gym
from collections import deque


# -------------------------- 1. 核心辅助工具 --------------------------
class Normalizer:
    """简化的状态/目标归一化器（加速训练）"""
    def __init__(self, shape, eps=1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
        self.eps = eps
        self.count = 1e-4  # 避免除以零

    def update(self, x):
        """滑动更新均值和标准差"""
        batch_mean = np.mean(x, axis=0)
        batch_std = np.std(x, axis=0)
        batch_count = x.shape[0]
        
        self.mean = (self.count * self.mean + batch_count * batch_mean) / (self.count + batch_count)
        self.std = np.sqrt((self.count * self.std**2 + batch_count * batch_std**2) / (self.count + batch_count))
        self.count += batch_count

    def normalize(self, x):
        """归一化数据"""
        return (x - self.mean) / (self.std + self.eps)


def her_sample_transitions(episode, replay_k=4):
    """
    HER核心：改写历史经验的目标（后见之明）
    输入：单条轨迹（episode）
    输出：包含HER改写的经验
    """
    T = len(episode["u"])  # 轨迹长度
    transitions = []

    # 原始经验（用原始目标）
    for t in range(T):
        transitions.append({
            "o": episode["o"][t],
            "g": episode["g"][t],
            "u": episode["u"][t],
            "o2": episode["o"][t+1],
            "g2": episode["g"][t+1],
            "r": episode["r"][t]
        })

    # HER改写经验（用未来达成的目标）
    future_p = 1 - (1. / (1 + replay_k))  # HER样本比例
    for t in range(T):
        if np.random.uniform() < future_p:
            # 随机选择未来时刻的达成目标作为新目标
            future_t = np.random.randint(t+1, T+1)
            new_g = episode["ag"][future_t]  # 未来时刻的达成目标
            
            # 重新计算奖励（是否达成新目标）
            new_r = compute_reward(episode["ag"][t+1], new_g, None)
            
            # 添加改写后的经验
            transitions.append({
                "o": episode["o"][t],
                "g": new_g,
                "u": episode["u"][t],
                "o2": episode["o"][t+1],
                "g2": new_g,  # 目标不变
                "r": new_r
            })

    return transitions


def compute_reward(ag, g, info):
    """目标任务奖励函数（以FetchReach为例：距离越小奖励越高）"""
    distance = np.linalg.norm(ag - g, axis=-1)
    return -(distance > 0.05).astype(np.float32)  # 距离<0.05视为成功，奖励0；否则奖励-1


# -------------------------- 2. 经验回放池（支持HER） --------------------------
class HERReplayBuffer:
    """存储轨迹经验，支持HER改写和随机采样"""
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def store_episode(self, episode):
        """存储一条轨迹，并通过HER生成额外经验"""
        # 用HER改写轨迹
        her_transitions = her_sample_transitions(episode)
        
        # 存入缓冲区
        for trans in her_transitions:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.pos] = trans
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        return {
            "o": np.array([self.buffer[i]["o"] for i in batch], dtype=np.float32),
            "g": np.array([self.buffer[i]["g"] for i in batch], dtype=np.float32),
            "u": np.array([self.buffer[i]["u"] for i in batch], dtype=np.float32),
            "o2": np.array([self.buffer[i]["o2"] for i in batch], dtype=np.float32),
            "g2": np.array([self.buffer[i]["g2"] for i in batch], dtype=np.float32),
            "r": np.array([self.buffer[i]["r"] for i in batch], dtype=np.float32)[:, None]
        }

    @property
    def size(self):
        return len(self.buffer)


# -------------------------- 3. 核心网络（Actor + Critic，支持目标输入） --------------------------
class Actor:
    """策略网络：输入（状态+目标），输出连续动作"""
    def __init__(self, obs_dim, goal_dim, action_dim, max_u, hidden_size=64, name="actor"):
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_u = max_u  # 动作边界（如[-1,1]）
        self.name = name

        with tf.variable_scope(self.name):
            # 输入：状态（obs）+ 目标（goal）
            self.obs = tf.placeholder(tf.float32, [None, obs_dim], name="obs")
            self.goal = tf.placeholder(tf.float32, [None, goal_dim], name="goal")
            x = tf.concat([self.obs, self.goal], axis=1)  # 拼接状态和目标

            # 全连接网络
            x = tf.layers.dense(x, hidden_size, activation=tf.nn.relu)
            x = tf.layers.dense(x, hidden_size, activation=tf.nn.relu)
            self.action = self.max_u * tf.layers.dense(x, action_dim, activation=tf.tanh)

        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Critic:
    """价值网络：输入（状态+目标+动作），输出Q值"""
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_size=64, name="critic"):
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.name = name

        with tf.variable_scope(self.name):
            # 输入：状态+目标+动作
            self.obs = tf.placeholder(tf.float32, [None, obs_dim], name="obs")
            self.goal = tf.placeholder(tf.float32, [None, goal_dim], name="goal")
            self.action = tf.placeholder(tf.float32, [None, action_dim], name="action")
            x = tf.concat([self.obs, self.goal, self.action], axis=1)  # 拼接三者

            # 全连接网络
            x = tf.layers.dense(x, hidden_size, activation=tf.nn.relu)
            x = tf.layers.dense(x, hidden_size, activation=tf.nn.relu)
            self.q_value = tf.layers.dense(x, 1, activation=None)  # 输出Q值

        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


# -------------------------- 4. HER-DDPG核心逻辑 --------------------------
class HERDDPG:
    def __init__(self, obs_dim, goal_dim, action_dim, max_u=1.0, gamma=0.98, tau=0.005, lr_actor=1e-4, lr_critic=1e-3):
        # 超参数
        self.gamma = gamma  # 折扣因子
        self.tau = tau      # 目标网络软更新系数
        self.max_u = max_u  # 动作边界

        # 1. 构建主网络和目标网络
        self.actor = Actor(obs_dim, goal_dim, action_dim, max_u, name="actor")
        self.critic = Critic(obs_dim, goal_dim, action_dim, name="critic")
        self.target_actor = Actor(obs_dim, goal_dim, action_dim, max_u, name="target_actor")
        self.target_critic = Critic(obs_dim, goal_dim, action_dim, name="target_critic")

        # 2. 目标网络更新操作（初始化+软更新）
        self.init_target_ops = [
            tf.assign(t_var, var) for t_var, var in zip(
                self.target_actor.trainable_vars + self.target_critic.trainable_vars,
                self.actor.trainable_vars + self.critic.trainable_vars
            )
        ]
        self.update_target_ops = [
            tf.assign(t_var, (1 - self.tau) * t_var + self.tau * var) for t_var, var in zip(
                self.target_actor.trainable_vars + self.target_critic.trainable_vars,
                self.actor.trainable_vars + self.critic.trainable_vars
            )
        ]

        # 3. 损失函数
        # Critic损失：拟合贝尔曼方程
        self.q_target = tf.placeholder(tf.float32, [None, 1], name="q_target")
        self.critic_loss = tf.reduce_mean(tf.square(self.q_target - self.critic.q_value))

        # Actor损失：最大化Q值
        self.actor_loss = -tf.reduce_mean(self.critic.q_value)
        # 动作L2正则化（避免动作过大）
        self.actor_loss += 1e-3 * tf.reduce_mean(tf.square(self.actor.action / self.max_u))

        # 4. 优化器
        self.opt_actor = tf.train.AdamOptimizer(lr_actor).minimize(self.actor_loss, var_list=self.actor.trainable_vars)
        self.opt_critic = tf.train.AdamOptimizer(lr_critic).minimize(self.critic_loss, var_list=self.critic.trainable_vars)

        # 5. 状态/目标归一化器
        self.obs_normalizer = Normalizer(obs_dim)
        self.goal_normalizer = Normalizer(goal_dim)

        # 6. TensorFlow会话
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.init_target_ops)  # 初始化目标网络=主网络

        # 7. 模型保存器
        self.saver = tf.train.Saver()

    def select_action(self, obs, goal, add_noise=True):
        """选择动作（训练加噪声探索，测试用确定性动作）"""
        # 归一化状态和目标
        obs_norm = self.obs_normalizer.normalize(obs[None, :])
        goal_norm = self.goal_normalizer.normalize(goal[None, :])

        # 主Actor输出动作
        action = self.sess.run(
            self.actor.action,
            feed_dict={self.actor.obs: obs_norm, self.actor.goal: goal_norm}
        )[0]  # 去掉batch维度

        # 加高斯噪声探索
        if add_noise:
            noise = 0.1 * self.max_u * np.random.randn(self.actor.action_dim)
            action = np.clip(action + noise, -self.max_u, self.max_u)

        return action

    def update(self, replay_buffer, batch_size=256):
        """用HER回放池的数据更新网络"""
        # 1. 采样一批经验
        batch = replay_buffer.sample(batch_size)
        o, g, u, o2, g2, r = batch["o"], batch["g"], batch["u"], batch["o2"], batch["g2"], batch["r"]

        # 2. 归一化
        o_norm = self.obs_normalizer.normalize(o)
        g_norm = self.goal_normalizer.normalize(g)
        o2_norm = self.obs_normalizer.normalize(o2)
        g2_norm = self.goal_normalizer.normalize(g2)

        # 3. 计算目标Q值（贝尔曼方程）
        # 目标Actor输出下一状态的动作 → 目标Critic计算Q值
        target_action = self.sess.run(
            self.target_actor.action,
            feed_dict={self.target_actor.obs: o2_norm, self.target_actor.goal: g2_norm}
        )
        target_q = self.sess.run(
            self.target_critic.q_value,
            feed_dict={
                self.target_critic.obs: o2_norm,
                self.target_critic.goal: g2_norm,
                self.target_critic.action: target_action
            }
        )
        q_target = r + self.gamma * target_q  # 目标Q值 = 即时奖励 + 未来Q值

        # 4. 更新Critic
        critic_loss, _ = self.sess.run(
            [self.critic_loss, self.opt_critic],
            feed_dict={
                self.critic.obs: o_norm,
                self.critic.goal: g_norm,
                self.critic.action: u,
                self.q_target: q_target
            }
        )

        # 5. 更新Actor（固定Critic，最大化Q值）
        actor_loss, _ = self.sess.run(
            [self.actor_loss, self.opt_actor],
            feed_dict={
                self.actor.obs: o_norm,
                self.actor.goal: g_norm,
                # Critic需要Actor的动作和对应的状态/目标
                self.critic.obs: o_norm,
                self.critic.goal: g_norm,
                self.critic.action: self.sess.run(
                    self.actor.action,
                    feed_dict={self.actor.obs: o_norm, self.actor.goal: g_norm}
                )
            }
        )

        # 6. 软更新目标网络
        self.sess.run(self.update_target_ops)

        return critic_loss, actor_loss

    def save(self, path="./her_ddpg_model"):
        """保存模型"""
        self.saver.save(self.sess, path)
        print(f"模型已保存至: {path}")

    def load(self, path="./her_ddpg_model"):
        """加载模型"""
        self.saver.restore(self.sess, path)
        print(f"模型已从: {path} 加载")


# -------------------------- 5. 轨迹收集器（与环境交互） --------------------------
def collect_episode(env, agent, max_steps=50):
    """与环境交互，收集一条轨迹"""
    obs_dict = env.reset()
    o = obs_dict["observation"]  # 状态
    ag = obs_dict["achieved_goal"]  # 已达成目标
    g = obs_dict["desired_goal"]   # 期望目标

    # 存储轨迹的列表
    episode = {
        "o": [o], "ag": [ag], "g": [g],
        "u": [], "r": []
    }

    for _ in range(max_steps):
        # 选动作
        u = agent.select_action(o, g, add_noise=True)
        # 执行动作
        obs_dict_new, _, done, info = env.step(u)
        o2 = obs_dict_new["observation"]
        ag2 = obs_dict_new["achieved_goal"]
        # 计算奖励
        r = compute_reward(ag2, g, info)

        # 存入轨迹
        episode["u"].append(u)
        episode["r"].append(r)
        episode["o"].append(o2)
        episode["ag"].append(ag2)
        episode["g"].append(g)  # 目标不变

        # 更新状态
        o, ag = o2, ag2

        # 任务成功或轨迹结束
        if info["is_success"] or done:
            break

    # 更新归一化器（用当前轨迹的状态和目标）
    agent.obs_normalizer.update(np.array(episode["o"]))
    agent.goal_normalizer.update(np.array(episode["g"]))

    return episode, info["is_success"]


# -------------------------- 6. 端到端训练流程 --------------------------
def train_her_ddpg(env_id="FetchReach-v1", total_episodes=2000, save_path="./her_ddpg_model"):
    # 1. 创建环境（FetchReach-v1：机械臂 Reach 任务）
    env = gym.make(env_id)
    obs_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    max_u = env.action_space.high[0]  # 动作边界（FetchReach是[-1,1]）

    # 2. 初始化HER-DDPG智能体和回放池
    agent = HERDDPG(obs_dim, goal_dim, action_dim, max_u)
    replay_buffer = HERReplayBuffer(capacity=1000000)

    # 3. 训练统计
    success_history = deque(maxlen=100)  # 最近100个回合的成功率
    start_time = time.time()

    print("开始HER-DDPG训练...")
    for episode in range(1, total_episodes + 1):
        # a. 收集一条轨迹
        episode_data, success = collect_episode(env, agent)
        success_history.append(success)
        success_rate = np.mean(success_history) if success_history else 0.0

        # b. 存入HER回放池（自动生成改写经验）
        replay_buffer.store_episode(episode_data)

        # c. 回放池满1000条经验后开始训练
        critic_loss, actor_loss = 0.0, 0.0
        if replay_buffer.size > 1000:
            for _ in range(40):  # 每回合更新40次网络
                critic_loss, actor_loss = agent.update(replay_buffer, batch_size=256)

        # d. 打印日志（每50回合）
        if episode % 50 == 0:
            print(f"回合: {episode:4d} | 成功率: {success_rate:.2f} | "
                  f"Critic损失: {critic_loss:.4f} | Actor损失: {actor_loss:.4f} | "
                  f"耗时: {time.time()-start_time:.1f}s")

        # e. 早停条件（成功率≥90%）
        if success_rate >= 0.9 and episode >= 100:
            print(f"\n训练提前完成！回合{episode}成功率达到90%")
            break

    # 4. 保存模型
    agent.save(save_path)
    env.close()
    return agent


# -------------------------- 7. 模型测试流程 --------------------------
def test_her_ddpg(env_id="FetchReach-v1", model_path="./her_ddpg_model", num_episodes=5):
    # 1. 创建测试环境
    env = gym.make(env_id)
    obs_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    max_u = env.action_space.high[0]

    # 2. 加载模型
    agent = HERDDPG(obs_dim, goal_dim, action_dim, max_u)
    agent.load(model_path)

    # 3. 测试
    print(f"\n开始测试{num_episodes}个回合（可视化开启）...")
    success_count = 0

    for episode in range(1, num_episodes + 1):
        obs_dict = env.reset()
        o = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        g = obs_dict["desired_goal"]
        ep_step = 0

        while True:
            env.render()  # 可视化机械臂动作
            # 选动作（不加噪声）
            u = agent.select_action(o, g, add_noise=False)
            # 执行动作
            obs_dict_new, _, done, info = env.step(u)
            o2 = obs_dict_new["observation"]
            ag2 = obs_dict_new["achieved_goal"]

            # 更新状态
            o, ag = o2, ag2
            ep_step += 1

            # 回合结束
            if info["is_success"] or done or ep_step >= 50:
                success = info["is_success"]
                success_count += success
                print(f"测试回合 {episode}: {'成功' if success else '失败'} | 步数: {ep_step}")
                break

    # 统计结果
    success_rate = success_count / num_episodes
    print(f"\n测试总结：成功率 = {success_rate:.2f}")
    env.close()


# -------------------------- 8. 主程序入口 --------------------------
if __name__ == "__main__":
    # 注意：需安装依赖：
    # pip install tensorflow==1.15 gym[robotics] mujoco-py（需申请Mujoco许可证）
    # 若没有Mujoco，可替换env_id为"HandReach-v0"（简化版）

    # 第一步：训练模型（约30分钟，FetchReach目标：成功率≥90%）
    trained_agent = train_her_ddpg(total_episodes=2000)

    # 第二步：测试模型（可视化机械臂执行任务）
    test_her_ddpg(model_path="./her_ddpg_model", num_episodes=5)