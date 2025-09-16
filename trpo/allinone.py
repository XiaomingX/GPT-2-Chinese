import time
import numpy as np
import tensorflow as tf
import gym
from collections import deque


# -------------------------- 1. 核心辅助函数 --------------------------
def gae_advantage(rewards, v_preds, dones, gamma=0.99, lam=0.95):
    """计算GAE优势（TRPO关键：减少优势估计的方差）"""
    T = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_adv = 0.0  # 反向累积优势
    
    # 最后一步的价值（回合结束则为0）
    next_v_pred = v_preds[-1] * (1 - dones[-1])
    
    for t in reversed(range(T)):
        # 时序差分误差：即时奖励 + 未来价值 - 当前价值
        delta = rewards[t] + gamma * next_v_pred - v_preds[t]
        # 累积优势：当前误差 + 折扣后的历史优势
        advantages[t] = running_adv = delta + gamma * lam * (1 - dones[t]) * running_adv
        next_v_pred = v_preds[t]
    
    # 优势归一化（加速训练稳定）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # 目标价值 = 优势 + 当前价值
    target_values = advantages + v_preds
    return advantages, target_values


def flatten_list(lst):
    """展平列表（如[[1,2],[3,4]]→[1,2,3,4]）"""
    return [item for sublist in lst for item in sublist]


# -------------------------- 2. 轨迹采样器（收集训练数据） --------------------------
def sample_trajectories(policy, env, timesteps_per_batch):
    """与环境交互，收集指定步数的轨迹数据"""
    # 存储轨迹数据
    obs_list, action_list, reward_list, v_pred_list, done_list = [], [], [], [], []
    ep_rewards, ep_lengths = [], []  # 记录每个回合的奖励和长度
    
    obs = env.reset()
    current_ep_reward = 0.0
    current_ep_length = 0
    
    for _ in range(timesteps_per_batch):
        # 策略预测动作和价值
        action, v_pred = policy.predict(obs)
        
        # 执行动作
        next_obs, reward, done, _ = env.step(action)
        
        # 存储数据
        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)
        v_pred_list.append(v_pred)
        done_list.append(done)
        
        # 更新状态
        obs = next_obs
        current_ep_reward += reward
        current_ep_length += 1
        
        # 回合结束：重置统计
        if done:
            ep_rewards.append(current_ep_reward)
            ep_lengths.append(current_ep_length)
            current_ep_reward = 0.0
            current_ep_length = 0
            obs = env.reset()
    
    # 转换为数组（适配TensorFlow输入）
    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(action_list, dtype=np.int32),  # 离散动作：整数
        "rewards": np.array(reward_list, dtype=np.float32),
        "v_preds": np.array(v_pred_list, dtype=np.float32),
        "dones": np.array(done_list, dtype=np.float32),
        "ep_rewards": ep_rewards,
        "ep_lengths": ep_lengths
    }


# -------------------------- 3. 核心网络（策略网络+价值网络） --------------------------
class PolicyValueNetwork:
    """
    共享特征提取的策略+价值网络（适合离散动作）
    策略网络：输出动作概率；价值网络：输出状态价值
    """
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64)):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 输入占位符
        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim], name="obs")
        self.actions_ph = tf.placeholder(tf.int32, [None], name="actions")  # 离散动作索引
        self.target_v_ph = tf.placeholder(tf.float32, [None], name="target_value")  # 价值目标
        self.advantages_ph = tf.placeholder(tf.float32, [None], name="advantages")  # 优势
        
        # 共享特征提取（2层全连接网络）
        with tf.variable_scope("shared"):
            x = tf.layers.dense(self.obs_ph, hidden_sizes[0], activation=tf.nn.tanh)
            x = tf.layers.dense(x, hidden_sizes[1], activation=tf.nn.tanh)
        
        # 1. 策略网络（输出动作概率）
        with tf.variable_scope("policy"):
            logits = tf.layers.dense(x, action_dim, activation=None)
            self.action_probs = tf.nn.softmax(logits, name="action_probs")
            # 动作对数概率（用于计算策略损失和KL散度）
            self.log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.actions_ph, logits=logits
            )
        
        # 2. 价值网络（输出状态价值）
        with tf.variable_scope("value"):
            self.value = tf.layers.dense(x, 1, activation=None, name="value")[:, 0]  # 标量价值
        
        # 3. 损失函数
        # 策略损失：优势加权的负对数似然（最大化期望回报）
        self.policy_loss = tf.reduce_mean(self.log_probs * self.advantages_ph)
        # 价值损失：MSE（拟合目标价值）
        self.value_loss = tf.reduce_mean(tf.square(self.target_v_ph - self.value))
        # 策略熵：鼓励探索（系数越小探索越弱）
        self.entropy = -tf.reduce_mean(tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 1e-8), axis=1))
        self.entropy_loss = -0.01 * self.entropy  # 负号转为损失
        
        # 4. 价值网络优化器（单独更新，TRPO不约束价值网络）
        self.value_optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.value_loss)
        
        # 5. 策略网络相关：用于TRPO的KL散度和Fisher向量积
        # 旧策略概率（用于计算KL和重要性权重）
        self.old_log_probs_ph = tf.placeholder(tf.float32, [None], name="old_log_probs")
        # KL散度（新旧策略之间的距离，TRPO的信任域约束）
        self.kl_div = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.softmax(logits) * (tf.log(tf.nn.softmax(logits) + 1e-8) - tf.stop_gradient(tf.log(self.action_probs + 1e-8))),
                axis=1
            )
        )
        
        # 6. TensorFlow会话和变量
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # 策略网络可训练参数
        self.policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="shared|policy")
        # 模型保存器
        self.saver = tf.train.Saver()

    def predict(self, obs):
        """预测动作（采样）和状态价值（用于轨迹收集）"""
        obs = obs[None, :] if obs.ndim == 1 else obs  # 扩展为batch维度
        probs, v_pred = self.sess.run(
            [self.action_probs, self.value],
            feed_dict={self.obs_ph: obs}
        )
        # 从概率分布中采样动作
        action = np.random.choice(self.action_dim, p=probs[0])
        return action, v_pred[0]

    def get_log_probs(self, obs, actions):
        """计算给定状态和动作的对数概率（用于TRPO）"""
        return self.sess.run(self.log_probs, feed_dict={self.obs_ph: obs, self.actions_ph: actions})

    def update_value_network(self, obs, target_values, batch_size=64):
        """更新价值网络（小批量训练，更稳定）"""
        num_samples = len(obs)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            self.sess.run(self.value_optimizer, feed_dict={
                self.obs_ph: obs[start:end],
                self.target_v_ph: target_values[start:end]
            })

    def save(self, path="./trpo_model"):
        """保存模型参数"""
        self.saver.save(self.sess, path)
        print(f"模型已保存至: {path}")

    def load(self, path="./trpo_model"):
        """加载模型参数"""
        self.saver.restore(self.sess, path)
        print(f"模型已从: {path} 加载")


# -------------------------- 4. TRPO核心逻辑（信任域策略优化） --------------------------
class TRPO:
    def __init__(self, obs_dim, action_dim, max_kl=0.01, cg_iters=10, cg_damping=0.1):
        # 超参数
        self.max_kl = max_kl  # 信任域最大KL散度（约束策略更新幅度）
        self.cg_iters = cg_iters  # 共轭梯度迭代次数
        self.cg_damping = cg_damping  # 共轭梯度阻尼（避免数值不稳定）
        
        # 初始化策略价值网络
        self.net = PolicyValueNetwork(obs_dim, action_dim)
        
        # 策略参数扁平化操作（方便共轭梯度计算）
        self.flat_policy_vars = tf.concat([tf.reshape(var, [-1]) for var in self.net.policy_vars], axis=0)
        self.set_flat_vars_ph = tf.placeholder(tf.float32, [None], name="set_flat_vars")
        
        # 构建"从扁平化参数恢复网络参数"的操作
        start = 0
        self.set_flat_vars_ops = []
        for var in self.net.policy_vars:
            var_shape = var.get_shape().as_list()
            var_size = np.prod(var_shape)
            var_flat = tf.reshape(self.set_flat_vars_ph[start:start+var_size], var_shape)
            self.set_flat_vars_ops.append(tf.assign(var, var_flat))
            start += var_size
        self.set_flat_vars = tf.group(*self.set_flat_vars_ops)

    def get_flat_policy_vars(self):
        """获取扁平化的策略参数"""
        return self.net.sess.run(self.flat_policy_vars)

    def set_flat_policy_vars(self, flat_vars):
        """用扁平化参数更新策略网络"""
        self.net.sess.run(self.set_flat_vars, feed_dict={self.set_flat_vars_ph: flat_vars})

    def fisher_vector_product(self, obs, actions, v):
        """计算Fisher信息矩阵与向量v的乘积（TRPO核心，避免直接求逆Fisher矩阵）"""
        # 先计算KL散度对策略参数的梯度
        kl_grad = tf.gradients(self.net.kl_div, self.net.policy_vars)
        kl_grad_flat = tf.concat([tf.reshape(g, [-1]) for g in kl_grad], axis=0)
        
        # 计算梯度与向量v的内积
        grad_v = tf.reduce_sum(kl_grad_flat * self.set_flat_vars_ph)
        # 计算内积对参数的梯度（即Fisher向量积）
        fisher_v = tf.gradients(grad_v, self.net.policy_vars)
        fisher_v_flat = tf.concat([tf.reshape(g, [-1]) for g in fisher_v], axis=0)
        
        # 加入阻尼项，避免数值问题
        fisher_v_flat += self.cg_damping * self.set_flat_vars_ph
        
        # 执行计算
        return self.net.sess.run(fisher_v_flat, feed_dict={
            self.net.obs_ph: obs,
            self.net.actions_ph: actions,
            self.set_flat_vars_ph: v
        })

    def conjugate_gradient(self, obs, actions, g):
        """共轭梯度算法：求解 F * x = g 的近似解（F是Fisher矩阵）"""
        x = np.zeros_like(g)
        r = g.copy()  # 残差 r = g - F*x（初始x=0，故r=g）
        p = r.copy()  # 搜索方向
        r_dot_r = np.dot(r, r)
        
        for _ in range(self.cg_iters):
            # 计算 F*p
            Fp = self.fisher_vector_product(obs, actions, p)
            # 计算步长
            alpha = r_dot_r / (np.dot(p, Fp) + 1e-8)
            # 更新x和r
            x += alpha * p
            r -= alpha * Fp
            # 更新残差点积
            new_r_dot_r = np.dot(r, r)
            # 更新搜索方向
            beta = new_r_dot_r / (r_dot_r + 1e-8)
            p = r + beta * p
            r_dot_r = new_r_dot_r
            
            # 残差足够小则提前退出
            if r_dot_r < 1e-10:
                break
        return x

    def update_policy(self, obs, actions, advantages):
        """TRPO策略更新：共轭梯度+线搜索"""
        # 1. 保存当前策略参数（用于线搜索回退）
        old_flat_vars = self.get_flat_policy_vars()
        # 2. 计算当前策略的损失和梯度
        old_loss, g = self.net.sess.run(
            [self.net.policy_loss, tf.gradients(self.net.policy_loss, self.flat_policy_vars)],
            feed_dict={
                self.net.obs_ph: obs,
                self.net.actions_ph: actions,
                self.net.advantages_ph: advantages
            }
        )
        g = np.concatenate([np.reshape(grad, [-1]) for grad in g])  # 梯度扁平化
        
        # 3. 共轭梯度求解搜索方向
        search_dir = self.conjugate_gradient(obs, actions, -g)  # 负梯度方向（最小化损失=最大化回报）
        
        # 4. 计算搜索方向的步长（满足KL约束）
        # 计算 Hessian * search_dir = F * search_dir
        hessian_dir = self.fisher_vector_product(obs, actions, search_dir)
        # 步长公式：sqrt(2*max_kl / (search_dir^T * F * search_dir))
        step_size = np.sqrt(2 * self.max_kl / (np.dot(search_dir, hessian_dir) + 1e-8))
        full_step = step_size * search_dir
        
        # 5. 线搜索：寻找满足KL约束且损失下降的最大步长
        best_loss = old_loss
        best_flat_vars = old_flat_vars
        # 尝试不同步长（从1.0开始衰减）
        for frac in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            new_flat_vars = old_flat_vars + frac * full_step
            self.set_flat_policy_vars(new_flat_vars)
            
            # 计算新策略的损失和KL散度
            new_loss, kl = self.net.sess.run(
                [self.net.policy_loss, self.net.kl_div],
                feed_dict={
                    self.net.obs_ph: obs,
                    self.net.actions_ph: actions,
                    self.net.advantages_ph: advantages
                }
            )
            
            # 满足条件：损失下降且KL≤max_kl
            if new_loss < best_loss and kl <= self.max_kl:
                best_loss = new_loss
                best_flat_vars = new_flat_vars
                break
        
        # 6. 更新为最优策略参数（若所有步长都不满足则回退原参数）
        self.set_flat_policy_vars(best_flat_vars)
        return best_loss


# -------------------------- 5. 端到端训练流程 --------------------------
def train_trpo(env_id="CartPole-v1", total_timesteps=1e4, timesteps_per_batch=512, save_path="./trpo_model"):
    # 1. 创建环境
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # 离散动作数量
    
    # 2. 初始化TRPO
    trpo = TRPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_kl=0.01,    # 信任域KL约束
        cg_iters=10,    # 共轭梯度迭代次数
        cg_damping=0.1  # 共轭梯度阻尼
    )
    
    # 3. 训练统计
    ep_reward_buf = deque(maxlen=10)  # 最近10个回合的奖励
    start_time = time.time()
    total_steps = 0
    
    # 4. 训练循环
    while total_steps < total_timesteps:
        # a. 采样轨迹数据
        traj_data = sample_trajectories(trpo.net, env, timesteps_per_batch)
        total_steps += timesteps_per_batch
        ep_reward_buf.extend(traj_data["ep_rewards"])
        
        # b. 计算GAE优势和目标价值
        advantages, target_values = gae_advantage(
            rewards=traj_data["rewards"],
            v_preds=traj_data["v_preds"],
            dones=traj_data["dones"],
            gamma=0.99,
            lam=0.95
        )
        
        # c. 更新策略网络（TRPO核心）
        policy_loss = trpo.update_policy(
            obs=traj_data["obs"],
            actions=traj_data["actions"],
            advantages=advantages
        )
        
        # d. 更新价值网络
        trpo.net.update_value_network(
            obs=traj_data["obs"],
            target_values=target_values
        )
        
        # e. 打印训练进度（每2个批次打印一次）
        if len(ep_reward_buf) > 0 and total_steps % (2 * timesteps_per_batch) == 0:
            avg_reward = np.mean(ep_reward_buf)
            avg_length = np.mean(traj_data["ep_lengths"]) if traj_data["ep_lengths"] else 0
            fps = total_steps / (time.time() - start_time)
            print(f"总步数: {int(total_steps):5d} | "
                  f"平均奖励: {avg_reward:5.1f} | "
                  f"平均长度: {avg_length:4.1f} | "
                  f"策略损失: {policy_loss:.4f} | "
                  f"FPS: {fps:4.0f}")
    
    # 5. 保存模型
    trpo.net.save(save_path)
    env.close()
    return trpo


# -------------------------- 6. 模型测试流程 --------------------------
def test_trpo(env_id="CartPole-v1", model_path="./trpo_model", num_episodes=5):
    # 1. 创建测试环境
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 2. 初始化网络并加载模型
    net = PolicyValueNetwork(obs_dim, action_dim)
    net.load(model_path)
    
    # 3. 测试循环
    print(f"\n开始测试{num_episodes}个回合（可视化开启）...")
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_length = 0
        
        while True:
            env.render()  # 可视化
            # 贪心选择动作（测试时不采样，选概率最大的动作）
            probs, _ = net.sess.run(
                [net.action_probs, net.value],
                feed_dict={net.obs_ph: obs[None, :]}
            )
            action = np.argmax(probs[0])  # 贪心动作
            
            # 执行动作
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            ep_length += 1
            
            if done:
                print(f"测试回合 {ep}: 奖励 = {ep_reward:4.1f}, 长度 = {ep_length}")
                break
    
    env.close()
    print("测试完成！")


# -------------------------- 7. 主程序入口 --------------------------
if __name__ == "__main__":
    # 第一步：训练模型（约2分钟，CartPole-v1目标：平均奖励≥200）
    trained_trpo = train_trpo(total_timesteps=2e4)
    
    # 第二步：测试模型
    test_trpo(model_path="./trpo_model", num_episodes=5)