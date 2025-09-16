import time
import numpy as np
import tensorflow as tf
import gym
from collections import deque


# -------------------------- 1. 核心辅助函数 --------------------------
def safemean(xs):
    """避免空列表求均值报错"""
    return np.nan if len(xs) == 0 else np.mean(xs)


def discount_with_dones(rewards, dones, gamma):
    """带终止状态的折扣回报计算（A2C/ACKTR核心）"""
    discounted = []
    running_reward = 0
    for r, done in zip(rewards[::-1], dones[::-1]):
        if done:
            running_reward = 0
        running_reward = r + gamma * running_reward
        discounted.append(running_reward)
    return discounted[::-1]


# -------------------------- 2. 经验收集（简化版，替代原Runner） --------------------------
def collect_experience(env, model, nsteps, gamma):
    """与环境交互，收集n步轨迹数据"""
    obs_list, acts_list, vals_list, rews_list, dones_list = [], [], [], [], []
    ep_infos = []  # 记录回合信息（总回报、长度）
    obs = env.reset()
    done = False
    ep_reward = 0
    ep_length = 0

    while len(obs_list) < nsteps:
        if done:
            # 回合结束，记录信息并重置
            ep_infos.append({"r": ep_reward, "l": ep_length})
            obs = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0

        # 模型预测动作和价值
        action, value = model.step(obs[None, :])  # 扩展batch维度
        action = action[0]  # 去掉batch维度

        # 存储当前步数据
        obs_list.append(obs)
        acts_list.append(action)
        vals_list.append(value[0])
        dones_list.append(done)

        # 执行动作
        obs, reward, done, _ = env.step(action)
        rews_list.append(reward)
        ep_reward += reward
        ep_length += 1

    # 处理折扣回报（用最后一步价值做Bootstrapping）
    last_value = model.value(obs[None, :])[0]
    rews_list += [last_value]
    dones_list += [done]
    discounted_rews = discount_with_dones(rews_list, dones_list, gamma)[:-1]  # 去掉最后一个bootstrap值

    # 转换为数组并返回
    return (
        np.array(obs_list, dtype=np.float32),
        np.array(acts_list, dtype=np.int32),  # 离散动作用int
        np.array(vals_list, dtype=np.float32),
        np.array(discounted_rews, dtype=np.float32),
        ep_infos
    )


# -------------------------- 3. KFAC优化器（核心简化版） --------------------------
class KFACOptimizer:
    """简化的KFAC优化器：只支持全连接层，保留核心预条件逻辑"""
    def __init__(self, lr=0.25, clip_kl=0.01, momentum=0.9, epsilon=1e-2):
        self.lr = lr
        self.clip_kl = clip_kl  # KL散度裁剪（控制更新幅度）
        self.momentum = momentum
        self.epsilon = epsilon  # 阻尼项（避免矩阵奇异）
        self.sess = tf.get_default_session()

        # 存储Fisher矩阵相关统计量
        self.stats = {}
        self.eigen = {}  # 特征值和特征向量
        self.momentum_vars = {}  # 动量缓存

    def compute_fisher_stats(self, loss, var_list):
        """计算Fisher信息矩阵的统计量（通过采样损失的梯度）"""
        grads = tf.gradients(loss, var_list)
        for var, grad in zip(var_list, grads):
            if grad is None:
                continue
            # Fisher矩阵近似：E[grad * grad^T]，这里用单个batch的梯度估计
            grad_flat = tf.reshape(grad, [-1, 1])
            fisher = tf.matmul(grad_flat, grad_flat, transpose_b=True) / tf.cast(tf.shape(grad_flat)[0], tf.float32)
            
            # 初始化统计量（滑动平均更新）
            if var not in self.stats:
                self.stats[var] = tf.Variable(tf.zeros_like(fisher), trainable=False)
            self.stats[var] = self.stats[var] * 0.99 + fisher * 0.01  # 滑动平均

        # 计算特征值和特征向量（Fisher矩阵对角化）
        for var in var_list:
            if var not in self.stats:
                continue
            # 特征分解（CPU上计算更稳定）
            with tf.device('/cpu:0'):
                eig_vals, eig_vecs = tf.self_adjoint_eig(self.stats[var])
            self.eigen[var] = (eig_vals, eig_vecs)

            # 初始化动量变量
            if var not in self.momentum_vars:
                self.momentum_vars[var] = tf.Variable(tf.zeros_like(var), trainable=False)

    def apply_gradients(self, grads, var_list):
        """用KFAC预条件梯度更新参数"""
        update_ops = []
        vf_v = 0.0  # 用于KL裁剪的指标

        for var, grad in zip(var_list, grads):
            if grad is None or var not in self.eigen:
                continue

            # 1. 梯度扁平化
            grad_flat = tf.reshape(grad, [-1, 1])
            eig_vals, eig_vecs = self.eigen[var]

            # 2. 特征空间投影（去相关性）
            grad_proj = tf.matmul(eig_vecs, grad_flat, transpose_a=True)

            # 3. 预条件：用特征值缩放梯度（Fisher矩阵的逆近似）
            eig_vals_clipped = tf.maximum(eig_vals, self.epsilon)  # 避免除以零
            grad_precond = grad_proj / tf.reshape(eig_vals_clipped, [-1, 1])

            # 4. 投影回原空间
            grad_final = tf.matmul(eig_vecs, grad_precond)
            grad_final = tf.reshape(grad_final, var.shape)

            # 5. 计算KL裁剪系数
            vf_v += tf.reduce_sum(grad_final * grad * self.lr ** 2)

            # 6. 动量更新
            self.momentum_vars[var] = self.momentum * self.momentum_vars[var] - self.lr * grad_final
            update_ops.append(tf.assign(var, var + self.momentum_vars[var]))

        # 7. KL散度裁剪（控制更新幅度）
        scaling = tf.minimum(1.0, tf.sqrt(self.clip_kl / (vf_v + 1e-8)))
        for var in var_list:
            if var in self.momentum_vars:
                update_ops.append(tf.assign(var, var + (scaling - 1.0) * self.momentum_vars[var]))

        return tf.group(*update_ops)


# -------------------------- 4. Actor-Critic核心网络 --------------------------
class ACKTRModel:
    """ACKTR模型：共享特征提取层，输出策略（动作分布）和价值"""
    def __init__(self, obs_dim, action_dim, hidden_size=64, lr=0.25):
        self.obs_dim = obs_dim
        self.action_dim = action_dim  # 离散动作数量

        # 输入占位符
        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim], name="obs")
        self.acts_ph = tf.placeholder(tf.int32, [None], name="actions")
        self.rews_ph = tf.placeholder(tf.float32, [None], name="rewards")
        self.vals_ph = tf.placeholder(tf.float32, [None], name="old_values")

        # 共享特征提取（2层全连接）
        with tf.variable_scope("acktr_net"):
            x = tf.layers.dense(self.obs_ph, hidden_size, activation=tf.nn.tanh)
            x = tf.layers.dense(x, hidden_size, activation=tf.nn.tanh)

            # 策略头（离散动作：输出动作概率）
            logits = tf.layers.dense(x, action_dim, activation=None)
            self.act_probs = tf.nn.softmax(logits)

            # 价值头（输出状态价值）
            self.value = tf.layers.dense(x, 1, activation=None)[:, 0]

        # 损失计算
        adv = self.rews_ph - self.vals_ph  # 优势值
        # 策略损失：带优势加权的负对数似然
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.acts_ph)
        self.policy_loss = tf.reduce_mean(adv * neg_log_prob)
        # 价值损失：MSE
        self.value_loss = tf.reduce_mean(tf.square(self.rews_ph - self.value))
        # 熵损失：鼓励探索
        entropy = -tf.reduce_sum(self.act_probs * tf.log(self.act_probs + 1e-8), axis=1)
        self.entropy_loss = -tf.reduce_mean(entropy)

        # 总损失
        self.total_loss = self.policy_loss + 0.5 * self.value_loss + 0.01 * self.entropy_loss

        # KFAC优化器
        self.var_list = tf.trainable_variables("acktr_net")
        self.kfac = KFACOptimizer(lr=lr)
        # 计算Fisher矩阵的采样损失（用策略损失近似）
        self.fisher_loss = self.policy_loss
        self.kfac.compute_fisher_stats(self.fisher_loss, self.var_list)

        # 梯度计算
        self.grads = tf.gradients(self.total_loss, self.var_list)
        # 优化操作
        self.train_op = self.kfac.apply_gradients(self.grads, self.var_list)

        # 初始化
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def step(self, obs):
        """预测动作和价值（用于收集经验）"""
        act_probs, value = self.sess.run(
            [self.act_probs, self.value],
            feed_dict={self.obs_ph: obs}
        )
        # 从概率分布采样动作
        action = [np.random.choice(self.action_dim, p=p) for p in act_probs]
        return np.array(action), value

    def value(self, obs):
        """单独预测价值（用于Bootstrapping）"""
        return self.sess.run(self.value, feed_dict={self.obs_ph: obs})

    def train(self, obs, acts, vals, rews):
        """训练模型"""
        # 先更新Fisher统计量
        self.sess.run(self.kfac.stats)
        # 再更新参数
        loss, _ = self.sess.run(
            [self.total_loss, self.train_op],
            feed_dict={
                self.obs_ph: obs,
                self.acts_ph: acts,
                self.vals_ph: vals,
                self.rews_ph: rews
            }
        )
        return loss

    def save(self, path="./acktr_model"):
        """保存模型"""
        self.saver.save(self.sess, path)
        print(f"模型保存至: {path}")

    def load(self, path="./acktr_model"):
        """加载模型"""
        self.saver.restore(self.sess, path)
        print(f"模型加载自: {path}")


# -------------------------- 5. 端到端训练流程 --------------------------
def train_acktr(env_id="CartPole-v1", total_timesteps=2e4, nsteps=5, gamma=0.99, save_path="./acktr_model"):
    # 1. 创建环境（CartPole是离散动作，适合简化版ACKTR）
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 2. 初始化模型
    model = ACKTRModel(obs_dim=obs_dim, action_dim=action_dim, lr=0.25)

    # 3. 训练配置
    nupdates = int(total_timesteps // nsteps)
    epinfobuf = deque(maxlen=100)  # 记录最近100个回合信息
    tstart = time.time()

    # 4. 训练循环
    for update in range(1, nupdates + 1):
        # a. 收集n步经验
        obs, acts, vals, rews, epinfos = collect_experience(env, model, nsteps, gamma)
        epinfobuf.extend(epinfos)

        # b. 训练模型
        loss = model.train(obs, acts, vals, rews)

        # c. 打印日志（每10轮更新）
        if update % 10 == 0 or update == 1:
            total_steps = update * nsteps
            fps = int(total_steps / (time.time() - tstart))
            print(f"更新轮数: {update:3d}/{nupdates} | "
                  f"总步数: {total_steps:5d} | "
                  f"FPS: {fps:3d} | "
                  f"损失: {loss:.4f} | "
                  f"平均回报: {safemean([e['r'] for e in epinfobuf]):.1f} | "
                  f"平均长度: {safemean([e['l'] for e in epinfobuf]):.1f}")

    # 5. 保存模型
    model.save(save_path)
    env.close()
    return model


# -------------------------- 6. 模型测试流程 --------------------------
def test_acktr(env_id="CartPole-v1", model_path="./acktr_model", num_episodes=5):
    # 1. 创建测试环境
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 2. 加载模型
    model = ACKTRModel(obs_dim=obs_dim, action_dim=action_dim)
    model.load(model_path)

    # 3. 测试循环
    print(f"\n开始测试{num_episodes}个回合（可视化开启）...")
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            env.render()  # 可视化
            # 贪心策略（取概率最大的动作）
            act_probs, _ = model.sess.run(
                [model.act_probs, model.value],
                feed_dict={model.obs_ph: obs[None, :]}
            )
            action = np.argmax(act_probs[0])

            # 执行动作
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            ep_length += 1

        print(f"测试回合 {ep}: 回报 = {ep_reward:3d}, 长度 = {ep_length:3d}")

    env.close()
    print("测试完成！")


# -------------------------- 7. 主程序入口 --------------------------
if __name__ == "__main__":
    # 第一步：训练模型（约1分钟，CartPole目标回报≥500）
    trained_model = train_acktr(total_timesteps=2e4)

    # 第二步：测试模型
    test_acktr(model_path="./acktr_model", num_episodes=5)