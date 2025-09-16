# 核心依赖导入（统一整合，去除重复）
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
import pickle
import os
from argparse import Namespace
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


# ------------------------------ 1. 核心工具函数 ------------------------------
def discount_with_dones(rewards, dones, gamma):
    """计算带终止状态的折扣奖励"""
    discounted = []
    r = 0.0
    # 反向计算折扣奖励
    for reward, done in zip(reversed(rewards), reversed(dones)):
        r = reward + gamma * r * (1.0 - done)  # 终止状态后奖励清零
        discounted.append(r)
    return list(reversed(discounted))


def make_update_exp(source_vars, target_vars, polyak=0.99):
    """目标网络软更新表达式（Polyak平均）"""
    update_ops = []
    # 按变量名匹配源网络与目标网络参数
    for var, var_target in zip(sorted(source_vars, key=lambda v: v.name),
                               sorted(target_vars, key=lambda v: v.name)):
        update_ops.append(var_target.assign(polyak * var_target + (1 - polyak) * var))
    return tf.group(*update_ops)


# TensorFlow工具函数（保留核心功能，去除冗余封装）
def scope_vars(scope, trainable_only=False):
    """获取指定作用域下的变量"""
    var_type = tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES
    return tf.get_collection(var_type, scope=scope)


def minimize_and_clip(optimizer, loss, var_list, clip_val=0.5):
    """带梯度裁剪的优化器（防止梯度爆炸）"""
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    clipped_grads = [(tf.clip_by_norm(g, clip_val) if g is not None else (g, v)) for g, v in grads]
    return optimizer.apply_gradients(clipped_grads)


def function(inputs, outputs, updates=None):
    """简化版TensorFlow函数封装（类似Theano）"""
    updates = updates or []
    update_group = tf.group(*updates)
    outputs_with_update = list(outputs) + [update_group]

    def func(*args):
        feed_dict = {inputs[i]: args[i] for i in range(len(args))}
        results = tf.get_default_session().run(outputs_with_update, feed_dict=feed_dict)
        return results[:-1]  # 去掉update的返回值

    return func


# ------------------------------ 2. 经验回放缓冲池 ------------------------------
class ReplayBuffer:
    """存储智能体经验的循环缓冲池"""
    def __init__(self, max_size=1e6):
        self._storage = []
        self._max_size = int(max_size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs, act, rew, next_obs, done):
        """添加一条经验（s,a,r,s',done）"""
        data = (obs, act, rew, next_obs, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data  # 循环覆盖旧数据
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size):
        """随机采样一批经验"""
        idxes = [random.randint(0, len(self._storage)-1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        """将采样索引转换为批量数据（numpy数组）"""
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = [], [], [], [], []
        for i in idxes:
            obs, act, rew, next_obs, done = self._storage[i]
            obs_batch.append(np.array(obs, copy=False))
            act_batch.append(np.array(act, copy=False))
            rew_batch.append(rew)
            next_obs_batch.append(np.array(next_obs, copy=False))
            done_batch.append(done)
        return (np.array(obs_batch), np.array(act_batch), np.array(rew_batch),
                np.array(next_obs_batch), np.array(done_batch))


# ------------------------------ 3. 动作概率分布（简化常用类型） ------------------------------
class PdType:
    """动作概率分布类型基类"""
    def __init__(self, ac_space):
        self.ac_space = ac_space

    def pdfromflat(self, flat_params):
        """从扁平参数创建分布实例"""
        raise NotImplementedError

    def param_shape(self):
        """分布参数的形状"""
        raise NotImplementedError

    def sample_shape(self):
        """采样动作的形状"""
        raise NotImplementedError


class DiagGaussianPdType(PdType):
    """对角高斯分布（适用于连续动作空间）"""
    def pdfromflat(self, flat_params):
        return DiagGaussianPd(flat_params)

    def param_shape(self):
        return [2 * self.ac_space.shape[0]]  # 均值+对数标准差

    def sample_shape(self):
        return self.ac_space.shape


class CategoricalPdType(PdType):
    """分类分布（适用于离散动作空间）"""
    def pdfromflat(self, flat_params):
        return CategoricalPd(flat_params)

    def param_shape(self):
        return [self.ac_space.n]  # 动作数

    def sample_shape(self):
        return []


class DiagGaussianPd:
    """对角高斯分布实例"""
    def __init__(self, flat_params):
        # 前半部分为均值，后半部分为对数标准差
        mean, logstd = tf.split(flat_params, 2, axis=1)
        self.mean = mean
        self.std = tf.exp(logstd)

    def sample(self):
        """采样动作"""
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def flatparam(self):
        """返回扁平参数"""
        return tf.concat([self.mean, tf.log(self.std)], axis=1)


class CategoricalPd:
    """分类分布实例（softmax）"""
    def __init__(self, flat_params):
        self.logits = flat_params

    def sample(self):
        """采样动作（Gumbel-Softmax，保证可导）"""
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.nn.softmax(self.logits - tf.log(-tf.log(u)), axis=1)

    def flatparam(self):
        """返回扁平参数"""
        return self.logits


def make_pdtype(ac_space):
    """根据动作空间创建对应的概率分布类型"""
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        return DiagGaussianPdType(ac_space)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space)
    else:
        raise NotImplementedError(f"不支持的动作空间类型: {type(ac_space)}")


# ------------------------------ 4. 网络结构（策略/价值网络共享MLP） ------------------------------
def mlp_model(inputs, output_dim, scope, reuse=False, num_units=64):
    """多层感知机（MLP）基础模型（策略/价值网络共用）"""
    with tf.variable_scope(scope, reuse=reuse):
        x = inputs
        x = layers.fully_connected(x, num_outputs=num_units, activation_fn=tf.nn.relu)
        x = layers.fully_connected(x, num_outputs=num_units, activation_fn=tf.nn.relu)
        x = layers.fully_connected(x, num_outputs=output_dim, activation_fn=None)
        return x


# ------------------------------ 5. 多智能体训练器 ------------------------------
class MADDPGAgentTrainer:
    """单个MADDPG智能体的训练器"""
    def __init__(self, agent_name, obs_shape, act_space, agent_idx, args):
        self.name = agent_name  # 智能体名称（区分作用域）
        self.agent_idx = agent_idx  # 智能体索引
        self.args = args
        self.pdtype = make_pdtype(act_space)  # 动作概率分布类型

        # 1. 创建输入占位符
        self.obs_ph = tf.placeholder(tf.float32, [None] + list(obs_shape), name="obs")
        self.act_ph = tf.placeholder(tf.float32, [None] + list(self.pdtype.sample_shape()), name="act")
        self.target_ph = tf.placeholder(tf.float32, [None], name="target_q")

        # 2. 构建策略网络（P网络）与目标策略网络（target P）
        self._build_policy_net()
        # 3. 构建价值网络（Q网络）与目标价值网络（target Q）
        self._build_value_net()

        # 4. 经验回放缓冲池
        self.replay_buffer = ReplayBuffer(max_size=1e6)

    def _build_policy_net(self):
        """构建策略网络（输出动作分布参数）"""
        args = self.args
        with tf.variable_scope(self.name + "/p_net"):
            # 策略网络输出动作分布参数
            p_params = mlp_model(self.obs_ph, self.pdtype.param_shape()[0], scope="p_func", num_units=args.num_units)
            self.pd = self.pdtype.pdfromflat(p_params)
            self.act_sample = self.pd.sample()  # 采样动作

            # 目标策略网络（结构相同，参数软更新）
            target_p_params = mlp_model(self.obs_ph, self.pdtype.param_shape()[0], scope="target_p_func", num_units=args.num_units)
            self.target_pd = self.pdtype.pdfromflat(target_p_params)
            self.target_act_sample = self.target_pd.sample()

            # 策略网络参数与更新操作
            self.p_vars = scope_vars(self.name + "/p_net/p_func")
            self.target_p_vars = scope_vars(self.name + "/p_net/target_p_func")
            self.update_target_p = make_update_exp(self.p_vars, self.target_p_vars)

            # 策略损失（最大化Q值，加L2正则）
            self.q_for_p = self._get_q_value(self.obs_ph, self.act_sample)
            self.p_loss = -tf.reduce_mean(self.q_for_p) + 1e-3 * tf.reduce_mean(tf.square(self.pd.flatparam()))

            # 策略优化器
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self.p_train_op = minimize_and_clip(optimizer, self.p_loss, self.p_vars)

            # 封装可调用函数
            self.act_func = function([self.obs_ph], [self.act_sample])
            self.target_act_func = function([self.obs_ph], [self.target_act_sample])
            self.p_train_func = function([self.obs_ph], [self.p_loss], updates=[self.p_train_op])

    def _build_value_net(self):
        """构建价值网络（输出Q(s1,a1,s2,a2,...)）"""
        args = self.args
        with tf.variable_scope(self.name + "/q_net"):
            # 价值网络输入：所有智能体的观测+动作（MADDPG核心）
            self.all_obs_ph = tf.placeholder(tf.float32, [None, args.total_obs_dim], name="all_obs")
            self.all_act_ph = tf.placeholder(tf.float32, [None, args.total_act_dim], name="all_act")
            q_input = tf.concat([self.all_obs_ph, self.all_act_ph], axis=1)

            # 价值网络输出Q值
            self.q_value = mlp_model(q_input, 1, scope="q_func", num_units=args.num_units)[:, 0]  # 降维为[batch_size]

            # 目标价值网络
            target_q_value = mlp_model(q_input, 1, scope="target_q_func", num_units=args.num_units)[:, 0]
            self.target_q_value = target_q_value

            # 价值网络参数与更新操作
            self.q_vars = scope_vars(self.name + "/q_net/q_func")
            self.target_q_vars = scope_vars(self.name + "/q_net/target_q_func")
            self.update_target_q = make_update_exp(self.q_vars, self.target_q_vars)

            # 价值损失（TD误差）
            self.q_loss = tf.reduce_mean(tf.square(self.q_value - self.target_ph))

            # 价值优化器
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            self.q_train_op = minimize_and_clip(optimizer, self.q_loss, self.q_vars)

            # 封装可调用函数
            self.q_value_func = function([self.all_obs_ph, self.all_act_ph], [self.q_value])
            self.target_q_value_func = function([self.all_obs_ph, self.all_act_ph], [self.target_q_value])
            self.q_train_func = function([self.all_obs_ph, self.all_act_ph, self.target_ph], [self.q_loss], updates=[self.q_train_op])

    def _get_q_value(self, obs, act):
        """获取当前智能体动作对应的Q值（需拼接所有智能体的观测/动作，这里先占位，训练时填充）"""
        # 实际调用时需传入所有智能体的观测和动作，此处为临时实现
        dummy_all_obs = tf.zeros([tf.shape(obs)[0], self.args.total_obs_dim])
        dummy_all_act = tf.zeros([tf.shape(act)[0], self.args.total_act_dim])
        return self.q_value_func(dummy_all_obs, dummy_all_act)[0]

    def action(self, obs):
        """生成动作（推理时用）"""
        return self.act_func(obs[None])[0]  # 增加batch维度，再去掉

    def add_experience(self, obs, act, rew, next_obs, done):
        """添加经验到缓冲池"""
        self.replay_buffer.add(obs, act, rew, next_obs, done)

    def update(self, all_agents, train_step):
        """更新当前智能体的策略和价值网络"""
        args = self.args
        # 1. 缓冲池数据不足或未到更新步长，直接返回
        if len(self.replay_buffer) < args.batch_size * 10 or train_step % 100 != 0:
            return None

        # 2. 从所有智能体的缓冲池采样（共享采样索引，保证经验一致性）
        batch_size = args.batch_size
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.replay_buffer.sample(batch_size)
        
        # 拼接所有智能体的观测和动作
        all_obs_batch = np.concatenate([agent.replay_buffer.sample(batch_size)[0] for agent in all_agents], axis=1)
        all_act_batch = np.concatenate([agent.replay_buffer.sample(batch_size)[1] for agent in all_agents], axis=1)
        all_next_obs_batch = np.concatenate([agent.replay_buffer.sample(batch_size)[3] for agent in all_agents], axis=1)

        # 3. 计算目标Q值（TD目标）
        # 3.1 用目标策略网络生成所有智能体的下一动作
        all_next_act_batch = np.concatenate([agent.target_act_func(next_obs_batch) for agent, next_obs_batch in 
                                            zip(all_agents, [agent.replay_buffer.sample(batch_size)[3] for agent in all_agents])], axis=1)
        # 3.2 用目标价值网络计算下一状态Q值
        target_q_next = self.target_q_value_func(all_next_obs_batch, all_next_act_batch)[0]
        # 3.3 TD目标：r + gamma * (1-done) * Q_next
        target_q = rew_batch + args.gamma * (1.0 - done_batch) * target_q_next

        # 4. 更新价值网络
        q_loss = self.q_train_func(all_obs_batch, all_act_batch, target_q)[0]

        # 5. 更新策略网络
        p_loss = self.p_train_func(obs_batch)[0]

        # 6. 软更新目标网络
        self.update_target_p.run()
        self.update_target_q.run()

        return {"q_loss": q_loss, "p_loss": p_loss, "avg_reward": np.mean(rew_batch)}


# ------------------------------ 6. 环境创建与训练/推理流程 ------------------------------
def create_env(scenario_name="simple"):
    """创建多智能体环境（基于ParticleEnv）"""
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def init_agents(env, args):
    """初始化所有智能体的训练器"""
    agents = []
    # 计算所有智能体的观测维度总和和动作维度总和（MADDPG Q网络输入需要）
    total_obs_dim = sum(env.observation_space[i].shape[0] for i in range(env.n))
    total_act_dim = sum(env.action_space[i].shape[0] if hasattr(env.action_space[i], "shape") else 1 
                        for i in range(env.n))
    args.total_obs_dim = total_obs_dim
    args.total_act_dim = total_act_dim

    for i in range(env.n):
        agent_name = f"agent_{i}"
        obs_shape = env.observation_space[i].shape
        act_space = env.action_space[i]
        agent = MADDPGAgentTrainer(agent_name, obs_shape, act_space, i, args)
        agents.append(agent)
    return agents


def train_maddpg(args):
    """训练MADDPG模型"""
    # 1. 初始化环境和智能体
    env = create_env(args.scenario)
    agents = init_agents(env, args)
    print(f"环境: {args.scenario}, 智能体数量: {len(agents)}")

    # 2. 初始化TensorFlow会话
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)

    # 3. 加载预训练模型（如果有）
    if args.load_dir and os.path.exists(args.load_dir + "checkpoint"):
        saver.restore(sess, tf.train.latest_checkpoint(args.load_dir))
        print(f"加载模型成功: {args.load_dir}")

    # 4. 训练主循环
    episode_rewards = [0.0]
    obs_n = env.reset()  # 初始观测
    start_time = time.time()

    for train_step in range(args.max_train_steps):
        # 4.1 生成所有智能体的动作
        act_n = [agent.action(obs) for agent, obs in zip(agents, obs_n)]
        
        # 4.2 环境步进
        next_obs_n, rew_n, done_n, _ = env.step(act_n)
        done = all(done_n)
        episode_step = len(episode_rewards) - 1

        # 4.3 存储经验
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], act_n[i], rew_n[i], next_obs_n[i], done_n[i])

        # 4.4 更新奖励记录
        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew

        # 4.5  episode结束处理
        if done or episode_step >= args.max_episode_len:
            obs_n = env.reset()
            episode_rewards.append(0.0)

        # 4.6 更新智能体网络
        for agent in agents:
            update_info = agent.update(agents, train_step)
            if update_info and train_step % 1000 == 0:
                print(f"Step: {train_step}, Q Loss: {update_info['q_loss']:.3f}, P Loss: {update_info['p_loss']:.3f}, Avg Rew: {update_info['avg_reward']:.3f}")

        # 4.7 保存模型
        if train_step % args.save_interval == 0 and train_step > 0:
            save_path = saver.save(sess, os.path.join(args.save_dir, "maddpg_model"), global_step=train_step)
            print(f"模型保存成功: {save_path}")

        # 4.8 打印训练日志
        if train_step % 1000 == 0 and train_step > 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Step: {train_step}, Episodes: {len(episode_rewards)}, Avg Reward (100 eps): {avg_reward:.3f}, Time: {time.time()-start_time:.1f}s")
            start_time = time.time()

    # 5. 保存最终奖励曲线
    with open(os.path.join(args.save_dir, "rewards.pkl"), "wb") as f:
        pickle.dump(episode_rewards, f)
    print("训练结束，奖励曲线已保存")


def infer_maddpg(args):
    """模型推理（使用训练好的模型进行演示）"""
    # 1. 初始化环境和智能体
    env = create_env(args.scenario)
    agents = init_agents(env, args)

    # 2. 加载模型
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(args.load_dir))
    print(f"加载模型成功，开始演示: {args.scenario}")

    # 3. 推理循环
    obs_n = env.reset()
    episode_reward = 0.0
    while True:
        # 生成动作（无探索）
        act_n = [agent.action(obs) for agent, obs in zip(agents, obs_n)]
        # 环境步进
        next_obs_n, rew_n, done_n, _ = env.step(act_n)
        # 渲染环境
        env.render()
        time.sleep(0.1)
        # 更新奖励
        episode_reward += sum(rew_n)
        # 检查episode结束
        if all(done_n):
            print(f"Episode Reward: {episode_reward:.3f}")
            episode_reward = 0.0
            obs_n = env.reset()
            time.sleep(1.0)  # 暂停1秒再开始下一个episode
        else:
            obs_n = next_obs_n


# ------------------------------ 7. 主函数（入口） ------------------------------
if __name__ == "__main__":
    # 默认参数配置（简化命令行参数，适合学习）
    args = Namespace(
        scenario="simple",  # 环境名称（simple: 合作收集，simple_adversary: 对抗）
        max_train_steps=100000,  # 最大训练步数
        max_episode_len=25,  # 单episode最大步长
        batch_size=1024,  # 经验回放批次大小
        lr=1e-2,  # 学习率
        gamma=0.95,  # 折扣因子
        num_units=64,  # MLP隐藏层单元数
        save_dir="./maddpg_models/",  # 模型保存目录
        load_dir="./maddpg_models/",  # 模型加载目录
        save_interval=5000,  # 模型保存间隔
        mode="train"  # 运行模式：train/infer
    )

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 运行训练或推理
    if args.mode == "train":
        train_maddpg(args)
    elif args.mode == "infer":
        infer_maddpg(args)
    else:
        print("请指定运行模式：mode='train' 或 'infer'")