import os
import time
import numpy as np
import tensorflow as tf
import sonnet as snt
from six.moves import xrange


# -------------------------- 1. 基础预处理模块（整合原preprocess） --------------------------
class Clamp(snt.AbstractModule):
    """梯度裁剪预处理：将值限制在[min_value, max_value]范围内"""
    def __init__(self, min_value=None, max_value=None, name="clamp"):
        super(Clamp, self).__init__(name=name)
        self._min = min_value
        self._max = max_value

    def _build(self, inputs):
        output = inputs
        if self._min is not None:
            output = tf.maximum(output, self._min)
        if self._max is not None:
            output = tf.minimum(output, self._max)
        return output


class LogAndSign(snt.AbstractModule):
    """梯度预处理：按论文《Learning to Learn》实现log+sign变换，增强梯度特征"""
    def __init__(self, k, name="log_and_sign"):
        super(LogAndSign, self).__init__(name=name)
        self._k = k

    def _build(self, gradients):
        eps = np.finfo(gradients.dtype.as_numpy_dtype).eps
        ndims = gradients.get_shape().ndims
        
        # log(|grad| + eps) 并裁剪，sign(grad) 并裁剪
        log = tf.log(tf.abs(gradients) + eps)
        clamped_log = Clamp(min_value=-1.0)(log / self._k)
        sign = Clamp(min_value=-1.0, max_value=1.0)(gradients * np.exp(self._k))
        
        return tf.concat([clamped_log, sign], ndims - 1)


# -------------------------- 2. 元优化器核心网络（整合原networks） --------------------------
class Network(snt.RNNCore):
    """所有元优化器网络的基类，定义统一接口"""
    @staticmethod
    def initial_state_for_inputs(inputs, **kwargs):
        """根据输入形状生成初始状态"""
        raise NotImplementedError


class CoordinateWiseDeepLSTM(Network):
    """逐坐标DeepLSTM元优化器：适合标量/低维参数优化"""
    def __init__(self, layers=(20, 20), preprocess_name="LogAndSign", 
                 preprocess_options={"k":5}, scale=0.01, initializer="zeros", name="cw_lstm"):
        super(CoordinateWiseDeepLSTM, self).__init__(name=name)
        self._scale = scale

        # 梯度预处理
        if preprocess_name == "LogAndSign":
            self._preprocess = LogAndSign(**preprocess_options)
        else:
            self._preprocess = getattr(tf, preprocess_name)  # 如tf.identity

        # 构建DeepLSTM和输出层
        with tf.variable_scope(self._template.variable_scope):
            # LSTM堆叠
            self._lstms = [snt.LSTM(size, initializers=self._get_initializer(initializer, f"lstm_{i}")) 
                           for i, size in enumerate(layers)]
            self._deep_rnn = snt.DeepRNN(self._lstms)
            # 输出层（逐坐标输出1个值）
            self._linear = snt.Linear(1, initializers=self._get_initializer(initializer, "linear"))

    def _get_initializer(self, init_config, layer_name):
        """解析初始化配置（支持字符串、字典或TensorFlow初始化器）"""
        if isinstance(init_config, str):
            return getattr(tf, init_config + "_initializer")(dtype=tf.float32)
        elif isinstance(init_config, dict) and layer_name in init_config:
            return self._get_initializer(init_config[layer_name], "")
        return init_config

    def _reshape_inputs(self, inputs):
        """将任意形状输入转为[num_coords, 1]（逐坐标处理）"""
        return tf.reshape(inputs, [-1, 1])

    def _build(self, inputs, prev_state):
        """输入梯度，输出参数更新量和新状态"""
        input_shape = inputs.get_shape().as_list()
        reshaped_inputs = self._reshape_inputs(inputs)

        # 预处理→LSTM→线性输出
        preprocessed = self._preprocess(tf.expand_dims(reshaped_inputs, -1))
        preprocessed = tf.reshape(preprocessed, [preprocessed.shape[0], -1])
        lstm_out, next_state = self._deep_rnn(preprocessed, prev_state)
        update = self._linear(lstm_out) * self._scale

        # 恢复原始形状
        return tf.reshape(update, input_shape), next_state

    @staticmethod
    def initial_state_for_inputs(inputs, **kwargs):
        """生成LSTM初始状态（全零）"""
        batch_size = int(np.prod(inputs.get_shape().as_list()))
        return snt.DeepRNN.initial_state_from_inputs(None, batch_size, tf.float32)


class Adam(Network):
    """传统Adam优化器（作为对比基准，非可训练元优化器）"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="adam"):
        super(Adam, self).__init__(name=name)
        self._lr = learning_rate
        self._b1 = beta1
        self._b2 = beta2
        self._eps = epsilon

    def _build(self, g, prev_state):
        """g: 梯度；prev_state: (t, m, v) 迭代次数/一阶矩/二阶矩"""
        t, m, v = prev_state
        t_next = t + 1

        # Adam核心公式
        m_next = self._b1 * m + (1 - self._b1) * g
        v_next = self._b2 * v + (1 - self._b2) * tf.square(g)
        m_hat = m_next / (1 - tf.pow(self._b1, t_next))
        v_hat = v_next / (1 - tf.pow(self._b2, t_next))
        update = -self._lr * m_hat / (tf.sqrt(v_hat) + self._eps)

        return update, (t_next, m_next, v_next)

    @staticmethod
    def initial_state_for_inputs(inputs, **kwargs):
        """生成Adam初始状态（t=0，m/v全零）"""
        batch_size = int(np.prod(inputs.get_shape().as_list()))
        dtype = kwargs.get("dtype", tf.float32)
        return (tf.zeros((), dtype=dtype),
                tf.zeros((batch_size, 1), dtype=dtype),
                tf.zeros((batch_size, 1), dtype=dtype))


class SGD(Network):
    """传统SGD优化器（对比基准）"""
    def __init__(self, learning_rate=0.001, name="sgd"):
        super(SGD, self).__init__(name=name)
        self._lr = learning_rate

    def _build(self, g, prev_state):
        return -self._lr * g, []  # SGD无状态，返回空列表

    @staticmethod
    def initial_state_for_inputs(inputs, **kwargs):
        return []


# -------------------------- 3. 待优化问题定义（整合原problems） --------------------------
def simple_problem():
    """简单二次函数问题：f(x) = x²（目标：找到x使f(x)最小）"""
    def build():
        x = tf.get_variable("x", shape=[], dtype=tf.float32, initializer=tf.ones_initializer())
        return tf.square(x, name="loss")
    return build


def quadratic_problem(batch_size=32, num_dims=5):
    """高维二次函数问题：f(x) = ||Wx - y||²（多参数优化）"""
    def build():
        # 可训练参数x
        x = tf.get_variable("x", shape=[batch_size, num_dims], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(stddev=0.01))
        # 固定参数W和y（模拟数据分布）
        w = tf.get_variable("w", shape=[batch_size, num_dims, num_dims], dtype=tf.float32,
                           initializer=tf.random_uniform_initializer(), trainable=False)
        y = tf.get_variable("y", shape=[batch_size, num_dims], dtype=tf.float32,
                           initializer=tf.random_uniform_initializer(), trainable=False)
        # 计算损失
        product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
        return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))
    return build


# -------------------------- 4. 元优化器核心实现（整合原meta） --------------------------
class MetaOptimizer:
    """L2L元优化器：用可训练网络学习优化策略"""
    def __init__(self, net_config):
        """
        Args:
            net_config: 网络配置，如{"cw_lstm": {"net": CoordinateWiseDeepLSTM, "options": {...}}}
        """
        self._net_config = net_config
        self._nets = None  # 实际初始化的网络实例

    def _get_optimizee_vars(self, make_loss):
        """获取待优化问题的可训练参数和固定参数"""
        trainable_vars = []
        non_trainable_vars = []

        def custom_getter(getter, name, **kwargs):
            """拦截变量创建，分类可训练/不可训练参数"""
            var = getter(name, **kwargs)
            if kwargs["trainable"]:
                trainable_vars.append(var)
            else:
                non_trainable_vars.append(var)
            return var

        # 临时创建图获取变量，不执行计算
        with tf.variable_scope("temp_optimizee"):
            with tf.variable_scope("", custom_getter=custom_getter):
                make_loss()
        return trainable_vars, non_trainable_vars

    def _build_update_step(self, make_loss, x, state, net):
        """构建单步优化：输入梯度→输出参数更新量和新状态"""
        # 计算损失对x的梯度
        loss = self._make_with_vars(make_loss, x)
        grads = tf.gradients(loss, x)
        
        # 网络生成参数更新量
        updates, new_states = [], []
        for g, s, var in zip(grads, state, x):
            update, new_s = net(g, s)
            updates.append(update)
            new_states.append(new_s)
        
        # 更新参数：x_new = x + update
        x_new = [var + update for var, update in zip(x, updates)]
        return loss, x_new, new_states

    def _make_with_vars(self, make_loss, vars_list):
        """用指定的参数列表执行待优化问题的损失计算"""
        var_iter = iter(vars_list)
        def custom_getter(getter, name, **kwargs):
            if kwargs["trainable"]:
                return next(var_iter)
            kwargs["reuse"] = True
            return getter(name, **kwargs)
        
        with tf.variable_scope("", custom_getter=custom_getter):
            return make_loss()

    def meta_minimize(self, make_loss, unroll_length, lr=0.001):
        """
        构建元优化图：展开unroll_length步优化过程，计算元损失并优化
        Args:
            make_loss: 生成待优化损失的函数
            unroll_length: 展开优化步数（元学习的序列长度）
            lr: 元优化器自身的学习率
        Returns:
            step: 元优化器的参数更新op
            update: 待优化问题的参数更新op
            reset: 重置状态的op
            final_loss: 展开结束后的损失
            final_x: 展开结束后的参数值
        """
        # 1. 获取待优化问题的参数
        x, constants = self._get_optimizee_vars(make_loss)
        if not x:
            raise ValueError("待优化问题没有可训练参数")

        # 2. 初始化元优化网络（默认用第一个配置的网络）
        net_key = next(iter(self._net_config.keys()))
        net_cls = self._net_config[net_key]["net"]
        net_opts = self._net_config[net_key]["options"]
        self._nets = {net_key: net_cls(**net_opts)}
        net = self._nets[net_key]

        # 3. 初始化网络状态（每个参数对应一个状态）
        state = [net.initial_state_for_inputs(var) for var in x]
        state_vars = [tf.Variable(s, trainable=False) for s in state]

        # 4. 构建多步优化展开（unroll）
        loss_history = []
        current_x = x
        current_state = state_vars

        for _ in xrange(unroll_length):
            loss, current_x, current_state = self._build_update_step(
                make_loss, current_x, current_state, net
            )
            loss_history.append(loss)

        # 5. 元损失：展开过程的损失和
        meta_loss = tf.reduce_sum(loss_history, name="meta_loss")

        # 6. 元优化器的更新op（用Adam优化元损失）
        meta_optimizer = tf.train.AdamOptimizer(lr)
        step = meta_optimizer.minimize(meta_loss)

        # 7. 重置op：重置待优化参数和网络状态
        reset_vars = x + constants + state_vars
        reset = tf.variables_initializer(reset_vars)

        # 8. 参数更新op：将当前参数和状态写入变量
        update_x = [tf.assign(var, new_val) for var, new_val in zip(x, current_x)]
        update_state = [tf.assign(var, new_val) for var, new_val in zip(state_vars, current_state)]
        update = update_x + update_state

        return step, update, reset, meta_loss, current_x

    def save(self, sess, save_path):
        """保存元优化器网络参数"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for name, net in self._nets.items():
            var_dict = {var.name: var for var in snt.get_variables_in_module(net)}
            np.savez(os.path.join(save_path, f"{name}.npz"), **sess.run(var_dict))
        print(f"元优化器已保存至 {save_path}")

    def load(self, sess, save_path):
        """加载元优化器网络参数"""
        for name, net in self._nets.items():
            data = np.load(os.path.join(save_path, f"{name}.npz"))
            for var in snt.get_variables_in_module(net):
                sess.run(tf.assign(var, data[var.name]))
        print(f"元优化器已从 {save_path} 加载")


# -------------------------- 5. 工具函数（整合原util） --------------------------
def run_epoch(sess, loss_op, update_ops, reset_op, num_unrolls):
    """执行一轮优化（多个unroll步）"""
    start_time = time.time()
    sess.run(reset_op)  # 重置状态
    final_loss = 0.0
    for _ in xrange(num_unrolls):
        loss, _ = sess.run([loss_op, update_ops])
        final_loss = loss  # 记录最后一步的损失
    return time.time() - start_time, final_loss


def print_stats(header, total_loss, total_time, epochs):
    """打印训练/测试统计信息"""
    print(f"\n{header}")
    print(f"平均最终损失: {total_loss / epochs:.6f}")
    print(f"平均耗时: {total_time / epochs:.2f}s")


# -------------------------- 6. 端到端流程：元训练→保存→元测试 --------------------------
def meta_train():
    """阶段1：在简单问题上训练元优化器"""
    # 1. 配置参数
    config = {
        "problem": simple_problem(),  # 待优化问题（元训练任务）
        "unroll_length": 20,          # 每轮展开的优化步数
        "meta_lr": 1e-3,              # 元优化器的学习率
        "num_epochs": 1000,           # 元训练轮数
        "log_period": 200,            # 日志打印间隔
        "save_path": "./l2l_meta_optimizer"  # 模型保存路径
    }

    # 2. 初始化元优化器（用CoordinateWiseDeepLSTM作为元优化网络）
    net_config = {
        "cw_lstm": {
            "net": CoordinateWiseDeepLSTM,
            "options": {
                "layers": (16, 16),    # LSTM层数和隐藏层大小
                "preprocess_name": "LogAndSign",
                "preprocess_options": {"k":5},
                "scale": 0.01
            }
        }
    }
    meta_opt = MetaOptimizer(net_config)

    # 3. 构建元优化图
    step, update, reset, loss_op, _ = meta_opt.meta_minimize(
        make_loss=config["problem"],
        unroll_length=config["unroll_length"],
        lr=config["meta_lr"]
    )

    # 4. 开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_time = 0.0
        total_loss = 0.0

        for epoch in xrange(1, config["num_epochs"] + 1):
            epoch_time, epoch_loss = run_epoch(
                sess=sess,
                loss_op=loss_op,
                update_ops=[step, update],  # 同时更新元优化器和待优化参数
                reset_op=reset,
                num_unrolls=1  # 每轮1个unroll（可根据需求调整）
            )
            total_time += epoch_time
            total_loss += epoch_loss

            # 打印日志
            if epoch % config["log_period"] == 0:
                print_stats(f"元训练轮次 {epoch}", total_loss, total_time, config["log_period"])
                total_time = 0.0
                total_loss = 0.0

        # 5. 保存训练好的元优化器
        meta_opt.save(sess, config["save_path"])


def meta_test():
    """阶段2：用训练好的元优化器解决新问题（对比传统Adam）"""
    # 1. 配置参数
    config = {
        "test_problem": quadratic_problem(batch_size=8, num_dims=5),  # 新问题（高维二次函数）
        "unroll_length": 30,          # 测试时的优化步数
        "num_test_epochs": 50,        # 测试轮数
        "load_path": "./l2l_meta_optimizer"  # 加载元优化器的路径
    }

    # 2. 定义两个优化器：L2L元优化器 vs Adam
    # 2.1 L2L元优化器（加载预训练参数）
    l2l_net_config = {
        "cw_lstm": {
            "net": CoordinateWiseDeepLSTM,
            "options": {
                "layers": (16, 16),
                "preprocess_name": "LogAndSign",
                "preprocess_options": {"k":5},
                "scale": 0.01
            }
        }
    }
    l2l_opt = MetaOptimizer(l2l_net_config)
    l2l_step, l2l_update, l2l_reset, l2l_loss, _ = l2l_opt.meta_minimize(
        make_loss=config["test_problem"],
        unroll_length=config["unroll_length"],
        lr=0.0  # 测试时不更新元优化器
    )

    # 2.2 传统Adam优化器（对比基准）
    def adam_optimize(make_loss):
        """用传统Adam构建优化流程"""
        loss = make_loss()
        vars_list = tf.trainable_variables(scope="temp_optimizee")
        adam = tf.train.AdamOptimizer(learning_rate=0.01)
        update = adam.minimize(loss)
        reset = tf.variables_initializer(vars_list + adam.get_slot_names())
        return loss, update, reset

    # 3. 执行测试
    with tf.Session() as sess:
        # 初始化所有变量
        sess.run(tf.global_variables_initializer())
        # 加载预训练的元优化器
        l2l_opt.load(sess, config["load_path"])

        # 3.1 测试L2L元优化器
        l2l_total_time = 0.0
        l2l_total_loss = 0.0
        for _ in xrange(config["num_test_epochs"]):
            epoch_time, epoch_loss = run_epoch(
                sess=sess,
                loss_op=l2l_loss,
                update_ops=l2l_update,
                reset_op=l2l_reset,
                num_unrolls=1
            )
            l2l_total_time += epoch_time
            l2l_total_loss += epoch_loss
        print_stats("=== L2L元优化器测试结果 ===", l2l_total_loss, l2l_total_time, config["num_test_epochs"])

        # 3.2 测试传统Adam优化器
        adam_loss, adam_update, adam_reset = adam_optimize(config["test_problem"])
        sess.run(tf.variables_initializer(tf.global_variables(scope="temp_optimizee")))
        
        adam_total_time = 0.0
        adam_total_loss = 0.0
        for _ in xrange(config["num_test_epochs"]):
            epoch_time, epoch_loss = run_epoch(
                sess=sess,
                loss_op=adam_loss,
                update_ops=adam_update,
                reset_op=adam_reset,
                num_unrolls=config["unroll_length"]
            )
            adam_total_time += epoch_time
            adam_total_loss += epoch_loss
        print_stats("=== 传统Adam优化器测试结果 ===", adam_total_loss, adam_total_time, config["num_test_epochs"])


if __name__ == "__main__":
    # 执行端到端流程
    print("="*50)
    print("开始元优化器预训练...")
    meta_train()

    print("\n" + "="*50)
    print("开始元测试（对比L2L与Adam）...")
    meta_test()