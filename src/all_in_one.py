import os
import json
import regex as re
import numpy as np
import tensorflow as tf
from functools import lru_cache
from tensorflow.contrib.training import HParams
from tensorflow.python.training import saver as saver_lib


# ------------------------------ 核心工具函数 ------------------------------
@lru_cache()
def bytes_to_unicode():
    """构建 UTF-8 字节与 Unicode 字符的映射表"""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """提取词（符号序列）中的相邻符号对"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# ------------------------------ BPE 编解码模块 ------------------------------
class BPEEncoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        """对单个分词后的 token 执行 BPE 合并"""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        result = ' '.join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        """文本转 token ID 序列（端到端编码）"""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token_unicode = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token_strs = self.bpe(token_unicode).split(' ')
            bpe_tokens.extend(self.encoder[token_str] for token_str in bpe_token_strs)
        return bpe_tokens

    def decode(self, tokens):
        """token ID 序列转文本（端到端解码）"""
        bpe_symbols = ''.join([self.decoder[token] for token in tokens])
        bytes_seq = bytearray([self.byte_decoder[c] for c in bpe_symbols])
        text = bytes_seq.decode('utf-8', errors=self.errors)
        return text


def load_encoder(model_dir):
    """从模型目录加载 BPE 编码器"""
    with open(os.path.join(model_dir, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(model_dir, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return BPEEncoder(encoder=encoder, bpe_merges=bpe_merges)


# ------------------------------ Transformer 模型核心（补全） ------------------------------
def default_hparams():
    """默认模型超参数"""
    return HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        learning_rate=5e-5,
        batch_size=4,
        max_train_steps=10000,
        save_interval=1000,
        log_interval=100
    )


def shape_list(x):
    """处理 TensorFlow 动态形状"""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def softmax(x, axis=-1):
    """数值稳定的 softmax 函数"""
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def gelu(x):
    """激活函数：Gaussian Error Linear Unit"""
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def norm(x, scope, axis=-1, epsilon=1e-5):
    """层归一化（Layer Normalization）"""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        return x * g + b


def conv1d(x, scope, nf, w_init_stdev=0.02):
    """1D 卷积"""
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        x_flat = tf.reshape(x, [-1, nx])
        w_flat = tf.reshape(w, [-1, nf])
        conv_out = tf.matmul(x_flat, w_flat) + b
        return tf.reshape(conv_out, start + [nf])


def attention_mask(nd, ns, dtype):
    """生成注意力掩码（下三角掩码）"""
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    mask = i >= j - ns + nd
    return tf.cast(mask, dtype)


def multihead_attention(x, scope, hparams, past=None):
    """多头自注意力层（修正原代码注释错误）"""
    assert x.shape.ndims == 3
    n_state = x.shape[-1].value
    assert n_state % hparams.n_head == 0
    embd_per_head = n_state // hparams.n_head

    def split_heads(t):
        return tf.transpose(tf.reshape(t, shape_list(t) + [hparams.n_head, embd_per_head]), [0, 2, 1, 3])

    def merge_heads(t):
        return tf.reshape(tf.transpose(t, [0, 2, 1, 3]), shape_list(t)[:2] + [n_state])

    def mask_weights(w):
        _, _, nd, ns = shape_list(w)
        mask = attention_mask(nd, ns, dtype=w.dtype)
        mask = tf.reshape(mask, [1, 1, nd, ns])
        return w * mask - tf.cast(1e10, w.dtype) * (1 - mask)

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state * 3)
        q, k, v = tf.split(c, 3, axis=2)
        q, k, v = map(split_heads, [q, k, v])

        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)

        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(embd_per_head, w.dtype))
        w = mask_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)

        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, hparams):
    """前馈神经网络层（MLP）"""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state * 4))
        h2 = conv1d(h, 'c_proj', n_state)
        return h2


def transformer_block(x, scope, hparams, past=None):
    """Transformer 解码器块"""
    with tf.variable_scope(scope):
        attn_out, present = multihead_attention(norm(x, 'ln_1'), 'attn', hparams, past=past)
        x = x + attn_out
        mlp_out = mlp(norm(x, 'ln_2'), 'mlp', hparams)
        x = x + mlp_out
        return x, present


def positions_for(tokens, past_length):
    """生成位置序列（用于位置嵌入）"""
    batch_size = tf.shape(tokens)[0]
    seq_len = tf.shape(tokens)[1]
    return tf.tile(tf.expand_dims(past_length + tf.range(seq_len), 0), [batch_size, 1])


def gpt_model(hparams, X, past=None, scope='model', reuse=False):
    """GPT 核心模型（补全：堆叠 Transformer 层 + 输出 Logits）"""
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch_size, seq_len = shape_list(X)

        # 1. 嵌入层（词嵌入 + 位置嵌入）
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.02))
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.01))
        past_length = 0 if past is None else tf.shape(past)[-2]
        position_ids = positions_for(X, past_length)
        h = tf.gather(wte, X) + tf.gather(wpe, position_ids)

        # 2. 堆叠多层 Transformer 解码器块
        presents = []
        if past is not None:
            past = tf.unstack(past, axis=1)  # [n_layer, 2, batch, head, seq, embd_per_head]
        for layer in range(hparams.n_layer):
            h, present = transformer_block(
                h, scope=f'h{layer}', hparams=hparams,
                past=past[layer] if past is not None else None
            )
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)  # 保存所有层的 K/V 状态（用于增量生成）

        # 3. 输出层（Logits：词表维度的分数，权重共享 wte）
        h = norm(h, 'ln_f')  # 最终层归一化
        logits = tf.matmul(tf.reshape(h, [-1, hparams.n_embd]), wte, transpose_b=True)
        logits = tf.reshape(logits, [batch_size, seq_len, hparams.n_vocab])
        results['logits'] = logits

        return results


# ------------------------------ 新增：预训练数据处理模块 ------------------------------
def load_tokenized_corpus(corpus_path):
    """加载预处理好的 Token ID 语料（每行是一个 token 列表，如 "[123, 456, ...]"）"""
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token_ids = json.loads(line)
            corpus.extend(token_ids)  # 拼接所有 token 为长序列
    return corpus


def create_train_dataset(corpus, hparams):
    """将长 Token 序列切分为训练样本（自回归任务：X 是前 n_ctx-1 个 token，Y 是后 n_ctx-1 个 token）"""
    n_ctx = hparams.n_ctx
    total_tokens = len(corpus)
    # 每个样本长度为 n_ctx，可生成 (n_ctx-1) 个预测目标
    num_samples = (total_tokens - 1) // n_ctx

    # 切分样本
    X = []
    Y = []
    for i in range(num_samples):
        start = i * n_ctx
        end = start + n_ctx
        sample = corpus[start:end]
        if len(sample) < n_ctx:
            continue  # 跳过最后不足长度的样本
        X.append(sample[:-1])  # 输入：前 1023 个 token
        Y.append(sample[1:])   # 目标：后 1023 个 token（错开一位）

    # 转为 TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(hparams.batch_size, drop_remainder=True)
    dataset = dataset.repeat()  # 无限重复（训练时按 steps 停止）
    return dataset.make_one_shot_iterator().get_next()


# ------------------------------ 新增：预训练目标与训练流程 ------------------------------
def create_train_ops(logits, labels, hparams):
    """创建训练操作（交叉熵损失 + Adam 优化器）"""
    # 1. 计算损失（忽略 pad token，这里默认无 pad，若有需加 mask）
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits[:, :-1],  # 输入 logits 比 labels 多一位（去掉最后一个）
        labels=labels[:, 1:]    # 目标 labels 比输入多一位（去掉第一个）
    )
    mean_loss = tf.reduce_mean(loss)

    # 2. 优化器（Adam）
    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

    # 3. 梯度裁剪（防止梯度爆炸）
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(mean_loss, tvars), clip_norm=1.0)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    return train_op, mean_loss


def train_gpt2(model_dir, corpus_path, hparams=None):
    """GPT-2 预训练主流程"""
    # 1. 初始化参数
    if hparams is None:
        hparams = default_hparams()
    os.makedirs(model_dir, exist_ok=True)

    # 2. 加载数据与编码器
    print("Loading corpus and encoder...")
    corpus = load_tokenized_corpus(corpus_path)
    encoder = load_encoder(model_dir)  # 需提前将 encoder.json/vocab.bpe 放入 model_dir
    X_train, Y_train = create_train_dataset(corpus, hparams)

    # 3. 构建模型
    print("Building GPT-2 model...")
    model_outputs = gpt_model(hparams, X_train)
    logits = model_outputs['logits']
    train_op, mean_loss = create_train_ops(logits, Y_train, hparams)

    # 4. 模型保存器
    saver = saver_lib.Saver(tf.trainable_variables(), max_to_keep=5)
    init_op = tf.global_variables_initializer()

    # 5. 启动训练
    with tf.Session() as sess:
        sess.run(init_op)
        print("Start training...")

        for step in range(1, hparams.max_train_steps + 1):
            _, loss_val = sess.run([train_op, mean_loss])

            # 打印日志
            if step % hparams.log_interval == 0:
                print(f"Step {step:5d} | Loss: {loss_val:.4f}")

            # 保存模型
            if step % hparams.save_interval == 0 or step == hparams.max_train_steps:
                save_path = os.path.join(model_dir, f"model_step_{step}")
                saver.save(sess, save_path)
                print(f"Model saved to {save_path}")

    print("Training completed!")


# ------------------------------ 新增：推理生成模块 ------------------------------
def generate_text(encoder, model_dir, prompt, hparams=None, max_len=50, top_k=40):
    """
    用训练好的 GPT-2 生成文本
    :param encoder: BPEEncoder 实例
    :param model_dir: 模型保存目录（含 checkpoint）
    :param prompt: 提示文本（str）
    :param max_len: 生成文本的最大长度（含 prompt）
    :param top_k: Top-K 采样（仅保留概率前 K 的 token）
    :return: 生成的文本（str）
    """
    if hparams is None:
        hparams = default_hparams()

    # 1. 编码 prompt 为 token ID
    prompt_tokens = encoder.encode(prompt)
    if len(prompt_tokens) >= max_len:
        return prompt  # 若 prompt 已超长，直接返回

    # 2. 构建推理模型（增量生成，复用 past 状态）
    tf.reset_default_graph()
    X = tf.placeholder(tf.int32, [1, None])  # [batch=1, seq_len]
    past = tf.placeholder(tf.float32, [hparams.n_layer, 2, 1, hparams.n_head, None, hparams.n_embd//hparams.n_head])
    model_outputs = gpt_model(hparams, X, past=past, reuse=tf.AUTO_REUSE)
    logits = model_outputs['logits']
    present = model_outputs['present']

    # 3. 加载模型权重
    saver = saver_lib.Saver()
    with tf.Session() as sess:
        # 加载最新的 checkpoint
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, latest_ckpt)
        print(f"Loaded model from {latest_ckpt}")

        # 4. 增量生成
        current_tokens = prompt_tokens
        past_state = None  # 初始无历史状态

        while len(current_tokens) < max_len:
            # 输入当前 token 序列（仅最后一个 token，或全序列？这里用全序列，简化实现）
            feed_dict = {X: [current_tokens]}
            if past_state is not None:
                feed_dict[past] = past_state

            # 预测下一个 token 的 logits
            logits_val, present_val = sess.run([logits, present], feed_dict=feed_dict)
            next_logits = logits_val[0, -1, :]  # 取最后一个位置的 logits

            # Top-K 采样（过滤低概率 token）
            top_k_indices = tf.math.top_k(next_logits, k=top_k).indices.numpy()
            top_k_logits = next_logits[top_k_indices]
            top_k_probs = softmax(top_k_logits).numpy()

            # 随机选择下一个 token
            next_token = np.random.choice(top_k_indices, p=top_k_probs)
            current_tokens.append(next_token)
            past_state = present_val  # 更新历史状态

    # 5. 解码为文本
    generated_text = encoder.decode(current_tokens)
    return generated_text


# ------------------------------ 新增：运行入口 ------------------------------
if __name__ == "__main__":
    # 配置参数
    MODEL_DIR = "./gpt2_small_model"  # 模型保存/加载目录
    CORPUS_PATH = "./tokenized_corpus.txt"  # 预处理好的 Token 语料路径

    # 1. 预训练（首次运行时执行，需提前准备语料和 BPE 文件）
    # 注意：预训练需大量计算资源（建议 GPU，12GB 以上显存）
    # train_gpt2(model_dir=MODEL_DIR, corpus_path=CORPUS_PATH)

    # 2. 推理生成（训练完成后执行）
    encoder = load_encoder(MODEL_DIR)
    prompt = "Artificial intelligence is "
    generated_text = generate_text(
        encoder=encoder,
        model_dir=MODEL_DIR,
        prompt=prompt,
        max_len=100,
        top_k=40
    )
    print("Generated Text:\n", generated_text)