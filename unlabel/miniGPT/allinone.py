import os
import sys
import json
import time
import random
import math
from collections import defaultdict
from ast import literal_eval

import regex as re
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------- 基础配置与工具函数 --------------------------
class CfgNode:
    """轻量级配置类，用于统一管理参数"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        for arg in args:
            keyval = arg.split('=')
            assert len(keyval) == 2, "参数格式应为 --arg=value"
            key, val = keyval[0][2:], keyval[1]  # 去除--前缀
            try:
                val = literal_eval(val)
            except ValueError:
                pass
            # 处理嵌套参数（如 model.n_layer）
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            setattr(obj, keys[-1], val)


def set_seed(seed):
    """固定随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(work_dir):
    """创建工作目录，保存配置信息"""
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))


# -------------------------- BPE分词器（简化版） --------------------------
def bytes_to_unicode():
    """字节到Unicode字符的映射，处理特殊字符"""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


class BPETokenizer:
    """整合BPE编码/解码功能，简化接口"""
    def __init__(self):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.encoder, self.bpe_ranks = self._load_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def _load_vocab(self):
        """加载GPT-2预训练的词表和BPE合并规则"""
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'mingpt')
        os.makedirs(cache_dir, exist_ok=True)

        # 下载encoder.json（词表映射）
        encoder_path = os.path.join(cache_dir, 'encoder.json')
        if not os.path.exists(encoder_path):
            resp = requests.get('https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json')
            open(encoder_path, 'wb').write(resp.content)
        with open(encoder_path, 'r') as f:
            encoder = json.load(f)

        # 下载vocab.bpe（BPE合并规则）
        vocab_path = os.path.join(cache_dir, 'vocab.bpe')
        if not os.path.exists(vocab_path):
            resp = requests.get('https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe')
            open(vocab_path, 'wb').write(resp.content)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            bpe_data = f.read().split('\n')[1:-1]  # 去除首行版本和尾行空行
        bpe_ranks = dict(zip([tuple(line.split()) for line in bpe_data], range(len(bpe_data))))

        return encoder, bpe_ranks

    def _get_pairs(self, word):
        """获取词的相邻字符对（bigram）"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token):
        """对单个token执行BPE合并"""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return token

        while True:
            # 找到优先级最高的可合并字符对
            bigram = min(pairs, key=lambda x: self.bpe_ranks.get(x, float('inf')))
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
                if i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        result = ' '.join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        """文本转token索引列表"""
        bpe_idx = []
        for token in re.findall(self.pat, text):
            # 字节编码→Unicode映射→BPE合并→索引转换
            token_bytes = token.encode('utf-8')
            translated = ''.join([self.byte_encoder[b] for b in token_bytes])
            merged = self._bpe(translated).split(' ')
            bpe_idx.extend([self.encoder[tok] for tok in merged])
        return torch.tensor(bpe_idx, dtype=torch.long).unsqueeze(0)  # 增加batch维度

    def decode(self, idx):
        """token索引列表转文本"""
        idx = idx.squeeze(0).tolist()  # 去除batch维度
        tokens = [self.decoder[i] for i in idx]
        bytes_seq = bytearray([self.byte_decoder[c] for c in ''.join(tokens)])
        return bytes_seq.decode('utf-8', errors='replace')


# -------------------------- GPT模型核心组件 --------------------------
class NewGELU(nn.Module):
    """GELU激活函数，GPT-2默认使用"""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """带掩码的多头自注意力层（因果掩码，保证时序合理性）"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # QKV投影（合并为一个线性层提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 正则化
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 因果掩码（下三角矩阵，防止关注未来token）
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, -1, -1))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape  # B=批量大小, T=序列长度, C=嵌入维度
        # QKV拆分与多头维度调整
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)  # (B, 头数, T, 头维度)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

        # 注意力计算
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # 缩放点积
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # 掩码未来token
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, 头数, T, 头维度)
        # 合并多头结果
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 输出投影与残差 dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer基础块（注意力+前馈网络）"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)  # 注意力前归一化
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)  # 前馈网络前归一化
        # 前馈网络（4倍扩张）
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            NewGELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT主模型"""
    @staticmethod
    def get_default_config():
        """默认模型配置，支持预定义模型类型"""
        C = CfgNode()
        C.model_type = 'gpt-mini'  # 预定义小型模型，适合学习
        C.n_layer = None  # 层数（由model_type自动填充）
        C.n_head = None   # 头数（由model_type自动填充）
        C.n_embd = None   # 嵌入维度（由model_type自动填充）
        C.vocab_size = 50257  # GPT-2标准词表大小
        C.block_size = 1024   # 最大序列长度
        # Dropout参数
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # 预定义模型参数映射
        C.merge_from_dict({
            'gpt-nano': dict(n_layer=3, n_head=3, n_embd=48),
            'gpt-micro': dict(n_layer=4, n_head=4, n_embd=128),
            'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192),
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
        }[C.model_type])
        return C

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        # Transformer主体
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入
            'wpe': nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入
            'drop': nn.Dropout(config.embd_pdrop),                  # 嵌入dropout
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer块
            'ln_f': nn.LayerNorm(config.n_embd),                    # 最终归一化
        })
        # 语言模型头（无偏置，与词嵌入权重共享）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化权重
        self.apply(self._init_weights)
        # 对残差投影层进行特殊初始化（GPT-2论文推荐）
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"模型参数数量: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        """前向传播（含损失计算）"""
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度{T}超过模型最大长度{self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)

        # 嵌入层：词嵌入+位置嵌入
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (1, T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # 经过Transformer块
        for block in self.transformer.h:
            x = block(x)

        # 最终归一化与语言模型头
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失（如果提供目标）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """文本生成（贪心或采样）"""
        self.eval()
        for _ in range(max_new_tokens):
            # 裁剪序列到最大长度（只保留最后block_size个token）
            idx_cond = idx[:, -self.block_size:] if idx.size(1) > self.block_size else idx
            # 前向传播获取logits
            logits, _ = self(idx_cond)
            # 只取最后一个token的logits
            logits = logits[:, -1, :] / temperature
            # Top-k过滤（可选）
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            # 转为概率
            probs = F.softmax(logits, dim=-1)
            # 采样或贪心选择
            idx_next = torch.multinomial(probs, num_samples=1) if do_sample else torch.argmax(probs, dim=-1, keepdim=True)
            # 拼接新token
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


# -------------------------- 数据集与训练器 --------------------------
class CharDataset(Dataset):
    """字符级语言模型数据集（简单易理解，适合入门）"""
    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.block_size = 128  # 序列长度
        return C

    def __init__(self, config, text_path):
        self.config = config
        # 读取文本数据
        with open(text_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        # 构建字符映射表
        self.chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        print(f"数据长度: {len(self.text)} 字符, 词表大小: {self.vocab_size}")

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.text) - self.config.block_size

    def __getitem__(self, idx):
        # 截取序列：输入x为前block_size个字符，目标y为后block_size个字符
        chunk = self.text[idx:idx + self.config.block_size + 1]
        dix = [self.stoi[c] for c in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class Trainer:
    """简化版训练器，适配GPT模型"""
    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.device = 'auto'  # 自动选择设备（cuda/cpu）
        C.batch_size = 32  # 批量大小
        C.max_iters = 5000  # 最大迭代次数
        C.learning_rate = 3e-4  # 学习率
        C.betas = (0.9, 0.95)  # AdamW参数
        C.weight_decay = 0.1  # 权重衰减（正则化）
        C.grad_norm_clip = 1.0  # 梯度裁剪阈值
        C.num_workers = 2  # 数据加载线程数
        return C

    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device
        self.model.to(self.device)
        # 优化器
        self.optimizer = self._get_optimizer()
        # 数据加载器
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        # 训练状态
        self.iter_num = 0
        self.iter_time = time.time()
        self.callbacks = defaultdict(list)

    def _get_optimizer(self):
        """分离需要权重衰减的参数（正则化）"""
        decay_params = []
        no_decay_params = []
        for pn, p in self.model.named_parameters():
            if pn.endswith('bias') or isinstance(p, (nn.LayerNorm, nn.Embedding)):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        # AdamW优化器
        return torch.optim.AdamW(
            [{'params': decay_params, 'weight_decay': self.config.weight_decay},
             {'params': no_decay_params, 'weight_decay': 0.0}],
            lr=self.config.learning_rate,
            betas=self.config.betas
        )

    def add_callback(self, event, func):
        self.callbacks[event].append(func)

    def _trigger_callbacks(self, event):
        for func in self.callbacks[event]:
            func(self)

    def run(self):
        """启动训练"""
        self.model.train()
        data_iter = iter(self.dataloader)
        while self.iter_num < self.config.max_iters:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            
            # 数据移至设备
            x, y = [t.to(self.device) for t in batch]
            # 前向传播计算损失
            logits, loss = self.model(x, y)
            # 反向传播与参数更新
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()

            # 记录状态并触发回调
            self.iter_num += 1
            iter_dt = time.time() - self.iter_time
            self.iter_time = time.time()
            self.loss = loss  # 供回调函数访问
            self.iter_dt = iter_dt  # 供回调函数访问
            self._trigger_callbacks('on_batch_end')


# -------------------------- 端到端主流程 --------------------------
if __name__ == '__main__':
    # 1. 配置初始化
    config = CfgNode()
    # 系统配置
    config.system = CfgNode(seed=3407, work_dir='./out/chargpt')
    # 数据配置
    config.data = CharDataset.get_default_config()
    # 模型配置
    config.model = GPT.get_default_config()
    # 训练配置
    config.trainer = Trainer.get_default_config()

    # 从命令行覆盖配置（例如 --trainer.batch_size=16 --model.model_type=gpt-micro）
    if len(sys.argv) > 1:
        config.merge_from_args(sys.argv[1:])

    # 2. 初始化准备
    set_seed(config.system.seed)
    setup_logging(config.system.work_dir)

    # 3. 数据准备（请确保当前目录有input.txt文件，内容可自选，如莎士比亚文本）
    if not os.path.exists('input.txt'):
        # 自动下载莎士比亚文本作为示例数据
        resp = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
        open('input.txt', 'wb').write(resp.content)
    dataset = CharDataset(config.data, 'input.txt')

    # 4. 模型初始化
    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()
    model = GPT(config.model)

    # 5. 训练器初始化与回调函数
    trainer = Trainer(config.trainer, model, dataset)

    # 训练回调（每10步打印损失，每500步测试生成并保存模型）
    def train_callback(trainer):
        # 打印训练状态
        if trainer.iter_num % 10 == 0:
            print(f"迭代 {trainer.iter_num:4d} | 耗时 {trainer.iter_dt*1000:.1f}ms | 损失 {trainer.loss.item():.4f}")
        
        # 测试生成与保存模型
        if trainer.iter_num % 500 == 0 and trainer.iter_num > 0:
            # 字符级生成（使用数据集的字符映射）
            print("\n=== 生成示例 ===")
            context = "To be or not to be, "
            x = torch.tensor([dataset.stoi[c] for c in context], dtype=torch.long).unsqueeze(0).to(trainer.device)
            y = model.generate(x, max_new_tokens=200, do_sample=True, top_k=20)
            completion = ''.join([dataset.itos[int(i)] for i in y[0]])
            print(completion)
            print("===============\n")

            # 保存模型
            model_path = os.path.join(config.system.work_dir, f'model_iter_{trainer.iter_num}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存至 {model_path}\n")

    trainer.add_callback('on_batch_end', train_callback)

    # 6. 启动训练
    print("开始训练...")
    trainer.run()

    # 7. 训练完成后，使用BPE分词器进行完整示例（GPT-2风格）
    print("\n=== 训练完成！BPE分词器生成示例 ===")
    tokenizer = BPETokenizer()
    # 加载最终模型
    final_model_path = os.path.join(config.system.work_dir, f'model_iter_{config.trainer.max_iters}.pt')
    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path))
        model.to(trainer.device)
    
    # BPE分词器生成
    prompt = "Hello! This is a GPT model generated text. "
    x = tokenizer.encode(prompt).to(trainer.device)
    y = model.generate(x, max_new_tokens=150, do_sample=True, top_k=30)
    print("生成结果:", tokenizer.decode(y))