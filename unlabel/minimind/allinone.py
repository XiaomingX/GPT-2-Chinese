import os
import sys
import argparse
import time
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

# 基础配置
warnings.filterwarnings('ignore')
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ========================== 共用工具函数 ==========================
def Logger(content):
    """分布式训练下仅主进程打印日志"""
    if not ddp or dist.get_rank() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {content}")


def get_lr(current_step, total_steps, base_lr):
    """余弦学习率调度（含基础偏移）"""
    return base_lr / 10 + 0.5 * base_lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode():
    """初始化分布式训练环境"""
    if not ddp:
        return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# ========================== 模型核心定义 ==========================
class LoRA(nn.Module):
    """LoRA低秩适配模块"""
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)  # 降维矩阵
        self.B = nn.Linear(rank, out_features, bias=False)  # 升维矩阵
        # 初始化：A高斯分布，B全零
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


class MiniMindConfig(PretrainedConfig):
    """MiniMind模型配置类"""
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 1024,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            # MoE配置（默认关闭）
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            aux_loss_alpha: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or 64 * ((int(hidden_size * 8 / 3) + 64 - 1) // 64)
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        # MoE参数
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.aux_loss_alpha = aux_loss_alpha


# ------------------------------ 模型组件 ------------------------------
class RMSNorm(nn.Module):
    """RMS归一化层"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 1e6):
    """预计算RoPE位置编码的cos和sin"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1), torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用RoPE位置编码到Q/K"""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    return (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1)), \
           (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复KV头以匹配Q头数量"""
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_kv_heads, n_rep, head_dim).reshape(bs, slen, num_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    """多头注意力层"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_kv_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.hidden_size // self.num_heads
        # 投影层
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.dropout = config.dropout
        self.flash = config.flash_attn and hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        bsz, seq_len, _ = x.shape
        # QKV投影与reshape
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # 应用RoPE
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # KV缓存
        if past_kv is not None:
            xk = torch.cat([past_kv[0], xk], dim=1)
            xv = torch.cat([past_kv[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 转置为[bs, heads, seq_len, dim]
        xq, xk, xv = xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 注意力计算
        if self.flash:
            attn_out = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1).unsqueeze(0).unsqueeze(0)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = F.dropout(scores, p=self.dropout, training=self.training)
            attn_out = scores @ xv

        # 输出投影
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out), past_kv


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """MoE门控网络"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_experts = config.n_routed_experts
        self.weight = nn.Parameter(torch.empty((self.n_experts, config.hidden_size)))
        self.alpha = config.aux_loss_alpha
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        bsz_seq = x.shape[0]
        logits = F.linear(x.view(bsz_seq, -1), self.weight)
        scores = logits.softmax(dim=-1)
        # 选择Top-K专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # 权重归一化
        if self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        # 辅助损失（负载均衡）
        aux_loss = 0.0
        if self.training and self.alpha > 0:
            mask = F.one_hot(topk_idx.view(-1), num_classes=self.n_experts).float()
            ce = mask.mean(0)
            pi = scores.mean(0)
            aux_loss = (pi * ce * self.n_experts).sum() * self.alpha
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """MoE前馈网络"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
        self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_shared_experts)])
        self.gate = MoEGate(config)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        orig_x = x
        # 门控选专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x_flat = x.view(-1, x.shape[-1])
        flat_idx = topk_idx.view(-1)

        # 专家计算
        if self.training:
            x_repeat = x_flat.repeat_interleave(self.gate.top_k, dim=0)
            y = torch.empty_like(x_repeat)
            for i, expert in enumerate(self.experts):
                y[flat_idx == i] = expert(x_repeat[flat_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        else:
            y = torch.zeros_like(x_flat)
            idxs = flat_idx.argsort()
            token_idxs = idxs // self.gate.top_k
            cnt = flat_idx.bincount().cumsum(0)
            for i, end in enumerate(cnt):
                start = 0 if i == 0 else cnt[i-1]
                if start == end:
                    continue
                exp_tokens = x_flat[token_idxs[start:end]]
                y[token_idxs[start:end]] += self.experts[i](exp_tokens) * topk_weight.view(-1, 1)[idxs[start:end]]

        # 共享专家
        y = y.view(bsz, seq_len, -1)
        for expert in self.shared_experts:
            y += expert(orig_x)
        self.aux_loss = aux_loss
        return y


class MiniMindBlock(nn.Module):
    """模型基础块（注意力+前馈）"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.attn = Attention(config)
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = MOEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        # 注意力残差
        attn_out, past_kv = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache)
        x = x + attn_out
        # 前馈残差
        x = x + self.ffn(self.ffn_norm(x))
        return x, past_kv


# ------------------------------ 完整模型 ------------------------------
class MiniMindModel(nn.Module):
    """MiniMind基础编码器"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, past_kvs=None, use_cache=False):
        bsz, seq_len = input_ids.shape
        past_kvs = past_kvs or [None] * len(self.layers)
        start_pos = past_kvs[0][0].shape[1] if past_kvs[0] else 0

        # 词嵌入
        x = self.dropout(self.embed(input_ids))
        # RoPE位置编码
        cos, sin = self.freqs_cos[start_pos:start_pos+seq_len], self.freqs_sin[start_pos:start_pos+seq_len]

        # 层堆叠
        presents = []
        for layer, past_kv in zip(self.layers, past_kvs):
            x, present = layer(x, cos, sin, past_kv, use_cache)
            presents.append(present)

        # 最终归一化
        x = self.norm(x)
        # 计算MoE辅助损失
        aux_loss = sum(layer.ffn.aux_loss for layer in self.layers if isinstance(layer.ffn, MOEFeedForward))
        return x, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """MiniMind因果语言模型（含LM头）"""
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        config = config or MiniMindConfig()
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 权重共享
        self.model.embed.weight = self.lm_head.weight

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        h, past_kvs, aux_loss = self.model(input_ids, past_key_values, use_cache)
        logits = self.lm_head(h)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_kvs,
            aux_loss=aux_loss,
            last_hidden_state=h
        )


# ========================== 数据加载（简化版） ==========================
class PretrainDataset(torch.utils.data.Dataset):
    """预训练数据集（简化版，实际需根据数据格式调整）"""
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # 加载数据（示例：每行一个文本）
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_seq_len)['input_ids']
        # 补齐到max_seq_len
        if len(tokens) < self.max_seq_len:
            tokens += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))
        tokens = tokens[:self.max_seq_len]
        # 预训练任务：输入=tokens[:-1], 标签=tokens[1:]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        loss_mask = torch.tensor([1]*(len(tokens)-1), dtype=torch.float32)  # 非pad位置计算损失
        return x, y, loss_mask


class SFTDataset(PretrainDataset):
    """SFT数据集（继承预训练数据集，可扩展指令格式）"""
    def __getitem__(self, idx):
        # 示例：指令微调格式 -> "指令: xxx 回答: xxx"
        item = self.data[idx]
        if isinstance(item, dict):
            text = f"指令: {item['instruction']} 回答: {item['response']}"
        else:
            text = item  # 兼容纯文本
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_seq_len)['input_ids']
        if len(tokens) < self.max_seq_len:
            tokens += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))
        tokens = tokens[:self.max_seq_len]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        loss_mask = torch.tensor([1]*(len(tokens)-1), dtype=torch.float32)
        return x, y, loss_mask


class DPODataset(torch.utils.data.Dataset):
    """DPO数据集（含优选/非优选样本）"""
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [eval(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 编码优选/非优选样本
        def encode(text):
            tokens = self.tokenizer(text, truncation=True, max_length=self.max_seq_len)['input_ids']
            if len(tokens) < self.max_seq_len:
                tokens += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))
            return torch.tensor(tokens[:self.max_seq_len], dtype=torch.long)

        x_chosen = encode(f"指令: {item['instruction']} 回答: {item['chosen']}")[:-1]
        y_chosen = encode(f"指令: {item['instruction']} 回答: {item['chosen']}")[1:]
        x_rejected = encode(f"指令: {item['instruction']} 回答: {item['rejected']}")[:-1]
        y_rejected = encode(f"指令: {item['instruction']} 回答: {item['rejected']}")[1:]
        # 损失掩码
        mask_chosen = (y_chosen != self.tokenizer.pad_token_id).float()
        mask_rejected = (y_rejected != self.tokenizer.pad_token_id).float()
        return {
            'x_chosen': x_chosen, 'y_chosen': y_chosen, 'mask_chosen': mask_chosen,
            'x_rejected': x_rejected, 'y_rejected': y_rejected, 'mask_rejected': mask_rejected
        }


# ========================== 训练阶段实现 ==========================
def init_common_components(args):
    """初始化模型、分词器、数据加载器等共用组件"""
    # 配置与设备
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=args.use_moe
    )
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained('../model')
    tokenizer.pad_token = tokenizer.eos_token  # 设pad_token为eos_token

    # 数据集与加载器
    if args.stage == "pretrain":
        dataset = PretrainDataset(args.data_path, tokenizer, args.max_seq_len)
    elif args.stage in ["sft", "distill", "lora_sft"]:
        dataset = SFTDataset(args.data_path, tokenizer, args.max_seq_len)
    elif args.stage == "dpo":
        dataset = DPODataset(args.data_path, tokenizer, args.max_seq_len)
    else:
        raise ValueError(f"未知阶段: {args.stage}")

    sampler = DistributedSampler(dataset) if ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 模型初始化
    model = MiniMindForCausalLM(lm_config).to(args.device)
    # 加载预训练权重（非pretrain阶段）
    if args.stage != "pretrain" and os.path.exists(args.load_ckp):
        state_dict = torch.load(args.load_ckp, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
        Logger(f"加载权重: {args.load_ckp}")

    # 分布式包装
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 优化器与混合精度
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    if args.stage == "lora_sft":
        # LoRA模式：仅优化LoRA参数
        apply_lora(model)
        lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
        optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
        Logger(f"LoRA参数量: {sum(p.numel() for p in lora_params)/1e6:.2f}M")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # WandB初始化
    wandb = None
    if args.use_wandb and (not ddp or dist.get_rank() == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=f"MiniMind-{args.stage}-{args.wandb_run_name}")

    return {
        "lm_config": lm_config, "ctx": ctx, "tokenizer": tokenizer,
        "dataloader": dataloader, "model": model, "optimizer": optimizer,
        "scaler": scaler, "wandb": wandb
    }


def apply_lora(model, rank=8):
    """为模型线性层添加LoRA"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:  # 仅方阵层（如注意力、FFN）
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            # 重写forward：原输出 + LoRA输出
            original_forward = module.forward
            def new_forward(x, orig=original_forward, lora=lora):
                return orig(x) + lora(x)
            module.forward = new_forward
    # 冻结非LoRA参数
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False


def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    """蒸馏损失（KL散度）"""
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return (temperature ** 2) * F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')


def dpo_loss(ref_probs, student_probs, mask, beta=0.1):
    """DPO损失"""
    # 计算序列平均概率
    seq_len = mask.sum(dim=1, keepdim=True)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_len.squeeze()
    student_probs = (student_probs * mask).sum(dim=1) / seq_len.squeeze()

    # 拆分优选/非优选样本
    batch_size = ref_probs.shape[0]
    chosen_ref, rejected_ref = ref_probs[:batch_size//2], ref_probs[batch_size//2:]
    chosen_stu, rejected_stu = student_probs[:batch_size//2], student_probs[batch_size//2:]

    # DPO核心损失
    log_ratio = (chosen_stu - rejected_stu) - (chosen_ref - rejected_ref)
    return -F.logsigmoid(beta * log_ratio).mean()


def logits_to_probs(logits, labels):
    """从logits计算标签对应的概率（对数概率）"""
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def train_pretrain(args, components):
    """预训练阶段"""
    ctx = components["ctx"]
    model = components["model"]
    optimizer = components["optimizer"]
    scaler = components["scaler"]
    dataloader = components["dataloader"]
    wandb = components["wandb"]
    iter_per_epoch = len(dataloader)
    total_steps = args.epochs * iter_per_epoch

    model.train()
    for epoch in range(args.epochs):
        if ddp:
            dataloader.sampler.set_epoch(epoch)
        start_time = time.time()
        for step, (x, y, loss_mask) in enumerate(dataloader):
            x, y, loss_mask = x.to(args.device), y.to(args.device), loss_mask.to(args.device)
            current_step = epoch * iter_per_epoch + step

            # 学习率调度
            lr = get_lr(current_step, total_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向计算
            with ctx:
                outputs = model(input_ids=x)
                logits = outputs.logits
                # 交叉熵损失
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction='none'
                ).view(y.size())
                # 应用损失掩码
                loss = (ce_loss * loss_mask).sum() / loss_mask.sum()
                # 加上MoE辅助损失
                if outputs.aux_loss is not None:
                    loss += outputs.aux_loss
                # 梯度累积
                loss = loss / args.accumulation_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 优化步骤
            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # 日志打印
            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                log_loss = loss.item() * args.accumulation_steps
                Logger(f"Epoch[{epoch+1}/{args.epochs}] Step[{step}/{iter_per_epoch}] "
                       f"Loss:{log_loss:.3f} LR:{lr:.6e} ETA:{spend_time/(step+1)*iter_per_epoch//60}min")
                if wandb:
                    wandb.log({"loss": log_loss, "lr": lr})

            # 模型保存
            if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
                save_path = f"{args.save_dir}/pretrain_epoch{epoch}_step{step}.pth"
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                torch.save({k: v.half() for k, v in state_dict.items()}, save_path)
                Logger(f"模型保存至: {save_path}")


def train_sft(args, components):
    """全参数SFT微调"""
    # 逻辑与预训练一致，仅数据集不同（已在init_common_components中处理）
    train_pretrain(args, components)


def train_distill(args, components):
    """知识蒸馏（学生模型蒸馏教师模型知识）"""
    ctx = components["ctx"]
    student_model = components["model"]
    optimizer = components["optimizer"]
    scaler = components["scaler"]
    dataloader = components["dataloader"]
    wandb = components["wandb"]
    iter_per_epoch = len(dataloader)
    total_steps = args.epochs * iter_per_epoch

    # 初始化教师模型
    teacher_config = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers)
    teacher_model = MiniMindForCausalLM(teacher_config).to(args.device)
    teacher_model.load_state_dict(torch.load(args.teacher_ckp, map_location=args.device), strict=False)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f"教师模型参数量: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")

    student_model.train()
    for epoch in range(args.epochs):
        if ddp:
            dataloader.sampler.set_epoch(epoch)
        start_time = time.time()
        for step, (x, y, loss_mask) in enumerate(dataloader):
            x, y, loss_mask = x.to(args.device), y.to(args.device), loss_mask.to(args.device)
            current_step = epoch * iter_per_epoch + step
            lr = get_lr(current_step, total_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向计算
            with ctx:
                # 学生模型
                student_out = student_model(input_ids=x)
                student_logits = student_out.logits
                # 教师模型（无梯度）
                with torch.no_grad():
                    teacher_out = teacher_model(input_ids=x)
                    teacher_logits = teacher_out.logits[..., :student_logits.size(-1)]  # 对齐词表

                # 计算损失
                ce_loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    y.view(-1),
                    reduction='none'
                ).view(y.size())
                ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()
                # 蒸馏损失
                distill_loss = distillation_loss(
                    student_logits.view(-1, student_logits.size(-1))[loss_mask.view(-1) == 1],
                    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask.view(-1) == 1],
                    temperature=args.distill_temp
                )
                # 总损失 = alpha*CE + (1-alpha)*蒸馏损失
                loss = (args.alpha * ce_loss + (1 - args.alpha) * distill_loss) / args.accumulation_steps

            # 反向传播与优化
            scaler.scale(loss).backward()
            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # 日志
            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                log_loss = loss.item() * args.accumulation_steps
                Logger(f"Epoch[{epoch+1}/{args.epochs}] Step[{step}/{iter_per_epoch}] "
                       f"Loss:{log_loss:.3f} CE:{ce_loss:.3f} Distill:{distill_loss:.3f} LR:{lr:.6e}")
                if wandb:
                    wandb.log({"loss": log_loss, "ce_loss": ce_loss, "distill_loss": distill_loss, "lr": lr})

            # 保存
            if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
                save_path = f"{args.save_dir}/distill_epoch{epoch}_step{step}.pth"
                state_dict = student_model.module.state_dict() if ddp else student_model.state_dict()
                torch.save({k: v.half() for k, v in state_dict.items()}, save_path)
                Logger(f"蒸馏模型保存至: {save_path}")


def train_dpo(args, components):
    """DPO（直接偏好优化）"""
    ctx = components["ctx"]
    model = components["model"]
    optimizer = components["optimizer"]
    scaler = components["scaler"]
    dataloader = components["dataloader"]
    wandb = components["wandb"]
    iter_per_epoch = len(dataloader)
    total_steps = args.epochs * iter_per_epoch

    # 初始化参考模型（与当前模型同权重，无梯度）
    ref_model = MiniMindForCausalLM(components["lm_config"]).to(args.device)
    ref_model.load_state_dict(model.state_dict() if not ddp else model.module.state_dict(), strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    model.train()
    for epoch in range(args.epochs):
        if ddp:
            dataloader.sampler.set_epoch(epoch)
        start_time = time.time()
        for step, batch in enumerate(dataloader):
            # 数据预处理
            x_chosen = batch['x_chosen'].to(args.device)
            x_rejected = batch['x_rejected'].to(args.device)
            y_chosen = batch['y_chosen'].to(args.device)
            y_rejected = batch['y_rejected'].to(args.device)
            mask_chosen = batch['mask_chosen'].to(args.device)
            mask_rejected = batch['mask_rejected'].to(args.device)

            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

            current_step = epoch * iter_per_epoch + step
            lr = get_lr(current_step, total_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向计算
            with ctx:
                # 参考模型概率
                with torch.no_grad():
                    ref_out = ref_model(input_ids=x)
                    ref_probs = logits_to_probs(ref_out.logits, y)
                # 当前模型概率
                model_out = model(input_ids=x)
                model_probs = logits_to_probs(model_out.logits, y)
                # DPO损失
                loss = dpo_loss(ref_probs, model_probs, mask, beta=args.dpo_beta) / args.accumulation_steps

            # 反向传播与优化
            scaler.scale(loss).backward()
            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # 日志
            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                log_loss = loss.item() * args.accumulation_steps
                Logger(f"Epoch[{epoch+1}/{args.epochs}] Step[{step}/{iter_per_epoch}] "
                       f"DPO Loss:{log_loss:.3f} LR:{lr:.6e}")
                if wandb:
                    wandb.log({"dpo_loss": log_loss, "lr": lr})

            # 保存
            if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
                save_path = f"{args.save_dir}/dpo_epoch{epoch}_step{step}.pth"
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                torch.save({k: v.half() for k, v in state_dict.items()}, save_path)
                Logger(f"DPO模型保存至: {save_path}")


def train_lora_sft(args, components):
    """LoRA微调（仅训练LoRA参数）"""
    # 逻辑与全参数SFT一致，仅优化器已在init_common_components中配置为LoRA参数
    train_pretrain(args, components)
    # 单独保存LoRA权重
    if (not ddp or dist.get_rank() == 0):
        lora_save_path = f"{args.save_dir}/lora_weights.pth"
        state_dict = {}
        for name, module in components["model"].named_modules():
            if hasattr(module, 'lora'):
                lora_state = {f"{name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
                state_dict.update(lora_state)
        torch.save(state_dict, lora_save_path)
        Logger(f"LoRA权重保存至: {lora_save_path}")


# ========================== 模型推理 ==========================
def model_inference(args, model_path, prompt):
    """模型推理（文本生成）"""
    # 加载模型与分词器
    tokenizer = AutoTokenizer.from_pretrained('../model')
    tokenizer.pad_token = tokenizer.eos_token
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=args.use_moe
    )
    model = MiniMindForCausalLM(lm_config).to(args.device)
    state_dict = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 编码输入
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)

    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.generate_max_len,
            temperature=args.generate_temp,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码输出
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"输入: {prompt}")
    print(f"输出: {result[len(prompt):]}")  # 去掉输入部分
    return result


# ========================== 主入口 ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind 端到端训练与推理")
    # 通用参数
    parser.add_argument("--stage", type=str, required=True, choices=["pretrain", "sft", "distill", "dpo", "lora_sft", "inference"], help="运行阶段")
    parser.add_argument("--out_dir", type=str, default="../out", help="输出目录")
    parser.add_argument("--data_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用WandB日志")
    parser.add_argument("--wandb_project", type=str, default="MiniMind", help="WandB项目名")
    parser.add_argument("--wandb_run_name", type=str, default="run1", help="WandB运行名")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--ddp", action="store_true", help="是否使用分布式训练")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层层数")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--use_moe", action="store_true", help="是否使用MoE结构")
    # 训练参数
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--load_ckp", type=str, default="", help="加载预训练权重路径（非pretrain阶段）")
    # 蒸馏参数
    parser.add_argument("--teacher_ckp", type=str, default="", help="教师模型权重路径（distill阶段）")
    parser.add_argument("--teacher_hidden_size", type=int, default=768, help="教师模型隐藏层维度（distill阶段）")
    parser.add_argument("--teacher_num_layers", type=int, default=12, help="教师模型隐藏层层数（distill阶段）")
    parser.add_argument("--distill_temp", type=float, default=2.0, help="蒸馏温度（distill阶段）")
    parser.add_argument("--alpha", type=float, default=0.5, help="CE损失权重（distill阶段）")
    # DPO参数
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta参数（dpo阶段）")
    # 推理参数
    parser.add_argument("--model_path", type=str, default="", help="推理模型权重路径（inference阶段）")
    parser.add_argument("--prompt", type=str, default="指令: 解释什么是人工智能", help="推理输入提示（inference阶段）")
    parser.add_argument("--generate_max_len", type=int, default=200, help="最大生成长度（inference阶段）")
    parser.add_argument("--generate_temp", type=float, default=0.7, help="生成温度（inference阶段）")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    args.save_dir = args.out_dir

    # 分布式初始化
    global ddp, ddp_local_rank, DEVICE
    ddp = args.ddp and (int(os.environ.get("RANK", -1)) != -1)
    ddp_local_rank, DEVICE = 0, args.device
    if ddp:
        init_distributed_mode()
        args.device = DEVICE

    # 随机种子
    base_seed = 1337
    torch.manual_seed(base_seed + (dist.get_rank() if ddp else 0))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed + (dist.get_rank() if ddp else 0))

    # 运行对应阶段
    if args.stage in ["pretrain", "sft", "dpo", "lora_sft"]:
        components = init_common_components(args)
        if args.stage == "pretrain":
            train_pretrain(args, components)
        elif args.stage == "sft":
            train_sft(args, components)
        elif args.stage == "dpo":
            train_dpo(args, components)
        elif args.stage == "lora_sft":
            train_lora_sft(args, components)
    elif args.stage == "distill":
        components = init_common_components(args)
        train_distill(args, components)
    elif args.stage == "inference":
        assert os.path.exists(args.model_path), f"模型权重不存在: {args.model_path}"
        model_inference(args, args.model_path, args.prompt)

    # 分布式清理
    if ddp:
        dist.destroy_process_group()