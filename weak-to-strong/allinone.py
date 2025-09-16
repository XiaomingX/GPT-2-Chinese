import gc
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel
)
import wandb
import fire


# ------------------------------
# 1. 基础工具函数
# ------------------------------
def clear_mem():
    """清理PyTorch占用的GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"清理后GPU内存占用: {used:.2f}GB")


def get_tokenizer(model_name: str):
    """获取模型对应的分词器"""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# ------------------------------
# 2. 数据集处理
# ------------------------------
def format_amazon_polarity(ex, rng):
    """Amazon情感分类数据集格式化：拼接标题+内容，标签直接使用"""
    return dict(txt=f"{ex['title']} {ex['content']}", hard_label=ex["label"])


def format_sciq(ex, rng):
    """SciQ问答数据集格式化：随机选择正确/干扰答案，构建文本"""
    hard_label = 1 if rng.random() < 0.5 else 0
    ans = ex["correct_answer"] if hard_label else rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    return dict(txt=f"Q: {ex['question']} A: {ans}", hard_label=hard_label)


def format_boolq(ex, rng):
    """BoolQ问答数据集格式化：拼接段落+问题，标签为原答案"""
    return dict(txt=f"Passage: {ex['passage']}\nQuestion: {ex['question']}", hard_label=int(ex["answer"]))


# 数据集名称 -> (加载路径, 格式化函数, 分裂映射)
DATASET_CONFIG = {
    "amazon_polarity": ("amazon_polarity", format_amazon_polarity, None),
    "sciq": ("sciq", format_sciq, None),
    "boolq": ("boolq", format_boolq, dict(test="validation"))  # boolq用validation作为测试集
}
VALID_DATASETS = list(DATASET_CONFIG.keys())


def load_and_prep_dataset(ds_name: str, seed: int = 0, split_sizes: dict = None):
    """
    加载并预处理数据集
    Args:
        ds_name: 数据集名称（需在VALID_DATASETS中）
        seed: 随机种子
        split_sizes: 各分裂的样本量（如{"train":1000, "test":200}）
    Returns:
        预处理后的数据集字典（包含train/test，含txt、hard_label、soft_label）
    """
    if split_sizes is None:
        split_sizes = dict(train=1000, test=200)
    assert ds_name in VALID_DATASETS, f"数据集{ds_name}未支持，可选：{VALID_DATASETS}"
    
    # 从配置中获取加载信息
    hf_path, formatter, split_map = DATASET_CONFIG[ds_name]
    rng = random.Random(seed)
    results = {}

    for split, n_docs in split_sizes.items():
        # 加载原始数据（处理分裂映射，如boolq的test→validation）
        actual_split = split_map[split] if (split_map and split in split_map) else split
        raw_ds = load_dataset(hf_path, split=actual_split)
        
        # 截取样本量（避免超出数据集大小）
        if n_docs and len(raw_ds) > n_docs:
            raw_ds = raw_ds.select(range(n_docs))
        
        # 格式化数据（添加txt和hard_label）
        ds = raw_ds.map(lambda x: formatter(x, rng))
        # 生成soft_label（二分类概率格式）
        ds = ds.map(lambda x: {"soft_label": [1 - x["hard_label"], x["hard_label"]]})
        # 打乱数据
        ds = ds.shuffle(seed=seed)
        results[split] = ds
    
    return results


def tokenize_dataset(raw_ds: Dataset, tokenizer, max_ctx: int = 512):
    """
    对数据集进行分词处理
    Args:
        raw_ds: 预处理后的数据集
        tokenizer: 分词器
        max_ctx: 最大上下文长度（过滤超长样本）
    Returns:
        分词后的数据集（含input_ids）
    """
    def process_fn(examples):
        # 批量分词，返回input_ids
        toks = tokenizer(examples["txt"], truncation=True, max_length=max_ctx)
        return {"input_ids": toks["input_ids"]}
    
    # 批量处理效率更高
    ds = raw_ds.map(process_fn, batched=True)
    # 过滤超长样本（避免训练报错）
    ds = ds.filter(lambda x: len(x["input_ids"]) < max_ctx)
    # 转换为PyTorch格式（方便训练）
    ds.set_format("torch", columns=["input_ids", "soft_label"])
    return ds


# ------------------------------
# 3. 模型定义
# ------------------------------
class TransformerWithHead(PreTrainedModel):
    """带分类头的Transformer模型（基于预训练LM）"""
    def __init__(self, model_name: str, num_labels: int = 2, linear_probe: bool = False):
        # 加载预训练模型配置
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.num_labels = num_labels
        self.linear_probe = linear_probe  # 是否固定Transformer权重，只训练分类头
        
        # 加载预训练语言模型
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.transformer = self.lm.transformer
        
        # 获取隐藏层维度（适配不同模型如gpt2、qwen）
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size
        # 分类头（初始化为0均值正态分布）
        self.classifier = nn.Linear(hidden_size, num_labels, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.02)
        
        # 转移设备并匹配数据类型
        self.to(self.lm.device if hasattr(self.lm, "device") else "cpu")
        self.classifier.to(self.lm.dtype)

    def forward(self, input_ids: torch.Tensor):
        """前向传播：取最后一个非padding token的隐藏态做分类"""
        # 获取输入长度（非padding的token数）
        input_lens = (input_ids != 0).sum(dim=-1)  # 假设0是padding token
        
        # Transformer编码
        transformer_out = self.transformer(input_ids)
        hidden_states = transformer_out[0]  # (batch_size, seq_len, hidden_size)
        
        # 取每个样本最后一个有效token的隐藏态
        final_hidden = torch.stack([
            hidden_states[i, input_lens[i]-1, :] for i in range(len(input_lens))
        ])
        
        # 线性探针模式：固定Transformer权重
        if self.linear_probe:
            final_hidden = final_hidden.detach()
        
        # 分类头计算logits
        logits = self.classifier(final_hidden)
        return logits


# ------------------------------
# 4. 损失函数
# ------------------------------
class XentLoss:
    """交叉熵损失（基础二分类损失）"""
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        return nn.functional.cross_entropy(logits, labels).mean()


class ProductLoss:
    """乘积损失（结合预测与标签的幂次）"""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor,** kwargs):
        preds = torch.softmax(logits, dim=-1)
        target = torch.pow(preds, self.beta) * torch.pow(labels, self.alpha)
        target /= target.sum(dim=-1, keepdim=True)
        return nn.functional.cross_entropy(logits, target.detach()).mean()


class LogConfLoss:
    """日志置信度损失（结合弱模型标签）"""
    def __init__(self, aux_coef: float = 0.5, warmup_frac: float = 0.1):
        self.aux_coef = aux_coef
        self.warmup_frac = warmup_frac

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, step_frac: float):
        coef = self.aux_coef * min(1.0, step_frac / self.warmup_frac)
        preds = torch.softmax(logits, dim=-1)
        # 计算阈值（匹配弱模型的均值分布）
        mean_weak = labels.mean(dim=0)
        threshold = torch.quantile(preds[:, 0], mean_weak[1])
        # 生成强模型伪标签
        strong_preds = torch.cat([
            (preds[:, 0] >= threshold)[:, None],
            (preds[:, 0] < threshold)[:, None]
        ], dim=1).float()
        # 混合标签
        target = labels * (1 - coef) + strong_preds.detach() * coef
        return nn.functional.cross_entropy(logits, target).mean()


# 损失函数映射（方便调用）
LOSS_MAP = {
    "xent": XentLoss(),
    "product": ProductLoss(),
    "logconf": LogConfLoss()
}
VALID_LOSSES = list(LOSS_MAP.keys())


# ------------------------------
# 5. 训练与评估
# ------------------------------
def evaluate_model(model, test_ds, batch_size: int = 32, device: str = "cuda"):
    """
    评估模型准确率
    Args:
        model: 待评估模型
        test_ds: 测试数据集（已分词）
        batch_size: 评估批次大小
        device: 计算设备
    Returns:
        平均准确率
    """
    model.eval()  # 切换评估模式
    total_acc = 0
    total_samples = 0

    with torch.no_grad():  # 关闭梯度计算
        # 批量迭代测试集
        for i in range(0, len(test_ds), batch_size):
            batch = test_ds[i:i+batch_size]
            input_ids = batch["input_ids"].to(device)
            labels = batch["soft_label"].to(device)  # (batch, 2)
            
            # 模型预测
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)  # 预测标签（0或1）
            true_labels = torch.argmax(labels, dim=-1)  # 真实标签
            
            # 计算准确率
            acc = (preds == true_labels).float().sum()
            total_acc += acc.item()
            total_samples += len(input_ids)
    
    avg_acc = total_acc / total_samples
    print(f"评估准确率: {avg_acc:.4f}")
    return avg_acc


def train_model(
    model,
    train_ds,
    test_ds,
    loss_fn,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 5e-5,
    log_every: int = 10,
    save_path: str = "./saved_model",
    device: str = "cuda"
):
    """
    模型训练主函数
    Args:
        model: 待训练模型
        train_ds: 训练数据集（已分词）
        test_ds: 测试数据集（已分词）
        loss_fn: 损失函数
        epochs: 训练轮次
        batch_size: 训练批次大小
        lr: 学习率
        log_every: 每多少步打印日志
        save_path: 模型保存路径
        device: 计算设备
    Returns:
        训练后的模型
    """
    # 数据加载器（支持批量迭代）
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    total_steps = len(train_loader) * epochs
    
    # 优化器与学习率调度器
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_samples = 0

        for step, batch in enumerate(train_loader):
            # 数据转移到设备
            input_ids = batch["input_ids"].to(device)
            labels = batch["soft_label"].to(device)  # (batch, 2)
            
            # 前向传播
            logits = model(input_ids)
            step_frac = (epoch * len(train_loader) + step) / total_steps  # 训练进度
            loss = loss_fn(logits, labels, step_frac=step_frac)
            
            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 计算当前批次指标
            preds = torch.argmax(logits, dim=-1)
            true_labels = torch.argmax(labels, dim=-1)
            acc = (preds == true_labels).float().sum().item()
            
            # 累计指标
            batch_size = len(input_ids)
            epoch_loss += loss.item() * batch_size
            epoch_acc += acc
            total_samples += batch_size
            
            # 打印日志
            if (step + 1) % log_every == 0:
                avg_step_loss = epoch_loss / total_samples
                avg_step_acc = epoch_acc / total_samples
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], "
                      f"Loss: {avg_step_loss:.4f}, Acc: {avg_step_acc:.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 每轮结束后评估
        print(f"\nEpoch [{epoch+1}/{epochs}] 训练完成，开始评估...")
        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_acc = epoch_acc / total_samples
        test_acc = evaluate_model(model, test_ds, batch_size, device)
        
        # 记录wandb日志（若初始化）
        if wandb.run:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": avg_epoch_loss,
                "train_acc": avg_epoch_acc,
                "test_acc": test_acc,
                "lr": scheduler.get_last_lr()[0]
            })
    
    # 保存模型
    model.save_pretrained(save_path)
    print(f"\n模型已保存至: {save_path}")
    return model


# ------------------------------
# 6. 端到端完整流程
# ------------------------------
def main(
    # 数据配置
    ds_name: str = "sciq",
    train_samples: int = 1000,
    test_samples: int = 200,
    max_ctx: int = 512,
    # 模型配置
    model_name: str = "gpt2",  # 基础预训练模型
    linear_probe: bool = False,
    # 训练配置
    loss_type: str = "xent",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 5e-5,
    # 其他配置
    seed: int = 42,
    save_path: str = "./saved_model",
    use_wandb: bool = False,
    wandb_project: str = "weak_to_strong_demo"
):
    """端到端模型训练与使用流程"""
    # 1. 初始化随机种子（保证可复现）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 2. 初始化wandb（可选，用于日志记录）
    if use_wandb:
        wandb.init(project=wandb_project, name=f"{model_name}_{ds_name}_{loss_type}")
        wandb.config.update({
            "ds_name": ds_name,
            "model_name": model_name,
            "loss_type": loss_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr
        })
    
    # 3. 设备选择（自动使用GPU if available）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 4. 加载与预处理数据集
    print(f"\n1. 加载数据集: {ds_name}")
    raw_ds = load_and_prep_dataset(
        ds_name=ds_name,
        seed=seed,
        split_sizes={"train": train_samples, "test": test_samples}
    )
    train_raw, test_raw = raw_ds["train"], raw_ds["test"]
    print(f"训练集样本数: {len(train_raw)}, 测试集样本数: {len(test_raw)}")
    
    # 5. 分词处理
    print(f"\n2. 加载分词器: {model_name}")
    tokenizer = get_tokenizer(model_name)
    # 添加padding token（gpt2默认无padding token，用eos_token填充）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("3. 对数据集进行分词...")
    train_ds = tokenize_dataset(train_raw, tokenizer, max_ctx=max_ctx)
    test_ds = tokenize_dataset(test_raw, tokenizer, max_ctx=max_ctx)
    print(f"分词后训练集: {len(train_ds)} 样本, 测试集: {len(test_ds)} 样本")
    
    # 6. 初始化模型
    print(f"\n4. 初始化模型: {model_name}")
    model = TransformerWithHead(
        model_name=model_name,
        linear_probe=linear_probe
    ).to(device)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 7. 选择损失函数
    assert loss_type in VALID_LOSSES, f"损失函数{loss_type}未支持，可选：{VALID_LOSSES}"
    loss_fn = LOSS_MAP[loss_type]
    print(f"使用损失函数: {loss_type}")
    
    # 8. 模型训练
    print(f"\n5. 开始训练（共{epochs}轮）...")
    model = train_model(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        loss_fn=loss_fn,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        log_every=5,
        save_path=save_path,
        device=device
    )
    
    # 9. 模型推理示例
    print(f"\n6. 推理示例...")
    model.eval()
    # 随机选1个测试样本
    sample = test_raw[random.randint(0, len(test_raw)-1)]
    print(f"输入文本: {sample['txt']}")
    print(f"真实标签: {'正确' if sample['hard_label'] == 1 else '错误'}")
    
    # 分词并推理
    inputs = tokenizer(sample["txt"], return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(inputs["input_ids"])
        pred = torch.argmax(logits, dim=-1).item()
    print(f"模型预测: {'正确' if pred == 1 else '错误'}")
    
    # 清理内存
    clear_mem()
    # 关闭wandb
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # 支持命令行参数调用
    fire.Fire(main)
