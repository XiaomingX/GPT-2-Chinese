import hashlib
import os
import urllib
import warnings
from packaging import version
from typing import Union, List, Tuple, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import ftfy
import regex as re
from functools import lru_cache


# -------------------------- 基础配置 --------------------------
# 图像插值方式（兼容不同PyTorch版本）
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# PyTorch版本检查
if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("建议使用PyTorch 1.7.1及以上版本")

# 预训练模型列表（名称->下载链接）
_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
}


# -------------------------- 1. 文本分词器（Tokenizer） --------------------------
@lru_cache()
def bytes_to_unicode():
    """字节与unicode字符的映射（用于BPE分词）"""
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


@lru_cache()
def default_bpe_path():
    """默认BPE词表路径（自动创建缓存目录）"""
    cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    bpe_path = os.path.join(cache_dir, "bpe_simple_vocab_16e6.txt.gz")
    
    # 自动下载BPE词表（原代码依赖本地文件，此处补全下载逻辑）
    if not os.path.exists(bpe_path):
        urllib.request.urlretrieve(
            "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz",
            bpe_path
        )
    return bpe_path


def get_pairs(word: Tuple[str]) -> set:
    """获取词的相邻字符对（用于BPE合并）"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class SimpleTokenizer:
    """CLIP专用文本分词器（基于BPE）"""
    def __init__(self, bpe_path: str = default_bpe_path()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 加载BPE合并规则
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = [tuple(merge.split()) for merge in merges[1:49152-256-2+1]]
        
        # 构建词表
        vocab = list(self.byte_encoder.values())
        vocab += [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])  # 起始/结束标记
        
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        
        # 文本匹配模式（提取单词、数字、特殊标记）
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token: str) -> str:
        """对单个token执行BPE分词"""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)  # 末尾加结束标记
        pairs = get_pairs(word)
        
        if not pairs:
            return token + '</w>'
        
        # 迭代合并最高优先级的字符对
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
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
                
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
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

    def encode(self, text: str) -> List[int]:
        """文本转token ID列表"""
        # 文本清洗
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # 字节转unicode
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # BPE分词并转ID
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        """token ID列表转文本"""
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace")
        return text.replace('</w>', ' ').strip()


# 全局分词器实例
_tokenizer = SimpleTokenizer()


# -------------------------- 2. 模型核心组件 --------------------------
class LayerNorm(nn.LayerNorm):
    """支持FP16的LayerNorm（继承自PyTorch，适配混合精度）"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """高效GELU激活函数（CLIP专用）"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Transformer残差注意力块"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """多头注意力计算"""
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差连接：注意力 + MLP"""
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Transformer编码器（文本/图像特征提取）"""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """Vision Transformer（图像编码器）"""
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        
        # 图像分块卷积（将图像转为patch嵌入）
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        # 初始化参数
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # 类别嵌入
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # 位置嵌入
        self.ln_pre = LayerNorm(width)
        
        # Transformer编码器
        self.transformer = Transformer(width, layers, heads)
        
        # 输出层
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))  # 投影到特征维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入：[batch, 3, H, W]，输出：[batch, output_dim]"""
        # 1. 图像分块 -> [batch, width, num_patches]
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # [batch, num_patches, width]
        
        # 2. 拼接类别嵌入和位置嵌入
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # [batch, num_patches+1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        # 3. Transformer编码
        x = x.permute(1, 0, 2)  # [num_patches+1, batch, width]（MultiheadAttention要求的形状）
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch, num_patches+1, width]
        
        # 4. 取类别嵌入的输出并投影
        x = self.ln_post(x[:, 0, :])  # 只取class_embedding对应的输出
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    """CLIP主模型（图像-文本双编码器+对比学习）"""
    def __init__(self,
                 embed_dim: int,          # 图像/文本特征的最终维度
                 # 图像编码器参数（ViT-B/32为例）
                 image_resolution: int = 224,
                 vision_layers: int = 12,
                 vision_width: int = 768,
                 vision_patch_size: int = 32,
                 # 文本编码器参数
                 context_length: int = 77,  # 文本最大长度
                 vocab_size: int = 49408,   # 分词器词表大小
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 12
                 ):
        super().__init__()
        self.context_length = context_length
        
        # 1. 图像编码器（ViT）
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,  # 头数=维度/64（CLIP默认）
            output_dim=embed_dim
        )
        
        # 2. 文本编码器（Transformer）
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self._build_attention_mask()  # 文本因果掩码
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # 词嵌入
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))  # 位置嵌入
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # 文本特征投影层
        
        # 3. 对比学习温度参数（初始值对应1/0.07）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # 初始化所有参数
        self._initialize_parameters()

    def _build_attention_mask(self) -> torch.Tensor:
        """构建文本Transformer的因果掩码（下三角可见，防止未来信息泄露）"""
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 上三角置为-inf，下三角（包括对角线）为0
        return mask

    def _initialize_parameters(self):
        """参数初始化（遵循CLIP原论文设置）"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # Transformer参数初始化
        proj_std = (self.transformer.resblocks[0].attn.out_proj.in_features ** -0.5) * ((2 * len(self.transformer.resblocks)) ** -0.5)
        attn_std = self.transformer.resblocks[0].attn.in_proj_weight.shape[1] ** -0.5
        fc_std = (2 * self.transformer.resblocks[0].mlp.c_fc.in_features) ** -0.5
        
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        # 文本投影层初始化
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.resblocks[0].attn.out_proj.in_features ** -0.5)

    @property
    def dtype(self) -> torch.dtype:
        """模型参数的数据类型（适配FP16）"""
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """图像编码：[batch, 3, H, W] -> [batch, embed_dim]"""
        return self.visual(image.type(self.dtype))

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """文本编码：[batch, context_length] -> [batch, embed_dim]"""
        # 1. 词嵌入 + 位置嵌入
        x = self.token_embedding(text).type(self.dtype)  # [batch, context_length, transformer_width]
        x = x + self.positional_embedding.type(self.dtype)
        
        # 2. Transformer编码
        x = x.permute(1, 0, 2)  # [context_length, batch, transformer_width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch, context_length, transformer_width]
        
        # 3. 取结束标记（EOT）的特征并投影
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  # 取每个序列中EOT的位置
        x = x @ self.text_projection
        return x

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播（对比学习）"""
        # 1. 编码图像和文本
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 2. 特征归一化（余弦相似度计算基础）
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # 3. 计算对比logits（温度缩放的余弦相似度）
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()  # [batch, batch]：图像->文本的相似度
        logits_per_text = logits_per_image.t()  # [batch, batch]：文本->图像的相似度
        
        return logits_per_image, logits_per_text


# -------------------------- 3. 工具函数（下载/加载/预处理） --------------------------
def _download_model(url: str, root: str = None) -> str:
    """下载预训练模型并校验SHA256"""
    root = root or os.path.expanduser("~/.cache/clip")
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_path = os.path.join(root, filename)
    
    # 校验文件是否已存在且完整
    expected_sha256 = url.split("/")[-2]
    if os.path.exists(download_path):
        with open(download_path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == expected_sha256:
                return download_path
        warnings.warn(f"文件{download_path}已存在但校验失败，重新下载")
    
    # 下载文件（带进度条）
    with urllib.request.urlopen(url) as source, open(download_path, "wb") as output:
        total_size = int(source.info().get("Content-Length"))
        with tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                pbar.update(len(buffer))
    
    # 二次校验
    with open(download_path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != expected_sha256:
            raise RuntimeError("模型下载成功但校验失败，请重试")
    return download_path


def _image_transform(n_px: int) -> Callable:
    """图像预处理流水线（适配CLIP输入）"""
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda x: x.convert("RGB"),  # 转为RGB
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # CLIP默认归一化
    ])


def available_models() -> List[str]:
    """获取可用的预训练模型名称"""
    return list(_MODELS.keys())


def load_model(model_name: str, device: Union[str, torch.device] = None) -> Tuple[CLIP, Callable]:
    """
    加载预训练CLIP模型
    
    参数：
        model_name: 模型名称（如"ViT-B/32"）或本地模型路径
        device: 运行设备（默认自动检测cuda/cpu）
    
    返回：
        model: CLIP模型（已放至指定设备）
        preprocess: 图像预处理函数
    """
    # 自动检测设备
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 下载/获取模型路径
    if model_name in _MODELS:
        model_path = _download_model(_MODELS[model_name])
    elif os.path.isfile(model_name):
        model_path = model_name
    else:
        raise RuntimeError(f"模型{model_name}不存在，可用模型：{available_models()}")
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location="cpu")
    
    # 从权重推断模型参数（简化原build_model逻辑，只支持ViT类模型）
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    
    # ViT参数推断
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    num_patches = state_dict["visual.positional_embedding"].shape[0] - 1
    image_resolution = int(np.sqrt(num_patches)) * vision_patch_size
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.transformer.resblocks.") and k.endswith(".attn.in_proj_weight")])
    
    # 文本Transformer参数推断
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len([k for k in state_dict.keys() if k.startswith("transformer.resblocks.") and k.endswith(".attn.in_proj_weight")])
    
    # 构建模型
    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers
    )
    
    # 加载权重并放至设备
    model.load_state_dict(state_dict)
    model = model.to(device).eval()  #  eval()模式（关闭dropout等）
    
    # 生成图像预处理函数
    preprocess = _image_transform(image_resolution)
    
    return model, preprocess


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.Tensor:
    """
    文本转模型输入的token ID张量
    
    参数：
        texts: 单个文本或文本列表
        context_length: 文本最大长度（CLIP固定为77）
        truncate: 过长时是否截断（否则报错）
    
    返回：
        tensor: [batch_size, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # 为每个文本添加起始(<|startoftext|>)和结束(<|endoftext|>)标记
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    # 初始化结果张量（用0填充）
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token  # 确保最后一个是结束标记
            else:
                raise RuntimeError(f"文本过长（{len(tokens)} > {context_length}），请设置truncate=True")
        result[i, :len(tokens)] = torch.tensor(tokens)
    
    return result


# -------------------------- 4. 预训练流程（简化版） --------------------------
class ImageTextDataset(Dataset):
    """
    图像-文本对数据集（预训练用）
    输入：列表形式的(image_path, text)对
    """
    def __init__(self, data: List[Tuple[str, str]], preprocess: Callable):
        self.data = data
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, text = self.data[idx]
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        # 文本分词
        text = tokenize(text, truncate=True)
        return image, text.squeeze(0)  # text维度从[1,77]转为[77]


def clip_pretrain_step(model: CLIP, image: torch.Tensor, text: torch.Tensor, device: str) -> torch.Tensor:
    """
    单步预训练（对比损失计算）
    CLIP的损失：图像和文本的对比交叉熵（正例是配对的(image, text)）
    """
    # 前向传播获取logits
    logits_per_image, logits_per_text = model(image, text)
    
    # 构建标签（对角线为正例，即第i个图像对应第i个文本）
    batch_size = image.shape[0]
    labels = torch.arange(batch_size, device=device)
    
    # 计算双向对比损失
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    total_loss = (loss_img + loss_txt) / 2
    
    return total_loss


def clip_pretrain(model: CLIP, train_data: List[Tuple[str, str]], epochs: int = 10, batch_size: int = 8, lr: float = 5e-5):
    """
    CLIP预训练主函数（简化版）
    
    参数：
        model: 未训练的CLIP模型
        train_data: 训练数据，格式[(image_path1, text1), (image_path2, text2), ...]
        epochs: 训练轮次
        batch_size: 批次大小
        lr: 学习率
    """
    device = next(model.parameters()).device
    preprocess = _image_transform(model.visual.input_resolution)
    
    # 1. 构建数据集和数据加载器
    dataset = ImageTextDataset(train_data, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 2. 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 3. 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, texts in progress_bar:
            # 数据放至设备
            images = images.to(device)
            texts = texts.to(device)
            
            # 前向传播计算损失
            loss = clip_pretrain_step(model, images, texts, device)
            
            # 反向传播更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计损失
            total_loss += loss.item() * images.shape[0]
            progress_bar.set_postfix(loss=loss.item())
        
        # 打印每轮平均损失
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    # 保存训练后的模型
    torch.save(model.state_dict(), "clip_pretrained.pth")
    print("预训练完成，模型已保存为 clip_pretrained.pth")


# -------------------------- 5. 推理流程（图像-文本匹配） --------------------------
def clip_inference(model: CLIP, preprocess: Callable, image_path: str, candidate_texts: List[str]) -> Tuple[List[str], List[float]]:
    """
    CLIP推理：给定图像和候选文本，返回匹配度排序
    
    参数：
        model: 预训练CLIP模型
        preprocess: 图像预处理函数
        image_path: 待匹配图像路径
        candidate_texts: 候选文本列表
    
    返回：
        sorted_texts: 按匹配度降序排列的文本
        sorted_scores: 对应的匹配分数（余弦相似度）
    """
    device = next(model.parameters()).device
    
    # 1. 预处理图像
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # 2. 预处理文本
    text_tokens = tokenize(candidate_texts, truncate=True).to(device)  # [num_texts, 77]
    
    # 3. 编码特征（关闭梯度计算，加速推理）
    with torch.no_grad():
        image_feat = model.encode_image(image)
        text_feat = model.encode_text(text_tokens)
    
    # 4. 计算相似度（余弦相似度）
    image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
    similarity = (image_feat @ text_feat.t()).squeeze(0).cpu().numpy()
    
    # 5. 排序
    sorted_idx = similarity.argsort()[::-1]  # 降序排列
    sorted_texts = [candidate_texts[i] for i in sorted_idx]
    sorted_scores = [similarity[i] for i in sorted_idx]
    
    return sorted_texts, sorted_scores


# -------------------------- 6. 示例：预训练 + 推理 --------------------------
if __name__ == "__main__":
    # 注意：以下为示例代码，实际预训练需准备大规模图像-文本数据集
    # 若没有数据集，可直接运行推理部分（加载预训练模型）
    
    # -------------------------- 示例1：预训练（需准备数据） --------------------------
    # # 1. 准备小规模演示数据（实际需替换为大规模数据）
    # demo_train_data = [
    #     ("cat1.jpg", "a photo of a cat"),
    #     ("dog1.jpg", "a photo of a dog"),
    #     ("cat2.jpg", "a cute cat sitting on sofa"),
    #     ("dog2.jpg", "a dog playing with ball"),
    #     # ... 更多图像-文本对
    # ]
    #
    # # 2. 初始化未训练的CLIP模型（ViT-B/32结构）
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # untrained_model = CLIP(
    #     embed_dim=512,
    #     image_resolution=224,
    #     vision_layers=12,
    #     vision_width=768,
    #     vision_patch_size=32,
    #     context_length=77,
    #     vocab_size=49408,
    #     transformer_width=512,
    #     transformer_heads=8,
    #     transformer_layers=12
    # ).to(device)
    #
    # # 3. 开始预训练
    # clip_pretrain(untrained_model, demo_train_data, epochs=3, batch_size=2)
    
    # -------------------------- 示例2：推理（加载预训练模型） --------------------------
    # 1. 加载预训练模型
    print("正在加载预训练模型...")
    model, preprocess = load_model("ViT-B/32")
    
    # 2. 推理配置
    test_image_path = "test_cat.jpg"  # 替换为你的测试图像路径
    candidate_texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a rabbit"
    ]
    
    # 3. 执行推理
    print(f"\n正在推理图像: {test_image_path}")
    sorted_texts, sorted_scores = clip_inference(model, preprocess, test_image_path, candidate_texts)
    
    # 4. 输出结果
    print("\n匹配结果（降序）:")
    for i, (text, score) in enumerate(zip(sorted_texts, sorted_scores), 1):
        print(f"{i}. {text}: {score:.4f}")