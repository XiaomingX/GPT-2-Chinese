import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms, box_area
from typing import Any, Dict, List, Optional, Tuple, Generator, Type
import cv2
from PIL import Image


# 1. 掩码数据管理类
class MaskData:
    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(v, (list, np.ndarray, torch.Tensor)), "不支持的输入类型"
        self._data = dict(** kwargs)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(item, (list, np.ndarray, torch.Tensor)), "不支持的输入类型"
        self._data[key] = item

    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                self._data[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._data[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list):
                self._data[k] = [v[i] for i in keep] if keep.dtype != torch.bool else [a for i, a in enumerate(v) if keep[i]]

    def cat(self, new_data: "MaskData") -> None:
        for k, v in new_data._data.items():
            if k not in self._data:
                self._data[k] = v.copy() if isinstance(v, (np.ndarray, list)) else v.clone()
            elif isinstance(v, torch.Tensor):
                self._data[k] = torch.cat([self._data[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._data[k] = np.concatenate([self._data[k], v], axis=0)
            elif isinstance(v, list):
                self._data[k] += v.copy()

    def to_numpy(self) -> None:
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                self._data[k] = v.detach().cpu().numpy()


# 2. 核心工具函数
def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx:idx+count] = parity
        idx += count
        parity ^= True
    return mask.reshape(w, h).transpose()

def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat([torch.tensor([0], device=cur_idxs.device), cur_idxs+1, torch.tensor([h*w], device=cur_idxs.device)])
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [0] if tensor[i, 0] == 0 else []
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out

def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])

def calculate_stability_score(masks: torch.Tensor, mask_threshold: float, offset: float) -> torch.Tensor:
    intersections = (masks > (mask_threshold + offset)).sum(-1).sum(-1)
    unions = (masks > (mask_threshold - offset)).sum(-1).sum(-1)
    return intersections / unions

def build_point_grid(n_per_side: int) -> np.ndarray:
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1-offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    return np.stack([points_x, points_y], axis=-1).reshape(-1, 2)

def build_all_layer_point_grids(n_per_side: int, n_layers: int, scale: int) -> List[np.ndarray]:
    return [build_point_grid(int(n_per_side / (scale**i))) for i in range(n_layers+1)]

def generate_crop_boxes(im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float) -> Tuple[List[List[int]], List[int]]:
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(np.ceil((overlap*(n_crops-1) + orig_len)/n_crops))

    for i in range(n_layers):
        n_crops = 2 ** (i+1)
        overlap = int(overlap_ratio * min(im_h, im_w) * (2 / n_crops))
        crop_w = crop_len(im_w, n_crops, overlap)
        crop_h = crop_len(im_h, n_crops, overlap)
        
        x0_list = [int((crop_w - overlap)*i) for i in range(n_crops)]
        y0_list = [int((crop_h - overlap)*i) for i in range(n_crops)]
        
        for x0 in x0_list:
            for y0 in y0_list:
                box = [x0, y0, min(x0+crop_w, im_w), min(y0+crop_h, im_h)]
                crop_boxes.append(box)
                layer_idxs.append(i+1)
    return crop_boxes, layer_idxs

def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0 = crop_box[0], crop_box[1]
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    return boxes + offset

def uncrop_points(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0 = crop_box[0], crop_box[1]
    offset = torch.tensor([[x0, y0]], device=points.device)
    return points + offset

def uncrop_masks(masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int) -> torch.Tensor:
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    pad = (x0, orig_w - x1, y0, orig_h - y1)
    return F.pad(masks, pad, value=0)

def remove_small_regions(mask: np.ndarray, area_thresh: int, mode: str) -> Tuple[np.ndarray, bool]:
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]
    small_regions = [i+1 for i, s in enumerate(sizes) if s < area_thresh]
    
    if not small_regions:
        return mask, False
    
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        if not fill_labels:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    
    shape = masks.shape
    h, w = shape[-2:]
    masks_flat = masks.flatten(0, -3) if len(shape) > 2 else masks.unsqueeze(0)
    
    # 计算上下边界
    in_h = masks_flat.max(-1)[0]
    h_coords = in_h * torch.arange(h, device=masks.device)[None, :]
    bottom = h_coords.max(-1)[0]
    h_coords += h * (~in_h)
    top = h_coords.min(-1)[0]
    
    # 计算左右边界
    in_w = masks_flat.max(-2)[0]
    w_coords = in_w * torch.arange(w, device=masks.device)[None, :]
    right = w_coords.max(-1)[0]
    w_coords += w * (~in_w)
    left = w_coords.min(-1)[0]
    
    # 过滤空掩码
    empty = (right < left) | (bottom < top)
    boxes = torch.stack([left, top, right, bottom], dim=-1)
    boxes = boxes * (~empty).unsqueeze(-1)
    
    return boxes.reshape(*shape[:-2], 4) if len(shape) > 2 else boxes[0]

def box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    box_xywh = box_xyxy.clone()
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args)
    n_batches = len(args[0]) // batch_size + (1 if len(args[0]) % batch_size != 0 else 0)
    for b in range(n_batches):
        yield [arg[b*batch_size : (b+1)*batch_size] for arg in args]

def is_box_near_crop_edge(boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0) -> torch.Tensor:
    crop_torch = torch.tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_torch = torch.tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes_uncrop = uncrop_boxes_xyxy(boxes, crop_box).float()
    
    near_crop = torch.isclose(boxes_uncrop, crop_torch[None, :], atol=atol)
    near_orig = torch.isclose(boxes_uncrop, orig_torch[None, :], atol=atol)
    return torch.any(torch.logical_and(near_crop, ~near_orig), dim=1)


# 3. 模型基础组件
class MLPBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embed_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# 4. 图像编码器
class ImageEncoderViT(nn.Module):
    def __init__(
        self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3,
        embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
        mlp_ratio: float = 4.0, out_chans: int = 256
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, img_size//patch_size, img_size//patch_size))

        # Transformer块
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            block = nn.Sequential(
                nn.LayerNorm(embed_dim),
                attn,
                nn.LayerNorm(embed_dim),
                MLPBlock(embed_dim, int(embed_dim * mlp_ratio))
            )
            self.blocks.append(block)

        # 特征颈部
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            LayerNorm2d(out_chans)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 图像->Patch嵌入
        x = self.patch_embed(x)  # B, C, H, W
        x = x + self.pos_embed  # 加位置嵌入

        # Transformer块计算
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  # B, N_patches, C
        for block in self.blocks:
            norm_x = block[0](x)
            attn_out, _ = block[1](norm_x, norm_x, norm_x)
            x = x + attn_out
            x = x + block[3](block[2](x))

        # 颈部处理
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        return self.neck(x)


# 5. 提示编码器
class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "pos_mat", scale * torch.randn((2, num_pos_feats))
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1  # 转为[-1,1]
        coords = coords @ self.pos_mat
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

class PromptEncoder(nn.Module):
    def __init__(
        self, embed_dim: int, image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int], mask_in_chans: int = 16
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe = PositionEmbeddingRandom(embed_dim // 2)

        # 点/框嵌入
        self.point_embeds = nn.ModuleList([nn.Embedding(1, embed_dim) for _ in range(4)])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # 掩码嵌入
        self.mask_downscale = nn.Sequential(
            nn.Conv2d(1, mask_in_chans//4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans//4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans//4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        h, w = self.image_embedding_size
        grid = torch.ones((h, w), device=self.pe.pos_mat.device)
        y = (grid.cumsum(0) - 0.5) / h
        x = (grid.cumsum(1) - 0.5) / w
        coords = torch.stack([x, y], dim=-1)
        return self.pe(coords).permute(2, 0, 1).unsqueeze(0)

    def forward(
        self, points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None, masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = points[0].shape[0] if points else (boxes.shape[0] if boxes else (masks.shape[0] if masks else 1))
        sparse_embeds = torch.empty((bs, 0, self.embed_dim), device=self.pe.pos_mat.device)

        # 点提示嵌入
        if points:
            coords, labels = points
            coords = coords / torch.tensor(self.input_image_size[::-1], device=coords.device)
            pe = self.pe(coords)
            pe[labels == -1] = 0.0
            pe[labels == -1] += self.not_a_point_embed.weight
            pe[labels == 0] += self.point_embeds[0].weight
            pe[labels == 1] += self.point_embeds[1].weight
            sparse_embeds = torch.cat([sparse_embeds, pe], dim=1)

        # 框提示嵌入
        if boxes:
            boxes = boxes / torch.tensor(self.input_image_size[::-1], device=boxes.device)
            coords = boxes.reshape(-1, 2, 2)
            pe = self.pe(coords)
            pe[:, 0] += self.point_embeds[2].weight
            pe[:, 1] += self.point_embeds[3].weight
            sparse_embeds = torch.cat([sparse_embeds, pe], dim=1)

        # 掩码提示嵌入
        if masks:
            dense_embeds = self.mask_downscale(masks)
        else:
            dense_embeds = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeds, dense_embeds


# 6. 掩码解码器
class TwoWayTransformer(nn.Module):
    def __init__(self, depth: int, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                    nn.LayerNorm(embed_dim),
                    MLPBlock(embed_dim, mlp_dim)
                )
            )

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.layers:
            # 自注意力
            norm_q = block[0](queries)
            attn_q, _ = block[1](norm_q, norm_q, norm_q)
            queries = queries + attn_q

            # 交叉注意力
            norm_q = block[2](queries)
            norm_k = block[2](keys)
            attn_qk, _ = block[1](norm_q, norm_k, norm_k)
            queries = queries + attn_qk

            # MLP
            queries = queries + block[3](block[2](queries))
        return queries, keys

class MaskDecoder(nn.Module):
    def __init__(
        self, transformer_dim: int, transformer: nn.Module,
        num_multimask_outputs: int = 3, iou_head_depth: int = 3, iou_head_hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        # 特殊token
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(num_multimask_outputs + 1, transformer_dim)

        # 掩码上采样
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # 掩码预测头
        self.output_hypernetworks = nn.ModuleList([
            MLPBlock(transformer_dim, transformer_dim, transformer_dim // 8)
            for _ in range(num_multimask_outputs + 1)
        ])

        # IoU预测头
        self.iou_prediction_head = nn.Sequential(
            MLPBlock(transformer_dim, iou_head_hidden_dim, iou_head_hidden_dim),
            nn.Linear(iou_head_hidden_dim, num_multimask_outputs + 1)
        )

    def forward(
        self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
        sparse_prompt_embeds: torch.Tensor, dense_prompt_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 准备输入
        bs, c, h, w = image_embeddings.shape
        src = image_embeddings + dense_prompt_embeds
        src = src.flatten(2).permute(0, 2, 1)  # B, H*W, C
        pos_src = image_pe.flatten(2).permute(0, 2, 1)  # B, H*W, C

        # 组合提示和特殊token
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(bs, -1, -1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeds], dim=1)

        # Transformer计算
        hs, _ = self.transformer(tokens, src)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1+self.num_multimask_outputs+1, :]

        # 上采样图像特征
        src = src.transpose(1, 2).view(bs, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        upscaled_embedding = upscaled_embedding.flatten(2)  # B, C, H*W

        # 预测掩码
        masks = []
        for i in range(self.num_multimask_outputs + 1):
            hyper_in = self.output_hypernetworks[i](mask_tokens_out[:, i, :])
            masks.append((hyper_in.unsqueeze(1) @ upscaled_embedding).view(bs, 1, h*4, w*4))
        masks = torch.cat(masks, dim=1)

        # 预测IoU
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# 7. 完整SAM模型
class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self, image_encoder: ImageEncoderViT, prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder, pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375]
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """图像预处理：归一化和填充"""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        return F.pad(x, (0, padw, 0, padh))

    def postprocess_masks(
        self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """掩码后处理：上采样到原始图像尺寸"""
        masks = F.interpolate(
            masks, (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear", align_corners=False
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    @torch.no_grad()
    def forward(
        self, image: torch.Tensor, original_size: Tuple[int, int],
        point_coords: Optional[torch.Tensor] = None, point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None, mask_inputs: Optional[torch.Tensor] = None,
        multimask_output: bool = True
    ) -> Dict[str, torch.Tensor]:
        """前向传播：从图像和提示生成掩码"""
        # 图像编码
        input_size = image.shape[-2:]
        image = self.preprocess(image)
        image_embedding = self.image_encoder(image.unsqueeze(0))[0]

        # 提示编码
        points = (point_coords, point_labels) if point_coords is not None else None
        sparse_embeds, dense_embeds = self.prompt_encoder(
            points=points, boxes=boxes, masks=mask_inputs
        )

        # 掩码解码
        low_res_masks, iou_preds = self.mask_decoder(
            image_embeddings=image_embedding.unsqueeze(0),
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeds=sparse_embeds,
            dense_prompt_embeds=dense_embeds
        )

        # 后处理
        masks = self.postprocess_masks(low_res_masks, input_size, original_size)
        masks = masks > self.mask_threshold

        # 选择输出掩码
        if multimask_output:
            return {
                "masks": masks[0, 1:],
                "iou_predictions": iou_preds[0, 1:],
                "low_res_logits": low_res_masks[0, 1:]
            }
        else:
            return {
                "masks": masks[0, :1],
                "iou_predictions": iou_preds[0, :1],
                "low_res_logits": low_res_masks[0, :1]
            }


# 8. 预测器类（简化接口）
class SamPredictor:
    def __init__(self, sam_model: Sam) -> None:
        self.model = sam_model
        self.reset_image()

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.orig_size = None
        self.input_size = None

    def set_image(self, image: np.ndarray) -> None:
        """设置输入图像，进行预处理和编码"""
        self.orig_size = image.shape[:2]
        self.input_size = image.shape[:2]
        
        # 转换为PyTorch张量
        image_tensor = torch.as_tensor(image, device=self.model.device)
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        # 图像编码
        input_image = self.model.preprocess(image_tensor)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self, point_coords: Optional[np.ndarray] = None, point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None, mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """根据提示预测掩码"""
        if not self.is_image_set:
            raise RuntimeError("请先调用set_image设置图像")

        # 准备输入
        coords_torch, labels_torch, box_torch, mask_torch = None, None, None, None
        
        if point_coords is not None:
            assert point_labels is not None
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.model.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.model.device)
            
        if box is not None:
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.model.device)
            box_torch = box_torch[None, :]
            
        if mask_input is not None:
            mask_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.model.device)
            mask_torch = mask_torch[None, None, :, :]

        # 模型预测
        outputs = self.model(
            image=self.features,
            original_size=self.orig_size,
            point_coords=coords_torch,
            point_labels=labels_torch,
            boxes=box_torch,
            mask_inputs=mask_torch,
            multimask_output=multimask_output
        )

        return (
            outputs["masks"].detach().cpu().numpy(),
            outputs["iou_predictions"].detach().cpu().numpy(),
            outputs["low_res_logits"].detach().cpu().numpy()
        )


# 9. 自动掩码生成器
class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
    ) -> None:
        self.model = model
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.min_mask_region_area = min_mask_region_area

        # 生成点网格
        self.point_grids = build_all_layer_point_grids(
            points_per_side, crop_n_layers, 2
        )

    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """自动生成图像中所有物体的掩码"""
        predictor = SamPredictor(self.model)
        predictor.set_image(image)
        orig_size = image.shape[:2]

        # 生成裁剪框
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, 0.25
        )

        # 处理每个裁剪区域
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            # 裁剪图像并处理
            x0, y0, x1, y1 = crop_box
            cropped_im = image[y0:y1, x0:x1, :]
            cropped_size = cropped_im.shape[:2]
            
            # 获取该层的点网格
            points_scale = np.array(cropped_size)[None, ::-1]
            points_for_image = self.point_grids[layer_idx] * points_scale

            # 批量处理点
            for (points,) in batch_iterator(self.points_per_batch, points_for_image):
                # 预测掩码
                points_torch = torch.as_tensor(points, dtype=torch.float, device=self.model.device)
                labels_torch = torch.ones(points_torch.shape[0], dtype=torch.int, device=self.model.device)
                
                masks, iou_preds, _ = predictor.predict(
                    point_coords=points_torch,
                    point_labels=labels_torch,
                    multimask_output=True
                )

                # 处理结果
                for i in range(masks.shape[0]):
                    mask = masks[i]
                    iou_pred = iou_preds[i]
                    
                    # 计算稳定性分数
                    stability_score = calculate_stability_score(
                        torch.as_tensor(mask[None, ...]),
                        self.model.mask_threshold,
                        self.stability_score_offset
                    ).item()

                    # 过滤低质量掩码
                    if iou_pred < self.pred_iou_thresh or stability_score < self.stability_score_thresh:
                        continue

                    # 转换为RLE并保存
                    rle = mask_to_rle_pytorch(torch.as_tensor(mask[None, ...]))[0]
                    box = batched_mask_to_box(torch.as_tensor(mask[None, ...])).tolist()
                    
                    data.cat(MaskData({
                        "rles": [rle],
                        "iou_preds": [iou_pred],
                        "stability_score": [stability_score],
                        "boxes": [box],
                        "points": [points[i]],
                        "crop_boxes": [crop_box]
                    }))

        # 过滤重复掩码
        if len(data["boxes"]) > 0:
            boxes = torch.as_tensor(data["boxes"], device=self.model.device)
            scores = torch.as_tensor(data["iou_preds"], device=self.model.device)
            keep = batched_nms(boxes.float(), scores, torch.zeros_like(boxes[:, 0]), self.box_nms_thresh)
            data.filter(keep)

        # 后处理：移除小区域
        if self.min_mask_region_area > 0 and len(data["rles"]) > 0:
            data = self.postprocess_small_regions(data, self.min_mask_region_area, self.box_nms_thresh)

        # 转换为字典列表并排序
        masks = []
        for i in range(len(data["rles"])):
            mask = rle_to_mask(data["rles"][i])
            masks.append({
                "segmentation": data["rles"][i],
                "area": area_from_rle(data["rles"][i]),
                "bbox": box_xyxy_to_xywh(torch.as_tensor(data["boxes"][i])).tolist(),
                "predicted_iou": data["iou_preds"][i],
                "stability_score": data["stability_score"][i],
                "point_coords": [data["points"][i].tolist()]
            })

        return sorted(masks, key=lambda x: x["area"], reverse=True)

    def postprocess_small_regions(
        self, mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        if len(mask_data["rles"]) == 0:
            return mask_data

        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)
            
            # 移除小区域和孔洞
            mask, _ = remove_small_regions(mask, min_area, mode="holes")
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            
            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            scores.append(float(not changed))  # 未改变的掩码得分更高

        # 重新计算边界框并去重
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),
            nms_thresh
        )

        # 更新数据
        for i in keep:
            if scores[i] == 0:
                mask_torch = masks[i].unsqueeze(0)
                mask_data["rles"][i] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i] = boxes[i].tolist()
        mask_data.filter(keep)

        return mask_data


# 10. 模型构建函数
def build_sam_vit_b() -> Sam:
    """构建基础版SAM模型 (ViT-B)"""
    image_encoder = ImageEncoderViT(
        img_size=1024,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_chans=256
    )
    
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024)
    )
    
    transformer = TwoWayTransformer(
        depth=2,
        embed_dim=256,
        num_heads=8,
        mlp_dim=2048
    )
    
    mask_decoder = MaskDecoder(
        transformer_dim=256,
        transformer=transformer,
        num_multimask_outputs=3
    )
    
    return Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder
    )


# 11. 使用示例
def main():
    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam_vit_b().to(device)
    
    # 加载示例图像（实际使用时替换为你的图像路径）
    try:
        image = np.array(Image.open("example.jpg").convert("RGB"))
    except:
        # 如果没有图像，创建一个简单的测试图像
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (255, 0, 0), -1)
        cv2.circle(image, (400, 400), 50, (0, 255, 0), -1)
    
    # 自动生成掩码
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    print(f"生成了 {len(masks)} 个掩码")
    
    # 可视化结果
    for i, mask_data in enumerate(masks):
        mask = rle_to_mask(mask_data["segmentation"])
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        
        # 在图像上绘制掩码
        image[mask] = image[mask] * 0.5 + color * 0.5
        
        # 绘制边界框
        bbox = mask_data["bbox"]
        x, y, w, h = bbox
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 255), 2)
    
    # 保存结果
    Image.fromarray(image).save("sam_result.jpg")
    print("结果已保存为 sam_result.jpg")


if __name__ == "__main__":
    main()
