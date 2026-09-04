"""Mask Encoders 专用数据：复用四块裁剪，只新增中心遮挡。"""

import torch
from torch.utils.data import Dataset

from anomaly_reproduction.dataset import MVTecFourCropTrainDataset


def center_mask(image: torch.Tensor, mask_size: int = 64) -> torch.Tensor:
    """遮住 CHW 图片中心；输入已归一化到 [-1,1]，黑色应填 -1。"""
    if image.ndim not in (3, 4) or mask_size <= 0 or mask_size > min(image.shape[-2:]):
        raise ValueError("图片形状或 mask_size 不正确")
    masked = image.clone()
    height, width = image.shape[-2:]
    top, left = (height - mask_size) // 2, (width - mask_size) // 2
    masked[..., top : top + mask_size, left : left + mask_size] = -1.0
    return masked


class MaskPretrainDataset(Dataset):
    """从四块训练集固定选取 50%，同时返回遮挡输入与完整目标。"""

    def __init__(self, data_root, category="grid", image_size=128,
                 mask_size=64, fraction=0.5, seed=42):
        self.base = MVTecFourCropTrainDataset(data_root, category, image_size)
        self.mask_size = mask_size
        if not 0 < fraction <= 1:
            raise ValueError("fraction 必须在 (0,1] 范围")
        generator = torch.Generator().manual_seed(seed)
        count = int(len(self.base) * fraction)
        self.indices = torch.randperm(len(self.base), generator=generator)[:count].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.base[self.indices[index]]
        complete = sample["image"]
        return {**sample, "image": center_mask(complete, self.mask_size),
                "target": complete}
