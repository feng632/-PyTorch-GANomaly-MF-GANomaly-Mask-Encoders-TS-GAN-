"""TS-GAN 异常训练集与无泄漏测试集。"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from anomaly_reproduction.dataset import FOUR_CROP_NAMES, four_crop_box


DEFECT_TYPES = ("bent", "broken", "glue", "metal_contamination", "thread")


def choose_anomaly_images(data_root, category="grid", per_type=2, seed=42):
    """每种缺陷固定随机选择原图；返回相对 category 根目录的 POSIX 路径。"""
    root = Path(data_root) / category
    rng = random.Random(seed)
    selected = []
    for defect in DEFECT_TYPES:
        candidates = sorted((root / "test" / defect).glob("*.png"))
        if len(candidates) < per_type:
            raise FileNotFoundError(f"{defect} 的异常图不足 {per_type} 张")
        selected.extend(path.relative_to(root).as_posix() for path in rng.sample(candidates, per_type))
    return sorted(selected)


def image_transform(image_size=64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


class TSAnomalyTrainDataset(Dataset):
    """把选中的 10 张异常原图各裁成四块，得到 40 个训练 patch。"""

    def __init__(self, data_root, selected_paths, category="grid", image_size=64):
        self.root = Path(data_root) / category
        self.transform = image_transform(image_size)
        self.samples = [(self.root / relative, crop) for relative in selected_paths for crop in range(4)]
        if not self.samples or not all(path.is_file() for path, _ in self.samples):
            raise FileNotFoundError("异常训练图片不存在")

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        path, crop_index = self.samples[index]
        with Image.open(path) as image:
            crop = image.convert("RGB").crop(four_crop_box(image, crop_index))
            tensor = self.transform(crop)
        return {"image": tensor, "path": str(path), "label": 1,
                "defect_type": path.parent.name, "crop_index": crop_index,
                "crop_name": FOUR_CROP_NAMES[crop_index]}


class TSTestDataset(Dataset):
    """测试时排除异常支路训练见过的 10 张原图。"""

    def __init__(self, data_root, excluded_paths, category="grid", image_size=64):
        self.root = Path(data_root) / category
        excluded = set(excluded_paths)
        paths = sorted((self.root / "test").glob("*/*.png"))
        self.paths = [p for p in paths if p.relative_to(self.root).as_posix() not in excluded]
        self.samples = [(path, crop) for path in self.paths for crop in range(4)]
        self.transform = image_transform(image_size)
        if not self.samples:
            raise FileNotFoundError("没有可用测试图片")

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        path, crop_index = self.samples[index]
        defect = path.parent.name
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb.crop(four_crop_box(rgb, crop_index)))
        return {"image": tensor, "path": str(path), "label": int(defect != "good"),
                "defect_type": defect, "crop_index": crop_index,
                "crop_name": FOUR_CROP_NAMES[crop_index]}
