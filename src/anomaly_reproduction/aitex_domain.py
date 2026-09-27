"""AITEX 域内训练的数据划分与图块数据集。

所有划分都以原始长图为单位完成，再把长图切成图块，避免同一原图的图块
同时出现在训练、验证和测试中。划分清单保存为 YAML，后续训练和评价只读取
该清单，不再临时随机抽样。
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from anomaly_reproduction.aitex_dataset import parse_aitex_codes, sliding_positions
from anomaly_reproduction.ts_dataset import image_transform


SPLIT_KEYS = (
    "normal_train",
    "normal_validation",
    "normal_test",
    "anomaly_train",
    "anomaly_test",
)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def create_aitex_manifest(
    data_root: str | Path,
    seed: int = 42,
    normal_train_per_fabric: int = 12,
    normal_validation_per_fabric: int = 4,
    anomaly_train_per_type: int = 1,
) -> dict:
    """创建固定且分层的 AITEX 原图级划分。

    正常图按织物代码分层。异常图按缺陷代码分层；只有样本数严格大于训练
    数量的类别才抽训练图，因此每个已见缺陷类型至少保留一张测试图，只有一张
    图的稀有缺陷全部作为未见类型留在测试集。
    """
    root = Path(data_root).resolve()
    normal_paths = sorted((root / "NODefect_images").rglob("*.png"))
    anomaly_paths = sorted((root / "Defect_images").glob("*.png"))
    if not normal_paths or not anomaly_paths:
        raise FileNotFoundError(
            "AITEX 根目录必须包含非空的 NODefect_images 与 Defect_images"
        )
    if min(
        normal_train_per_fabric,
        normal_validation_per_fabric,
        anomaly_train_per_type,
    ) <= 0:
        raise ValueError("各划分数量必须为正数")

    rng = random.Random(seed)
    normal_by_fabric: dict[str, list[Path]] = defaultdict(list)
    for path in normal_paths:
        _, fabric_code = parse_aitex_codes(path)
        normal_by_fabric[fabric_code].append(path)

    split: dict[str, list[str]] = {key: [] for key in SPLIT_KEYS}
    normal_counts = {}
    for fabric_code, paths in sorted(normal_by_fabric.items()):
        paths = sorted(paths)
        rng.shuffle(paths)
        required = normal_train_per_fabric + normal_validation_per_fabric + 1
        if len(paths) < required:
            raise ValueError(
                f"织物 {fabric_code} 只有 {len(paths)} 张正常图，至少需要 {required} 张"
            )
        train_end = normal_train_per_fabric
        validation_end = train_end + normal_validation_per_fabric
        groups = {
            "normal_train": paths[:train_end],
            "normal_validation": paths[train_end:validation_end],
            "normal_test": paths[validation_end:],
        }
        for key, group in groups.items():
            split[key].extend(_relative(path, root) for path in group)
        normal_counts[fabric_code] = {key: len(group) for key, group in groups.items()}

    anomaly_by_type: dict[str, list[Path]] = defaultdict(list)
    for path in anomaly_paths:
        defect_code, _ = parse_aitex_codes(path)
        anomaly_by_type[defect_code].append(path)

    anomaly_counts = {}
    seen_types = []
    unseen_types = []
    for defect_code, paths in sorted(anomaly_by_type.items()):
        paths = sorted(paths)
        rng.shuffle(paths)
        train_count = anomaly_train_per_type if len(paths) > anomaly_train_per_type else 0
        train_paths = paths[:train_count]
        test_paths = paths[train_count:]
        split["anomaly_train"].extend(_relative(path, root) for path in train_paths)
        split["anomaly_test"].extend(_relative(path, root) for path in test_paths)
        (seen_types if train_paths else unseen_types).append(defect_code)
        anomaly_counts[defect_code] = {
            "train": len(train_paths),
            "test": len(test_paths),
        }

    for key in SPLIT_KEYS:
        split[key] = sorted(split[key])
    manifest = {
        "version": 1,
        "dataset": "AITEX-AFID",
        "seed": seed,
        "settings": {
            "normal_train_per_fabric": normal_train_per_fabric,
            "normal_validation_per_fabric": normal_validation_per_fabric,
            "anomaly_train_per_type": anomaly_train_per_type,
            "split_unit": "original image",
        },
        "splits": split,
        "summary": {
            "normal_by_fabric": normal_counts,
            "anomaly_by_defect": anomaly_counts,
            "seen_defect_types": sorted(seen_types),
            "unseen_defect_types": sorted(unseen_types),
            "counts": {key: len(value) for key, value in split.items()},
        },
    }
    validate_aitex_manifest(manifest, root)
    return manifest


def save_aitex_manifest(manifest: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, allow_unicode=True, sort_keys=False)


def load_aitex_manifest(path: str | Path, data_root: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        manifest = yaml.safe_load(file)
    validate_aitex_manifest(manifest, Path(data_root))
    return manifest


def validate_aitex_manifest(manifest: dict, data_root: str | Path) -> None:
    root = Path(data_root)
    if manifest.get("dataset") != "AITEX-AFID" or "splits" not in manifest:
        raise ValueError("不是有效的 AITEX-AFID 划分清单")
    split = manifest["splits"]
    missing_keys = set(SPLIT_KEYS) - set(split)
    if missing_keys:
        raise ValueError(f"划分清单缺少字段：{sorted(missing_keys)}")
    sets = {key: set(split[key]) for key in SPLIT_KEYS}
    for index, left in enumerate(SPLIT_KEYS):
        for right in SPLIT_KEYS[index + 1:]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"{left} 与 {right} 存在数据泄漏：{sorted(overlap)[:3]}")
    missing = [relative for key in SPLIT_KEYS for relative in split[key]
               if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"划分清单中的图片不存在：{missing[:3]}")
    if not all(split[key] for key in SPLIT_KEYS):
        raise ValueError("训练、验证和测试划分均不能为空")


class AITEXManifestTileDataset(Dataset):
    """将清单中的原始长图展开为固定正方形图块。"""

    def __init__(
        self,
        data_root: str | Path,
        relative_paths: list[str],
        label: int,
        split_name: str,
        image_size: int = 64,
        source_tile_size: int = 256,
        source_stride: int = 256,
        selected_tiles: list[dict] | None = None,
    ) -> None:
        self.root = Path(data_root)
        self.transform = image_transform(image_size)
        allowed = set(relative_paths)
        self.samples: list[dict] = []
        if selected_tiles is not None:
            for item in selected_tiles:
                if item["path"] not in allowed:
                    raise ValueError(f"选中图块不属于当前划分：{item['path']}")
                self.samples.append({
                    "path": self.root / item["path"],
                    "relative_path": item["path"],
                    "tile_index": int(item["tile_index"]),
                    "box": tuple(item["box"]),
                })
        else:
            for relative in sorted(relative_paths):
                path = self.root / relative
                with Image.open(path) as image:
                    width, height = image.size
                boxes = [
                    (left, top, left + source_tile_size, top + source_tile_size)
                    for top in sliding_positions(height, source_tile_size, source_stride)
                    for left in sliding_positions(width, source_tile_size, source_stride)
                ]
                for tile_index, box in enumerate(boxes):
                    self.samples.append({
                        "path": path,
                        "relative_path": relative,
                        "tile_index": tile_index,
                        "box": box,
                    })
        if not self.samples:
            raise FileNotFoundError(f"{split_name} 没有可用图块")
        self.label = int(label)
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample["path"]) as image:
            tensor = self.transform(image.convert("RGB").crop(sample["box"]))
        defect_code, fabric_code = parse_aitex_codes(sample["path"])
        return {
            "image": tensor,
            "path": str(sample["path"]),
            "relative_path": sample["relative_path"],
            "label": self.label,
            "split": self.split_name,
            "defect_code": defect_code,
            "fabric_code": fabric_code,
            "tile_index": sample["tile_index"],
            "box": torch.tensor(sample["box"], dtype=torch.int64),
        }


class AITEXManifestPixelDataset(AITEXManifestTileDataset):
    """只读取清单中的测试图，同时返回与图块对齐的像素掩码。"""

    def __init__(
        self,
        data_root: str | Path,
        normal_paths: list[str],
        anomaly_paths: list[str],
        image_size: int = 64,
        source_tile_size: int = 256,
        source_stride: int = 256,
    ) -> None:
        root = Path(data_root)
        self.mask_dir = root / "Mask_images"
        usable_anomalies = []
        self.missing_mask_paths = []
        self.mask_paths: dict[str, list[Path]] = {}
        for relative in anomaly_paths:
            stem = Path(relative).stem
            masks = sorted(self.mask_dir.glob(f"{stem}_mask*.png"))
            if masks:
                usable_anomalies.append(relative)
                self.mask_paths[relative] = masks
            else:
                self.missing_mask_paths.append(root / relative)
        combined = list(normal_paths) + usable_anomalies
        super().__init__(
            root, combined, label=0, split_name="test", image_size=image_size,
            source_tile_size=source_tile_size, source_stride=source_stride,
        )
        anomaly_set = set(usable_anomalies)
        for sample in self.samples:
            sample["label"] = int(sample["relative_path"] in anomaly_set)
        self.normal_image_count = len(normal_paths)
        self.anomaly_image_count = len(usable_anomalies)
        self.image_count = self.normal_image_count + self.anomaly_image_count
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        result = super().__getitem__(index)
        label = int(sample["label"])
        result["label"] = label
        if label:
            union = None
            for mask_path in self.mask_paths[sample["relative_path"]]:
                with Image.open(mask_path) as mask:
                    current = self.mask_transform(mask.convert("L").crop(sample["box"]))
                union = current if union is None else torch.maximum(union, current)
            result["mask"] = union
        else:
            size = result["image"].shape[-1]
            result["mask"] = torch.zeros((1, size, size))
        return result
