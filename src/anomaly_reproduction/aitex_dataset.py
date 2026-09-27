"""AITEX-AFID 外部验证数据读取。

AITEX 原图通常是 4096x256 的长条图。TS-GAN 固定接收 64x64，直接把整张
长条图压缩成正方形会严重破坏纹理和细小缺陷，因此这里先沿原图切取
256x256 正方形区域，再缩放到模型要求的 64x64。
"""

from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from anomaly_reproduction.ts_dataset import image_transform


def sliding_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """返回覆盖完整边长的裁剪起点，最后一个块一定贴住右侧或下侧边缘。"""
    if length < tile_size:
        raise ValueError(f"原图边长 {length} 小于裁剪尺寸 {tile_size}")
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def parse_aitex_codes(path: Path) -> tuple[str, str]:
    """由 nnnn_ddd_ff.png 解析缺陷代码 ddd 与织物代码 ff。"""
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"AITEX 文件名不符合 nnnn_ddd_ff.png：{path.name}")
    return parts[1], parts[2]


def split_normal_images(
    paths: list[Path], calibration_per_fabric: int
) -> tuple[set[Path], set[Path]]:
    """每种织物固定取前若干张正常图校准阈值，其余正常图才参与测试。"""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        _, fabric_code = parse_aitex_codes(path)
        grouped[fabric_code].append(path)

    calibration: set[Path] = set()
    test: set[Path] = set()
    for fabric_code, fabric_paths in sorted(grouped.items()):
        ordered = sorted(fabric_paths)
        if len(ordered) <= calibration_per_fabric:
            raise ValueError(
                f"织物 {fabric_code} 的正常图只有 {len(ordered)} 张，"
                f"无法取 {calibration_per_fabric} 张校准图后继续测试"
            )
        calibration.update(ordered[:calibration_per_fabric])
        test.update(ordered[calibration_per_fabric:])
    return calibration, test


class AITEXTileDataset(Dataset):
    """把 AITEX 长条图展开为可供 TS-GAN 推理的正方形图块。"""

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 64,
        source_tile_size: int = 256,
        source_stride: int = 256,
        calibration_per_fabric: int = 5,
    ) -> None:
        self.root = Path(data_root)
        self.normal_dir = self.root / "NODefect_images"
        self.defect_dir = self.root / "Defect_images"
        if not self.normal_dir.is_dir() or not self.defect_dir.is_dir():
            raise FileNotFoundError(
                "AITEX 目录应同时包含 NODefect_images 和 Defect_images："
                f"{self.root}"
            )

        normal_paths = sorted(self.normal_dir.rglob("*.png"))
        defect_paths = sorted(self.defect_dir.glob("*.png"))
        if not normal_paths or not defect_paths:
            raise FileNotFoundError("AITEX 正常图或缺陷图为空")

        calibration_paths, normal_test_paths = split_normal_images(
            normal_paths, calibration_per_fabric
        )
        image_records = []
        for path in normal_paths:
            split = "calibration" if path in calibration_paths else "test"
            image_records.append((path, 0, split))
        image_records.extend((path, 1, "test") for path in defect_paths)

        self.transform = image_transform(image_size)
        self.samples: list[dict] = []
        for path, label, split in image_records:
            with Image.open(path) as image:
                width, height = image.size
            left_positions = sliding_positions(width, source_tile_size, source_stride)
            top_positions = sliding_positions(height, source_tile_size, source_stride)
            defect_code, fabric_code = parse_aitex_codes(path)
            tile_index = 0
            for top in top_positions:
                for left in left_positions:
                    self.samples.append({
                        "path": path,
                        "label": label,
                        "split": split,
                        "defect_code": defect_code,
                        "fabric_code": fabric_code,
                        "tile_index": tile_index,
                        "box": (
                            left,
                            top,
                            left + source_tile_size,
                            top + source_tile_size,
                        ),
                    })
                    tile_index += 1

        self.image_count = len(image_records)
        self.calibration_image_count = len(calibration_paths)
        self.normal_test_image_count = len(normal_test_paths)
        self.defect_image_count = len(defect_paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample["path"]) as image:
            tile = image.convert("RGB").crop(sample["box"])
            tensor = self.transform(tile)
        return {
            "image": tensor,
            "path": str(sample["path"]),
            "label": sample["label"],
            "split": sample["split"],
            "defect_code": sample["defect_code"],
            "fabric_code": sample["fabric_code"],
            "tile_index": sample["tile_index"],
        }
