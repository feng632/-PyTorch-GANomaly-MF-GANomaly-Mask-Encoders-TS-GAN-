"""为 AITEX 跨数据集实验生成代表性的 TS-GAN PSNR 热力图。

默认选择若干漏检异常图和误报正常图，分别保存独立面板，用于分析模型在
跨数据集零样本迁移时为什么表现不好。该程序只推理，不训练模型。
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score

from anomaly_reproduction.aitex_dataset import sliding_positions
from anomaly_reproduction.evaluate import denormalize
from anomaly_reproduction.localize_ts import (
    normalize_heatmap,
    sliding_window_psnr_heatmap,
)
from anomaly_reproduction.models_ts import TSGenerator
from anomaly_reproduction.train import PROJECT_ROOT
from anomaly_reproduction.ts_dataset import image_transform

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "runs" / "_cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "runs" / "ts_gan_grid_seed42"
        / "stage2_abnormal" / "checkpoint_final.pt",
    )
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--false-negatives", type=int, default=6)
    parser.add_argument("--false-positives", type=int, default=2)
    parser.add_argument("--source-tile-size", type=int, default=256)
    parser.add_argument("--source-stride", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def choose_distinct_defects(rows: list[dict], count: int) -> list[dict]:
    """优先选择不同缺陷代码，再用剩余最低分漏检样本补足数量。"""
    ordered = sorted(rows, key=lambda row: float(row["anomaly_score_mean"]))
    selected, used_codes = [], set()
    for row in ordered:
        if row["defect_code"] not in used_codes:
            selected.append(row)
            used_codes.add(row["defect_code"])
        if len(selected) == count:
            return selected
    for row in ordered:
        if row not in selected:
            selected.append(row)
        if len(selected) == count:
            break
    return selected


def load_union_mask(mask_dir: Path, image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """同一缺陷图存在多个 mask1/mask2 时逐像素取并集。"""
    mask_paths = sorted(mask_dir.glob(f"{image_path.stem}_mask*.png"))
    union = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    for path in mask_paths:
        with Image.open(path) as mask:
            current = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        if current.shape != union.shape:
            raise ValueError(f"掩码尺寸与原图不同：{path}")
        union = np.maximum(union, current)
    return union


def tile_boxes(width: int, height: int, tile_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for top in sliding_positions(height, tile_size, stride):
        for left in sliding_positions(width, tile_size, stride):
            boxes.append((left, top, left + tile_size, top + tile_size))
    return boxes


def resize_rgb(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255))
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def resize_float(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def save_panel(case: dict, output_path: Path) -> None:
    residual = np.abs(case["original"] - case["reconstruction"]).mean(axis=2)
    figure, axes = plt.subplots(1, 6, figsize=(19, 3.5))
    panels = (
        (case["original"], "Original tile", None),
        (case["reconstruction"], "Normal reconstruction", None),
        (residual, "Absolute residual", "magma"),
        (case["mask"], "Ground truth", "gray"),
        (case["heatmap"], "PSNR heatmap", "jet"),
    )
    for axis, (image, title, color_map) in zip(axes[:5], panels):
        axis.imshow(image, cmap=color_map, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    axes[5].imshow(case["original"])
    axes[5].imshow(case["heatmap"], cmap="jet", vmin=0, vmax=1, alpha=0.55)
    axes[5].set_title("Overlay")
    axes[5].axis("off")
    figure.suptitle(
        f"{case['case_type']} | defect={case['defect_code']} | "
        f"fabric={case['fabric_code']} | score={case['image_score']:.2f} | "
        f"threshold={case['threshold']:.2f}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    evaluation_dir = args.evaluation_dir or args.checkpoint.parent / "external_aitex"
    image_csv = evaluation_dir / "aitex_image_scores.csv"
    tile_csv = evaluation_dir / "aitex_tile_scores.csv"
    metrics_path = evaluation_dir / "aitex_metrics.yaml"
    for path in (args.checkpoint, image_csv, tile_csv, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"找不到所需文件：{path}")

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = yaml.safe_load(file)
    threshold = float(metrics["mean"]["threshold"])
    image_rows = read_csv(image_csv)
    tile_rows = read_csv(tile_csv)
    tiles_by_path: dict[str, list[dict]] = defaultdict(list)
    for row in tile_rows:
        tiles_by_path[row["path"]].append(row)

    false_negatives = [
        row for row in image_rows
        if int(row["label"]) == 1 and int(row["prediction_mean"]) == 0
        and row["path"]
    ]
    false_positives = sorted(
        [row for row in image_rows
         if int(row["label"]) == 0 and int(row["prediction_mean"]) == 1],
        key=lambda row: float(row["anomaly_score_mean"]), reverse=True,
    )[:args.false_positives]
    selected = [
        ("false_negative", row)
        for row in choose_distinct_defects(false_negatives, args.false_negatives)
    ] + [("false_positive", row) for row in false_positives]
    if not selected:
        raise ValueError("没有找到可视化所需的漏检或误报样本")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TSGenerator(
        model_config["image_channels"], model_config["latent_dim"]
    ).to(device).eval()
    generator.load_state_dict(checkpoint["normal_generator"])
    transform = image_transform(config["dataset"]["image_size"])
    mask_dir = args.data_root / "Mask_images"
    output_dir = args.output or evaluation_dir / "heatmap_failure_cases"

    summary_rows = []
    for case_index, (case_type, row) in enumerate(selected, start=1):
        image_path = Path(row["path"])
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            full_mask = load_union_mask(mask_dir, image_path, rgb.size)
            boxes = tile_boxes(
                width, height, args.source_tile_size, args.source_stride
            )

            if case_type == "false_negative" and full_mask.any():
                mask_areas = [
                    full_mask[top:bottom, left:right].sum()
                    for left, top, right, bottom in boxes
                ]
                tile_index = int(np.argmax(mask_areas))
            else:
                scored_tiles = tiles_by_path[str(image_path)]
                tile_index = int(max(
                    scored_tiles, key=lambda tile: float(tile["anomaly_score"])
                )["tile_index"])

            left, top, right, bottom = boxes[tile_index]
            original_tile = rgb.crop((left, top, right, bottom))
            input_tensor = transform(original_tile).unsqueeze(0).to(device)
            with torch.no_grad():
                reconstruction_tensor = generator(input_tensor)[0].cpu()
            original_small = denormalize(input_tensor[0].cpu())
            reconstruction_small = denormalize(reconstruction_tensor)
            raw_heatmap, overall_psnr = sliding_window_psnr_heatmap(
                original_small, reconstruction_small, args.window_size, args.stride
            )
            heatmap = normalize_heatmap(raw_heatmap)

            target_size = (right - left, bottom - top)
            original = np.asarray(original_tile, dtype=np.float32) / 255.0
            reconstruction = resize_rgb(reconstruction_small, target_size)
            heatmap_large = resize_float(heatmap, target_size)
            mask = full_mask[top:bottom, left:right]

        binary_mask = mask > 0.5
        pixel_auroc = ""
        if binary_mask.any() and (~binary_mask).any():
            pixel_auroc = float(roc_auc_score(binary_mask.ravel(), heatmap_large.ravel()))
        case = {
            "case_type": case_type,
            "defect_code": row["defect_code"],
            "fabric_code": row["fabric_code"],
            "image_score": float(row["anomaly_score_mean"]),
            "threshold": threshold,
            "original": original,
            "reconstruction": reconstruction,
            "mask": mask,
            "heatmap": heatmap_large,
        }
        filename = (
            f"{case_index:02d}_{case_type}_{image_path.stem}_tile{tile_index}.png"
        )
        save_panel(case, output_dir / filename)
        summary_rows.append({
            "case_type": case_type,
            "path": str(image_path),
            "defect_code": row["defect_code"],
            "fabric_code": row["fabric_code"],
            "tile_index": tile_index,
            "image_score_mean": float(row["anomaly_score_mean"]),
            "threshold_mean": threshold,
            "pixel_auroc_selected_tile": pixel_auroc,
            "overall_psnr_selected_tile": overall_psnr,
            "panel": filename,
        })

    with (output_dir / "heatmap_case_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"已生成 {len(summary_rows)} 个跨数据集失败案例：{output_dir}")


if __name__ == "__main__":
    main()
