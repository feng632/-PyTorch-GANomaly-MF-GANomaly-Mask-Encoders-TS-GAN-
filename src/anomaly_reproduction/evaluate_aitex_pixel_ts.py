"""在全部可用 AITEX 测试图上评价 TS-GAN 的像素级异常定位。"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.aitex_dataset import AITEXPixelTileDataset
from anomaly_reproduction.aitex_domain import (
    AITEXManifestPixelDataset,
    load_aitex_manifest,
)
from anomaly_reproduction.evaluate import denormalize
from anomaly_reproduction.localize_aitex_ts import resize_rgb
from anomaly_reproduction.localize_ts import normalize_heatmap, sliding_window_psnr_heatmap
from anomaly_reproduction.models_ts import TSGenerator
from anomaly_reproduction.train import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "runs" / "_cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "runs" / "ts_gan_grid_seed42"
        / "stage2_abnormal" / "checkpoint_final.pt",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--manifest", type=Path,
        help="AITEX 域内训练使用的固定划分清单；不提供时执行 Grid 到 AITEX 外部评价",
    )
    parser.add_argument("--source-tile-size", type=int, default=256)
    parser.add_argument("--source-stride", type=int, default=256)
    parser.add_argument("--calibration-per-fabric", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cases-per-group", type=int, default=3)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def concatenate_tiles(tiles: dict[int, np.ndarray]) -> np.ndarray:
    """AITEX 高度等于一个源图块，因此按编号从左到右拼接模型尺度图块。"""
    indices = sorted(tiles)
    if indices != list(range(len(indices))):
        raise ValueError(f"图块编号不连续：{indices}")
    return np.concatenate([tiles[index] for index in indices], axis=1)


def save_pixel_panel(case: dict, output_path: Path) -> None:
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
        f"{case['group']} | defect={case['defect_code']} | "
        f"fabric={case['fabric_code']} | pixel AUROC={case['pixel_auroc']}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 TS-GAN 权重：{args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    dataset_config = config["dataset"]
    manifest_path = args.manifest
    if manifest_path is None and dataset_config.get("split_manifest"):
        candidate = Path(dataset_config["split_manifest"])
        manifest_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    domain_training = manifest_path is not None
    if domain_training:
        manifest = load_aitex_manifest(manifest_path, args.data_root)
        split = manifest["splits"]
        dataset = AITEXManifestPixelDataset(
            args.data_root,
            normal_paths=split["normal_test"],
            anomaly_paths=split["anomaly_test"],
            image_size=dataset_config["image_size"],
            source_tile_size=args.source_tile_size,
            source_stride=args.source_stride,
        )
    else:
        manifest = None
        dataset = AITEXPixelTileDataset(
            args.data_root,
            image_size=dataset_config["image_size"],
            source_tile_size=args.source_tile_size,
            source_stride=args.source_stride,
            calibration_per_fabric=args.calibration_per_fabric,
        )
    loader = DataLoader(
        dataset, batch_size=config["dataset"]["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True,
    )

    device = torch.device("cuda")
    generator = TSGenerator(
        model_config["image_channels"], model_config["latent_dim"]
    ).to(device).eval()
    generator.load_state_dict(checkpoint["normal_generator"])

    grouped: dict[str, dict] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="TS-GAN AITEX pixel evaluation"):
            images = batch["image"].to(device, non_blocking=True)
            reconstructions = generator(images)
            for index in range(images.size(0)):
                path = batch["path"][index]
                tile_index = int(batch["tile_index"][index])
                original = denormalize(images[index].cpu())
                reconstruction = denormalize(reconstructions[index].cpu())
                raw_heatmap, overall_psnr = sliding_window_psnr_heatmap(
                    original, reconstruction, args.window_size, args.stride
                )
                if path not in grouped:
                    grouped[path] = {
                        "label": int(batch["label"][index]),
                        "defect_code": batch["defect_code"][index],
                        "fabric_code": batch["fabric_code"][index],
                        "raw": {}, "mask": {}, "psnr": {},
                    }
                grouped[path]["raw"][tile_index] = raw_heatmap
                grouped[path]["mask"][tile_index] = batch["mask"][index, 0].numpy()
                grouped[path]["psnr"][tile_index] = overall_psnr

    image_rows = []
    all_masks, all_heatmaps = [], []
    anomaly_masks, anomaly_heatmaps = [], []
    per_defect_masks: dict[str, list[np.ndarray]] = defaultdict(list)
    per_defect_heatmaps: dict[str, list[np.ndarray]] = defaultdict(list)
    normalized_tiles_by_path: dict[str, dict[int, np.ndarray]] = {}

    for path, item in sorted(grouped.items()):
        raw_full = concatenate_tiles(item["raw"])
        mask_full = concatenate_tiles(item["mask"]) > 0.5
        heatmap_full = normalize_heatmap(raw_full)
        tile_width = next(iter(item["raw"].values())).shape[1]
        normalized_tiles_by_path[path] = {
            index: heatmap_full[:, index * tile_width:(index + 1) * tile_width]
            for index in sorted(item["raw"])
        }
        all_masks.append(mask_full.ravel())
        all_heatmaps.append(heatmap_full.ravel())
        if item["label"]:
            anomaly_masks.append(mask_full.ravel())
            anomaly_heatmaps.append(heatmap_full.ravel())
            per_defect_masks[item["defect_code"]].append(mask_full.ravel())
            per_defect_heatmaps[item["defect_code"]].append(heatmap_full.ravel())

        pixel_auroc = ""
        pixel_average_precision = ""
        if mask_full.any() and (~mask_full).any():
            pixel_auroc = float(roc_auc_score(mask_full.ravel(), heatmap_full.ravel()))
            pixel_average_precision = float(
                average_precision_score(mask_full.ravel(), heatmap_full.ravel())
            )
        image_rows.append({
            "path": path,
            "label": item["label"],
            "defect_code": item["defect_code"],
            "fabric_code": item["fabric_code"],
            "tile_count": len(item["raw"]),
            "pixel_auroc": pixel_auroc,
            "pixel_average_precision": pixel_average_precision,
            "heatmap_mean": float(heatmap_full.mean()),
            "heatmap_max": float(heatmap_full.max()),
            "patch_psnr_mean": float(np.mean(list(item["psnr"].values()))),
        })

    all_masks_array = np.concatenate(all_masks)
    all_heatmaps_array = np.concatenate(all_heatmaps)
    anomaly_masks_array = np.concatenate(anomaly_masks)
    anomaly_heatmaps_array = np.concatenate(anomaly_heatmaps)
    per_defect = {}
    for defect_code in sorted(per_defect_masks):
        masks = np.concatenate(per_defect_masks[defect_code])
        heatmaps = np.concatenate(per_defect_heatmaps[defect_code])
        per_defect[defect_code] = {
            "images": len(per_defect_masks[defect_code]),
            "pixel_auroc": float(roc_auc_score(masks, heatmaps)),
            "pixel_average_precision": float(average_precision_score(masks, heatmaps)),
        }

    metrics = {
        "model": "ts_gan",
        "training_dataset": (
            "AITEX-AFID train split" if domain_training else "MVTec AD grid"
        ),
        "test_dataset": (
            "AITEX-AFID held-out test split" if domain_training else "AITEX-AFID"
        ),
        "retraining_on_aitex": domain_training,
        "split_seed": int(manifest["seed"]) if manifest else None,
        "split_unit": "original image before tiling" if manifest else None,
        "normal_test_images": dataset.normal_image_count,
        "anomaly_images_with_masks": dataset.anomaly_image_count,
        "excluded_anomaly_images_without_mask": [
            path.name for path in dataset.missing_mask_paths
        ],
        "evaluated_images": dataset.image_count,
        "evaluation_resolution": "64x64 per 256x256 source tile",
        "heatmap_method": "paper sliding-window PSNR",
        "normalization": "per-image min-max to [0,1] after stitching tiles",
        "all_test_pixel_auroc": float(
            roc_auc_score(all_masks_array, all_heatmaps_array)
        ),
        "all_test_pixel_average_precision": float(
            average_precision_score(all_masks_array, all_heatmaps_array)
        ),
        "anomaly_only_pixel_auroc": float(
            roc_auc_score(anomaly_masks_array, anomaly_heatmaps_array)
        ),
        "anomaly_only_pixel_average_precision": float(
            average_precision_score(anomaly_masks_array, anomaly_heatmaps_array)
        ),
        "per_defect": per_defect,
    }

    default_parent = (
        args.checkpoint.parent / "aitex_domain_evaluation"
        if domain_training else args.checkpoint.parent / "external_aitex"
    )
    output_dir = args.output or default_parent / "pixel_localization"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "aitex_pixel_image_metrics.csv", image_rows)
    with (output_dir / "aitex_pixel_metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, allow_unicode=True, sort_keys=False)

    anomaly_rows = [row for row in image_rows if row["pixel_auroc"] != ""]
    best = sorted(anomaly_rows, key=lambda row: row["pixel_auroc"], reverse=True)[:args.cases_per_group]
    worst = sorted(anomaly_rows, key=lambda row: row["pixel_auroc"])[:args.cases_per_group]
    normal_high = sorted(
        [row for row in image_rows if int(row["label"]) == 0],
        key=lambda row: row["heatmap_mean"], reverse=True,
    )[:args.cases_per_group]
    case_groups = (
        ("best_localization", best),
        ("worst_localization", worst),
        ("normal_high_response", normal_high),
    )

    for group_name, rows in case_groups:
        for rank, row in enumerate(rows, start=1):
            path = row["path"]
            item = grouped[path]
            if int(row["label"]):
                tile_index = max(item["mask"], key=lambda index: item["mask"][index].sum())
            else:
                tile_index = max(
                    normalized_tiles_by_path[path],
                    key=lambda index: normalized_tiles_by_path[path][index].mean(),
                )
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                left = tile_index * args.source_stride
                left = min(left, rgb.width - args.source_tile_size)
                tile = rgb.crop((left, 0, left + args.source_tile_size, args.source_tile_size))
            input_tensor = dataset.image_transform(tile).unsqueeze(0).to(device)
            with torch.no_grad():
                reconstruction = denormalize(generator(input_tensor)[0].cpu())
            original = np.asarray(tile, dtype=np.float32) / 255.0
            reconstruction_large = resize_rgb(
                reconstruction, (args.source_tile_size, args.source_tile_size)
            )
            heatmap_small = normalized_tiles_by_path[path][tile_index]
            heatmap_large = np.asarray(
                Image.fromarray(heatmap_small.astype(np.float32), mode="F").resize(
                    (args.source_tile_size, args.source_tile_size), Image.Resampling.BILINEAR
                ), dtype=np.float32,
            )
            mask_small = item["mask"][tile_index]
            mask_large = np.asarray(
                Image.fromarray(np.uint8(mask_small > 0.5) * 255).resize(
                    (args.source_tile_size, args.source_tile_size), Image.Resampling.NEAREST
                ), dtype=np.float32,
            ) / 255.0
            save_pixel_panel({
                "group": group_name,
                "defect_code": row["defect_code"],
                "fabric_code": row["fabric_code"],
                "pixel_auroc": (
                    f"{float(row['pixel_auroc']):.4f}" if row["pixel_auroc"] != "" else "N/A"
                ),
                "original": original,
                "reconstruction": reconstruction_large,
                "mask": mask_large,
                "heatmap": heatmap_large,
            }, output_dir / "cases" / group_name /
                f"{rank:02d}_{Path(path).stem}_tile{tile_index}.png")

    print(metrics)
    print(f"AITEX 像素级评价结果目录：{output_dir}")


if __name__ == "__main__":
    main()
