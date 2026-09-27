"""按照论文方法，用滑动窗口 PSNR 差异生成 TS-GAN 异常热图。

本程序只读取训练完成的模型，不重新训练。论文没有给出像素二值化阈值，
因此这里只输出连续热图，不生成预测掩码。
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.evaluate import denormalize
from anomaly_reproduction.models_ts import TSGenerator
from anomaly_reproduction.train import PROJECT_ROOT
from anomaly_reproduction.train_ts import DEFAULT_CONFIG, load_config
from anomaly_reproduction.ts_dataset import TSPixelTestDataset

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "runs" / "_cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-size", type=int, default=8,
                        help="局部PSNR窗口边长；论文未公开，默认采用8")
    parser.add_argument("--stride", type=int, default=1,
                        help="滑动窗口步长；论文未公开，默认采用1")
    return parser.parse_args()


def stitch_four(patches: dict[int, np.ndarray]) -> np.ndarray:
    """按左上、右上、左下、右下，把四个模型输入块重新拼成整图。"""
    if set(patches) != {0, 1, 2, 3}:
        raise ValueError(f"四块裁剪不完整，实际编号：{sorted(patches)}")
    top = np.concatenate([patches[0], patches[1]], axis=1)
    bottom = np.concatenate([patches[2], patches[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def sliding_positions(length: int, window_size: int, stride: int) -> list[int]:
    """生成窗口起点，并保证最后一个窗口覆盖到图像边缘。"""
    positions = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def psnr_from_mse(mse: float) -> float:
    """图像范围为[0,1]时，由均方误差计算PSNR。"""
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def sliding_window_psnr_heatmap(
    original: np.ndarray,
    reconstruction: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, float]:
    """实现论文描述的热图：整体PSNR减去每个局部窗口的PSNR。

    局部重建越差，局部PSNR越低，差值越大，对应位置的异常响应越强。
    重叠窗口对同一像素的响应取平均，这是论文未公开细节的实现假设。
    """
    height, width, _ = original.shape
    if window_size > min(height, width):
        raise ValueError("滑动窗口不能大于输入裁剪块")

    squared_error = np.square(original - reconstruction).mean(axis=2)
    overall_psnr = psnr_from_mse(float(squared_error.mean()))
    score_sum = np.zeros((height, width), dtype=np.float64)
    score_count = np.zeros((height, width), dtype=np.float64)

    for top in sliding_positions(height, window_size, stride):
        for left in sliding_positions(width, window_size, stride):
            patch_error = squared_error[top:top + window_size, left:left + window_size]
            local_psnr = psnr_from_mse(float(patch_error.mean()))
            response = max(overall_psnr - local_psnr, 0.0)
            score_sum[top:top + window_size, left:left + window_size] += response
            score_count[top:top + window_size, left:left + window_size] += 1.0

    return (score_sum / np.maximum(score_count, 1.0)).astype(np.float32), overall_psnr


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """按照论文描述，将单张热图线性归一化到[0,1]。"""
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum - minimum < 1e-12:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - minimum) / (maximum - minimum)).astype(np.float32)


def save_case(case: dict, output_path: Path) -> None:
    """按照论文展示思路保存原图、正常重建、真值、热图及叠加图。"""
    figure, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    panels = (
        (case["original"], "Original", None),
        (case["normal_reconstruction"], "Normal reconstruction", None),
        (case["mask"], "Ground truth", "gray"),
        (case["heatmap"], "Sliding-window PSNR heatmap", "jet"),
    )
    for axis, (image, title, color_map) in zip(axes[:4], panels):
        axis.imshow(image, cmap=color_map, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    axes[4].imshow(case["original"])
    axes[4].imshow(case["heatmap"], cmap="jet", vmin=0, vmax=1, alpha=0.55)
    axes[4].set_title("Overlay")
    axes[4].axis("off")
    figure.suptitle(f"{case['defect_type']} | {case['name']}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.window_size <= 0 or args.stride <= 0:
        raise ValueError("窗口大小和步长必须为正整数")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法运行TS-GAN像素定位")

    disk_config = load_config(args.config)
    checkpoint_path = args.checkpoint or (
        PROJECT_ROOT / "runs" / disk_config["experiment"]["name"]
        / "stage2_abnormal" / "checkpoint_final.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到TS-GAN最终检查点：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint.get("phase") != "abnormal" or config["model"]["name"] != "ts_gan":
        raise ValueError("该检查点不是完成异常阶段训练的TS-GAN")

    ds = config["dataset"]
    model_config = config["model"]
    excluded = ds.get("anomaly_train_paths")
    if not excluded or len(excluded) != 10:
        raise ValueError("检查点缺少10张异常训练图清单，为避免数据泄漏而停止")

    dataset = TSPixelTestDataset(ds["root"], excluded, ds["category"], ds["image_size"])
    loader = DataLoader(dataset, batch_size=ds["batch_size"], shuffle=False,
                        num_workers=ds["num_workers"], pin_memory=True)

    device = torch.device("cuda")
    normal_generator = TSGenerator(
        model_config["image_channels"], model_config["latent_dim"]
    ).to(device).eval()
    normal_generator.load_state_dict(checkpoint["normal_generator"])

    # 论文在Grid热图实验中只使用正常路完成重建，再计算异常响应热图。
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    with torch.no_grad():
        for batch in tqdm(loader, desc="TS-GAN paper PSNR heatmap"):
            images = batch["image"].to(device, non_blocking=True)
            reconstructed = normal_generator(images)
            for index in range(images.size(0)):
                original_np = denormalize(images[index])
                reconstructed_np = denormalize(reconstructed[index])
                raw_heatmap, overall_psnr = sliding_window_psnr_heatmap(
                    original_np, reconstructed_np, args.window_size, args.stride
                )
                path = batch["path"][index]
                crop_index = int(batch["crop_index"][index])
                grouped[path][crop_index] = {
                    "original": original_np,
                    "normal_reconstruction": reconstructed_np,
                    "mask": batch["mask"][index, 0].numpy(),
                    "raw_heatmap": raw_heatmap,
                    "overall_psnr": overall_psnr,
                    "label": int(batch["label"][index]),
                    "defect_type": batch["defect_type"][index],
                }

    cases = []
    for path, patches in grouped.items():
        first = patches[0]
        raw_heatmap = stitch_four({i: p["raw_heatmap"] for i, p in patches.items()})
        cases.append({
            "path": path,
            "name": Path(path).stem,
            "label": first["label"],
            "defect_type": first["defect_type"],
            "original": stitch_four({i: p["original"] for i, p in patches.items()}),
            "normal_reconstruction": stitch_four(
                {i: p["normal_reconstruction"] for i, p in patches.items()}
            ),
            "mask": stitch_four({i: p["mask"] for i, p in patches.items()}),
            "raw_heatmap": raw_heatmap,
            "heatmap": normalize_heatmap(raw_heatmap),
            "patch_psnr_mean": float(np.mean(
                [p["overall_psnr"] for p in patches.values()]
            )),
        })

    labels = [case["label"] for case in cases]
    if labels.count(0) != 21 or labels.count(1) != 47:
        raise ValueError(f"无泄漏测试划分异常：正常{labels.count(0)}张，异常{labels.count(1)}张")

    all_masks = np.concatenate([(case["mask"] > 0.5).ravel() for case in cases])
    all_heatmaps = np.concatenate([case["heatmap"].ravel() for case in cases])
    pixel_auroc = float(roc_auc_score(all_masks, all_heatmaps))

    output_dir = args.output or checkpoint_path.parent / f"pixel_localization_{checkpoint_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_defect_masks: dict[str, list[np.ndarray]] = defaultdict(list)
    per_defect_heatmaps: dict[str, list[np.ndarray]] = defaultdict(list)

    for case in cases:
        binary_mask = case["mask"] > 0.5
        image_pixel_auroc = ""
        if binary_mask.any() and (~binary_mask).any():
            image_pixel_auroc = float(
                roc_auc_score(binary_mask.ravel(), case["heatmap"].ravel())
            )
        rows.append({
            "path": case["path"],
            "defect_type": case["defect_type"],
            "label": case["label"],
            "pixel_auroc": image_pixel_auroc,
            "patch_psnr_mean": case["patch_psnr_mean"],
            "heatmap_mean": float(case["heatmap"].mean()),
            "heatmap_max": float(case["heatmap"].max()),
        })
        per_defect_masks[case["defect_type"]].append(binary_mask.ravel())
        per_defect_heatmaps[case["defect_type"]].append(case["heatmap"].ravel())

        defect_dir = output_dir / "cases" / case["defect_type"]
        defect_dir.mkdir(parents=True, exist_ok=True)
        np.save(defect_dir / f"{case['name']}_heatmap_raw.npy", case["raw_heatmap"])
        np.save(defect_dir / f"{case['name']}_heatmap_normalized.npy", case["heatmap"])
        save_case(case, defect_dir / f"{case['name']}_panel.png")

    per_defect_pixel_auroc = {}
    for defect_type in sorted(per_defect_masks):
        defect_masks = np.concatenate(per_defect_masks[defect_type])
        defect_heatmaps = np.concatenate(per_defect_heatmaps[defect_type])
        if defect_masks.any() and (~defect_masks).any():
            per_defect_pixel_auroc[defect_type] = float(
                roc_auc_score(defect_masks, defect_heatmaps)
            )

    metrics = {
        "model": "ts_gan",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_images": len(cases),
        "normal_images": labels.count(0),
        "anomaly_images": labels.count(1),
        "excluded_anomaly_train_images": len(excluded),
        "stitched_resolution": 2 * ds["image_size"],
        "heatmap_method": "paper sliding-window PSNR: max(overall_psnr - local_psnr, 0)",
        "normalization": "per-image min-max to [0,1]",
        "normal_generator_only": True,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "implementation_assumption": (
            "overlapping window responses are averaged per pixel; paper does not "
            "specify window size, stride, or overlap aggregation"
        ),
        "binary_threshold": None,
        "pixel_auroc_extra_evaluation_not_reported_by_paper": pixel_auroc,
        "per_defect_pixel_auroc_extra_evaluation": per_defect_pixel_auroc,
    }
    write_csv(output_dir / "pixel_case_metrics.csv", rows)
    with (output_dir / "pixel_metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, allow_unicode=True, sort_keys=False)

    print(metrics)
    print(f"论文PSNR热图结果目录：{output_dir}")


if __name__ == "__main__":
    main()
