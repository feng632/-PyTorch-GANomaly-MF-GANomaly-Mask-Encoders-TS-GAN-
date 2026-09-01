"""Evaluate a trained GANomaly checkpoint on the MVTec AD grid test set."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / "runs" / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.dataset import MVTecFourCropTestDataset, MVTecTestDataset
from anomaly_reproduction.models import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "runs"
            / "ganomaly_grid_seed42"
            / "checkpoint_final.pt"
        ),
    )
    return parser.parse_args()


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Convert a CHW tensor from [-1, 1] into a displayable HWC array."""
    image = image.detach().cpu().clamp(-1, 1)
    image = (image + 1.0) / 2.0
    return image.permute(1, 2, 0).numpy()


def save_preview(samples: dict[str, dict], output_path: Path) -> None:
    """Save one normal example and one example from every defect type."""
    ordered_types = ["good", "bent", "broken", "glue", "metal_contamination", "thread"]
    selected = [samples[name] for name in ordered_types if name in samples]
    figure, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))

    for row, sample in enumerate(selected):
        original = denormalize(sample["image"])
        reconstructed = denormalize(sample["reconstructed"])
        residual = np.abs(original - reconstructed).mean(axis=2)

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{sample['defect_type']} | original")
        axes[row, 1].imshow(reconstructed)
        axes[row, 1].set_title("reconstruction")
        axes[row, 2].imshow(original)
        axes[row, 2].imshow(residual, cmap="jet", alpha=0.55)
        axes[row, 2].set_title(f"residual | score={sample['score']:.4f}")

        for column in range(3):
            axes[row, column].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"找不到模型检查点：{args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，测试无法继续。")

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset_config = config["dataset"]
    model_config = config["model"]

    preprocessing = dataset_config.get("preprocessing", "whole_image")
    dataset_class = (
        MVTecFourCropTestDataset
        if preprocessing == "four_crops"
        else MVTecTestDataset
    )
    dataset = dataset_class(
        data_root=dataset_config["root"],
        category=dataset_config["category"],
        image_size=int(dataset_config["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(dataset_config["batch_size"]),
        shuffle=False,
        num_workers=int(dataset_config["num_workers"]),
        pin_memory=True,
    )

    generator = Generator(
        image_channels=int(model_config["image_channels"]),
        latent_dim=int(model_config["latent_dim"]),
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    rows: list[dict] = []
    preview_samples: dict[str, dict] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device, non_blocking=True)
            reconstructed, latent_original, latent_reconstructed = generator(images)

            # 原始 GANomaly 使用 z 与 z_hat 的差异作为图片异常分数。
            scores = torch.mean(
                torch.abs(latent_original - latent_reconstructed),
                dim=(1, 2, 3),
            )

            for index in range(images.size(0)):
                defect_type = batch["defect_type"][index]
                row = {
                    "path": batch["path"][index],
                    "defect_type": defect_type,
                    "label": int(batch["label"][index].item()),
                    "anomaly_score": float(scores[index].item()),
                }
                if preprocessing == "four_crops":
                    row["crop_index"] = int(batch["crop_index"][index].item())
                    row["crop_name"] = batch["crop_name"][index]
                rows.append(row)

                if defect_type not in preview_samples:
                    preview_samples[defect_type] = {
                        "image": images[index].cpu(),
                        "reconstructed": reconstructed[index].cpu(),
                        "defect_type": defect_type,
                        "score": row["anomaly_score"],
                    }

    run_dir = args.checkpoint.parent
    raw_scores_name = (
        "test_patch_scores.csv" if preprocessing == "four_crops" else "test_scores.csv"
    )
    with (run_dir / raw_scores_name).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if preprocessing == "four_crops":
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["path"], []).append(row)

        image_rows = []
        for path, patch_rows in grouped.items():
            assert len(patch_rows) == 4
            patch_scores = [row["anomaly_score"] for row in patch_rows]
            image_rows.append(
                {
                    "path": path,
                    "defect_type": patch_rows[0]["defect_type"],
                    "label": patch_rows[0]["label"],
                    "anomaly_score_max": max(patch_scores),
                    "anomaly_score_mean": sum(patch_scores) / 4.0,
                }
            )

        with (run_dir / "test_image_scores.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(image_rows)

        labels = np.asarray([row["label"] for row in image_rows], dtype=np.int64)
        scores = np.asarray(
            [row["anomaly_score_max"] for row in image_rows], dtype=np.float64
        )
        mean_scores = np.asarray(
            [row["anomaly_score_mean"] for row in image_rows], dtype=np.float64
        )
        mean_auroc = float(roc_auc_score(labels, mean_scores))
    else:
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        scores = np.asarray([row["anomaly_score"] for row in rows], dtype=np.float64)
        mean_auroc = None

    image_auroc = float(roc_auc_score(labels, scores))

    metrics = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "preprocessing": preprocessing,
        "test_images": int(labels.size),
        "normal_images": int((labels == 0).sum()),
        "anomaly_images": int((labels == 1).sum()),
        "image_auroc": image_auroc,
        "normal_score_mean": float(scores[labels == 0].mean()),
        "anomaly_score_mean": float(scores[labels == 1].mean()),
    }
    if mean_auroc is not None:
        metrics["image_auroc_mean_aggregation"] = mean_auroc
    with (run_dir / "test_metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, allow_unicode=True, sort_keys=False)

    save_preview(preview_samples, run_dir / "test_reconstruction_preview.png")

    print(f"测试图片：{metrics['test_images']}")
    print(f"正常图片平均分：{metrics['normal_score_mean']:.6f}")
    print(f"异常图片平均分：{metrics['anomaly_score_mean']:.6f}")
    print(f"Image-level AUROC：{image_auroc:.6f}")
    if mean_auroc is not None:
        print(f"Image-level AUROC（四块平均分对照）：{mean_auroc:.6f}")
    print(f"结果目录：{run_dir}")


if __name__ == "__main__":
    main()
