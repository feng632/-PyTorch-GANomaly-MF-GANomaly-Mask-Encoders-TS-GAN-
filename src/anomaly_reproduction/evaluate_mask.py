"""Mask Encoders 测试入口；PyCharm 直接运行。"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.dataset import MVTecFourCropTestDataset
from anomaly_reproduction.evaluate import denormalize, save_preview
from anomaly_reproduction.evaluate_mf import aggregate_rows
from anomaly_reproduction.losses_mask import l2_per_sample
from anomaly_reproduction.models_mask import MaskGenerator
from anomaly_reproduction.train import PROJECT_ROOT
from anomaly_reproduction.train_mask import DEFAULT_CONFIG, load_config


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    config = load_config(args.config)
    ds, model = config["dataset"], config["model"]
    checkpoint_path = args.checkpoint or (
        PROJECT_ROOT / "runs" / config["experiment"]["name"] /
        "stage2_finetune" / "checkpoint_final.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到第二阶段最终模型：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["config"]["model"]["name"] != "mask_encoders":
        raise ValueError("检查点不是 Mask Encoders")
    device = torch.device("cuda")
    generator = MaskGenerator(model["image_channels"], model["bottleneck_dim"]).to(device)
    generator.load_state_dict(checkpoint["generator"]); generator.eval()
    dataset = MVTecFourCropTestDataset(ds["root"], ds["category"], ds["image_size"])
    loader = DataLoader(dataset, batch_size=ds["batch_size"], shuffle=False,
        num_workers=ds["num_workers"], pin_memory=True)
    rows, preview = [], {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Mask Encoders evaluating"):
            images = batch["image"].to(device, non_blocking=True)
            reconstructed = generator(images)
            scores = l2_per_sample(images, reconstructed)
            for i in range(images.size(0)):
                original_np = denormalize(images[i])
                reconstructed_np = denormalize(reconstructed[i])
                row = dict(path=batch["path"][i], defect_type=batch["defect_type"][i],
                    label=int(batch["label"][i]), crop_index=int(batch["crop_index"][i]),
                    anomaly_score=float(scores[i]),
                    psnr=float(peak_signal_noise_ratio(original_np, reconstructed_np, data_range=1.0)),
                    ssim=float(structural_similarity(original_np, reconstructed_np,
                                                     channel_axis=2, data_range=1.0)))
                rows.append(row)
                if row["defect_type"] not in preview:
                    preview[row["defect_type"]] = dict(image=images[i].cpu(),
                        reconstructed=reconstructed[i].cpu(), defect_type=row["defect_type"],
                        score=row["anomaly_score"])
    image_rows = aggregate_rows(rows)
    labels = [row["label"] for row in image_rows]
    metrics = dict(checkpoint_epoch=int(checkpoint["epoch"]), model="mask_encoders",
        test_images=len(image_rows), normal_images=labels.count(0), anomaly_images=labels.count(1),
        image_auroc_max=float(roc_auc_score(labels, [r["anomaly_score_max"] for r in image_rows])),
        image_auroc_mean=float(roc_auc_score(labels, [r["anomaly_score_mean"] for r in image_rows])),
        patch_psnr_mean=float(np.mean([r["psnr"] for r in rows])),
        patch_ssim_mean=float(np.mean([r["ssim"] for r in rows])))
    output_dir = checkpoint_path.parent / f"evaluation_{checkpoint_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "test_patch_scores.csv", rows)
    write_csv(output_dir / "test_image_scores.csv", image_rows)
    with (output_dir / "test_metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, allow_unicode=True, sort_keys=False)
    save_preview(preview, output_dir / "reconstruction_residual_preview.png")
    print(metrics); print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()
