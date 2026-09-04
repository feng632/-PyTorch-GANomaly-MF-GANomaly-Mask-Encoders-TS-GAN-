"""MF-GANomaly 测试入口；仅加载自己生成、可信的本地检查点。"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.dataset import MVTecFourCropTestDataset
from anomaly_reproduction.evaluate import save_preview
from anomaly_reproduction.losses_mf import anomaly_scores
from anomaly_reproduction.models_mf import MFGenerator
from anomaly_reproduction.train import PROJECT_ROOT
from anomaly_reproduction.train_mf import load_config


def normalize_scores(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("异常分数含 NaN/Inf")
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["path"], []).append(row)
    result = []
    for path, patches in grouped.items():
        if len(patches) != 4 or {r["crop_index"] for r in patches} != {0, 1, 2, 3}:
            raise ValueError(f"图片没有恰好四个不同裁剪块：{path}")
        if len({(r["label"], r["defect_type"]) for r in patches}) != 1:
            raise ValueError(f"裁剪标签不一致：{path}")
        scores = [r["anomaly_score"] for r in patches]
        result.append(dict(path=path, label=patches[0]["label"], defect_type=patches[0]["defect_type"],
                           anomaly_score_max=max(scores), anomaly_score_mean=sum(scores) / 4))
    for mode in ("max", "mean"):
        normalized = normalize_scores([r[f"anomaly_score_{mode}"] for r in result])
        for row, score in zip(result, normalized):
            row[f"normalized_score_{mode}"] = float(score)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"还没有检查点，请先完成训练：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["model"].get("name") != "mf_ganomaly":
        raise ValueError("这是原版或其他模型的检查点，请勿混用")
    ds, model = config["dataset"], config["model"]
    dataset = MVTecFourCropTestDataset(ds["root"], ds["category"], ds["image_size"])
    loader = DataLoader(dataset, batch_size=ds["batch_size"], shuffle=False,
                        num_workers=ds["num_workers"], pin_memory=device.type == "cuda")
    generator = MFGenerator(model["image_channels"], model["latent_dim"], model["encoder_variant"]).to(device).eval()
    generator.load_state_dict(checkpoint["generator"])
    rows, preview = [], {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="MF evaluating"):
            images = batch["image"].to(device)
            output = generator(images)
            parts = anomaly_scores(images, output, config["evaluation"]["alpha"],
                                    config["training"]["reconstruction_metric"])
            for i in range(images.size(0)):
                row = dict(path=batch["path"][i], defect_type=batch["defect_type"][i],
                           label=int(batch["label"][i]), crop_index=int(batch["crop_index"][i]),
                           anomaly_score=float(parts["score"][i]),
                           pixel_distance=float(parts["pixel"][i]), shallow_distance=float(parts["shallow"][i]),
                           encoding_distance=float(parts["encoding"][i]))
                rows.append(row)
                if row["defect_type"] not in preview:
                    preview[row["defect_type"]] = dict(image=images[i].cpu(),
                        reconstructed=output.reconstructed[i].cpu(), defect_type=row["defect_type"],
                        score=row["anomaly_score"])
    image_rows = aggregate_rows(rows)
    labels = [row["label"] for row in image_rows]
    if len(set(labels)) != 2:
        raise ValueError("AUROC 需要正常和异常两类测试图片")
    metrics = dict(checkpoint_epoch=checkpoint["epoch"], model="mf_ganomaly",
                   encoder_variant=model["encoder_variant"],
                   reconstruction_metric=config["training"]["reconstruction_metric"],
                   alpha=config["evaluation"]["alpha"], test_images=len(image_rows),
                   normal_images=labels.count(0), anomaly_images=labels.count(1))
    for mode in ("max", "mean"):
        metrics[f"image_auroc_{mode}"] = float(roc_auc_score(labels, [r[f"anomaly_score_{mode}"] for r in image_rows]))
    # 不覆盖原版结果；不同检查点的测试报告也分开保存。
    output_dir = checkpoint_path.parent / f"evaluation_{checkpoint_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "test_patch_scores.csv", rows)
    write_csv(output_dir / "test_image_scores.csv", image_rows)
    with (output_dir / "test_metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, allow_unicode=True, sort_keys=False)
    # 这是像素残差预览，不是论文的滑窗 PSNR 热图；不报告 Pixel AUROC。
    if len(preview) >= 2:
        save_preview(preview, output_dir / "reconstruction_residual_preview.png")
    print(metrics)
    print(f"结果目录：{output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    checkpoint = args.checkpoint
    if checkpoint is None:
        config = load_config()
        checkpoint = PROJECT_ROOT / "runs" / config["experiment"]["name"] / "checkpoint_final.pt"
    evaluate_checkpoint(checkpoint, torch.device(args.device))


if __name__ == "__main__":
    main()
