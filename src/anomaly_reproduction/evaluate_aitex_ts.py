"""使用 Grid 上训练完成的 TS-GAN 对 AITEX-AFID 做跨数据集直接验证。

该程序不重新训练模型。正常校准集只用于确定判定阈值；AUROC、F1、精确率
和召回率只在未参与阈值校准的测试图上计算。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.aitex_dataset import AITEXTileDataset
from anomaly_reproduction.losses_ts import branch_score
from anomaly_reproduction.models_ts import TSDiscriminator, TSGenerator
from anomaly_reproduction.train import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="解压后的 AITEX 根目录")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "runs" / "ts_gan_grid_seed42"
        / "stage2_abnormal" / "checkpoint_final.pt",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-tile-size", type=int, default=256,
                        help="AITEX 原图上的正方形裁剪边长")
    parser.add_argument("--source-stride", type=int, default=256,
                        help="AITEX 原图上的裁剪步长")
    parser.add_argument("--calibration-per-fabric", type=int, default=5,
                        help="每种织物用于无异常阈值校准的正常图片数")
    parser.add_argument("--threshold-quantile", type=float, default=0.99,
                        help="正常校准分数的阈值分位数")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_tiles(rows: list[dict]) -> list[dict]:
    """把同一张长条图的多个图块汇总成最大值和平均值两个整图分数。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["path"]].append(row)

    images = []
    for path, tiles in sorted(grouped.items()):
        metadata = {
            (tile["label"], tile["split"], tile["defect_code"], tile["fabric_code"])
            for tile in tiles
        }
        if len(metadata) != 1:
            raise ValueError(f"同一张图的标签信息不一致：{path}")
        label, split, defect_code, fabric_code = metadata.pop()
        scores = [tile["anomaly_score"] for tile in tiles]
        images.append({
            "path": path,
            "label": label,
            "split": split,
            "defect_code": defect_code,
            "fabric_code": fabric_code,
            "tile_count": len(tiles),
            "anomaly_score_max": max(scores),
            "anomaly_score_mean": float(np.mean(scores)),
        })
    return images


def threshold_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    predictions = [int(score > threshold) for score in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold_quantile <= 1:
        raise ValueError("threshold-quantile 必须位于 (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 TS-GAN 权重：{args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint.get("phase") != "abnormal" or config["model"]["name"] != "ts_gan":
        raise ValueError("该 checkpoint 不是完成异常阶段训练的 TS-GAN")

    ds_config = config["dataset"]
    model_config = config["model"]
    evaluation = config["evaluation"]
    dataset = AITEXTileDataset(
        args.data_root,
        image_size=ds_config["image_size"],
        source_tile_size=args.source_tile_size,
        source_stride=args.source_stride,
        calibration_per_fabric=args.calibration_per_fabric,
    )
    loader = DataLoader(
        dataset,
        batch_size=ds_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda")
    normal_generator = TSGenerator(
        model_config["image_channels"], model_config["latent_dim"]
    ).to(device).eval()
    abnormal_generator = TSGenerator(
        model_config["image_channels"], model_config["latent_dim"]
    ).to(device).eval()
    discriminator = TSDiscriminator(model_config["image_channels"]).to(device).eval()
    normal_generator.load_state_dict(checkpoint["normal_generator"])
    abnormal_generator.load_state_dict(checkpoint["abnormal_generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])

    tile_rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="TS-GAN AITEX external evaluation"):
            images = batch["image"].to(device, non_blocking=True)
            normal_reconstruction = normal_generator(images)
            abnormal_reconstruction = abnormal_generator(images)
            _, real_features = discriminator(images)
            _, normal_features = discriminator(normal_reconstruction)
            _, abnormal_features = discriminator(abnormal_reconstruction)
            normal_scores = branch_score(
                images, normal_reconstruction, real_features, normal_features,
                evaluation["alpha"],
            )
            abnormal_scores = branch_score(
                images, abnormal_reconstruction, real_features, abnormal_features,
                evaluation["beta"],
            )
            scores = normal_scores - evaluation["eta"] * abnormal_scores

            for index in range(images.size(0)):
                tile_rows.append({
                    "path": batch["path"][index],
                    "label": int(batch["label"][index]),
                    "split": batch["split"][index],
                    "defect_code": batch["defect_code"][index],
                    "fabric_code": batch["fabric_code"][index],
                    "tile_index": int(batch["tile_index"][index]),
                    "normal_score": float(normal_scores[index]),
                    "abnormal_score": float(abnormal_scores[index]),
                    "anomaly_score": float(scores[index]),
                })

    image_rows = aggregate_tiles(tile_rows)
    calibration_rows = [row for row in image_rows if row["split"] == "calibration"]
    test_rows = [row for row in image_rows if row["split"] == "test"]
    if not calibration_rows or any(row["label"] != 0 for row in calibration_rows):
        raise ValueError("阈值校准集必须非空且只能包含正常图片")
    labels = [row["label"] for row in test_rows]
    if set(labels) != {0, 1}:
        raise ValueError("正式测试集必须同时包含正常图和异常图")

    metrics = {
        "model": "ts_gan",
        "training_dataset": "MVTec AD grid",
        "external_dataset": "AITEX-AFID",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "retraining_on_aitex": False,
        "source_tile_size": args.source_tile_size,
        "source_stride": args.source_stride,
        "model_input_size": ds_config["image_size"],
        "threshold_source": "normal calibration images only",
        "threshold_quantile": args.threshold_quantile,
        "calibration_images": len(calibration_rows),
        "test_images": len(test_rows),
        "normal_test_images": labels.count(0),
        "anomaly_test_images": labels.count(1),
    }
    for mode in ("max", "mean"):
        key = f"anomaly_score_{mode}"
        calibration_scores = [row[key] for row in calibration_rows]
        threshold = float(np.quantile(calibration_scores, args.threshold_quantile))
        scores = [row[key] for row in test_rows]
        metrics[mode] = threshold_metrics(labels, scores, threshold)
        for row in image_rows:
            row[f"prediction_{mode}"] = int(row[key] > threshold)

    output_dir = args.output or args.checkpoint.parent / "external_aitex"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "aitex_tile_scores.csv", tile_rows)
    write_csv(output_dir / "aitex_image_scores.csv", image_rows)
    with (output_dir / "aitex_metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, allow_unicode=True, sort_keys=False)

    print(metrics)
    print(f"AITEX 外部验证结果目录：{output_dir}")


if __name__ == "__main__":
    main()
