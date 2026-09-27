"""评价在 AITEX 上训练的 TS-GAN；验证集只含正常图，测试集完全独立。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from anomaly_reproduction.aitex_domain import (
    AITEXManifestTileDataset,
    load_aitex_manifest,
)
from anomaly_reproduction.evaluate_aitex_ts import (
    aggregate_tiles,
    threshold_metrics,
    write_csv,
)
from anomaly_reproduction.losses_ts import branch_score
from anomaly_reproduction.models_ts import TSDiscriminator, TSGenerator
from anomaly_reproduction.train import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "runs" / "ts_gan_aitex_seed42"
        / "stage2_abnormal" / "checkpoint_final.pt",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold-quantile", type=float)
    return parser.parse_args()


def resolve_manifest(config: dict, override: Path | None) -> Path:
    path = override or Path(config["dataset"]["split_manifest"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def per_defect_metrics(
    image_rows: list[dict], mode: str, threshold: float, seen_types: set[str]
) -> dict:
    score_key = f"anomaly_score_{mode}"
    normal = [row for row in image_rows if row["label"] == 0]
    defects: dict[str, list[dict]] = defaultdict(list)
    for row in image_rows:
        if row["label"] == 1:
            defects[row["defect_code"]].append(row)
    result = {}
    for code, rows in sorted(defects.items()):
        labels = [0] * len(normal) + [1] * len(rows)
        scores = [row[score_key] for row in normal + rows]
        result[code] = {
            "images": len(rows),
            "seen_in_anomaly_training": code in seen_types,
            "auroc_vs_all_normal_test": float(roc_auc_score(labels, scores)),
            "recall_at_normal_validation_threshold": float(np.mean([
                row[score_key] > threshold for row in rows
            ])),
            "mean_score": float(np.mean([row[score_key] for row in rows])),
        }
    return result


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到检查点：{args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint.get("phase") != "abnormal":
        raise ValueError("必须使用异常阶段完成后的 TS-GAN 检查点")
    ds = config["dataset"]
    data_root = args.data_root or Path(ds["root"])
    manifest_path = resolve_manifest(config, args.manifest)
    manifest = load_aitex_manifest(manifest_path, data_root)
    split = manifest["splits"]
    threshold_quantile = (
        args.threshold_quantile
        if args.threshold_quantile is not None
        else float(config["evaluation"].get("threshold_quantile", 0.99))
    )
    if not 0 < threshold_quantile <= 1:
        raise ValueError("threshold-quantile 必须位于 (0, 1]")

    common = {
        "image_size": int(ds["image_size"]),
        "source_tile_size": int(ds["source_tile_size"]),
        "source_stride": int(ds["source_stride"]),
    }
    datasets = [
        AITEXManifestTileDataset(
            data_root, split["normal_validation"], 0, "validation", **common
        ),
        AITEXManifestTileDataset(
            data_root, split["normal_test"], 0, "test", **common
        ),
        AITEXManifestTileDataset(
            data_root, split["anomaly_test"], 1, "test", **common
        ),
    ]
    loader = DataLoader(
        ConcatDataset(datasets), batch_size=int(ds["batch_size"]), shuffle=False,
        num_workers=int(ds["num_workers"]), pin_memory=True,
    )

    device = torch.device("cuda")
    model = config["model"]
    normal_generator = TSGenerator(model["image_channels"], model["latent_dim"]).to(device).eval()
    abnormal_generator = TSGenerator(model["image_channels"], model["latent_dim"]).to(device).eval()
    discriminator = TSDiscriminator(model["image_channels"]).to(device).eval()
    normal_generator.load_state_dict(checkpoint["normal_generator"])
    abnormal_generator.load_state_dict(checkpoint["abnormal_generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])

    evaluation = config["evaluation"]
    tile_rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="TS-GAN AITEX domain evaluation"):
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
    validation_rows = [row for row in image_rows if row["split"] == "validation"]
    test_rows = [row for row in image_rows if row["split"] == "test"]
    if any(row["label"] for row in validation_rows):
        raise ValueError("阈值验证集不得含异常图")
    labels = [row["label"] for row in test_rows]
    seen_types = set(manifest["summary"]["seen_defect_types"])
    metrics = {
        "model": "ts_gan",
        "training_dataset": "AITEX-AFID train split",
        "test_dataset": "AITEX-AFID held-out test split",
        "retraining_on_aitex": True,
        "split_seed": int(manifest["seed"]),
        "split_unit": "original image before tiling",
        "threshold_source": "normal validation images only",
        "threshold_quantile": threshold_quantile,
        "validation_normal_images": len(validation_rows),
        "test_normal_images": labels.count(0),
        "test_anomaly_images": labels.count(1),
        "seen_defect_types": sorted(seen_types),
        "unseen_defect_types": manifest["summary"]["unseen_defect_types"],
    }
    for mode in ("max", "mean"):
        score_key = f"anomaly_score_{mode}"
        threshold = float(np.quantile(
            [row[score_key] for row in validation_rows], threshold_quantile
        ))
        metrics[mode] = threshold_metrics(
            labels, [row[score_key] for row in test_rows], threshold
        )
        metrics[mode]["per_defect"] = per_defect_metrics(
            test_rows, mode, threshold, seen_types
        )
        for row in image_rows:
            row[f"prediction_{mode}"] = int(row[score_key] > threshold)

    output_dir = args.output or args.checkpoint.parent / "aitex_domain_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tile_scores.csv", tile_rows)
    write_csv(output_dir / "image_scores.csv", image_rows)
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, allow_unicode=True, sort_keys=False)
    print(yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False))
    print(f"AITEX 域内图片级结果目录：{output_dir}")


if __name__ == "__main__":
    main()
