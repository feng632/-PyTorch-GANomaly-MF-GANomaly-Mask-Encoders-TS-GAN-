"""TS-GAN 无数据泄漏测试入口；PyCharm 直接运行。"""

import argparse, csv
from pathlib import Path

import torch, yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.evaluate import save_preview
from anomaly_reproduction.evaluate_mf import aggregate_rows
from anomaly_reproduction.losses_ts import branch_score
from anomaly_reproduction.models_ts import TSDiscriminator, TSGenerator
from anomaly_reproduction.train import PROJECT_ROOT
from anomaly_reproduction.train_ts import DEFAULT_CONFIG, load_config
from anomaly_reproduction.ts_dataset import TSTestDataset


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA 不可用")
    config_file = load_config(args.config)
    checkpoint_path = args.checkpoint or (PROJECT_ROOT / "runs" /
        config_file["experiment"]["name"] / "stage2_abnormal" / "checkpoint_final.pt")
    if not checkpoint_path.is_file(): raise FileNotFoundError(f"找不到异常阶段最终模型：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint["phase"] != "abnormal" or config["model"]["name"] != "ts_gan":
        raise ValueError("检查点不是完成异常阶段的 TS-GAN")
    ds, model, ev = config["dataset"], config["model"], config["evaluation"]
    excluded = ds.get("anomaly_train_paths")
    if not excluded or len(excluded) != 10:
        raise ValueError("检查点未保存完整的 10 张异常训练图清单，拒绝测试以避免泄漏")
    dataset = TSTestDataset(ds["root"], excluded, ds["category"], ds["image_size"])
    loader = DataLoader(dataset, batch_size=ds["batch_size"], shuffle=False,
                        num_workers=ds["num_workers"], pin_memory=True)
    device = torch.device("cuda")
    normal_g = TSGenerator(model["image_channels"], model["latent_dim"]).to(device).eval()
    abnormal_g = TSGenerator(model["image_channels"], model["latent_dim"]).to(device).eval()
    discriminator = TSDiscriminator(model["image_channels"]).to(device).eval()
    normal_g.load_state_dict(checkpoint["normal_generator"])
    abnormal_g.load_state_dict(checkpoint["abnormal_generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    rows, preview = [], {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="TS-GAN evaluating"):
            images = batch["image"].to(device, non_blocking=True)
            normal_recon, abnormal_recon = normal_g(images), abnormal_g(images)
            _, real_features = discriminator(images)
            _, normal_features = discriminator(normal_recon)
            _, abnormal_features = discriminator(abnormal_recon)
            sn = branch_score(images, normal_recon, real_features, normal_features, ev["alpha"])
            sa = branch_score(images, abnormal_recon, real_features, abnormal_features, ev["beta"])
            scores = sn - ev["eta"] * sa
            for i in range(images.size(0)):
                row = {"path": batch["path"][i], "defect_type": batch["defect_type"][i],
                    "label": int(batch["label"][i]), "crop_index": int(batch["crop_index"][i]),
                    "normal_score": float(sn[i]), "abnormal_score": float(sa[i]),
                    "anomaly_score": float(scores[i])}
                rows.append(row)
                if row["defect_type"] not in preview:
                    preview[row["defect_type"]] = {"image": images[i].cpu(),
                        "reconstructed": normal_recon[i].cpu(), "defect_type": row["defect_type"],
                        "score": row["anomaly_score"]}
    image_rows = aggregate_rows(rows)
    labels = [row["label"] for row in image_rows]
    # 21 正常 + (57-10) 异常，保证训练异常图确实没有进入评价。
    if labels.count(0) != 21 or labels.count(1) != 47:
        raise ValueError(f"测试划分异常：正常 {labels.count(0)}，异常 {labels.count(1)}")
    metrics = {"checkpoint_epoch": int(checkpoint["epoch"]), "model": "ts_gan",
        "test_images": len(image_rows), "normal_images": labels.count(0),
        "anomaly_images": labels.count(1), "excluded_anomaly_train_images": len(excluded),
        "alpha": ev["alpha"], "beta": ev["beta"], "eta": ev["eta"],
        "image_auroc_max": float(roc_auc_score(labels, [r["anomaly_score_max"] for r in image_rows])),
        "image_auroc_mean": float(roc_auc_score(labels, [r["anomaly_score_mean"] for r in image_rows]))}
    output_dir = checkpoint_path.parent / f"evaluation_{checkpoint_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "test_patch_scores.csv", rows)
    write_csv(output_dir / "test_image_scores.csv", image_rows)
    with (output_dir / "test_metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, allow_unicode=True, sort_keys=False)
    save_preview(preview, output_dir / "normal_branch_residual_preview.png")
    print(metrics); print(f"结果目录：{output_dir}")


if __name__ == "__main__": main()
