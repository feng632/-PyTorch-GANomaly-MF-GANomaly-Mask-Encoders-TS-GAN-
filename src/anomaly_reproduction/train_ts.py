"""TS-GAN 两阶段训练入口；PyCharm 直接运行。"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.dataset import MVTecFourCropTrainDataset
from anomaly_reproduction.losses import DiscriminatorLoss
from anomaly_reproduction.losses_ts import TSGeneratorLoss
from anomaly_reproduction.models_ts import TSDiscriminator, TSGenerator
from anomaly_reproduction.train import PROJECT_ROOT, save_history, set_seed
from anomaly_reproduction.training import set_requires_grad
from anomaly_reproduction.training_ts import train_one_batch
from anomaly_reproduction.ts_dataset import TSAnomalyTrainDataset, choose_anomaly_images

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ts_gan_grid.yaml"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as file: config = yaml.safe_load(file)
    ds, model, tr, ev = config["dataset"], config["model"], config["training"], config["evaluation"]
    if model["name"] != "ts_gan" or ds["image_size"] != 64 or model["latent_dim"] != 512:
        raise ValueError("TS-GAN 配置必须使用 64x64 输入和 512 维隐向量")
    if ds["anomaly_images_per_type"] != 2:
        raise ValueError("Grid 五类缺陷各选 2 张，合计 10 张异常训练图")
    if min(tr["normal_epochs"], tr["abnormal_epochs"], tr["normal_learning_rate"],
           tr["abnormal_learning_rate"], ds["batch_size"]) <= 0:
        raise ValueError("训练配置必须为正数")
    if [ev["alpha"], ev["beta"], ev["eta"]] != [40.0, 10.0, 0.1]:
        raise ValueError("测试权重应为 alpha=40、beta=10、eta=0.1")
    return config


def save_ts(path, phase, epoch, normal_g, abnormal_g, discriminator,
            optimizer_g, optimizer_d, history, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"phase": phase, "epoch": epoch,
        "normal_generator": normal_g.state_dict(), "abnormal_generator": abnormal_g.state_dict(),
        "discriminator": discriminator.state_dict(), "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(), "history": history, "config": config}, path)


def run_phase(name, generator, epochs, dataset, learning_rate, models, config, device, output_dir):
    normal_g, abnormal_g, discriminator = models
    tr, ds = config["training"], config["dataset"]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate,
        betas=(tr["beta1"], tr["beta2"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,
        betas=(tr["beta1"], tr["beta2"]))
    loss_g = TSGeneratorLoss(tr["reconstruction_weight"], tr["feature_weight"], tr["adversarial_weight"])
    loss_d = DiscriminatorLoss()
    loader = DataLoader(dataset, batch_size=ds["batch_size"], shuffle=True,
        num_workers=ds["num_workers"], pin_memory=True,
        generator=torch.Generator().manual_seed(config["experiment"]["seed"] + (name == "abnormal")))
    history, completed = [], 0

    def save(filename):
        save_ts(output_dir / filename, name, completed, normal_g, abnormal_g, discriminator,
                optimizer_g, optimizer_d, history, config)
    try:
        for epoch in range(1, epochs + 1):
            sums, count = {}, 0
            progress = tqdm(loader, desc=f"TS {name} {epoch}/{epochs}")
            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                metrics = train_one_batch(images, generator, discriminator, optimizer_g,
                                          optimizer_d, loss_g, loss_d)
                count += images.size(0)
                for key, value in metrics.items(): sums[key] = sums.get(key, 0) + value * images.size(0)
                progress.set_postfix(g=metrics["generator_total"], d=metrics["discriminator_total"])
            completed = epoch
            history.append({"epoch": float(epoch), **{k: v / count for k,v in sums.items()}})
            save_history(history, output_dir)
            if epoch % tr["checkpoint_every"] == 0: save("checkpoint_latest.pt")
    except KeyboardInterrupt:
        save("checkpoint_interrupted.pt")
        raise
    save("checkpoint_final.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA 不可用")
    config = load_config(args.config); set_seed(config["experiment"]["seed"])
    ds, tr = config["dataset"], config["training"]
    selected = choose_anomaly_images(ds["root"], ds["category"],
        ds["anomaly_images_per_type"], config["experiment"]["seed"])
    # 把真实划分写入检查点和配置，测试必须读取同一列表并排除这些图片。
    config["dataset"]["anomaly_train_paths"] = selected
    run_dir = PROJECT_ROOT / "runs" / config["experiment"]["name"]
    if run_dir.exists(): raise FileExistsError(f"为避免覆盖，结果目录已存在：{run_dir}")
    normal_dir, abnormal_dir = run_dir / "stage1_normal", run_dir / "stage2_abnormal"
    normal_dir.mkdir(parents=True); abnormal_dir.mkdir()
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    with (run_dir / "anomaly_split.txt").open("w", encoding="utf-8") as file:
        file.write("\n".join(selected) + "\n")

    device = torch.device("cuda")
    normal_g = TSGenerator(config["model"]["image_channels"], config["model"]["latent_dim"]).to(device)
    abnormal_g = TSGenerator(config["model"]["image_channels"], config["model"]["latent_dim"]).to(device)
    discriminator = TSDiscriminator(config["model"]["image_channels"]).to(device)
    models = normal_g, abnormal_g, discriminator

    normal_data = MVTecFourCropTrainDataset(ds["root"], ds["category"], ds["image_size"])
    set_requires_grad(abnormal_g, False)
    print(f"阶段1：{len(normal_data)} 个正常 patch，{tr['normal_epochs']} epochs")
    try:
        run_phase("normal", normal_g, tr["normal_epochs"], normal_data,
                  tr["normal_learning_rate"], models, config, device, normal_dir)
    except KeyboardInterrupt:
        print("正常阶段中止，已保存当前权重；不会开始异常阶段。"); return

    set_requires_grad(normal_g, False); set_requires_grad(abnormal_g, True)
    anomaly_data = TSAnomalyTrainDataset(ds["root"], selected, ds["category"], ds["image_size"])
    print(f"阶段2：10 张异常原图展开为 {len(anomaly_data)} 个 patch，{tr['abnormal_epochs']} epochs")
    try:
        run_phase("abnormal", abnormal_g, tr["abnormal_epochs"], anomaly_data,
                  tr["abnormal_learning_rate"], models, config, device, abnormal_dir)
    except KeyboardInterrupt:
        print("异常阶段中止，已保存当前权重。"); return
    print("TS-GAN 两阶段训练完成。现在运行 evaluate_ts.py。")


if __name__ == "__main__": main()
