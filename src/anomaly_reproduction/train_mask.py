"""Mask Encoders 两阶段训练入口；PyCharm 直接运行。"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.dataset import MVTecFourCropTrainDataset
from anomaly_reproduction.losses import DiscriminatorLoss
from anomaly_reproduction.losses_mask import MaskGeneratorLoss
from anomaly_reproduction.mask_dataset import MaskPretrainDataset
from anomaly_reproduction.models_mask import MaskDiscriminator, MaskGenerator
from anomaly_reproduction.train import PROJECT_ROOT, save_checkpoint, save_history, set_seed
from anomaly_reproduction.training_mask import train_one_batch

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mask_encoders_grid.yaml"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    ds, model, tr = config["dataset"], config["model"], config["training"]
    if model["name"] != "mask_encoders" or ds["preprocessing"] != "four_crops":
        raise ValueError("此入口仅支持四块裁剪的 Mask Encoders")
    if ds["image_size"] != 128 or ds["mask_size"] != 64:
        raise ValueError("论文设置要求 128x128 输入、中心 64x64 遮挡")
    if ds["batch_size"] < 2 or not 0 < ds["pretrain_fraction"] <= 1:
        raise ValueError("batch_size 或预训练比例不正确")
    if min(tr["pretrain_epochs"], tr["finetune_epochs"], tr["learning_rate"]) <= 0:
        raise ValueError("训练轮数和学习率必须为正数")
    return config


def build(config, device):
    model, tr = config["model"], config["training"]
    generator = MaskGenerator(model["image_channels"], model["bottleneck_dim"]).to(device)
    discriminator = MaskDiscriminator(model["image_channels"]).to(device)
    kwargs = dict(lr=tr["learning_rate"], betas=(tr["beta1"], tr["beta2"]))
    return (generator, discriminator,
            torch.optim.Adam(generator.parameters(), **kwargs),
            torch.optim.Adam(discriminator.parameters(), **kwargs),
            MaskGeneratorLoss(tr["reconstruction_weight"], tr["adversarial_weight"]),
            DiscriminatorLoss())


def train_phase(name, epochs, dataset, components, config, device, output_dir):
    ds, tr = config["dataset"], config["training"]
    if len(dataset) % ds["batch_size"] == 1:
        raise ValueError("最后一批只有一个样本，不适合 BatchNorm；请调整 batch_size")
    loader = DataLoader(dataset, batch_size=ds["batch_size"], shuffle=True,
        num_workers=ds["num_workers"], pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config["experiment"]["seed"] + (name == "finetune")))
    generator, discriminator, optimizer_g, optimizer_d, _, _ = components
    history, completed = [], 0

    def save(filename):
        save_checkpoint(output_dir / filename, completed, generator, discriminator,
                        optimizer_g, optimizer_d, history, config)
    try:
        for epoch in range(1, epochs + 1):
            sums, count = {}, 0
            progress = tqdm(loader, desc=f"{name} {epoch}/{epochs}")
            for batch in progress:
                inputs = batch["image"].to(device, non_blocking=True)
                targets = batch.get("target", batch["image"]).to(device, non_blocking=True)
                metrics = train_one_batch(inputs, targets, *components)
                count += inputs.size(0)
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value * inputs.size(0)
                progress.set_postfix(g=metrics["generator_total"], d=metrics["discriminator_total"])
            completed = epoch
            history.append({"epoch": float(epoch), **{k: v / count for k, v in sums.items()}})
            save_history(history, output_dir)
            if epoch % tr["checkpoint_every"] == 0:
                save("checkpoint_latest.pt")
    except KeyboardInterrupt:
        save("checkpoint_interrupted.pt")
        raise
    save("checkpoint_final.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = torch.device("cuda")
    ds, tr = config["dataset"], config["training"]
    run_dir = PROJECT_ROOT / "runs" / config["experiment"]["name"]
    if run_dir.exists():
        raise FileExistsError(f"为避免覆盖，结果目录已存在：{run_dir}")
    pretrain_dir, finetune_dir = run_dir / "stage1_pretrain", run_dir / "stage2_finetune"
    pretrain_dir.mkdir(parents=True); finetune_dir.mkdir()
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    components = build(config, device)
    pretrain_data = MaskPretrainDataset(ds["root"], ds["category"], ds["image_size"],
        ds["mask_size"], ds["pretrain_fraction"], config["experiment"]["seed"])
    print(f"第一阶段：{len(pretrain_data)} 个遮挡裁剪样本，{tr['pretrain_epochs']} epochs")
    try:
        train_phase("stage1", tr["pretrain_epochs"], pretrain_data, components,
                    config, device, pretrain_dir)
    except KeyboardInterrupt:
        print("第一阶段被中止，已保存当前权重；不会继续第二阶段。")
        return

    # 第二阶段载入第一阶段 G/D 权重，但按新任务重新创建 Adam 优化器。
    pretrained = torch.load(pretrain_dir / "checkpoint_final.pt", map_location=device, weights_only=False)
    components = build(config, device)
    components[0].load_state_dict(pretrained["generator"])
    components[1].load_state_dict(pretrained["discriminator"])
    finetune_data = MVTecFourCropTrainDataset(ds["root"], ds["category"], ds["image_size"])
    print(f"第二阶段：{len(finetune_data)} 个完整裁剪样本，{tr['finetune_epochs']} epochs")
    try:
        train_phase("stage2", tr["finetune_epochs"], finetune_data, components,
                    config, device, finetune_dir)
    except KeyboardInterrupt:
        print("第二阶段被中止，已保存当前权重。")
        return
    print("两阶段训练完成。现在运行 evaluate_mask.py。")


if __name__ == "__main__":
    main()
