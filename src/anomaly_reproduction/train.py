"""Train the GANomaly baseline on MVTec AD grid.

Run from the project root with:
    python -m anomaly_reproduction.train
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

# train.py 位于 <项目根目录>/src/anomaly_reproduction/，向上两级得到项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep Matplotlib's writable cache inside the project.
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.dataset import MVTecFourCropTrainDataset, MVTecTrainDataset
from anomaly_reproduction.losses import DiscriminatorLoss, GeneratorLoss
from anomaly_reproduction.models import Discriminator, Generator
from anomaly_reproduction.training import train_one_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ganomaly_grid.yaml",
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Make repeated runs with the same seed as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path: Path,
    epoch: int,
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    history: list[dict[str, float]],
    config: dict,
) -> None:
    """Save enough state to inspect or resume the experiment later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )


def save_history(history: list[dict[str, float]], run_dir: Path) -> None:
    """Write numeric logs and a simple loss curve after every epoch."""
    if not history:
        return

    csv_path = run_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    epochs = [int(row["epoch"]) for row in history]
    generator_losses = [row["generator_total"] for row in history]
    discriminator_losses = [row["discriminator_total"] for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, generator_losses)
    axes[0].set_title("Generator total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, discriminator_losses)
    axes[1].set_title("Discriminator total loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(run_dir / "loss_curves.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    experiment_config = config["experiment"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = config["training"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；本实验配置要求使用 NVIDIA GPU。")

    seed = int(experiment_config["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    run_dir = PROJECT_ROOT / "runs" / experiment_config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存本次实际使用的配置，避免以后忘记训练参数。
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    preprocessing = dataset_config.get("preprocessing", "whole_image")
    dataset_class = (
        MVTecFourCropTrainDataset
        if preprocessing == "four_crops"
        else MVTecTrainDataset
    )
    dataset = dataset_class(
        data_root=dataset_config["root"],
        category=dataset_config["category"],
        image_size=int(dataset_config["image_size"]),
    )
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(dataset_config["batch_size"]),
        shuffle=True,
        num_workers=int(dataset_config["num_workers"]),
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    generator = Generator(
        image_channels=int(model_config["image_channels"]),
        latent_dim=int(model_config["latent_dim"]),
    ).to(device)
    discriminator = Discriminator(
        image_channels=int(model_config["image_channels"]),
    ).to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(training_config["learning_rate"]),
        betas=(
            float(training_config["beta1"]),
            float(training_config["beta2"]),
        ),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(training_config["learning_rate"]),
        betas=(
            float(training_config["beta1"]),
            float(training_config["beta2"]),
        ),
    )
    generator_loss_function = GeneratorLoss(
        reconstruction_weight=float(training_config["reconstruction_weight"]),
        encoding_weight=float(training_config["encoding_weight"]),
        adversarial_weight=float(training_config["adversarial_weight"]),
    ).to(device)
    discriminator_loss_function = DiscriminatorLoss().to(device)

    epochs = int(training_config["epochs"])
    checkpoint_every = int(training_config["checkpoint_every"])
    history: list[dict[str, float]] = []
    last_completed_epoch = 0

    print(f"设备：{torch.cuda.get_device_name(device)}")
    print(f"训练图片：{len(dataset)}")
    print(f"预处理：{preprocessing}")
    print(f"每轮批次：{len(loader)}")
    print(f"训练轮数：{epochs}")
    print(f"结果目录：{run_dir.resolve()}")

    try:
        for epoch in range(1, epochs + 1):
            sums: dict[str, float] = {}
            sample_count = 0
            progress = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=True)

            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                metrics = train_one_batch(
                    images,
                    generator,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    generator_loss_function,
                    discriminator_loss_function,
                )

                batch_size = images.size(0)
                sample_count += batch_size
                for name, value in metrics.items():
                    sums[name] = sums.get(name, 0.0) + value * batch_size

                progress.set_postfix(
                    g=f"{metrics['generator_total']:.3f}",
                    d=f"{metrics['discriminator_total']:.3f}",
                )

            epoch_metrics = {
                "epoch": float(epoch),
                **{name: value / sample_count for name, value in sums.items()},
            }
            history.append(epoch_metrics)
            last_completed_epoch = epoch
            save_history(history, run_dir)

            print(
                f"Epoch {epoch:03d}: "
                f"G={epoch_metrics['generator_total']:.4f}, "
                f"D={epoch_metrics['discriminator_total']:.4f}"
            )

            if epoch % checkpoint_every == 0:
                # 覆盖同一个 latest 文件，避免150轮产生大量大文件。
                save_checkpoint(
                    run_dir / "checkpoint_latest.pt",
                    epoch,
                    generator,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    history,
                    config,
                )

    except KeyboardInterrupt:
        print("检测到手动停止，正在保存当前进度……")
        save_checkpoint(
            run_dir / "checkpoint_interrupted.pt",
            last_completed_epoch,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            history,
            config,
        )
        save_history(history, run_dir)
        print("已保存中断检查点。")
        return

    save_checkpoint(
        run_dir / "checkpoint_final.pt",
        epochs,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        history,
        config,
    )
    print("训练完成，最终检查点与损失曲线已保存。")


if __name__ == "__main__":
    main()
