"""Verify one real GANomaly training step changes both models."""

import torch
from torch.utils.data import DataLoader

from anomaly_reproduction.dataset import MVTecTrainDataset
from anomaly_reproduction.losses import DiscriminatorLoss, GeneratorLoss
from anomaly_reproduction.models import Discriminator, Generator
from anomaly_reproduction.training import train_one_batch


def first_parameter_copy(model: torch.nn.Module) -> torch.Tensor:
    """Copy one parameter so it can be compared after optimizer.step()."""
    return next(model.parameters()).detach().clone()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，单步训练检查无法继续。")

    device = torch.device("cuda")
    dataset = MVTecTrainDataset(
        data_root="E:/datasets/mvtec_ad",
        category="grid",
        image_size=64,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    images = next(iter(loader))["image"].to(device)

    generator = Generator(image_channels=3, latent_dim=128).to(device)
    discriminator = Discriminator(image_channels=3).to(device)
    generator_loss_function = GeneratorLoss(40.0, 1.0, 1.0).to(device)
    discriminator_loss_function = DiscriminatorLoss().to(device)

    # 两个模型参数不同，必须分别创建优化器。
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    generator_before = first_parameter_copy(generator)
    discriminator_before = first_parameter_copy(discriminator)

    metrics = train_one_batch(
        images,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        generator_loss_function,
        discriminator_loss_function,
    )

    generator_after = first_parameter_copy(generator)
    discriminator_after = first_parameter_copy(discriminator)
    generator_changed = not torch.equal(generator_before, generator_after)
    discriminator_changed = not torch.equal(discriminator_before, discriminator_after)

    print("生成器总损失：", metrics["generator_total"])
    print("判别器总损失：", metrics["discriminator_total"])
    print("生成器参数发生变化：", generator_changed)
    print("判别器参数发生变化：", discriminator_changed)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert generator_changed
    assert discriminator_changed
    print("One training step 检查：PASS")


if __name__ == "__main__":
    main()
