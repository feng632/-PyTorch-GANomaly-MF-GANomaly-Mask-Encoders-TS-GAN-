"""Check all GANomaly losses and one generator backward pass."""

import torch
from torch.utils.data import DataLoader

from anomaly_reproduction.dataset import MVTecTrainDataset
from anomaly_reproduction.losses import DiscriminatorLoss, GeneratorLoss
from anomaly_reproduction.models import Discriminator, Generator


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，损失函数 GPU 检查无法继续。")

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

    generator.train()
    discriminator.train()
    reconstructed, latent_original, latent_reconstructed = generator(images)
    real_probabilities, real_features = discriminator(images)
    reconstructed_probabilities, reconstructed_features = discriminator(reconstructed)

    generator_losses = generator_loss_function(
        images,
        reconstructed,
        latent_original,
        latent_reconstructed,
        real_features,
        reconstructed_features,
    )
    discriminator_losses = discriminator_loss_function(
        real_probabilities,
        # 训练判别器时切断生成器方向的梯度。
        discriminator(reconstructed.detach())[0],
    )

    expected_total = (
        40.0 * generator_losses["reconstruction"]
        + generator_losses["encoding"]
        + generator_losses["adversarial"]
    )
    assert torch.allclose(generator_losses["total"], expected_total)

    # 验证生成器总损失可以产生梯度，但这里不执行参数更新。
    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    generator_losses["total"].backward()
    generator_has_gradients = any(
        parameter.grad is not None for parameter in generator.parameters()
    )

    print("重建损失：", generator_losses["reconstruction"].item())
    print("编码损失：", generator_losses["encoding"].item())
    print("对抗特征损失：", generator_losses["adversarial"].item())
    print("生成器加权总损失：", generator_losses["total"].item())
    print("判别器真实图损失：", discriminator_losses["real"].item())
    print("判别器重建图损失：", discriminator_losses["reconstructed"].item())
    print("判别器总损失：", discriminator_losses["total"].item())
    print("生成器成功产生梯度：", generator_has_gradients)

    all_losses = [*generator_losses.values(), *discriminator_losses.values()]
    assert all(torch.isfinite(loss).item() for loss in all_losses)
    assert generator_has_gradients
    print("Loss functions 检查：PASS")


if __name__ == "__main__":
    main()

