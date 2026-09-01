"""Training utilities for the GANomaly baseline."""

from __future__ import annotations

import torch
from torch import nn

from anomaly_reproduction.losses import DiscriminatorLoss, GeneratorLoss
from anomaly_reproduction.models import Discriminator, Generator


def set_requires_grad(model: nn.Module, enabled: bool) -> None:
    """Enable or disable gradient storage for every parameter in a model."""
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def train_one_batch(
    images: torch.Tensor,
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    generator_loss_function: GeneratorLoss,
    discriminator_loss_function: DiscriminatorLoss,
) -> dict[str, float]:
    """Update the discriminator once and then update the generator once."""

    # ---- 1. 更新判别器 ----
    generator.train()
    discriminator.train()
    set_requires_grad(discriminator, True)
    optimizer_d.zero_grad(set_to_none=True)

    # 判别器更新时不需要生成器梯度，因此使用 no_grad 和 detach。
    with torch.no_grad():
        reconstructed_for_d, _, _ = generator(images)

    real_probabilities, _ = discriminator(images)
    reconstructed_probabilities, _ = discriminator(reconstructed_for_d.detach())
    discriminator_losses = discriminator_loss_function(
        real_probabilities,
        reconstructed_probabilities,
    )
    discriminator_losses["total"].backward()
    optimizer_d.step()

    # ---- 2. 更新生成器 ----
    # 冻结判别器参数，但仍允许梯度穿过判别器传回重建图和生成器。
    set_requires_grad(discriminator, False)
    discriminator.eval()
    optimizer_g.zero_grad(set_to_none=True)

    reconstructed, latent_original, latent_reconstructed = generator(images)
    with torch.no_grad():
        _, real_features = discriminator(images)
    _, reconstructed_features = discriminator(reconstructed)

    generator_losses = generator_loss_function(
        images,
        reconstructed,
        latent_original,
        latent_reconstructed,
        real_features,
        reconstructed_features,
    )
    generator_losses["total"].backward()
    optimizer_g.step()

    # 为下一个 batch 恢复判别器的可训练状态。
    set_requires_grad(discriminator, True)

    return {
        "generator_total": generator_losses["total"].item(),
        "generator_reconstruction": generator_losses["reconstruction"].item(),
        "generator_encoding": generator_losses["encoding"].item(),
        "generator_adversarial": generator_losses["adversarial"].item(),
        "discriminator_total": discriminator_losses["total"].item(),
        "discriminator_real": discriminator_losses["real"].item(),
        "discriminator_reconstructed": discriminator_losses["reconstructed"].item(),
    }

