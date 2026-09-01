"""Loss functions used by the GANomaly baseline."""

from __future__ import annotations

import torch
from torch import nn


class GeneratorLoss(nn.Module):
    """Combine reconstruction, encoding, and adversarial feature losses.

    The thesis experiment uses weights 40:1:1. Reconstruction therefore has
    the strongest direct contribution to the generator objective.
    """

    def __init__(
        self,
        reconstruction_weight: float = 40.0,
        encoding_weight: float = 1.0,
        adversarial_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.encoding_weight = encoding_weight
        self.adversarial_weight = adversarial_weight

        # L1 比较原图与重建图的逐像素绝对差。
        self.l1 = nn.L1Loss()
        # MSE 是平方差，这里用于潜在编码和判别器特征。
        self.mse = nn.MSELoss()

    def forward(
        self,
        original_images: torch.Tensor,
        reconstructed_images: torch.Tensor,
        latent_original: torch.Tensor,
        latent_reconstructed: torch.Tensor,
        real_features: torch.Tensor,
        reconstructed_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        reconstruction = self.l1(reconstructed_images, original_images)
        encoding = self.mse(latent_reconstructed, latent_original)

        # 真实图特征只作为目标；detach 防止生成器更新时反向修改这条分支。
        adversarial = self.mse(reconstructed_features, real_features.detach())

        total = (
            self.reconstruction_weight * reconstruction
            + self.encoding_weight * encoding
            + self.adversarial_weight * adversarial
        )

        return {
            "total": total,
            "reconstruction": reconstruction,
            "encoding": encoding,
            "adversarial": adversarial,
        }


class DiscriminatorLoss(nn.Module):
    """Train the discriminator to output 1 for real and 0 for reconstructed."""

    def __init__(self) -> None:
        super().__init__()
        self.binary_cross_entropy = nn.BCELoss()

    def forward(
        self,
        real_probabilities: torch.Tensor,
        reconstructed_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        real_targets = torch.ones_like(real_probabilities)
        reconstructed_targets = torch.zeros_like(reconstructed_probabilities)

        real = self.binary_cross_entropy(real_probabilities, real_targets)
        reconstructed = self.binary_cross_entropy(
            reconstructed_probabilities,
            reconstructed_targets,
        )
        total = 0.5 * (real + reconstructed)

        return {
            "total": total,
            "real": real,
            "reconstructed": reconstructed,
        }

