"""Mask Encoders 的 L2 重建损失与生成器对抗损失。"""

import torch
from torch import nn


def l2_per_sample(left, right):
    return torch.linalg.vector_norm((left - right).flatten(1), ord=2, dim=1)


class MaskGeneratorLoss(nn.Module):
    def __init__(self, reconstruction_weight=40.0, adversarial_weight=1.0):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.adversarial_weight = adversarial_weight
        self.bce = nn.BCELoss()

    def forward(self, target, reconstructed, fake_probabilities):
        reconstruction = l2_per_sample(target, reconstructed).mean()
        # 生成器希望判别器把重建图判断为真实图（1）。
        adversarial = self.bce(fake_probabilities, torch.ones_like(fake_probabilities))
        total = self.reconstruction_weight * reconstruction + self.adversarial_weight * adversarial
        return {"total": total, "reconstruction": reconstruction,
                "adversarial": adversarial}
