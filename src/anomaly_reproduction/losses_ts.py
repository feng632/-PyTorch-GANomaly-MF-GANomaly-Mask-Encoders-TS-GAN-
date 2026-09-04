"""TS-GAN 的重建、对抗与共享判别器特征损失。"""

import torch
from torch import nn


def l2_per_sample(left, right):
    return torch.linalg.vector_norm((left - right).flatten(1), ord=2, dim=1)


class TSGeneratorLoss(nn.Module):
    def __init__(self, reconstruction_weight=40, feature_weight=1, adversarial_weight=1):
        super().__init__()
        self.weights = reconstruction_weight, feature_weight, adversarial_weight
        self.bce = nn.BCELoss()

    def forward(self, images, reconstructed, real_features, fake_features, fake_probabilities):
        reconstruction = l2_per_sample(images, reconstructed).mean()
        feature = l2_per_sample(real_features.detach(), fake_features).mean()
        adversarial = self.bce(fake_probabilities, torch.ones_like(fake_probabilities))
        wr, wf, wa = self.weights
        total = wr * reconstruction + wf * feature + wa * adversarial
        return {"total": total, "reconstruction": reconstruction,
                "feature": feature, "adversarial": adversarial}


def branch_score(images, reconstructed, real_features, fake_features, pixel_weight):
    return pixel_weight * l2_per_sample(images, reconstructed) + l2_per_sample(real_features, fake_features)
