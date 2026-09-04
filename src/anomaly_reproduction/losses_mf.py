"""论文式 (3.4)-(3.8)；明确区分 L2 范数、MSE 和 L1。"""

import torch
from torch import nn

from anomaly_reproduction.models_mf import MFOutput


def distance_per_sample(left, right, metric="l2"):
    """返回 [B]：先对每张图展平，再算距离，绝不跨 batch 汇总。

    l2 是欧氏范数，不是 MSE。l1 是绝对差之和，用于图示 L1 的对照。
    """
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("输入必须形状相同并包含 batch 维度")
    if metric not in ("l1", "l2"):
        raise ValueError("metric 必须是 l1 或 l2")
    return torch.linalg.vector_norm((left - right).flatten(1), ord=1 if metric == "l1" else 2, dim=1)


def score_components(images, output: MFOutput, reconstruction_metric="l2"):
    return {
        "pixel": distance_per_sample(images, output.reconstructed, reconstruction_metric),
        "shallow": distance_per_sample(output.shallow_original, output.shallow_reconstructed, reconstruction_metric),
        "encoding": distance_per_sample(output.latent_original, output.latent_reconstructed),
    }


def anomaly_scores(images, output, alpha=40.0, reconstruction_metric="l2"):
    parts = score_components(images, output, reconstruction_metric)
    parts["score"] = parts["encoding"] + alpha * (parts["pixel"] + parts["shallow"])
    return parts


class MFGeneratorLoss(nn.Module):
    def __init__(self, reconstruction_weight=40.0, encoding_weight=1.0,
                 adversarial_weight=1.0, reconstruction_metric="l2"):
        super().__init__()
        self.weights = (reconstruction_weight, encoding_weight, adversarial_weight)
        self.reconstruction_metric = reconstruction_metric

    def forward(self, images, output, real_features, reconstructed_features):
        # 两边编码器都参与梯度计算；不能 detach 浅层特征。
        parts = {k: v.mean() for k, v in score_components(images, output, self.reconstruction_metric).items()}
        parts["reconstruction"] = parts["pixel"] + parts["shallow"]
        parts["adversarial"] = distance_per_sample(real_features.detach(), reconstructed_features).mean()
        wr, we, wd = self.weights
        parts["total"] = wr * parts["reconstruction"] + we * parts["encoding"] + wd * parts["adversarial"]
        return parts
