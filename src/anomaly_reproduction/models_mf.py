"""独立的 MF-GANomaly 模型，不修改原版 GANomaly 的行为。"""

import torch
from torch import nn
from typing import NamedTuple

from anomaly_reproduction.models import Decoder, Encoder


class MFEncoder(Encoder):
    """复用 Encoder 的层定义，但同时返回最终编码和第一层特征。

    每次 MFEncoder() 都创建独立参数，不会共享或加载原版训练权重。
    """

    def __init__(self, input_channels=3, latent_dim=128, encoder_variant="baseline"):
        super().__init__(input_channels, latent_dim)
        if encoder_variant not in ("baseline", "paper"):
            raise ValueError("encoder_variant 必须为 baseline 或 paper")
        if encoder_variant == "paper":
            # 论文正文描述每层都有 BN + LeakyReLU；保留 baseline 供消融对照。
            self.conv1 = nn.Sequential(
                self.conv1[0], nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv5 = nn.Sequential(
                self.conv5, nn.BatchNorm2d(latent_dim), nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 保留第一层模块的输出（卷积 + LeakyReLU 之后）。
        # [B, 3, 64, 64] -> [B, 64, 32, 32]
        shallow_features = self.conv1(images)

        # 仍然按原顺序完成后面的编码，没有新增卷积层。
        features = self.conv2(shallow_features)
        features = self.conv3(features)
        features = self.conv4(features)
        latent_code = self.conv5(features)

        # 不使用 detach()：以后浅层特征损失需要通过这里反向传播。
        return latent_code, shallow_features


class MFOutput(NamedTuple):
    reconstructed: torch.Tensor
    latent_original: torch.Tensor
    latent_reconstructed: torch.Tensor
    shallow_original: torch.Tensor
    shallow_reconstructed: torch.Tensor


class MFGenerator(nn.Module):
    """Encoder1 -> Decoder -> Encoder2；额外暴露两边的浅层特征。"""

    def __init__(self, image_channels=3, latent_dim=128, encoder_variant="paper"):
        super().__init__()
        self.encoder1 = MFEncoder(image_channels, latent_dim, encoder_variant)
        self.decoder = Decoder(image_channels, latent_dim)
        # 结构一致，但不是同一个对象，参数不会共享。
        self.encoder2 = MFEncoder(image_channels, latent_dim, encoder_variant)

    def forward(self, images: torch.Tensor) -> MFOutput:
        z, shallow_original = self.encoder1(images)
        reconstructed = self.decoder(z)
        z_hat, shallow_reconstructed = self.encoder2(reconstructed)
        return MFOutput(reconstructed, z, z_hat, shallow_original, shallow_reconstructed)
