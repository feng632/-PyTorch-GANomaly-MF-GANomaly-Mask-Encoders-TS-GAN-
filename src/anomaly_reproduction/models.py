"""GANomaly network components.

This module is built one component at a time. At the current stage it only
contains the first encoder used to turn an image into a compact latent code.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    """Encode a 64x64 RGB image into a 128-dimensional latent code.

    Shape progression for a batch of 16 images:
    [16, 3, 64, 64] -> [16, 64, 32, 32] -> [16, 128, 16, 16]
    -> [16, 256, 8, 8] -> [16, 512, 4, 4] -> [16, 128, 1, 1]
    """

    def __init__(self, input_channels: int = 3, latent_dim: int = 128) -> None:
        super().__init__()

        # 第一层直接处理 RGB 像素。按照 DCGAN 的常见设计，这一层不使用 BN。
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = self._down_block(64, 128)
        self.conv3 = self._down_block(128, 256)
        self.conv4 = self._down_block(256, 512)

        # 输入已经是 4x4，使用 4x4 卷积后空间尺寸恰好变成 1x1。
        # 最后一层不加激活函数，让潜在特征可以自由取正值或负值。
        self.conv5 = nn.Conv2d(
            512,
            latent_dim,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        )

    @staticmethod
    def _down_block(input_channels: int, output_channels: int) -> nn.Sequential:
        """Create one block that halves width and height."""
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run a batch of images through all five convolutional layers."""
        features = self.conv1(images)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        latent_code = self.conv5(features)
        return latent_code


class Decoder(nn.Module):
    """Decode a 128-dimensional latent code into a 64x64 RGB image.

    Shape progression for a batch of 16 latent codes:
    [16, 128, 1, 1] -> [16, 512, 4, 4] -> [16, 256, 8, 8]
    -> [16, 128, 16, 16] -> [16, 64, 32, 32] -> [16, 3, 64, 64]
    """

    def __init__(self, output_channels: int = 3, latent_dim: int = 128) -> None:
        super().__init__()

        # 1x1 的潜在特征先扩展为 4x4，并增加到 512 个特征通道。
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim,
                512,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.deconv2 = self._up_block(512, 256)
        self.deconv3 = self._up_block(256, 128)
        self.deconv4 = self._up_block(128, 64)

        # 最后一层把 64 张特征图还原成 RGB 三通道图片。
        # Tanh 将输出限制到 [-1, 1]，与输入图片的归一化范围一致。
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                64,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    @staticmethod
    def _up_block(input_channels: int, output_channels: int) -> nn.Sequential:
        """Create one transposed-convolution block that doubles image size."""
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, latent_code: torch.Tensor) -> torch.Tensor:
        """Turn a batch of latent codes back into RGB images."""
        features = self.deconv1(latent_code)
        features = self.deconv2(features)
        features = self.deconv3(features)
        features = self.deconv4(features)
        reconstructed_images = self.deconv5(features)
        return reconstructed_images


class Generator(nn.Module):
    """Complete GANomaly generator: Encoder 1 -> Decoder -> Encoder 2.

    The two encoders share the same architecture but own independent weights.
    The forward pass returns everything needed by later loss functions:
    reconstructed image, original-image code, and reconstructed-image code.
    """

    def __init__(self, image_channels: int = 3, latent_dim: int = 128) -> None:
        super().__init__()

        # 两次调用 Encoder() 会创建两个结构相同、参数独立的编码器。
        self.encoder1 = Encoder(
            input_channels=image_channels,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            output_channels=image_channels,
            latent_dim=latent_dim,
        )
        self.encoder2 = Encoder(
            input_channels=image_channels,
            latent_dim=latent_dim,
        )

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct images and encode both original and reconstruction."""
        latent_original = self.encoder1(images)
        reconstructed_images = self.decoder(latent_original)
        latent_reconstructed = self.encoder2(reconstructed_images)

        return reconstructed_images, latent_original, latent_reconstructed


class Discriminator(nn.Module):
    """Judge whether an image is real and expose its internal features.

    The feature extractor maps [B, 3, 64, 64] to [B, 512, 4, 4].
    The classifier then maps those features to one probability per image.
    """

    def __init__(self, image_channels: int = 3) -> None:
        super().__init__()

        # 判别器的前半部分负责提取用于判断真实性的视觉特征。
        self.features = nn.Sequential(
            nn.Conv2d(
                image_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            self._down_block(64, 128),
            self._down_block(128, 256),
            self._down_block(256, 512),
        )

        # 4x4 特征经过最后一次卷积后变成每张图一个数，
        # Sigmoid 再把这个数压缩到 [0, 1]。
        self.classifier = nn.Sequential(
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    @staticmethod
    def _down_block(input_channels: int, output_channels: int) -> nn.Sequential:
        """Create one discriminator block that halves image size."""
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return real/fake probabilities and pre-classification features."""
        feature_maps = self.features(images)
        probabilities = self.classifier(feature_maps)

        # 分类层原始形状是 [B, 1, 1, 1]，flatten(1) 后为 [B, 1]，
        # squeeze(1) 最终得到更方便计算损失的 [B]。
        probabilities = probabilities.flatten(start_dim=1).squeeze(dim=1)
        return probabilities, feature_maps
