"""论文第 4.4.1 节描述的 128x128 Mask Encoders。"""

import torch
from torch import nn


def down(in_channels, out_channels, first=False):
    layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
    if not first:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class MaskEncoder(nn.Module):
    """[B,3,128,128] -> [B,512,1,1]，六层卷积。"""

    def __init__(self, image_channels=3, bottleneck_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            down(image_channels, 64, first=True), down(64, 64), down(64, 128),
            down(128, 256), down(256, 512),
            nn.Conv2d(512, bottleneck_dim, 4, 1, 0, bias=False),
        )

    def forward(self, images):
        return self.layers(images)


class MaskDecoder(nn.Module):
    """[B,512,1,1] -> [B,3,128,128]，与编码器对称。"""

    def __init__(self, image_channels=3, bottleneck_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.ConvTranspose2d(bottleneck_dim, 512, 4, 1, 0, bias=False),
                          nn.BatchNorm2d(512), nn.ReLU(inplace=True)),
            up(512, 256), up(256, 128), up(128, 64), up(64, 64),
            nn.Sequential(nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False), nn.Tanh()),
        )

    def forward(self, code):
        return self.layers(code)


class MaskGenerator(nn.Module):
    def __init__(self, image_channels=3, bottleneck_dim=512):
        super().__init__()
        self.encoder = MaskEncoder(image_channels, bottleneck_dim)
        self.decoder = MaskDecoder(image_channels, bottleneck_dim)

    def forward(self, images):
        return self.decoder(self.encoder(images))


class MaskDiscriminator(nn.Module):
    """六层卷积产生 100 维特征，再映射为一个真假概率。

    论文只说明 100 维特征后接 sigmoid，未交代 100 到 1 的汇总；这里明确采用线性层。
    """

    def __init__(self, image_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            down(image_channels, 64, first=True), down(64, 64), down(64, 128),
            down(128, 256), down(256, 512),
            nn.Conv2d(512, 100, 4, 1, 0, bias=False), nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(100, 1), nn.Sigmoid())

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features).squeeze(1), features
