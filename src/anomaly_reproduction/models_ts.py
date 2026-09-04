"""TS-GAN：两套独立的单跳跃连接生成器，共享一个判别器。"""

import torch
from torch import nn


def down(in_channels, out_channels, first=False):
    layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
    if not first: layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class TSGenerator(nn.Module):
    def __init__(self, image_channels=3, latent_dim=512):
        super().__init__()
        self.e1 = down(image_channels, 64, first=True)   # 64 -> 32
        self.e2 = down(64, 128)                          # 32 -> 16
        self.e3 = down(128, 256)                         # 16 -> 8
        self.e4 = down(256, 512)                         # 8 -> 4（跳跃来源）
        self.e5 = nn.Conv2d(512, latent_dim, 4, 1, 0, bias=False)  # 4 -> 1
        self.d1 = nn.Sequential(nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
                                nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.d2 = up(1024, 256)  # d1 与 e4 按通道拼接：512+512
        self.d3 = up(256, 128)
        self.d4 = up(128, 64)
        self.d5 = nn.Sequential(nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, images):
        e1 = self.e1(images); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        z = self.e5(e4)
        decoded = self.d1(z)
        reconstructed = self.d5(self.d4(self.d3(self.d2(torch.cat([decoded, e4], dim=1)))))
        return reconstructed


class TSDiscriminator(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.features = nn.Sequential(down(image_channels, 64, first=True), down(64, 128),
            down(128, 256), down(256, 512), nn.Conv2d(512, 512, 4, 1, 0, bias=False))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features).squeeze(1), features
