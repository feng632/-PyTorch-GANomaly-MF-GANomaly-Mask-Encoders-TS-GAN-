"""Check discriminator outputs for real and reconstructed image batches."""

import torch
from torch.utils.data import DataLoader

from anomaly_reproduction.dataset import MVTecTrainDataset
from anomaly_reproduction.models import Discriminator, Generator


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，判别器 GPU 检查无法继续。")

    device = torch.device("cuda")
    dataset = MVTecTrainDataset(
        data_root="E:/datasets/mvtec_ad",
        category="grid",
        image_size=64,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    real_images = next(iter(loader))["image"].to(device)

    generator = Generator(image_channels=3, latent_dim=128).to(device)
    discriminator = Discriminator(image_channels=3).to(device)
    generator.eval()
    discriminator.eval()

    # 这里只验证结构，生成器和判别器都不更新参数。
    with torch.no_grad():
        reconstructed_images, _, _ = generator(real_images)
        real_probabilities, real_features = discriminator(real_images)
        fake_probabilities, fake_features = discriminator(reconstructed_images)

    print("真实图概率形状：", real_probabilities.shape)
    print("重建图概率形状：", fake_probabilities.shape)
    print("真实图特征形状：", real_features.shape)
    print("重建图特征形状：", fake_features.shape)
    print("输出设备：", real_probabilities.device)
    print("真实图平均概率（尚未训练）：", real_probabilities.mean().item())
    print("重建图平均概率（尚未训练）：", fake_probabilities.mean().item())

    assert real_probabilities.shape == (16,)
    assert fake_probabilities.shape == (16,)
    assert real_features.shape == (16, 512, 4, 4)
    assert fake_features.shape == (16, 512, 4, 4)
    assert real_probabilities.device.type == "cuda"
    assert torch.all((real_probabilities >= 0) & (real_probabilities <= 1))
    assert torch.all((fake_probabilities >= 0) & (fake_probabilities <= 1))
    assert torch.isfinite(real_features).all()
    assert torch.isfinite(fake_features).all()
    print("Discriminator 检查：PASS")


if __name__ == "__main__":
    main()
