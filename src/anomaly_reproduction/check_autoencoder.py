"""Run real images through Encoder -> Decoder on the GPU."""

import torch
from torch.utils.data import DataLoader

from anomaly_reproduction.dataset import MVTecTrainDataset
from anomaly_reproduction.models import Decoder, Encoder


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，自编码器 GPU 检查无法继续。")

    device = torch.device("cuda")
    dataset = MVTecTrainDataset(
        data_root="E:/datasets/mvtec_ad",
        category="grid",
        image_size=64,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    encoder = Encoder(input_channels=3, latent_dim=128).to(device)
    decoder = Decoder(output_channels=3, latent_dim=128).to(device)
    images = next(iter(loader))["image"].to(device)

    # 现在只检查网络连接，不训练，因此关闭梯度计算。
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        latent_code = encoder(images)
        reconstructed_images = decoder(latent_code)

    print("输入图片形状：", images.shape)
    print("潜在特征形状：", latent_code.shape)
    print("重建图片形状：", reconstructed_images.shape)
    print("重建图片设备：", reconstructed_images.device)
    print("重建图片最小值：", reconstructed_images.min().item())
    print("重建图片最大值：", reconstructed_images.max().item())

    assert latent_code.shape == (16, 128, 1, 1)
    assert reconstructed_images.shape == images.shape
    assert reconstructed_images.device.type == "cuda"
    assert reconstructed_images.min().item() >= -1.0
    assert reconstructed_images.max().item() <= 1.0
    assert torch.isfinite(reconstructed_images).all()
    print("Autoencoder 结构检查：PASS")


if __name__ == "__main__":
    main()
