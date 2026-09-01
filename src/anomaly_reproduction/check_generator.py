"""Check the complete GANomaly generator with a real image batch."""

import torch
from torch.utils.data import DataLoader

from anomaly_reproduction.dataset import MVTecTrainDataset
from anomaly_reproduction.models import Generator


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，生成器 GPU 检查无法继续。")

    device = torch.device("cuda")
    dataset = MVTecTrainDataset(
        data_root="E:/datasets/mvtec_ad",
        category="grid",
        image_size=64,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    images = next(iter(loader))["image"].to(device)

    generator = Generator(image_channels=3, latent_dim=128).to(device)
    generator.eval()
    with torch.no_grad():
        reconstructed, latent_original, latent_reconstructed = generator(images)

    # 对象不同、首层权重内存地址也不同，证明两个编码器没有共享参数。
    encoder_objects_are_different = generator.encoder1 is not generator.encoder2
    weights_are_independent = (
        generator.encoder1.conv1[0].weight.data_ptr()
        != generator.encoder2.conv1[0].weight.data_ptr()
    )

    print("输入图片形状：", images.shape)
    print("重建图片形状：", reconstructed.shape)
    print("原图特征 z 形状：", latent_original.shape)
    print("重建图特征 z_hat 形状：", latent_reconstructed.shape)
    print("输出设备：", reconstructed.device)
    print("两个编码器是不同对象：", encoder_objects_are_different)
    print("两个编码器参数相互独立：", weights_are_independent)

    assert reconstructed.shape == (16, 3, 64, 64)
    assert latent_original.shape == (16, 128, 1, 1)
    assert latent_reconstructed.shape == (16, 128, 1, 1)
    assert reconstructed.device.type == "cuda"
    assert encoder_objects_are_different
    assert weights_are_independent
    assert torch.isfinite(reconstructed).all()
    assert torch.isfinite(latent_original).all()
    assert torch.isfinite(latent_reconstructed).all()
    print("Generator 检查：PASS")


if __name__ == "__main__":
    main()
