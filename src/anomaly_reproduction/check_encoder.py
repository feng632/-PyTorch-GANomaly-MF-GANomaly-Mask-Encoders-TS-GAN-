"""Run one real data batch through the GANomaly encoder."""

import torch
from torch.utils.data import DataLoader

from anomaly_reproduction.dataset import MVTecTrainDataset
from anomaly_reproduction.models import Encoder


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，编码器 GPU 检查无法继续。")

    device = torch.device("cuda")
    dataset = MVTecTrainDataset(
        data_root="E:/datasets/mvtec_ad",
        category="grid",
        image_size=64,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # 创建编码器，并把模型参数放到显卡上。
    encoder = Encoder(input_channels=3, latent_dim=128).to(device)

    # 取出一批真实训练图片，再把图片也放到显卡上。
    batch = next(iter(loader))
    images = batch["image"].to(device)

    # 当前只是检查前向传播，不计算梯度，也不更新参数。
    encoder.eval()
    with torch.no_grad():
        latent_code = encoder(images)

    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())

    print("输入形状：", images.shape)
    print("输入设备：", images.device)
    print("编码结果形状：", latent_code.shape)
    print("编码结果设备：", latent_code.device)
    print("编码器参数数量：", f"{parameter_count:,}")

    assert images.shape == (16, 3, 64, 64)
    assert latent_code.shape == (16, 128, 1, 1)
    assert latent_code.device.type == "cuda"
    assert torch.isfinite(latent_code).all()
    print("Encoder 检查：PASS")


if __name__ == "__main__":
    main()
