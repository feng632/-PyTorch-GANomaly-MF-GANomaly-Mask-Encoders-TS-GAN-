"""直接运行本文件，检查 MFEncoder 的两个输出；不训练、不读取数据集。"""

import torch

from anomaly_reproduction.models_mf import MFEncoder


def main() -> None:
    encoder = MFEncoder().eval()
    # 用两张随机图片检查接口，不代表真实训练或检测效果。
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        latent_code, shallow_features = encoder(images)

    print("输入图片形状：", images.shape)
    print("最终编码形状：", latent_code.shape)
    print("第一层特征形状：", shallow_features.shape)
    assert latent_code.shape == (2, 128, 1, 1)
    assert shallow_features.shape == (2, 64, 32, 32)
    print("MFEncoder 检查通过（未启动训练）。")


if __name__ == "__main__":
    main()
