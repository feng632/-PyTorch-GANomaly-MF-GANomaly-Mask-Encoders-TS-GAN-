"""检查 DataLoader 是否能把单张样本正确组成批次。"""

import torch
from torch.utils.data import DataLoader

from anomaly_reproduction.dataset import MVTecTestDataset, MVTecTrainDataset


def main() -> None:
    train_dataset = MVTecTrainDataset(
        data_root="E:/datasets/mvtec_ad",
        category="grid",
        image_size=64,
    )
    test_dataset = MVTecTestDataset(
        data_root="E:/datasets/mvtec_ad",
        category="grid",
        image_size=64,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,  # 训练时打乱顺序，避免每轮都按相同顺序学习。
        num_workers=0,  # Windows 初始阶段用主进程读取最稳定。
        pin_memory=torch.cuda.is_available(),
        drop_last=False,  # 最后不足 16 张的批次也保留。
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,  # 测试时保持固定顺序，便于对应文件名。
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    print("训练图片总数：", len(train_dataset))
    print("训练批次数量：", len(train_loader))
    print("训练批次图片形状：", train_batch["image"].shape)
    print("训练批次标签形状：", train_batch["label"].shape)
    print("训练批次路径数量：", len(train_batch["path"]))

    print("测试图片总数：", len(test_dataset))
    print("测试批次数量：", len(test_loader))
    print("测试批次图片形状：", test_batch["image"].shape)
    print("测试批次掩码形状：", test_batch["mask"].shape)
    print("测试批次标签形状：", test_batch["label"].shape)

    assert train_batch["image"].shape == (16, 3, 64, 64)
    assert train_batch["label"].shape == (16,)
    assert torch.all(train_batch["label"] == 0)
    assert test_batch["image"].shape == (16, 3, 64, 64)
    assert test_batch["mask"].shape == (16, 1, 64, 64)
    assert test_batch["label"].shape == (16,)

    # 264 张训练图会形成 16 个满批次和 1 个包含 8 张图的尾批次。
    last_train_batch = list(train_loader)[-1]
    assert last_train_batch["image"].shape == (8, 3, 64, 64)
    print("最后一个训练批次形状：", last_train_batch["image"].shape)
    print("DataLoader 检查：PASS")


if __name__ == "__main__":
    main()
