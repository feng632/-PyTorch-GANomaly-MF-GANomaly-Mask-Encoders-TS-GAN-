"""Validate deterministic four-crop preprocessing before training."""

from collections import Counter

from anomaly_reproduction.dataset import (
    MVTecFourCropTestDataset,
    MVTecFourCropTrainDataset,
)


def main() -> None:
    train_dataset = MVTecFourCropTrainDataset("E:/datasets/mvtec_ad", "grid", 64)
    test_dataset = MVTecFourCropTestDataset("E:/datasets/mvtec_ad", "grid", 64)

    train_counts = Counter(sample["path"] for sample in train_dataset)
    test_counts = Counter(sample["path"] for sample in test_dataset)

    assert len(train_dataset) == 1056
    assert len(test_dataset) == 312
    assert len(train_counts) == 264
    assert len(test_counts) == 78
    assert set(train_counts.values()) == {4}
    assert set(test_counts.values()) == {4}

    for sample in test_dataset:
        assert sample["image"].shape == (3, 64, 64)
        assert sample["mask"].shape == (1, 64, 64)
        assert sample["crop_index"] in range(4)
        if sample["label"] == 0:
            assert sample["mask"].max().item() == 0

    print("训练原图：264")
    print("训练 patch：", len(train_dataset))
    print("测试原图：78")
    print("测试 patch：", len(test_dataset))
    print("每张原图对应 patch 数：4")
    print("Four-crop 数据检查：PASS")


if __name__ == "__main__":
    main()
