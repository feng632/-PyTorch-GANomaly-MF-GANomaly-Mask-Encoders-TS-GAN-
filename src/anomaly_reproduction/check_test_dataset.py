"""检查测试图片、标签与掩码能否正确对应。"""

from anomaly_reproduction.dataset import MVTecTestDataset


def main() -> None:
    dataset = MVTecTestDataset(
        data_root="E:/datasets/mvtec_ad",
        category="grid",
        image_size=64,
    )

    normal_count = 0
    anomaly_count = 0
    first_anomaly = None

    for sample in dataset:
        image = sample["image"]
        mask = sample["mask"]

        assert image.shape == (3, 64, 64)
        assert mask.shape == (1, 64, 64)

        if sample["label"] == 0:
            normal_count += 1
            assert mask.max().item() == 0
        else:
            anomaly_count += 1
            assert sample["mask_path"] is not None
            if first_anomaly is None:
                first_anomaly = sample

    assert first_anomaly is not None

    print("测试图片总数：", len(dataset))
    print("正常测试图片：", normal_count)
    print("异常测试图片：", anomaly_count)
    print("示例缺陷类型：", first_anomaly["defect_type"])
    print("示例图片路径：", first_anomaly["path"])
    print("示例掩码路径：", first_anomaly["mask_path"])
    print("图片形状：", first_anomaly["image"].shape)
    print("掩码形状：", first_anomaly["mask"].shape)
    print("掩码最小值：", first_anomaly["mask"].min().item())
    print("掩码最大值：", first_anomaly["mask"].max().item())
    print("测试数据检查：PASS")


if __name__ == "__main__":
    main()
