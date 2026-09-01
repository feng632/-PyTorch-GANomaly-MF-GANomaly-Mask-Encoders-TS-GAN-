from anomaly_reproduction.dataset import MVTecTrainDataset


def main() -> None:
    # 创建训练数据集。
    dataset = MVTecTrainDataset(
        data_root="E:/datasets/mvtec_ad",
        category="grid",
        image_size=64,
    )

    print("训练图片数量：", len(dataset))

    # 读取第一张训练图片。
    sample = dataset[0]

    image = sample["image"]

    print("图片路径：", sample["path"])
    print("图片标签：", sample["label"])
    print("Tensor 形状：", image.shape)
    print("最小像素值：", image.min().item())
    print("最大像素值：", image.max().item())


if __name__ == "__main__":
    main()