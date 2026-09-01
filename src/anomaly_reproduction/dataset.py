from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


FOUR_CROP_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


def four_crop_box(image: Image.Image, crop_index: int) -> tuple[int, int, int, int]:
    """Return one deterministic quadrant box as (left, top, right, bottom)."""
    if crop_index not in range(4):
        raise ValueError(f"crop_index 必须是 0、1、2、3，实际为：{crop_index}")

    width, height = image.size
    middle_x = width // 2
    middle_y = height // 2
    boxes = (
        (0, 0, middle_x, middle_y),
        (middle_x, 0, width, middle_y),
        (0, middle_y, middle_x, height),
        (middle_x, middle_y, width, height),
    )
    return boxes[crop_index]


class MVTecTrainDataset(Dataset):
    """读取 MVTec AD 某个类别的正常训练图片。

    GANomaly 训练时只学习正常图片，
    因此这里固定读取 train/good 文件夹。
    """

    def __init__(
        self,
        data_root: str | Path,
        category: str = "grid",
        image_size: int = 64,
    ) -> None:
        # 将字符串路径转换成 Path 对象，方便后续拼接路径。
        self.data_root = Path(data_root)
        self.category = category
        self.image_size = image_size

        # 最终路径示例：
        # E:/datasets/mvtec_ad/grid/train/good
        self.image_dir = (
            self.data_root
            / self.category
            / "train"
            / "good"
        )

        # 找出文件夹中的全部 PNG 图片，并按文件名排序。
        # 排序可以让每次运行时图片顺序保持一致。
        self.image_paths = sorted(self.image_dir.glob("*.png"))

        # 如果没有找到图片，尽早报错。
        # 否则可能等到训练时才发现数据路径写错了。
        if not self.image_paths:
            raise FileNotFoundError(
                f"没有在以下目录找到 PNG 图片：{self.image_dir}"
            )

        # 定义图片预处理流程。
        self.transform = transforms.Compose(
            [
                # 确保所有图片都变成相同尺寸。
                transforms.Resize((image_size, image_size)),

                # 将 PIL 图片转换成 PyTorch Tensor。
                # 形状会从 [高度, 宽度, 通道]
                # 变成 [通道, 高度, 宽度]。
                #
                # 同时，像素范围会从 [0, 255]
                # 变成 [0, 1]。
                transforms.ToTensor(),

                # GAN 中常用 [-1, 1] 的像素范围。
                #
                # 计算过程是：
                # (像素值 - 0.5) / 0.5
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def __len__(self) -> int:
        """返回训练图片总数。"""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        """根据下标读取一张图片。"""

        # 找到第 index 张图片的路径。
        image_path = self.image_paths[index]

        # 打开图片并统一转换成 RGB 三通道。
        #
        # 即使某张原图是灰度图，经过 convert("RGB") 后
        # 也会得到形状为 [3, 高度, 宽度] 的 Tensor。
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        # 使用字典返回数据。
        # 后续还可以继续加入标签、掩码等信息。
        return {
            "image": image_tensor,
            "path": str(image_path),
            "label": 0,  # 0 表示正常
        }


class MVTecTestDataset(Dataset):
    """读取 MVTec AD 的正常和异常测试图片。

    测试阶段不仅需要图片，还需要图片级标签和像素级掩码。
    掩码相当于人工标出的“标准答案”，只用于评价，不参与训练。
    """

    def __init__(
        self,
        data_root: str | Path,
        category: str = "grid",
        image_size: int = 64,
    ) -> None:
        self.data_root = Path(data_root)
        self.category = category
        self.image_size = image_size
        self.category_root = self.data_root / self.category
        self.test_dir = self.category_root / "test"

        if not self.test_dir.exists():
            raise FileNotFoundError(f"测试集目录不存在：{self.test_dir}")

        # 搜索 test 的各个缺陷子目录以及 good 子目录。
        self.image_paths = sorted(self.test_dir.glob("*/*.png"))
        if not self.image_paths:
            raise FileNotFoundError(
                f"没有在以下目录找到测试图片：{self.test_dir}"
            )

        # 测试图片与训练图片采用相同的尺寸和归一化方式。
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

        # 掩码使用最近邻缩放，避免黑白边界产生没有意义的灰色标签。
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        """返回测试图片总数。"""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        """读取一张测试图片、标签和对应掩码。"""
        image_path = self.image_paths[index]
        defect_type = image_path.parent.name
        is_anomaly = defect_type != "good"
        label = 1 if is_anomaly else 0

        with Image.open(image_path) as image:
            image_tensor = self.image_transform(image.convert("RGB"))

        if is_anomaly:
            mask_path = (
                self.category_root
                / "ground_truth"
                / defect_type
                / f"{image_path.stem}_mask.png"
            )
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"异常图片缺少对应掩码：{mask_path}"
                )

            with Image.open(mask_path) as mask:
                mask_tensor = self.mask_transform(mask.convert("L"))
        else:
            # 正常图没有缺陷文件，返回全黑掩码以保持数据格式统一。
            mask_path = None
            mask_tensor = torch.zeros(
                (1, self.image_size, self.image_size),
                dtype=image_tensor.dtype,
            )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label,
            "defect_type": defect_type,
            "path": str(image_path),
            # 正常图用空字符串代替 None，确保 DataLoader 可以批量整理该字段。
            "mask_path": str(mask_path) if mask_path is not None else "",
        }


class MVTecFourCropTrainDataset(Dataset):
    """Treat the four non-overlapping quadrants of every normal image as samples."""

    def __init__(
        self,
        data_root: str | Path,
        category: str = "grid",
        image_size: int = 64,
    ) -> None:
        self.data_root = Path(data_root)
        self.category = category
        self.image_size = image_size
        self.image_dir = self.data_root / self.category / "train" / "good"
        image_paths = sorted(self.image_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"没有在以下目录找到训练图片：{self.image_dir}")

        # 每张原图固定展开成四个样本：(图片路径, 象限编号)。
        self.samples = [
            (image_path, crop_index)
            for image_path in image_paths
            for crop_index in range(4)
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_path, crop_index = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = image.crop(four_crop_box(image, crop_index))
            image_tensor = self.transform(crop)

        return {
            "image": image_tensor,
            "path": str(image_path),
            "label": 0,
            "crop_index": crop_index,
            "crop_name": FOUR_CROP_NAMES[crop_index],
        }


class MVTecFourCropTestDataset(Dataset):
    """Return four aligned image/mask quadrants for every MVTec test image."""

    def __init__(
        self,
        data_root: str | Path,
        category: str = "grid",
        image_size: int = 64,
    ) -> None:
        self.data_root = Path(data_root)
        self.category = category
        self.image_size = image_size
        self.category_root = self.data_root / self.category
        image_paths = sorted((self.category_root / "test").glob("*/*.png"))
        if not image_paths:
            raise FileNotFoundError("没有找到四块裁剪测试图片。")

        self.samples = [
            (image_path, crop_index)
            for image_path in image_paths
            for crop_index in range(4)
        ]
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_path, crop_index = self.samples[index]
        defect_type = image_path.parent.name
        is_anomaly = defect_type != "good"

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = image.crop(four_crop_box(image, crop_index))
            image_tensor = self.image_transform(crop)

        if is_anomaly:
            mask_path = (
                self.category_root
                / "ground_truth"
                / defect_type
                / f"{image_path.stem}_mask.png"
            )
            with Image.open(mask_path) as mask:
                mask = mask.convert("L")
                mask_crop = mask.crop(four_crop_box(mask, crop_index))
                mask_tensor = self.mask_transform(mask_crop)
        else:
            mask_path = None
            mask_tensor = torch.zeros(
                (1, self.image_size, self.image_size),
                dtype=image_tensor.dtype,
            )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": 1 if is_anomaly else 0,
            "defect_type": defect_type,
            "path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None else "",
            "crop_index": crop_index,
            "crop_name": FOUR_CROP_NAMES[crop_index],
        }
