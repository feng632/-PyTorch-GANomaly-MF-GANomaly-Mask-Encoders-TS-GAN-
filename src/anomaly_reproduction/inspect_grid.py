"""Inspect MVTec AD grid images before any model training.

This script intentionally contains no neural network code. Its only job is to
confirm that we understand the samples and labels that will enter an experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


DEFECTS = ("bent", "broken", "glue", "metal_contamination", "thread")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("E:/datasets/mvtec_ad"),
        help="Directory that contains the MVTec AD category folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/grid_data_preview.png"),
        help="Where to save the preview figure.",
    )
    return parser.parse_args()


def first_png(folder: Path) -> Path:
    images = sorted(folder.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in: {folder}")
    return images[0]


def mask_for(image_path: Path, category_root: Path) -> Path:
    defect = image_path.parent.name
    return category_root / "ground_truth" / defect / f"{image_path.stem}_mask.png"


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_mask(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def main() -> None:
    args = parse_args()
    category_root = args.data_root / "grid"

    train_image = first_png(category_root / "train" / "good")
    test_good = first_png(category_root / "test" / "good")
    anomaly_images = [first_png(category_root / "test" / defect) for defect in DEFECTS]

    rows = 3
    columns = 5
    figure, axes = plt.subplots(rows, columns, figsize=(15, 9))

    # Row 1 contains the two kinds of normal samples; unused cells stay blank.
    for axis in axes[0]:
        axis.axis("off")
    for axis, image_path, title in zip(
        axes[0, :2], [train_image, test_good], ["train/good", "test/good"], strict=True
    ):
        axis.imshow(load_rgb(image_path))
        axis.set_title(title)
        axis.axis("off")

    # Row 2 contains one example from each anomaly type.
    for axis, image_path, defect in zip(axes[1], anomaly_images, DEFECTS, strict=True):
        axis.imshow(load_rgb(image_path))
        axis.set_title(defect)
        axis.axis("off")

    # Row 3 shows matching masks. White pixels are the human-labelled defect.
    for axis, image_path, defect in zip(axes[2], anomaly_images, DEFECTS, strict=True):
        mask_path = mask_for(image_path, category_root)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path}: {mask_path}")
        axis.imshow(load_mask(mask_path), cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"{defect} mask")
        axis.axis("off")

    figure.suptitle("MVTec AD grid: images and pixel-level ground truth", fontsize=14)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Category root: {category_root}")
    print(f"Preview saved to: {args.output.resolve()}")
    print("Data preview: PASS")


if __name__ == "__main__":
    main()
