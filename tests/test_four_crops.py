from PIL import Image

from anomaly_reproduction.dataset import four_crop_box


def test_four_crop_boxes_cover_image_without_overlap() -> None:
    image = Image.new("RGB", (100, 80))
    boxes = [four_crop_box(image, index) for index in range(4)]

    assert boxes == [
        (0, 0, 50, 40),
        (50, 0, 100, 40),
        (0, 40, 50, 80),
        (50, 40, 100, 80),
    ]
