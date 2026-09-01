from pathlib import Path

from anomaly_reproduction.inspect_grid import mask_for


def test_mask_path_matches_mvtec_naming() -> None:
    root = Path("E:/datasets/mvtec_ad/grid")
    image = root / "test" / "broken" / "001.png"

    assert mask_for(image, root) == (
        root / "ground_truth" / "broken" / "001_mask.png"
    )

