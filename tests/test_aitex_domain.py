from pathlib import Path

from PIL import Image

from anomaly_reproduction.aitex_domain import (
    AITEXManifestPixelDataset,
    AITEXManifestTileDataset,
    create_aitex_manifest,
    load_aitex_manifest,
    save_aitex_manifest,
)


def write_image(path: Path, size=(512, 256), value=128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, value).save(path)


def make_tiny_aitex(root: Path) -> None:
    for fabric in ("00", "01"):
        for index in range(4):
            write_image(
                root / "NODefect_images" / f"fabric_{fabric}"
                / f"{index:04d}_000_{fabric}.png"
            )
    for index in range(3):
        write_image(root / "Defect_images" / f"{index:04d}_002_00.png")
        write_image(root / "Mask_images" / f"{index:04d}_002_00_mask.png", value=255)
    write_image(root / "Defect_images" / "0100_036_01.png")


def test_manifest_is_image_level_disjoint_and_reproducible(tmp_path):
    make_tiny_aitex(tmp_path)
    first = create_aitex_manifest(
        tmp_path, seed=42, normal_train_per_fabric=2,
        normal_validation_per_fabric=1, anomaly_train_per_type=1,
    )
    second = create_aitex_manifest(
        tmp_path, seed=42, normal_train_per_fabric=2,
        normal_validation_per_fabric=1, anomaly_train_per_type=1,
    )
    assert first == second
    assert first["summary"]["counts"] == {
        "normal_train": 4,
        "normal_validation": 2,
        "normal_test": 2,
        "anomaly_train": 1,
        "anomaly_test": 3,
    }
    assert first["summary"]["seen_defect_types"] == ["002"]
    assert first["summary"]["unseen_defect_types"] == ["036"]
    all_sets = [set(first["splits"][key]) for key in first["splits"]]
    assert sum(map(len, all_sets)) == len(set().union(*all_sets))

    path = tmp_path / "split.yaml"
    save_aitex_manifest(first, path)
    assert load_aitex_manifest(path, tmp_path) == first


def test_tile_and_pixel_datasets_follow_manifest(tmp_path):
    make_tiny_aitex(tmp_path)
    manifest = create_aitex_manifest(
        tmp_path, seed=7, normal_train_per_fabric=2,
        normal_validation_per_fabric=1, anomaly_train_per_type=1,
    )
    split = manifest["splits"]
    tiles = AITEXManifestTileDataset(
        tmp_path, split["normal_train"], 0, "normal_train",
        image_size=64, source_tile_size=256, source_stride=256,
    )
    assert len(tiles) == 8
    sample = tiles[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["split"] == "normal_train"

    pixels = AITEXManifestPixelDataset(
        tmp_path, split["normal_test"], split["anomaly_test"],
        image_size=64, source_tile_size=256, source_stride=256,
    )
    assert pixels.normal_image_count == 2
    assert pixels.anomaly_image_count == 2
    assert [path.name for path in pixels.missing_mask_paths] == ["0100_036_01.png"]
    assert len(pixels) == 8
    assert pixels[0]["mask"].shape == (1, 64, 64)
