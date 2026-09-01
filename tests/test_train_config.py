from pathlib import Path

import yaml


def test_ganomaly_grid_config_has_expected_paper_settings() -> None:
    config_path = Path("configs/ganomaly_grid.yaml")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["dataset"]["image_size"] == 64
    assert config["model"]["latent_dim"] == 128
    assert config["training"]["epochs"] == 150
    assert config["training"]["learning_rate"] == 0.0002
    assert config["training"]["reconstruction_weight"] == 40.0


def test_four_crop_config_changes_only_preprocessing_and_name() -> None:
    with Path("configs/ganomaly_grid.yaml").open("r", encoding="utf-8") as file:
        baseline = yaml.safe_load(file)
    with Path("configs/ganomaly_grid_four_crops.yaml").open(
        "r", encoding="utf-8"
    ) as file:
        four_crops = yaml.safe_load(file)

    assert four_crops["dataset"]["preprocessing"] == "four_crops"
    assert baseline["dataset"]["preprocessing"] == "whole_image"
    assert four_crops["model"] == baseline["model"]
    assert four_crops["training"] == baseline["training"]
