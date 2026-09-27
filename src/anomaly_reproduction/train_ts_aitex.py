"""在固定 AITEX 划分上进行 TS-GAN 两阶段域内训练。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.aitex_domain import (
    AITEXManifestTileDataset,
    load_aitex_manifest,
)
from anomaly_reproduction.models_ts import TSDiscriminator, TSGenerator
from anomaly_reproduction.train import PROJECT_ROOT, set_seed
from anomaly_reproduction.train_ts import run_phase
from anomaly_reproduction.training import set_requires_grad


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ts_gan_aitex.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    ds, model, training = config["dataset"], config["model"], config["training"]
    if model["name"] != "ts_gan" or ds["image_size"] != 64:
        raise ValueError("AITEX TS-GAN 必须使用 ts_gan 模型与 64x64 输入")
    positive = (
        ds["source_tile_size"], ds["source_stride"], ds["batch_size"],
        ds["anomaly_tiles_per_image"], training["normal_epochs"],
        training["abnormal_epochs"], training["normal_learning_rate"],
        training["abnormal_learning_rate"],
    )
    if min(positive) <= 0:
        raise ValueError("数据与训练参数必须为正数")
    return config


def select_high_error_anomaly_tiles(
    dataset: AITEXManifestTileDataset,
    generator: TSGenerator,
    batch_size: int,
    num_workers: int,
    tiles_per_image: int,
    device: torch.device,
) -> list[dict]:
    """仅凭正常生成器重建误差，为每张弱标注异常图挑选候选异常图块。"""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    generator.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="选择异常训练图块"):
            images = batch["image"].to(device, non_blocking=True)
            reconstruction = generator(images)
            errors = (images - reconstruction).abs().flatten(1).mean(1)
            for index in range(images.size(0)):
                relative = batch["relative_path"][index]
                grouped[relative].append({
                    "path": relative,
                    "tile_index": int(batch["tile_index"][index]),
                    "box": [int(value) for value in batch["box"][index].tolist()],
                    "normal_reconstruction_mae": float(errors[index]),
                })

    selected = []
    for relative, rows in sorted(grouped.items()):
        chosen = sorted(
            rows,
            key=lambda row: (-row["normal_reconstruction_mae"], row["tile_index"]),
        )[:tiles_per_image]
        if len(chosen) < tiles_per_image:
            raise ValueError(f"{relative} 可用图块不足 {tiles_per_image} 个")
        selected.extend(chosen)
    return selected


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    ds, training = config["dataset"], config["training"]
    data_root = Path(ds["root"])
    manifest_path = Path(ds["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = load_aitex_manifest(manifest_path, data_root)
    if int(manifest["seed"]) != seed:
        raise ValueError("配置随机种子与 AITEX 划分清单不一致")

    config["dataset"]["split_manifest"] = str(manifest_path)
    config["dataset"]["split_counts"] = manifest["summary"]["counts"]
    run_dir = PROJECT_ROOT / "runs" / config["experiment"]["name"]
    if run_dir.exists():
        raise FileExistsError(f"为避免覆盖，结果目录已存在：{run_dir}")
    normal_dir = run_dir / "stage1_normal"
    abnormal_dir = run_dir / "stage2_abnormal"
    normal_dir.mkdir(parents=True)
    abnormal_dir.mkdir()

    split = manifest["splits"]
    common = {
        "image_size": int(ds["image_size"]),
        "source_tile_size": int(ds["source_tile_size"]),
        "source_stride": int(ds["source_stride"]),
    }
    normal_data = AITEXManifestTileDataset(
        data_root, split["normal_train"], label=0, split_name="normal_train", **common
    )
    anomaly_pool = AITEXManifestTileDataset(
        data_root, split["anomaly_train"], label=1, split_name="anomaly_train", **common
    )

    device = torch.device("cuda")
    model_config = config["model"]
    normal_generator = TSGenerator(
        model_config["image_channels"], model_config["latent_dim"]
    ).to(device)
    abnormal_generator = TSGenerator(
        model_config["image_channels"], model_config["latent_dim"]
    ).to(device)
    discriminator = TSDiscriminator(model_config["image_channels"]).to(device)
    models = normal_generator, abnormal_generator, discriminator

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    with (run_dir / "split_manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, allow_unicode=True, sort_keys=False)

    set_requires_grad(abnormal_generator, False)
    print(
        f"阶段1：{len(split['normal_train'])} 张正常原图，"
        f"{len(normal_data)} 个图块，{training['normal_epochs']} epochs"
    )
    try:
        run_phase(
            "normal", normal_generator, training["normal_epochs"], normal_data,
            training["normal_learning_rate"], models, config, device, normal_dir,
        )
    except KeyboardInterrupt:
        print("正常阶段中止，已保存当前权重；不会开始异常阶段。")
        return

    selected_tiles = select_high_error_anomaly_tiles(
        anomaly_pool,
        normal_generator,
        batch_size=int(ds["batch_size"]),
        num_workers=int(ds["num_workers"]),
        tiles_per_image=int(ds["anomaly_tiles_per_image"]),
        device=device,
    )
    with (run_dir / "selected_anomaly_tiles.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(selected_tiles, file, allow_unicode=True, sort_keys=False)
    config["dataset"]["selected_anomaly_tiles"] = selected_tiles
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    anomaly_data = AITEXManifestTileDataset(
        data_root, split["anomaly_train"], label=1, split_name="anomaly_train",
        selected_tiles=selected_tiles, **common,
    )

    set_requires_grad(normal_generator, False)
    set_requires_grad(abnormal_generator, True)
    print(
        f"阶段2：{len(split['anomaly_train'])} 张弱标注异常原图，"
        f"自动选择 {len(anomaly_data)} 个高重建误差图块，"
        f"{training['abnormal_epochs']} epochs"
    )
    try:
        run_phase(
            "abnormal", abnormal_generator, training["abnormal_epochs"], anomaly_data,
            training["abnormal_learning_rate"], models, config, device, abnormal_dir,
        )
    except KeyboardInterrupt:
        print("异常阶段中止，已保存当前权重。")
        return
    print("AITEX TS-GAN 两阶段训练完成。下一步运行 evaluate_ts_aitex_domain。")


if __name__ == "__main__":
    main()
