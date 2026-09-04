"""PyCharm 直接运行开始 MF-GANomaly；--smoke-test 仅验证一个小批次。"""

import argparse
from pathlib import Path
import tempfile

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly_reproduction.dataset import MVTecFourCropTrainDataset
from anomaly_reproduction.losses import DiscriminatorLoss
from anomaly_reproduction.losses_mf import MFGeneratorLoss, anomaly_scores
from anomaly_reproduction.models import Discriminator
from anomaly_reproduction.models_mf import MFGenerator
from anomaly_reproduction.train import PROJECT_ROOT, save_checkpoint, save_history, set_seed
from anomaly_reproduction.training_mf import train_one_batch

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mf_ganomaly_grid.yaml"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    ds, model, training = config["dataset"], config["model"], config["training"]
    name = config["experiment"]["name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("实验名称只能使用英文字母、数字、下划线和横线")
    if model["name"] != "mf_ganomaly" or ds["preprocessing"] != "four_crops":
        raise ValueError("此入口仅支持 MF-GANomaly 的四块裁剪实验")
    if ds["image_size"] != 64 or model["image_channels"] != 3:
        raise ValueError("当前模型要求 64x64 RGB 输入")
    if ds["batch_size"] < 2 or ds["num_workers"] < 0:
        raise ValueError("batch_size 至少为 2（最后一层 BN 的要求）")
    if model["encoder_variant"] not in ("paper", "baseline") or training["reconstruction_metric"] not in ("l1", "l2"):
        raise ValueError("不支持的网络或距离配置")
    for key in ("epochs", "checkpoint_every", "learning_rate"):
        if training[key] <= 0:
            raise ValueError(f"{key} 必须为正数")
    if config["evaluation"]["alpha"] < 0:
        raise ValueError("alpha 不得为负数")
    if config["evaluation"]["aggregations"] != ["max", "mean"]:
        raise ValueError("本实验同时报告 max 和 mean，不挑选测试集最佳结果")
    return config


def build_components(config, device):
    model, tr = config["model"], config["training"]
    generator = MFGenerator(model["image_channels"], model["latent_dim"], model["encoder_variant"]).to(device)
    discriminator = Discriminator(model["image_channels"]).to(device)
    kwargs = dict(lr=tr["learning_rate"], betas=(tr["beta1"], tr["beta2"]))
    optimizer_g = torch.optim.Adam(generator.parameters(), **kwargs)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), **kwargs)
    loss_g = MFGeneratorLoss(tr["reconstruction_weight"], tr["encoding_weight"],
                             tr["adversarial_weight"], tr["reconstruction_metric"])
    return generator, discriminator, optimizer_g, optimizer_d, loss_g, DiscriminatorLoss()


def make_dataset(config):
    ds = config["dataset"]
    return MVTecFourCropTrainDataset(ds["root"], ds["category"], ds["image_size"])


def reserve_run_dir(config):
    path = PROJECT_ROOT / "runs" / config["experiment"]["name"]
    # 重复点击不会覆盖此前的模型；新实验请改配置里的 name。
    if path.exists():
        raise FileExistsError(f"结果目录已存在，为避免覆盖已停止：{path}。新实验请修改 experiment.name。")
    path.mkdir(parents=True)
    return path


def smoke_test(config, device):
    """只更新一个配置大小的真实批次，检查保存/加载；不用于正式训练。"""
    dataset = make_dataset(config)
    batch = next(iter(DataLoader(dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)))
    images = batch["image"].to(device)
    components = build_components(config, device)
    generator, discriminator, opt_g, opt_d, _, _ = components
    metrics = train_one_batch(images, *components)
    generator.eval()
    with torch.no_grad():
        expected = generator(images).reconstructed
    # 临时目录自动清理；不写正式 runs/<experiment.name>。
    with tempfile.TemporaryDirectory(prefix="mf_check_") as folder:
        path = Path(folder) / "checkpoint.pt"
        save_checkpoint(path, 0, generator, discriminator, opt_g, opt_d, [], config)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        restored = MFGenerator(config["model"]["image_channels"], config["model"]["latent_dim"],
                               config["model"]["encoder_variant"]).to(device).eval()
        restored.load_state_dict(checkpoint["generator"])
        with torch.no_grad():
            output = restored(images)
            torch.testing.assert_close(output.reconstructed, expected)
            scores = anomaly_scores(images, output, config["evaluation"]["alpha"],
                                     config["training"]["reconstruction_metric"])["score"]
            assert scores.shape == (images.size(0),) and torch.isfinite(scores).all()
    print(f"SMOKE PASS | device={device} | train crops={len(dataset)} | metrics={metrics}")
    print(f"仅完成 1 个 {images.size(0)} 样本批次的检查；临时权重已清理，正式训练未开始。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请检查环境。小批量检查可加 --device cpu。")
    device = torch.device(args.device)
    set_seed(config["experiment"]["seed"])
    if args.smoke_test:
        smoke_test(config, device)
        return

    ds, tr = config["dataset"], config["training"]
    dataset = make_dataset(config)
    # 1x1 特征的 BN 在训练模式不能处理单样本；不静默丢掉数据。
    if len(dataset) % ds["batch_size"] == 1:
        raise ValueError("最后一批只有 1 个样本，请调整 batch_size；不会静默丢弃图片。")
    loader = DataLoader(dataset, batch_size=ds["batch_size"], shuffle=True,
                        num_workers=ds["num_workers"], pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(config["experiment"]["seed"]))
    components = build_components(config, device)
    generator, discriminator, opt_g, opt_d, _, _ = components
    run_dir = reserve_run_dir(config)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    history = []
    completed_epoch = 0
    print(f"MF-GANomaly | {device} | {len(dataset)} 裁剪样本 | {tr['epochs']} epochs")
    print(f"结果目录：{run_dir}")

    def checkpoint(filename):
        save_checkpoint(run_dir / filename, completed_epoch, generator, discriminator,
                        opt_g, opt_d, history, config)

    try:
        for epoch in range(1, tr["epochs"] + 1):
            sums, count = {}, 0
            progress = tqdm(loader, desc=f"MF Epoch {epoch}/{tr['epochs']}")
            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                metrics = train_one_batch(images, *components)
                count += images.size(0)
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value * images.size(0)
                progress.set_postfix(g=metrics["generator_total"], d=metrics["discriminator_total"])
            completed_epoch = epoch
            history.append({"epoch": float(epoch), **{k: v / count for k, v in sums.items()}})
            save_history(history, run_dir)
            if epoch % tr["checkpoint_every"] == 0:
                checkpoint("checkpoint_latest.pt")
    except KeyboardInterrupt:
        checkpoint("checkpoint_interrupted.pt")
        print("已保存中断时权重；epoch 记录完整完成轮数，权重可能包含下一轮的部分更新。")
        return
    checkpoint("checkpoint_final.pt")
    print("正式训练完成。可运行 evaluate_mf.py 测试；不会自动使用测试集选模型。")


if __name__ == "__main__":
    main()
