import copy

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from anomaly_reproduction.models_mf import MFGenerator, MFOutput
from anomaly_reproduction.losses_mf import MFGeneratorLoss, anomaly_scores, distance_per_sample
from anomaly_reproduction.evaluate_mf import aggregate_rows, normalize_scores, evaluate_checkpoint
from anomaly_reproduction import train_mf
from anomaly_reproduction.training_mf import train_one_batch


def test_exact_distances_and_formula():
    a = torch.tensor([[3., 4.], [0., 0.]])
    b = torch.zeros_like(a)
    torch.testing.assert_close(distance_per_sample(a, b), torch.tensor([5., 0.]))
    torch.testing.assert_close(distance_per_sample(a, b, "l1"), torch.tensor([7., 0.]))
    output = MFOutput(b, a, b, a, b)
    scores = anomaly_scores(a, output, alpha=2.)
    torch.testing.assert_close(scores["score"], torch.tensor([25., 0.]))
    losses = MFGeneratorLoss()(a, output, a, b)
    torch.testing.assert_close(losses["total"], torch.tensor(205.))
    zero = torch.zeros(2, 3, requires_grad=True)
    distance_per_sample(zero, torch.zeros_like(zero)).mean().backward()
    assert torch.isfinite(zero.grad).all()


def test_aggregation_checks_and_normalization():
    rows = [dict(path="a", crop_index=i, label=0, defect_type="good", anomaly_score=float(i)) for i in range(4)]
    aggregated = aggregate_rows(rows)[0]
    assert aggregated["anomaly_score_max"] == 3
    assert aggregated["anomaly_score_mean"] == 1.5
    assert aggregated["normalized_score_max"] == 0
    np.testing.assert_equal(normalize_scores([2, 2]), [0, 0])
    with pytest.raises(ValueError):
        aggregate_rows(rows[:3])
    with pytest.raises(ValueError):
        normalize_scores([0, float("nan")])


def test_paper_generator_gradients_and_independence():
    model = MFGenerator().eval()
    assert isinstance(model.encoder1.conv1[1], torch.nn.BatchNorm2d)
    assert isinstance(model.encoder1.conv5[1], torch.nn.BatchNorm2d)
    assert model.encoder1.conv1[0].weight is not model.encoder2.conv1[0].weight
    output = model(torch.randn(2, 3, 64, 64))
    assert output.reconstructed.shape == (2, 3, 64, 64)
    assert output.shallow_original.shape == (2, 64, 32, 32)
    distance_per_sample(output.shallow_original, output.shallow_reconstructed).mean().backward()
    for part in (model.encoder1, model.decoder, model.encoder2):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())


def test_alternating_update_changes_both_models():
    config = train_mf.load_config()
    components = train_mf.build_components(config, torch.device("cpu"))
    g, d, *_ = components
    before_g = next(g.parameters()).detach().clone()
    before_d = next(d.parameters()).detach().clone()
    metrics = train_one_batch(torch.randn(2, 3, 64, 64), *components)
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.equal(before_g, next(g.parameters()))
    assert not torch.equal(before_d, next(d.parameters()))
    assert all(p.requires_grad and p.grad is None for p in d.parameters())
    assert d.training
    # 单个 G forward 应只让每层 BN 的计数增加一次。
    assert g.encoder1.conv1[1].num_batches_tracked == 1


def test_synthetic_train_save_evaluate(tmp_path, monkeypatch):
    """合成图片的小型集成测试，绝不使用真实实验的结果目录。"""
    config = copy.deepcopy(train_mf.load_config())
    data = tmp_path / "data"
    rng = np.random.default_rng(1)
    for relative in ("train/good/000.png", "test/good/000.png", "test/bent/000.png"):
        path = data / "grid" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
    mask = data / "grid/ground_truth/bent/000_mask.png"
    mask.parent.mkdir(parents=True)
    Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(mask)
    config["dataset"]["root"] = str(data)
    config["dataset"]["batch_size"] = 4
    config["training"]["epochs"] = 1
    config["training"]["checkpoint_every"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(train_mf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["train_mf", "--config", str(config_path), "--device", "cpu"])
    train_mf.main()
    run = tmp_path / "runs" / config["experiment"]["name"]
    assert (run / "history.csv").is_file()
    assert (run / "loss_curves.png").is_file()
    checkpoint = run / "checkpoint_final.pt"
    metrics = evaluate_checkpoint(checkpoint, torch.device("cpu"))
    assert metrics["test_images"] == 2
    assert 0 <= metrics["image_auroc_mean"] <= 1
    assert (run / "evaluation_checkpoint_final/test_image_scores.csv").is_file()
    with pytest.raises(FileExistsError):
        train_mf.reserve_run_dir(config)
