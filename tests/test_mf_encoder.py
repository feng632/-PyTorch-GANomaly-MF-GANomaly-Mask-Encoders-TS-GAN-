import torch

from anomaly_reproduction.models import Encoder
from anomaly_reproduction.models_mf import MFEncoder


def test_mf_encoder_preserves_baseline_encoding():
    baseline = Encoder().eval()
    encoder = MFEncoder().eval()
    encoder.load_state_dict(baseline.state_dict())
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        latent_code, shallow_features = encoder(images)
        torch.testing.assert_close(latent_code, baseline(images))
        torch.testing.assert_close(shallow_features, baseline.conv1(images))
    assert latent_code.shape == (2, 128, 1, 1)
    assert shallow_features.shape == (2, 64, 32, 32)
    assert next(encoder.parameters()) is not next(baseline.parameters())


def test_shallow_features_keep_gradient_path():
    encoder = MFEncoder()
    _, shallow_features = encoder(torch.randn(2, 3, 64, 64))
    shallow_features.square().mean().backward()
    gradient = encoder.conv1[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
