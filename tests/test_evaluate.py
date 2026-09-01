import torch

from anomaly_reproduction.evaluate import denormalize


def test_denormalize_maps_expected_range() -> None:
    image = torch.tensor([[[-1.0, 1.0]]]).repeat(3, 1, 1)
    result = denormalize(image)

    assert result.shape == (1, 2, 3)
    assert result.min() == 0.0
    assert result.max() == 1.0
