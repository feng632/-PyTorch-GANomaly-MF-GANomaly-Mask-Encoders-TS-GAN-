import torch

from anomaly_reproduction.losses import DiscriminatorLoss, GeneratorLoss


def test_generator_loss_weighted_sum() -> None:
    loss_function = GeneratorLoss(40.0, 1.0, 1.0)
    original = torch.zeros(2, 3, 4, 4)
    reconstructed = torch.ones_like(original)
    latent_original = torch.zeros(2, 8, 1, 1)
    latent_reconstructed = torch.ones_like(latent_original)
    real_features = torch.zeros(2, 4, 2, 2)
    reconstructed_features = torch.ones_like(real_features)

    losses = loss_function(
        original,
        reconstructed,
        latent_original,
        latent_reconstructed,
        real_features,
        reconstructed_features,
    )

    assert torch.isclose(losses["reconstruction"], torch.tensor(1.0))
    assert torch.isclose(losses["encoding"], torch.tensor(1.0))
    assert torch.isclose(losses["adversarial"], torch.tensor(1.0))
    assert torch.isclose(losses["total"], torch.tensor(42.0))


def test_discriminator_loss_prefers_correct_predictions() -> None:
    loss_function = DiscriminatorLoss()
    good = loss_function(torch.tensor([0.9]), torch.tensor([0.1]))["total"]
    bad = loss_function(torch.tensor([0.1]), torch.tensor([0.9]))["total"]

    assert good < bad

