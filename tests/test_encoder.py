import torch

from anomaly_reproduction.models import Decoder, Discriminator, Encoder, Generator


def test_encoder_output_shape() -> None:
    encoder = Encoder(input_channels=3, latent_dim=128)
    images = torch.randn(2, 3, 64, 64)

    latent_code = encoder(images)

    assert latent_code.shape == (2, 128, 1, 1)


def test_decoder_restores_image_shape_and_range() -> None:
    decoder = Decoder(output_channels=3, latent_dim=128)
    latent_code = torch.randn(2, 128, 1, 1)

    reconstructed_images = decoder(latent_code)

    assert reconstructed_images.shape == (2, 3, 64, 64)
    assert reconstructed_images.min().item() >= -1.0
    assert reconstructed_images.max().item() <= 1.0


def test_generator_shapes_and_independent_encoders() -> None:
    generator = Generator(image_channels=3, latent_dim=128)
    images = torch.randn(2, 3, 64, 64)

    reconstructed, latent_original, latent_reconstructed = generator(images)

    assert reconstructed.shape == (2, 3, 64, 64)
    assert latent_original.shape == (2, 128, 1, 1)
    assert latent_reconstructed.shape == (2, 128, 1, 1)
    assert generator.encoder1 is not generator.encoder2
    assert (
        generator.encoder1.conv1[0].weight.data_ptr()
        != generator.encoder2.conv1[0].weight.data_ptr()
    )


def test_discriminator_probability_and_feature_shapes() -> None:
    discriminator = Discriminator(image_channels=3)
    images = torch.randn(2, 3, 64, 64)

    probabilities, features = discriminator(images)

    assert probabilities.shape == (2,)
    assert features.shape == (2, 512, 4, 4)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
