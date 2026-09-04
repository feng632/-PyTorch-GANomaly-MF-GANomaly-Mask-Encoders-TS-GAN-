"""Mask Encoders 单批训练：先更新判别器，再更新生成器。"""

import torch

from anomaly_reproduction.training import set_requires_grad


def train_one_batch(inputs, targets, generator, discriminator, optimizer_g,
                    optimizer_d, generator_loss_fn, discriminator_loss_fn):
    generator.train(); discriminator.train()
    reconstructed = generator(inputs)

    set_requires_grad(discriminator, True)
    optimizer_d.zero_grad(set_to_none=True)
    real_probabilities, _ = discriminator(targets)
    fake_probabilities, _ = discriminator(reconstructed.detach())
    d_losses = discriminator_loss_fn(real_probabilities, fake_probabilities)
    d_losses["total"].backward()
    optimizer_d.step()

    set_requires_grad(discriminator, False)
    discriminator.eval()
    optimizer_g.zero_grad(set_to_none=True)
    fake_probabilities, _ = discriminator(reconstructed)
    g_losses = generator_loss_fn(targets, reconstructed, fake_probabilities)
    g_losses["total"].backward()
    optimizer_g.step()
    set_requires_grad(discriminator, True)
    discriminator.train()
    return {**{f"generator_{k}": v.item() for k, v in g_losses.items()},
            **{f"discriminator_{k}": v.item() for k, v in d_losses.items()}}
