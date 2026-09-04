"""训练当前支路生成器和共享判别器一次。"""

import torch
from anomaly_reproduction.training import set_requires_grad


def train_one_batch(images, generator, discriminator, optimizer_g, optimizer_d,
                    generator_loss_fn, discriminator_loss_fn):
    generator.train(); discriminator.train()
    reconstructed = generator(images)
    set_requires_grad(discriminator, True); optimizer_d.zero_grad(set_to_none=True)
    real_prob, _ = discriminator(images)
    fake_prob, _ = discriminator(reconstructed.detach())
    d_losses = discriminator_loss_fn(real_prob, fake_prob)
    d_losses["total"].backward(); optimizer_d.step()

    set_requires_grad(discriminator, False); discriminator.eval(); optimizer_g.zero_grad(set_to_none=True)
    with torch.no_grad(): _, real_features = discriminator(images)
    fake_prob, fake_features = discriminator(reconstructed)
    g_losses = generator_loss_fn(images, reconstructed, real_features, fake_features, fake_prob)
    g_losses["total"].backward(); optimizer_g.step()
    set_requires_grad(discriminator, True); discriminator.train()
    return {**{f"generator_{k}": v.item() for k,v in g_losses.items()},
            **{f"discriminator_{k}": v.item() for k,v in d_losses.items()}}
