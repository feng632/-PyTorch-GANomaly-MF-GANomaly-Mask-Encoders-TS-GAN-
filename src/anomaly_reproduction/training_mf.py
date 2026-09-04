"""MF-GANomaly 的交替更新；每批先更新 D，再更新 G。"""

import torch

from anomaly_reproduction.training import set_requires_grad


def checked_backward(loss, model):
    if not torch.isfinite(loss):
        raise FloatingPointError("损失出现 NaN/Inf，已停止本次更新")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("梯度出现 NaN/Inf，已停止本次更新")


def train_one_batch(images, generator, discriminator, optimizer_g, optimizer_d,
                    generator_loss_function, discriminator_loss_function):
    # G 只 forward 一次，避免一个 batch 更新两次 BN 统计量。
    generator.train()
    discriminator.train()
    set_requires_grad(discriminator, True)
    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)
    output = generator(images)

    real_probabilities, _ = discriminator(images)
    fake_probabilities, _ = discriminator(output.reconstructed.detach())
    d_losses = discriminator_loss_function(real_probabilities, fake_probabilities)
    checked_backward(d_losses["total"], discriminator)
    optimizer_d.step()
    optimizer_d.zero_grad(set_to_none=True)

    # G 更新时冻结 D 参数和 BN 统计量，但保留通向重建图的梯度路径。
    set_requires_grad(discriminator, False)
    discriminator.eval()
    try:
        with torch.no_grad():
            _, real_features = discriminator(images)
        _, fake_features = discriminator(output.reconstructed)
        g_losses = generator_loss_function(images, output, real_features, fake_features)
        checked_backward(g_losses["total"], generator)
        optimizer_g.step()
    finally:
        set_requires_grad(discriminator, True)
        discriminator.train()

    return {**{f"generator_{k}": v.item() for k, v in g_losses.items()},
            **{f"discriminator_{k}": v.item() for k, v in d_losses.items()}}
