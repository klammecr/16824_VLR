import os

import torch
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_wgan_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    1.5.1: Implement WGAN-GP loss for discriminator.
    loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    """
    # Calculate the gradient of discrim_interp
    grad_disc_interp = torch.autograd.grad(discrim_interp.sum(), interp, retain_graph=True, create_graph=True)[0]

    # Gradient penalization, find the norm over each image and average it
    grad_disc_interp  = grad_disc_interp.view(grad_disc_interp.shape[0], -1)
    grad_peanlty      = lamb * (torch.norm(grad_disc_interp, dim = 1) - 1)**2

    loss = discrim_fake.mean() - discrim_real.mean() + grad_peanlty.mean()
    return loss


def compute_generator_loss(discrim_fake):
    """
    1.5.1: Implement WGAN-GP loss for generator.
    loss = - E[D(fake_data)]
    """
    return -discrim_fake.mean()


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_wgan_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
