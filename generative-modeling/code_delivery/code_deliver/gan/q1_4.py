import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp = None, interp = None, lamb = None
):
    """
    TODO 1.4.1: Implement LSGAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    a = torch.zeros_like(discrim_fake)
    b = torch.ones_like(discrim_real)
    loss = 0.5 * ((discrim_real - b) ** 2).mean() + 0.5 * ((discrim_fake - a) ** 2).mean()
    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.4.1: Implement LSGAN loss for generator.
    """
    # Value we want D to believe which is that the fakes are real
    c = torch.ones_like(discrim_fake)
    loss = 0.5 * ((discrim_fake - c)**2).mean()
    return loss

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.4.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
