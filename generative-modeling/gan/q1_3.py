import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp = None, interp = None, lamb = None
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    # Term 1:
    # - High loss if the real has a low probability of being real
    # - No loss if the real has a high probability to be real
    # Term 2:
    # - High loss if the fake (generated) has a high probability of being real
    # - Small loss if the fake has a low probability of being real
    # discrim_real = torch.sigmoid(discrim_real)
    # discrim_fake = torch.sigmoid(discrim_fake)
    # loss = -torch.sum(torch.log(discrim_real+1e-12) + torch.log(1+1e-12 - discrim_fake))

    loss = F.binary_cross_entropy_with_logits(discrim_real, torch.ones_like(discrim_real)) + \
           F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_fake))

    return loss


def compute_generator_loss(discrim_fake):
    """
    1.3.1: Implement GAN loss for generator.
    """
    # Low loss if the fake has a low probability of being real
    # Very negative loss if the fake has a high probability of being real (discriminator was fooled)
    # discrim_fake = torch.sigmoid(discrim_fake)
    # loss = torch.sum(torch.log(1+1e-12 - discrim_fake))

    # If fakes are fake, this is a very high loss, if fakes are thought to be real this is low
    loss = F.binary_cross_entropy_with_logits(discrim_fake, torch.ones_like(discrim_fake))
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
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
