import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
from torchvision.utils import save_image


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # 1.2: Generate and save out latent space interpolations.
    # Generate 100 samples of 128-dim vectors
    samples = torch.zeros((100, 128)).cuda()

    # Do so by linearly interpolating for 10 steps across each of the first two dimensions between -1 and 1.
    d1 = torch.linspace(-1, 1, 10).cuda()
    d2 = torch.linspace(-1, 1, 10).cuda()
    cross = torch.cartesian_prod(d1, d2)
    samples[:, 0:2] = cross

    # Keep the rest of the z vector for the samples to be some fixed value (e.g. 0).
    # Forward the samples through the generator.
    img = gen.forward_given_samples(samples)

    # Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    save_image(img, fp = path)

    return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
