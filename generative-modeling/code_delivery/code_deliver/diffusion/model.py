import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas

        # 3.1: compute the cumulative products for current and previous timesteps
        self.alphas_cumprod      = torch.cumprod(alphas, dim = 0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.]).cuda(), self.alphas_cumprod[0:-1]))

        # 3.1: pre-compute values needed for forward process
        # This is the coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = 1/torch.sqrt(self.alphas_cumprod)
        # This is the coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = -torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)

        # 3.1: compute the coefficients for the mean
        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = (torch.sqrt(self.alphas_cumprod_prev) * self.betas) / (1 - self.alphas_cumprod)
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = (torch.sqrt(alphas) * (1 - self.alphas_cumprod_prev)) / (1 - self.alphas_cumprod)

        # 3.1: compute posterior variance
        # calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # 3.1: Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0
        # hint: can use extract function from utils.py

        posterior_mean     =  extract(self.posterior_mean_coef1, t, x_0.shape) * x_0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, posterior_mean.shape)
        posterior_log_variance_clipped = torch.log(posterior_variance)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):
        # 3.1: given a noised image x_t, predict x_0 and the additive noise
        # to predict the additive noise, use the denoising model.
        # Hint: You can use extract function from utils.py.
        # clamp x_0 to [-1, 1]
        eps_t = self.model(x_t, t)
        x_0   = extract(self.x_0_pred_coef_1, t, x_t.shape) * x_t + extract(self.x_0_pred_coef_2, t, eps_t.shape) * eps_t
        x_0   = torch.clamp(x_0, -1, 1)

        return (eps_t, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # 3.1: given x at timestep t, predict the denoised image at x_{t-1}.
        # also return the predicted starting image.
        # Hint: To do this, you will need a predicted x_0. Which function can do this for you?
        eps_t, x_0 = self.model_predictions(x, t)

        posterior_mean, posterior_variance, _ = self.get_posterior_parameters(x_0, x, t)

        # Find the mean
        if torch.all(t > 0):
            z = torch.randn_like(posterior_mean)
        else:
            z = 0
        
        pred_img   = posterior_mean +  z * torch.sqrt(posterior_variance)

        # Sample standard normal and scale by the standard deviation
        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps-1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        # 3.2: Generate a list of times to sample from.
        return torch.flip(torch.linspace(0, total_timesteps-1, sampling_timesteps).cuda(), dims = (0,)).type(torch.int64)

    def get_time_pairs(self, times):
        # 3.2: Generate a list of adjacent time pairs to sample from.
        out = torch.zeros(times.shape[0]-1, 2)
        out[:, 0] = times[0:-1]
        out[:, 1] = times[1:]
        return out

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        # 3.2: Compute the output image for a single step of the DDIM sampling process.
        tau_i     = tau_i.repeat(batch).cuda().type(torch.int64)
        tau_isub1 = tau_isub1.repeat(batch).cuda().type(torch.int64)

        # predict x_0 and the additive noise for tau_i
        eps_t, x_0 = model_predictions(img, tau_i)
        
        if torch.all(tau_i > 0):
            z = torch.randn_like(img)
        else:
            return img, x_0

        # extract \alpha_{\tau_{i - 1}} and \alpha_{\tau_{i}}
        alpha_tau_prev = extract(alphas_cumprod, tau_isub1, img.shape)
        alpha_tau      = extract(alphas_cumprod, tau_i, img.shape)

        # compute \sigma_{\tau_{i}}
        beta_tau  = ((1 - alpha_tau_prev) / (1 - alpha_tau)) * extract(self.betas, tau_isub1, img.shape)
        sigma_tau = eta * beta_tau

        # compute the coefficient of \epsilon_{\tau_{i}}
        coeff_eps = torch.sqrt(1 - alpha_tau_prev - sigma_tau)

        # sample from q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0)
        mean_tau = torch.sqrt(alpha_tau_prev) * x_0 + coeff_eps * eps_t
        
        # HINT: use the reparameterization trick
        img = mean_tau + z * sigma_tau**0.5

        return img, x_0

    def sample_ddim(self, shape, z):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device = self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        # 3.3: fill out based on the sample function above
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        # Permuting seems to help here, we need to move the channels to the second dim, gave a good boost.
        imgs      = sample_fn(shape, z.view(shape[0], shape[2], shape[3], shape[1]).permute(0,3,1,2))
        return imgs * 255.
