import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple


class VarianceScheduler:
    def __init__(self, beta_start: int=0.0001, beta_end: int=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        self.num_steps = num_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            # TODO: complete the linear interpolation of betas here
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
            self.betas = self.betas.to(self.device)
        elif interpolation == 'quadratic':
            # TODO: complete the quadratic interpolation of betas here
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
            self.betas = self.betas.to(self.device)
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')
        

        # TODO: add other statistics such alphas alpha_bars and all the other things you might need here
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bars = self.alpha_bars.to(self.device)

    def add_noise(self, x:torch.Tensor, time_step:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        batch_size = x.size(0)
        
        noise = torch.randn_like(x, device=device)
        alpha_bar_t = self.alpha_bars[time_step].view(batch_size, 1, 1, 1).to(device)
        
        noisy_input = x * alpha_bar_t.sqrt() + noise * (1 - alpha_bar_t).sqrt()
        
        return noisy_input, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.conv = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)  
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)  
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)  
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  
        
        self.bnorm = nn.GroupNorm(8 ,out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.relu(self.bnorm(self.conv(x)))  
        
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]  
        h = h + time_emb  
        return self.transform(h)


class UNet(nn.Module):
    def __init__(self, in_channels=1,
                 down_channels=(64, 128, 256, 512),
                 up_channels=(512, 256, 128, 64),
                 time_emb_dim=128,
                 num_classes=10):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, 2 * time_emb_dim),
            nn.ReLU(),
            nn.Linear(2 * time_emb_dim, time_emb_dim),
        )

        self.class_embedding = nn.Embedding(num_classes, time_emb_dim)

        self.conv0 = nn.Conv2d(self.in_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([
            Block(down_channels[0], down_channels[1], time_emb_dim),
            Block(down_channels[1], down_channels[2], time_emb_dim),  
            Block(down_channels[2], down_channels[3], time_emb_dim) 
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(down_channels[3], down_channels[3], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(down_channels[3], down_channels[3], 3, padding=1),
            nn.GroupNorm(8, down_channels[3]),
            nn.ReLU()
        )

        self.ups = nn.ModuleList([
            Block(up_channels[0], up_channels[1], time_emb_dim, True),
            Block(up_channels[1], up_channels[2], time_emb_dim, True),  
            Block(up_channels[2], up_channels[3], time_emb_dim, True) 
        ])

        self.output = nn.Conv2d(up_channels[-1], in_channels, 1)

    def forward(self, x, timestep, class_idx):
      
        class_emb = self.class_embedding(class_idx)
        t = self.time_mlp(timestep)
        emb = t + class_emb 
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, emb)
            residual_inputs.append(x)

        x = self.bottleneck(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, emb)

        return self.output(x)

class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int=1, 
                 height: int=32, 
                 width: int=32, 
                 mid_channels: List=[64, 128, 256, 512], 
                 latent_dim: int=128, 
                 num_classes: int=10) -> None:
        
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.mid_size = [mid_channels[-1], height // (2 ** (len(mid_channels)-1)), width // (2 ** (len(mid_channels)-1))]
        self.class_emb = nn.Embedding(self.num_classes, self.latent_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, mid_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[0]),
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[1]),
            nn.Conv2d(mid_channels[1], mid_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[1]), 
            nn.Conv2d(mid_channels[1], mid_channels[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[2]),
            nn.Conv2d(mid_channels[2], mid_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[2]),
            nn.Conv2d(mid_channels[2], mid_channels[3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[3]),
            nn.Conv2d(mid_channels[3], mid_channels[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[3]),   
            nn.Flatten()
        )

        input_dim = self.mid_size[0] * self.mid_size[1] * self.mid_size[2]
        self.mean_net = nn.Linear(input_dim, self.latent_dim)
        self.logvar_net = nn.Linear(input_dim, self.latent_dim)

     
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim, input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim),  
            nn.Unflatten(1, self.mid_size),
            nn.ConvTranspose2d(mid_channels[-1], mid_channels[-2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[-2]), 
            nn.Conv2d(mid_channels[-2], mid_channels[-2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[-2]), 
            nn.ConvTranspose2d(mid_channels[-2], mid_channels[-3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[-3]),
            nn.Conv2d(mid_channels[-3], mid_channels[-3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[-3]),  
            nn.ConvTranspose2d(mid_channels[-3], mid_channels[-4], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels[-4]),  
            nn.Conv2d(mid_channels[-4], self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
         
        out = self.encoder(x) 
        mean = self.mean_net(out)
        logvar = self.logvar_net(out) 
        sample = self.reparameterize(mean, logvar)
        out = self.decode(sample, label) 
        return out, mean, logvar


    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss
       
    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            # randomly consider some labels
            labels = torch.randint(0, self.num_classes, [num_samples,], device=device)

        noise = torch.randn(num_samples, self.latent_dim, device=device)

        return self.decode(noise, labels)
    
    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: use you decoder to decode a given sample and their corresponding labels
        label_emb = self.class_emb(labels).view(labels.size(0), -1)
        sample = torch.cat([sample, label_emb], dim=1)
        out = self.decoder(sample)
        return out


class LDDPM(nn.Module):
    def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.vae = vae
        self.network = network

        # freeze vae
        self.vae.requires_grad_(False)
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, (x.size(0),), device=x.device)

        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)
        # TODO: Estimate the noise 
        estimated_noise = self.network(noisy_input, t, label)

        # compute the loss (either L1 or L2 loss)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        betas = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1)
        alphas = self.var_scheduler.alphas[timestep].view(-1, 1, 1, 1)
        alpha_bars = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1)
        alpha_bars_prev = self.var_scheduler.alpha_bars[timestep - 1].view(-1, 1, 1, 1)
        alphaprevbyalphabar = (1-alpha_bars)/(1-alpha_bars_prev)
        multi = alpha_bars_prev * betas


        if timestep[0] > 0:
            noise_scale = multi.sqrt() * torch.randn_like(noisy_sample)
        else:
            noise_scale = 0

        sample = (1 / alphas.sqrt()) * (noisy_sample - (betas / (1 - alpha_bars).sqrt()) * estimated_noise) + noise_scale
        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.vae.num_classes, [num_samples,], device=device)
        
        # TODO: using the diffusion model generate a sample inside the latent space of the vae
        # NOTE: you need to recover the dimensions of the image in the latent space of your VAE
        latent_shape = (num_samples, self.vae.latent_dim, 4, 4)
        sample = torch.randn(latent_shape, device=device)
        for t in reversed(range(1,self.var_scheduler.num_timesteps)):
            timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(sample, timestep, labels)
            sample = self.recover_sample(sample, estimated_noise, timestep)

        sample = self.vae.decode(sample, labels)
        
        return sample


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, (x.size(0),), device=x.device)

        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # TODO: Estimate the noise 
        estimated_noise = self.network(noisy_input, t, label)

        # TODO: compute the loss (either L1, or L2 loss)
        loss = F.l1_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        betas = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1)
        alphas = self.var_scheduler.alphas[timestep].view(-1, 1, 1, 1)
        alpha_bars = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1)
        alpha_bars_prev = self.var_scheduler.alpha_bars[timestep - 1].view(-1, 1, 1, 1)
        alphaprevbyalphabar = (1-alpha_bars)/(1-alpha_bars_prev)
        multi = alpha_bars_prev * betas
        self.var_scheduler.alpha_bars[0] = 1


        if timestep[0] > 0:
            noise_scale = multi.sqrt() * torch.randn_like(noisy_sample)
        else:
            noise_scale = 0

        sample = (1 / alphas.sqrt()) * (noisy_sample - (betas / (1 - alpha_bars).sqrt()) * estimated_noise) + noise_scale
        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'), labels: torch.Tensor = None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, (num_samples,), device=device)
        else:
            labels = None

        sample = torch.randn((num_samples, 1, 32, 32), device=device)
        for t in reversed(range(1,self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(sample, t_tensor, labels)
            sample = self.recover_sample(sample, estimated_noise, t_tensor)
           

        return sample

        

class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.randint(0, self.var_scheduler.num_steps, (x.size(0),), device=x.device)

        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # TODO: Estimate the noise 
        estimated_noise = self.network(noisy_input, t, label)

        # TODO: compute the loss (either L1, or L2 loss)
        loss = F.l1_loss(estimated_noise, noise)

        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: apply the sample recovery strategy of the DDIM
        betas = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1)
        alphas = self.var_scheduler.alphas[timestep].view(-1, 1, 1, 1)
        alpha_bars = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1)
        alpha_bars_prev = self.var_scheduler.alpha_bars[timestep - 1].view(-1, 1, 1, 1)
        alphaprevbyalphabar = (1-alpha_bars)/(1-alpha_bars_prev)
        self.var_scheduler.alpha_bars[0] = 1
        multi = alpha_bars_prev * betas

        sample = (alpha_bars_prev.sqrt()) * ((noisy_sample - ((1-alpha_bars).sqrt() * estimated_noise))/ (alpha_bars).sqrt()) + (1-alpha_bars_prev).sqrt() * estimated_noise
        return sample

        
    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None
        # TODO: apply the iterative sample generation of DDIM (similar to DDPM)
        sample = torch.randn((num_samples, 1, 32, 32), device=device)
        for t in reversed(range(1,self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(sample, t_tensor, labels)
            sample = self.recover_sample(sample, estimated_noise, t_tensor)
           

        return sample
    
