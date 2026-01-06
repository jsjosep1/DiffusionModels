## Overview
This project implements and compares several **class-conditional generative models** capable of creating new FashionMNIST images on demand.  
All models are trained on padded 32×32 images and can generate samples for a chosen clothing category.

## Models Implemented
- **Conditional DDPM** – Probabilistic diffusion model with custom UNet and noise scheduler  
- **Conditional DDIM** – Faster, deterministic diffusion variant  
- **Conditional VAE** – Latent-variable model with reparameterization  
- **Latent Diffusion (Bonus)** – Diffusion applied in the VAE latent space for efficiency  

## Key Features
- Class-conditional image generation  
- Custom UNet backbone with timestep & label conditioning  
- Linear & quadratic noise schedules  
- Batch-based generation (no per-sample loops)  
- Saved checkpoints for reproducibility  
- Quantitative evaluation using a pretrained classifier  
