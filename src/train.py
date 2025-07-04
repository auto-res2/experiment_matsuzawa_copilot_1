import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def gen_noise(batch_size, z_dim, device):
    """Generate random noise for latent space."""
    return torch.randn(batch_size, z_dim, device=device)

def mmd_penalty(z, z_prior, kernel="IMQ", sigma2_p=1.0):
    """
    Maximum Mean Discrepancy penalty for regularization.
    Simplified implementation using L2 difference between means.
    """
    mean_z = z.mean(dim=0)
    mean_prior = z_prior.mean(dim=0)
    return torch.norm(mean_z - mean_prior)

class DummyNetQ(nn.Module):
    """Encoder network mapping input to latent space."""
    def __init__(self, input_dim=28*28, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc(x)

class DummyNetG(nn.Module):
    """Generator network mapping latent space to output."""
    def __init__(self, latent_dim=128, output_dim=28*28):
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_dim)
    
    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, 1, 28, 28)

class DRAL_WGAN(nn.Module):
    """Diffusion-Regularized Adaptive Latent Wasserstein GAN."""
    def __init__(self, netQ, netG):
        super().__init__()
        self.netQ = netQ
        self.netG = netG
    
    def forward(self, x):
        latent = self.netQ(x)
        diffused_latent = latent + 0.1 * torch.randn_like(latent)
        x_rec = self.netG(diffused_latent)
        return x_rec

class LWGAN(nn.Module):
    """Latent Wasserstein GAN baseline."""
    def __init__(self, netQ, netG):
        super().__init__()
        self.netQ = netQ
        self.netG = netG
    
    def forward(self, x):
        latent = self.netQ(x)
        x_rec = self.netG(latent)
        return x_rec

class DRAL_WGAN_Full(nn.Module):
    """Full DRAL-WGAN with dynamic noise and diffusion."""
    def __init__(self, netQ, netG):
        super().__init__()
        self.netQ = netQ
        self.netG = netG
        self.dynamic_noise = True
        self.use_diffusion = True
    
    def forward(self, x):
        latent = self.netQ(x)
        if self.use_diffusion:
            latent = latent + 0.1 * torch.randn_like(latent)
        x_rec = self.netG(latent)
        return x_rec

class DRAL_WGAN_FixedNoise(nn.Module):
    """DRAL-WGAN with fixed noise schedule."""
    def __init__(self, netQ, netG):
        super().__init__()
        self.netQ = netQ
        self.netG = netG
        self.dynamic_noise = False
        self.use_diffusion = True
    
    def forward(self, x):
        latent = self.netQ(x)
        fixed_noise_std = 0.1
        if self.use_diffusion:
            latent = latent + fixed_noise_std * torch.randn_like(latent)
        x_rec = self.netG(latent)
        return x_rec

class DRAL_WGAN_NoDiffusion(nn.Module):
    """DRAL-WGAN without latent diffusion."""
    def __init__(self, netQ, netG):
        super().__init__()
        self.netQ = netQ
        self.netG = netG
        self.dynamic_noise = True
        self.use_diffusion = False
    
    def forward(self, x):
        latent = self.netQ(x)
        if self.dynamic_noise:
            noise_std = 0.05
            latent = latent + noise_std * torch.randn_like(latent)
        x_rec = self.netG(latent)
        return x_rec

def train_one_epoch(model, optimizer, dataloader, criterion, device, noise_std=0.0, max_batches=20):
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        optimizer: Optimizer for training
        dataloader: Data loader
        criterion: Loss criterion
        device: Device to train on
        noise_std: Standard deviation for input noise
        max_batches: Maximum number of batches to process
    
    Returns:
        List of losses for each batch
    """
    model.train()
    losses = []
    
    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        images = images.to(device)
        
        if noise_std > 0:
            noise = torch.randn_like(images) * noise_std
            noisy_images = images + noise
        else:
            noisy_images = images
        
        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return losses

def create_models(device, latent_dim=128):
    """
    Create encoder and generator networks.
    
    Args:
        device: Device to create models on
        latent_dim: Dimension of latent space
    
    Returns:
        Tuple of (encoder, generator)
    """
    netQ = DummyNetQ(latent_dim=latent_dim).to(device)
    netG = DummyNetG(latent_dim=latent_dim).to(device)
    return netQ, netG
