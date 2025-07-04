#!/usr/bin/env python3
"""
DRAL-WGAN Experiment Implementation

This script implements three experiments comparing DRAL-WGAN ("New Method") versus LWGAN ("Base Method")
under noisy conditions, latent space analysis, and an ablation study on the latent diffusion module.

Experiments:
1. Stability Under Noisy Conditions - Compare robustness to input noise
2. Latent Space Analysis - Analyze importance weight dynamics and latent representations
3. Ablation Study - Compare full model vs variants (fixed noise, no diffusion)

All plots are saved as high-quality PDF files in .research/iteration1/images/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

from preprocess import prepare_data_for_experiments, add_noise
from train import (
    DRAL_WGAN, LWGAN, DRAL_WGAN_Full, DRAL_WGAN_FixedNoise, DRAL_WGAN_NoDiffusion,
    create_models, train_one_epoch
)
from evaluate import (
    extract_latent_representations, plot_training_losses, plot_latent_distributions,
    plot_tsne_latent_space, plot_ablation_comparison, evaluate_model_performance
)

def setup_experiment_environment():
    """Setup experiment environment and directories."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    return device

def run_experiment1(device, data_loaders, test_mode=True):
    """
    Experiment 1: Stability Under Noisy Conditions
    Compare DRAL-WGAN vs LWGAN robustness to Gaussian noise.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Stability Under Noisy Conditions")
    print("="*60)
    
    max_batches = 20 if test_mode else 100
    noise_std = 0.2
    
    netQ_dral, netG_dral = create_models(device)
    netQ_lwa, netG_lwa = create_models(device)
    
    model_dral = DRAL_WGAN(netQ_dral, netG_dral).to(device)
    model_lwa = LWGAN(netQ_lwa, netG_lwa).to(device)
    
    criterion = nn.MSELoss()
    optimizer_dral = optim.Adam(model_dral.parameters(), lr=1e-3)
    optimizer_lwa = optim.Adam(model_lwa.parameters(), lr=1e-3)
    
    print(f"Training DRAL-WGAN with noise_std={noise_std}...")
    losses_dral = train_one_epoch(
        model_dral, optimizer_dral, data_loaders['train'], 
        criterion, device, noise_std=noise_std, max_batches=max_batches
    )
    
    print(f"Training LWGAN with noise_std={noise_std}...")
    losses_lwa = train_one_epoch(
        model_lwa, optimizer_lwa, data_loaders['train'], 
        criterion, device, noise_std=noise_std, max_batches=max_batches
    )
    
    losses_dict = {
        "DRAL-WGAN": losses_dral,
        "LWGAN": losses_lwa
    }
    
    plot_training_losses(
        losses_dict, 
        ".research/iteration1/images/training_loss.pdf"
    )
    
    print(f"DRAL-WGAN final loss: {losses_dral[-1]:.4f}")
    print(f"LWGAN final loss: {losses_lwa[-1]:.4f}")
    print("Experiment 1 completed successfully!")

def run_experiment2(device, data_loaders, test_mode=True):
    """
    Experiment 2: Latent Space Analysis and Importance Weight Dynamics
    Analyze latent representations before and after diffusion.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Latent Space Analysis and Importance Weight Dynamics")
    print("="*60)
    
    max_batches = 10 if test_mode else 50
    
    netQ, netG = create_models(device)
    model_dral = DRAL_WGAN(netQ, netG).to(device)
    
    print("Extracting latent representations...")
    pre_latents, post_latents = extract_latent_representations(
        model_dral, data_loaders['train'], device, max_batches=max_batches
    )
    
    print(f"Pre-diffusion latents shape: {pre_latents.shape}")
    print(f"Post-diffusion latents shape: {post_latents.shape}")
    
    plot_latent_distributions(
        pre_latents, post_latents,
        ".research/iteration1/images/latent_distribution.pdf"
    )
    
    plot_tsne_latent_space(
        post_latents,
        ".research/iteration1/images/tsne_latent.pdf"
    )
    
    print(f"Pre-diffusion mean: {pre_latents.mean():.4f}, std: {pre_latents.std():.4f}")
    print(f"Post-diffusion mean: {post_latents.mean():.4f}, std: {post_latents.std():.4f}")
    print("Experiment 2 completed successfully!")

def run_experiment3(device, data_loaders, test_mode=True):
    """
    Experiment 3: Ablation Study on Latent Diffusion Process and Dynamic Noise Schedule
    Compare full model vs variants (fixed noise, no diffusion).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Ablation Study on Latent Diffusion Process")
    print("="*60)
    
    max_batches = 20 if test_mode else 100
    
    netQ_full, netG_full = create_models(device)
    netQ_fixed, netG_fixed = create_models(device)
    netQ_nodiff, netG_nodiff = create_models(device)
    
    model_full = DRAL_WGAN_Full(netQ_full, netG_full).to(device)
    model_fixed = DRAL_WGAN_FixedNoise(netQ_fixed, netG_fixed).to(device)
    model_nodiff = DRAL_WGAN_NoDiffusion(netQ_nodiff, netG_nodiff).to(device)
    
    criterion = nn.MSELoss()
    optimizer_full = optim.Adam(model_full.parameters(), lr=1e-3)
    optimizer_fixed = optim.Adam(model_fixed.parameters(), lr=1e-3)
    optimizer_nodiff = optim.Adam(model_nodiff.parameters(), lr=1e-3)
    
    print("Training full model variant...")
    loss_full = train_one_epoch(
        model_full, optimizer_full, data_loaders['train'], 
        criterion, device, max_batches=max_batches
    )
    
    print("Training fixed noise variant...")
    loss_fixed = train_one_epoch(
        model_fixed, optimizer_fixed, data_loaders['train'], 
        criterion, device, max_batches=max_batches
    )
    
    print("Training no diffusion variant...")
    loss_nodiff = train_one_epoch(
        model_nodiff, optimizer_nodiff, data_loaders['train'], 
        criterion, device, max_batches=max_batches
    )
    
    loss_dict = {
        "Full Model": loss_full,
        "Fixed Noise": loss_fixed,
        "No Diffusion": loss_nodiff
    }
    
    plot_ablation_comparison(
        loss_dict,
        ".research/iteration1/images/ablation_loss_comparison.pdf"
    )
    
    print(f"Full model final loss: {loss_full[-1]:.4f}")
    print(f"Fixed noise final loss: {loss_fixed[-1]:.4f}")
    print(f"No diffusion final loss: {loss_nodiff[-1]:.4f}")
    print("Experiment 3 completed successfully!")

def test_experiments():
    """Quick test function to verify experiment code runs without errors."""
    print("="*60)
    print("RUNNING QUICK TEST MODE")
    print("="*60)
    
    device = setup_experiment_environment()
    
    print("Preparing MNIST dataset...")
    data_loaders = prepare_data_for_experiments(device, batch_size=64)
    
    try:
        run_experiment1(device, data_loaders, test_mode=True)
        run_experiment2(device, data_loaders, test_mode=True)
        run_experiment3(device, data_loaders, test_mode=True)
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN TEST MODE!")
        print("="*60)
        
        print("\nGenerated PDF files:")
        pdf_files = [
            ".research/iteration1/images/training_loss.pdf",
            ".research/iteration1/images/latent_distribution.pdf", 
            ".research/iteration1/images/tsne_latent.pdf",
            ".research/iteration1/images/ablation_loss_comparison.pdf"
        ]
        
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                print(f"✓ {pdf_file}")
            else:
                print(f"✗ {pdf_file} - NOT FOUND")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during test execution: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for running the complete experiments."""
    print("="*60)
    print("DRAL-WGAN vs LWGAN EXPERIMENTAL COMPARISON")
    print("="*60)
    
    device = setup_experiment_environment()
    
    print("Preparing MNIST dataset...")
    data_loaders = prepare_data_for_experiments(device, batch_size=64)
    
    try:
        run_experiment1(device, data_loaders, test_mode=False)
        run_experiment2(device, data_loaders, test_mode=False)
        run_experiment3(device, data_loaders, test_mode=False)
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        status_enum = "stopped"
        print(f"Setting status_enum to: {status_enum}")
        
    except Exception as e:
        print(f"\nERROR during experiment execution: {e}")
        import traceback
        traceback.print_exc()
        status_enum = "error"
        print(f"Setting status_enum to: {status_enum}")
        sys.exit(1)

if __name__ == "__main__":
    test_experiments()
