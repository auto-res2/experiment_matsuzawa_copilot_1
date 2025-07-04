"""
Configuration file for DRAL-WGAN experiments.
Contains parameters optimized for NVIDIA Tesla T4 (16GB VRAM).
"""

import torch

EXPERIMENT_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,  # Optimized for Tesla T4
    "latent_dim": 128,
    "learning_rate": 1e-3,
    "max_epochs": 1,
    "max_batches_test": 20,
    "max_batches_full": 100,
    
    "noise_schedules": {
        "experiment1_noise_std": 0.2,
        "diffusion_noise_std": 0.1,
        "fixed_noise_std": 0.1,
        "dynamic_noise_std": 0.05
    },
    
    "output_paths": {
        "images_dir": ".research/iteration1/images",
        "training_loss_plot": ".research/iteration1/images/training_loss.pdf",
        "latent_distribution_plot": ".research/iteration1/images/latent_distribution.pdf",
        "tsne_plot": ".research/iteration1/images/tsne_latent.pdf",
        "ablation_plot": ".research/iteration1/images/ablation_loss_comparison.pdf"
    }
}

TESLA_T4_OPTIMIZATION = {
    "memory_fraction": 0.9,
    "allow_growth": True,
    "batch_size_limit": 128,
    "model_complexity": "medium"
}
