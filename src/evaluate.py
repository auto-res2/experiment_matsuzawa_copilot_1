import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def extract_latent_representations(model, dataloader, device, max_batches=10):
    """
    Extract latent representations from model.
    
    Args:
        model: Model to extract representations from
        dataloader: Data loader
        device: Device to run on
        max_batches: Maximum number of batches to process
    
    Returns:
        Tuple of (pre_diffusion_latents, post_diffusion_latents)
    """
    model.eval()
    pre_diffusion_list, post_diffusion_list = [], []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            images = images.to(device)
            latent = model.netQ(images)
            pre_diffusion_list.append(latent.cpu().numpy())
            
            diffused_latent = latent + 0.1 * torch.randn_like(latent)
            post_diffusion_list.append(diffused_latent.cpu().numpy())
    
    pre_latents = np.concatenate(pre_diffusion_list, axis=0)
    post_latents = np.concatenate(post_diffusion_list, axis=0)
    
    return pre_latents, post_latents

def plot_training_losses(losses_dict, save_path):
    """
    Plot training loss curves and save as PDF.
    
    Args:
        losses_dict: Dictionary with model names as keys and loss lists as values
        save_path: Path to save the PDF plot
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, losses in losses_dict.items():
        plt.plot(losses, label=model_name)
    
    plt.xlabel("Iteration")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.title("Training Loss Comparison")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches="tight", format='pdf')
    plt.close()
    print(f"Training loss plot saved as {save_path}")

def plot_latent_distributions(pre_latents, post_latents, save_path):
    """
    Plot latent weight distributions and save as PDF.
    
    Args:
        pre_latents: Pre-diffusion latent representations
        post_latents: Post-diffusion latent representations
        save_path: Path to save the PDF plot
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(pre_latents.flatten(), bins=50, kde=True)
    plt.title("Pre-Diffusion Latent Weights Distribution")
    
    plt.subplot(1, 2, 2)
    sns.histplot(post_latents.flatten(), bins=50, kde=True)
    plt.title("Post-Diffusion Latent Weights Distribution")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", format='pdf')
    plt.close()
    print(f"Latent distribution plot saved as {save_path}")

def plot_tsne_latent_space(latents, save_path):
    """
    Create t-SNE visualization of latent space and save as PDF.
    
    Args:
        latents: Latent representations to visualize
        save_path: Path to save the PDF plot
    """
    tsne = TSNE(n_components=2, random_state=0)
    latent_2d = tsne.fit_transform(latents)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=20)
    plt.title("t-SNE of Post-Diffusion Latent Space")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches="tight", format='pdf')
    plt.close()
    print(f"t-SNE latent space plot saved as {save_path}")

def plot_ablation_comparison(loss_dict, save_path):
    """
    Plot ablation study comparison and save as PDF.
    
    Args:
        loss_dict: Dictionary with variant names and their losses
        save_path: Path to save the PDF plot
    """
    plt.figure(figsize=(10, 6))
    
    for variant_name, losses in loss_dict.items():
        plt.plot(losses, label=variant_name, linewidth=2)
    
    plt.xlabel("Batch Iteration")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.title("Ablation Study: Loss Comparison")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches="tight", format='pdf')
    plt.close()
    print(f"Ablation study plot saved as {save_path}")

def evaluate_model_performance(model, dataloader, criterion, device, max_batches=10):
    """
    Evaluate model performance on given dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        criterion: Loss criterion
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate
    
    Returns:
        Average loss over the evaluation set
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0
