import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size=64, train=True, download=True):
    """
    Get MNIST dataloader with normalization.
    
    Args:
        batch_size: Batch size for the dataloader
        train: Whether to load training or test set
        download: Whether to download the dataset if not present
    
    Returns:
        DataLoader for MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=train, 
        download=download, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train
    )
    
    return dataloader

def add_noise(images, noise_std=0.1):
    """
    Add Gaussian noise to images for robustness testing.
    
    Args:
        images: Input tensor of images
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Noisy images tensor
    """
    noise = torch.randn_like(images) * noise_std
    return images + noise

def prepare_data_for_experiments(device, batch_size=64):
    """
    Prepare data loaders for all experiments.
    
    Args:
        device: Device to load data on
        batch_size: Batch size for data loaders
    
    Returns:
        Dictionary containing train and test data loaders
    """
    train_loader = get_mnist_dataloader(batch_size=batch_size, train=True)
    test_loader = get_mnist_dataloader(batch_size=batch_size, train=False)
    
    return {
        'train': train_loader,
        'test': test_loader
    }
