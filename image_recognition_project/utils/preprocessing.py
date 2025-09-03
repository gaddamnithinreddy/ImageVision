import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def get_transform():
    """
    Get the standard transformation pipeline for ImageNet models.
    
    Returns:
        torchvision.transforms.Compose: Transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(pil_image, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Preprocess a PIL image for model inference.
    
    Args:
        pil_image (PIL.Image): Input image
        device (str): Device to move the tensor to
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = get_transform()
    image_tensor = transform(pil_image)
    return image_tensor.unsqueeze(0).to(device)

def denormalize(tensor, mean=None, std=None):
    """
    Denormalize a tensor image with mean and std.
    
    Args:
        tensor (torch.Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (tuple): Mean for each channel.
        std (tuple): Standard deviation for each channel.
        
    Returns:
        torch.Tensor: Denormalized image.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean

def load_imagenet_classes(filepath='data/imagenet_classes.txt'):
    """
    Load ImageNet class names from a file.
    
    Args:
        filepath (str): Path to the file containing class names
        
    Returns:
        list: List of class names
    """
    with open(filepath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
