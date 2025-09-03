import torch
import torch.nn as nn
from torchvision import models

class ResNetModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize ResNet101 model with pretrained weights.
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.model = models.resnet101(weights='DEFAULT')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store the preprocessing parameters
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        
        # Load ImageNet class names
        import os
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        classes_path = os.path.join(script_dir, 'data', 'imagenet_classes.txt')
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
    def preprocess_image(self, pil_image):
        """
        Preprocess image for ResNet101 model.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        
        return preprocess(pil_image).unsqueeze(0).to(self.device)
    
    def predict(self, image_tensor, topk=5):
        """
        Make prediction on a single image.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            topk (int): Number of top predictions to return
            
        Returns:
            tuple: (probabilities, class_indices)
        """
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, topk)
        
        return top_probs.cpu().numpy(), top_indices.cpu().numpy()
