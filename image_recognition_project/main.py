#!/usr/bin/env python3
"""
Image Recognition with PyTorch

This script demonstrates how to use pre-trained models for image recognition.
It supports both AlexNet and ResNet101 models.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Union
import torch
from PIL import Image

# Import local modules
from models import AlexNetModel, ResNetModel
from utils import load_image, load_imagenet_classes

# Model mapping
MODEL_MAP = {
    'alexnet': AlexNetModel,
    'resnet101': ResNetModel
}

class ImageRecognizer:
    """Main class for image recognition using pre-trained models."""
    
    def __init__(self, model_name: str = 'resnet101', device: str = None):
        """
        Initialize the image recognizer with the specified model.
        
        Args:
            model_name (str): Name of the model to use ('alexnet' or 'resnet101')
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). 
                                  Auto-detects if None.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if model_name not in MODEL_MAP:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(MODEL_MAP.keys())}")
            
        self.model_name = model_name
        self.model = MODEL_MAP[model_name](device=self.device)
        
        # Load ImageNet class names
        script_dir = os.path.dirname(os.path.abspath(__file__))
        classes_path = os.path.join(script_dir, 'data', 'imagenet_classes.txt')
        self.classes = load_imagenet_classes(classes_path)
    
    def predict(self, image_path: str, topk: int = 5) -> Dict:
        """
        Make a prediction on an image.
        
        Args:
            image_path (str): Path or URL to the image
            topk (int): Number of top predictions to return
            
        Returns:
            dict: Prediction results including top classes and confidence scores
        """
        start_time = time.time()
        
        try:
            # Load and preprocess the image
            image = load_image(image_path)
            image_tensor = self.model.preprocess_image(image)
            
            # Make prediction
            probs, indices = self.model.predict(image_tensor, topk=topk)
            
            # Prepare results
            result = {
                'predicted_class': self.classes[indices[0]],
                'confidence': float(probs[0]) * 100,
                'top_predictions': [
                    {
                        'class': self.classes[idx],
                        'confidence': float(prob) * 100
                    }
                    for prob, idx in zip(probs, indices)
                ],
                'model': self.model_name,
                'device': str(self.device),
                'processing_time': time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Image Recognition with PyTorch')
    parser.add_argument('image', type=str, help='Path or URL to the image')
    parser.add_argument('--model', type=str, default='resnet101',
                       choices=['alexnet', 'resnet101'],
                       help='Model to use for prediction')
    parser.add_argument('--topk', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to run the model on (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the output as JSON')
    
    args = parser.parse_args()
    
    # Initialize the recognizer
    try:
        recognizer = ImageRecognizer(model_name=args.model, device=args.device)
        print(f"Using {args.model} on {recognizer.device}")
        
        # Make prediction
        result = recognizer.predict(args.image, topk=args.topk)
        
        # Print results
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nPredicted class: {result['predicted_class']} ({result['confidence']:.2f}%)")
            print("\nTop predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"{i}. {pred['class']}: {pred['confidence']:.2f}%")
            
            print(f"\nProcessing time: {result['processing_time']:.4f} seconds")
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {args.output}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
