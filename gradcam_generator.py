"""
Grad-CAM Generator
Based on: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
via Gradient-based Localization" (ICCV 2017)
Uses pytorch-grad-cam library: https://github.com/jacobgil/pytorch-grad-cam
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from PIL import Image
import cv2


class GradCAMGenerator:
    """Generates attention heatmaps from CNNs using Grad-CAM."""
    
    def __init__(self, model_name='resnet50', device=None):
        # Auto-detect GPU if available
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load pretrained model
        self.model = self._load_model(model_name)
        self.model.eval()
        self.model.to(self.device)
        
        # Need to specify which layer to compute gradients from
        self.target_layers = self._get_target_layers(model_name)
        
        # Initialize GradCAM with the target layer
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        
        # Standard ImageNet preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Initialized {model_name} on {self.device}")
    
    def _load_model(self, model_name):
        """Load pretrained model from torchvision."""
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        elif model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Model {model_name} not supported")
        return model
    
    def _get_target_layers(self, model_name):
        """Get the last convolutional layer for each architecture."""
        if 'resnet' in model_name:
            return [self.model.layer4[-1]]
        elif 'vgg' in model_name:
            return [self.model.features[-1]]
        else:
            raise ValueError(f"Target layer not defined for {model_name}")
    
    def generate_heatmap(self, image_path, target_class=None, return_prediction=False):
        """
        Generate Grad-CAM heatmap for an image.
        Returns heatmap as 2D array normalized to [0,1], same size as original image.
        """
        # Load image (can pass path or PIL Image)
        if isinstance(image_path, str):
            rgb_img = Image.open(image_path).convert('RGB')
        else:
            rgb_img = image_path
        
        original_size = rgb_img.size  # need this to resize heatmap back
        
        # Preprocess for model
        input_tensor = self.preprocess(rgb_img).unsqueeze(0).to(self.device)
        
        # Get model prediction
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                pred_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, pred_class].item()
        else:
            pred_class = target_class
            with torch.no_grad():
                output = self.model(input_tensor)
                confidence = F.softmax(output, dim=1)[0, pred_class].item()
        
        # Generate Grad-CAM for the predicted class
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        heatmap = grayscale_cam[0, :]
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, original_size)
        
        if return_prediction:
            prediction_info = {
                'class_idx': pred_class,
                'confidence': confidence
            }
            return heatmap_resized, prediction_info
        
        return heatmap_resized


if __name__ == "__main__":
    # Quick test
    generator = GradCAMGenerator(model_name='resnet50')