"""
Visualization Tools
Creates comparison plots of human vs model attention.

Reference: Standard matplotlib/OpenCV visualization techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def normalize_heatmap(heatmap):
    """Normalize heatmap to [0, 1] range for consistent visualization."""
    heatmap = heatmap.astype(np.float32)
    if heatmap.max() > heatmap.min():
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap


def apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    """
    Apply colormap to grayscale heatmap.
    Jet colormap: blue (low attention) -> red (high attention)
    """
    # Convert to 0-255 range
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert BGR to RGB for matplotlib
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored_rgb


def create_overlay(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image with transparency.
    
    Args:
        image: Original image (PIL Image or numpy array)
        heatmap: Attention heatmap (H, W) in [0, 1]
        alpha: Transparency (0=invisible, 1=opaque)
    
    Returns:
        Overlay image as numpy array
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize heatmap to match image if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize and apply colormap
    heatmap_norm = normalize_heatmap(heatmap)
    heatmap_colored = apply_colormap(heatmap_norm, colormap)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay


def plot_comparison(image, human_heatmap, model_heatmap, 
                   title="Human vs Model Attention", 
                   save_path=None):
    """
    Create side-by-side comparison: original image, human attention, model attention.
    
    Args:
        image: Original image
        human_heatmap: Human saliency map from SALICON
        model_heatmap: Model attention from Grad-CAM
        title: Plot title
        save_path: If provided, saves figure instead of showing
    """
    # Convert image if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Normalize heatmaps
    human_norm = normalize_heatmap(human_heatmap)
    model_norm = normalize_heatmap(model_heatmap)
    
    # Create overlays
    human_overlay = create_overlay(image, human_norm, alpha=0.5)
    model_overlay = create_overlay(image, model_norm, alpha=0.5)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(human_overlay)
    axes[1].set_title('Human Attention')
    axes[1].axis('off')
    
    axes[2].imshow(model_overlay)
    axes[2].set_title('Model Attention (Grad-CAM)')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_heatmaps_only(human_heatmap, model_heatmap, 
                       title="Attention Heatmaps",
                       save_path=None):
    """
    Plot just the heatmaps side-by-side without image overlay.
    Useful for seeing raw attention patterns.
    """
    human_norm = normalize_heatmap(human_heatmap)
    model_norm = normalize_heatmap(model_heatmap)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = axes[0].imshow(human_norm, cmap='jet', vmin=0, vmax=1)
    axes[0].set_title('Human Attention')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(model_norm, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Model Attention')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()