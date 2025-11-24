"""
Main Analysis Script
Compares human attention (SALICON) with model attention (Grad-CAM).

Usage:
    python compare_attention.py
"""

import os
import numpy as np
import pandas as pd
from gradcam_generator import GradCAMGenerator
from salicon_loader import SALICONDataset
import visualization as viz
import metrics as met


def process_single_image(dataset, gradcam, idx, save_dir='results'):
    """
    Process one image: load data, generate heatmaps, compute metrics, visualize.
    
    Args:
        dataset: SALICONDataset instance
        gradcam: GradCAMGenerator instance
        idx: Image index to process
        save_dir: Where to save results
    
    Returns:
        Dictionary with metrics for this image
    """
    # Load image and human saliency
    image, human_heatmap = dataset.get_pair(idx)
    
    # Generate model attention heatmap
    model_heatmap = gradcam.generate_heatmap(image)
    
    # Compute comparison metrics
    metrics_dict = met.compute_all_metrics(human_heatmap, model_heatmap)
    
    # Add image info
    metrics_dict['image_idx'] = idx
    metrics_dict['image_name'] = dataset.image_files[idx]
    
    # Save visualization
    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.splitext(dataset.image_files[idx])[0]
    save_path = os.path.join(save_dir, f'{img_name}_comparison.png')
    
    viz.plot_comparison(image, human_heatmap, model_heatmap,
                       title=f"Comparison: {img_name}",
                       save_path=save_path)
    
    print(f"Processed {idx}: {dataset.image_files[idx]}")
    print(f"  Pearson correlation: {metrics_dict['pearson_correlation']:.3f}")
    
    return metrics_dict


def run_analysis(n_images=10, data_root='data/salicon', output_dir='results'):
    """
    Run analysis on multiple images.
    
    Args:
        n_images: Number of images to process
        data_root: Path to SALICON dataset
        output_dir: Where to save results
    """
    print("="*80)
    print("SALIENCY COMPARISON ANALYSIS")
    print("="*80)
    print()
    
    # Initialize dataset and model
    print("Loading SALICON dataset...")
    dataset = SALICONDataset(data_root, split='val')
    
    print("Initializing Grad-CAM (ResNet50)...")
    gradcam = GradCAMGenerator(model_name='resnet50')
    print()
    
    # Process images
    print(f"Processing {n_images} images...")
    print()
    
    all_metrics = []
    
    for i in range(n_images):
        metrics_dict = process_single_image(dataset, gradcam, i, output_dir)
        all_metrics.append(metrics_dict)
        print()
    
    # Save metrics to CSV
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_dir, 'metrics.csv')
    df.to_csv(csv_path, index=False)
    
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()
    print(df[['pearson_correlation', 'spearman_correlation', 'similarity']].describe())
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    # Run on 10 images to start
    run_analysis(n_images=10, output_dir='results')