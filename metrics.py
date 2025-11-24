"""
Comparison Metrics for Saliency Maps
Quantifies similarity between human and model attention heatmaps.

Key metrics:
- Pearson/Spearman correlation: measure linear/rank relationship
- Similarity (SIM): measures distribution overlap
- KL/JS divergence: measures distribution difference
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import cv2


def pearson_correlation(heatmap1, heatmap2):
    """
    Pearson correlation between two heatmaps.
    Measures linear relationship - higher means more similar attention patterns.
    
    Returns:
        correlation: float in [-1, 1], higher is more similar
        p_value: statistical significance
    """
    # Resize if needed
    if heatmap1.shape != heatmap2.shape:
        heatmap2 = cv2.resize(heatmap2, (heatmap1.shape[1], heatmap1.shape[0]))
    
    # Flatten to 1D arrays
    h1_flat = heatmap1.flatten()
    h2_flat = heatmap2.flatten()
    
    correlation, p_value = stats.pearsonr(h1_flat, h2_flat)
    
    return correlation, p_value


def spearman_correlation(heatmap1, heatmap2):
    """
    Spearman rank correlation between two heatmaps.
    Less sensitive to outliers than Pearson, good for non-linear relationships.
    
    Returns:
        correlation: float in [-1, 1], higher is more similar
        p_value: statistical significance
    """
    if heatmap1.shape != heatmap2.shape:
        heatmap2 = cv2.resize(heatmap2, (heatmap1.shape[1], heatmap1.shape[0]))
    
    h1_flat = heatmap1.flatten()
    h2_flat = heatmap2.flatten()
    
    correlation, p_value = stats.spearmanr(h1_flat, h2_flat)
    
    return correlation, p_value


def similarity_metric(heatmap1, heatmap2):
    """
    Similarity (SIM) metric commonly used in saliency evaluation.
    Treats heatmaps as probability distributions and measures overlap.
    
    Returns:
        similarity: float in [0, 1], higher means more overlap
    """
    if heatmap1.shape != heatmap2.shape:
        heatmap2 = cv2.resize(heatmap2, (heatmap1.shape[1], heatmap1.shape[0]))
    
    # Normalize to probability distributions (sum to 1)
    h1_norm = heatmap1 / (heatmap1.sum() + 1e-10)
    h2_norm = heatmap2 / (heatmap2.sum() + 1e-10)
    
    # Similarity is sum of pointwise minimums
    similarity = np.minimum(h1_norm, h2_norm).sum()
    
    return similarity


def kl_divergence(heatmap1, heatmap2):
    """
    Kullback-Leibler divergence from heatmap1 to heatmap2.
    Measures how different heatmap2 is from heatmap1.
    
    Returns:
        kl_div: float in [0, inf], lower means more similar
    """
    if heatmap1.shape != heatmap2.shape:
        heatmap2 = cv2.resize(heatmap2, (heatmap1.shape[1], heatmap1.shape[0]))
    
    h1_flat = heatmap1.flatten()
    h2_flat = heatmap2.flatten()
    
    # Normalize and add small epsilon to avoid log(0)
    epsilon = 1e-10
    h1_norm = (h1_flat + epsilon) / (h1_flat.sum() + epsilon)
    h2_norm = (h2_flat + epsilon) / (h2_flat.sum() + epsilon)
    
    kl_div = np.sum(h1_norm * np.log(h1_norm / h2_norm))
    
    return kl_div


def jensen_shannon_divergence(heatmap1, heatmap2):
    """
    Jensen-Shannon divergence - symmetric version of KL divergence.
    Bounded between 0 and 1, easier to interpret.
    
    Returns:
        js_div: float in [0, 1], lower means more similar
    """
    if heatmap1.shape != heatmap2.shape:
        heatmap2 = cv2.resize(heatmap2, (heatmap1.shape[1], heatmap1.shape[0]))
    
    h1_flat = heatmap1.flatten()
    h2_flat = heatmap2.flatten()
    
    epsilon = 1e-10
    h1_norm = (h1_flat + epsilon) / (h1_flat.sum() + epsilon)
    h2_norm = (h2_flat + epsilon) / (h2_flat.sum() + epsilon)
    
    js_div = jensenshannon(h1_norm, h2_norm)
    
    return js_div


def compute_all_metrics(human_heatmap, model_heatmap):
    """
    Compute all metrics between human and model attention maps.
    Returns dict with all scores for easy comparison.
    """
    metrics = {}
    
    # Correlation metrics
    pearson_corr, pearson_p = pearson_correlation(human_heatmap, model_heatmap)
    metrics['pearson_correlation'] = pearson_corr
    metrics['pearson_p_value'] = pearson_p
    
    spearman_corr, spearman_p = spearman_correlation(human_heatmap, model_heatmap)
    metrics['spearman_correlation'] = spearman_corr
    metrics['spearman_p_value'] = spearman_p
    
    # Similarity metrics
    metrics['similarity'] = similarity_metric(human_heatmap, model_heatmap)
    metrics['kl_divergence'] = kl_divergence(human_heatmap, model_heatmap)
    metrics['js_divergence'] = jensen_shannon_divergence(human_heatmap, model_heatmap)
    
    return metrics


if __name__ == "__main__":
    # Quick check with random data
    import numpy as np
    h1 = np.random.rand(100, 100)
    h2 = np.random.rand(100, 100)
    
    metrics = compute_all_metrics(h1, h2)
    print("Metrics computed on random data:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")