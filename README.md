# Large-scale Receptive Field to Saliency Correlation Study

## Abstract

My project looks at human visual attention patterns and compares it to deep neural network attention patterns. I used the SALICON dataset (Jiang et al., 2015) for human eye tracking saliency maps, and for deep neural networks I used Gradient-weighted Class Activation Mapping -- Grad-CAM -- to see attention from a pre-trained ResNet50 CNN model. I ran this for 50 images, which could be variable amount by adjusting the code, and it shows moderate correlation between human and model attention, with stronger agreement on text containing images, and divergence on textured or multi-object scenes.

## Problem and Approach

**Problem:** Do deep learning models attend to images the same way humans do?

**Solution:** 
1. Load images and human saliency maps from SALICON dataset
2. Generate model attention using Grad-CAM on a pre-trained ResNet50
3. Compute similarity metrics using Pearson correlation, Spearman correlation, and similarity index
4. Visualize (and analyze) patterns of agreement or disagreement

## Framework and Implementation

**Frameworks:**
- PyTorch 2.0+ for deep learning framework
- pytorch-grad-cam library for Grad-CAM implementation
- OpenCV, matplotlib for visualization
- NumPy, SciPy, pandas for analysis

**Code Base:**
- Pre-trained ResNet50 from PyTorch torchvision model zoo
- Grad-CAM implementation from pytorch-grad-cam library by Jacob Gildenblat
- Custom modules for dataset loading, metrics computation, and visualization

## Results

### Quantitative Summary

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Pearson Correlation | 0.288 | 0.226 | -0.251 | 0.702 |
| Spearman Correlation | 0.394 | 0.239 | -0.217 | 0.709 |
| Similarity | 0.439 | 0.127 | 0.118 | 0.645 |

### Distribution of Metrics

![Metric Distributions](results/metric_distributions.png)

### Example Comparisons

**High Agreement (Pearson r = 0.70)**  
Human and model strongly attend to text and branding on the truck.

![High Agreement Example](results/COCO_val2014_000000001799_comparison.png)

**Medium Agreement (Pearson r = 0.62)**  
Model looks at multiple figures almost evenly, humans mostly focus on main figure.

![Medium Agreement Example](results/COCO_val2014_000000000192_comparison.png)

**Low Agreement (Pearson r = -0.25)**  
Model attention goes across texture and edges, humans focus on center and specific toppings.

![Low Agreement Example](https://github.com/mclopton-fau/CAP6415_F25_project-large-scale-receptive-field-saliency-correlation/blob/main/results/COCO_val2014_000000000397_comparison.png)

### Key Findings

- **High agreement (r>0.6):** Images with text and branding (8% of images)
- **Low agreement (r<0.2):** Textured or complex scenes with many objects (32% of images)
- **Pattern:** Model attention tends to be more spread evenly; human attention more focused on specific objects
- **Text advantage:** Both humans and CNNs strongly attend to readable text in images

## References

**Dataset:**  
Jiang, M., Huang, S., Duan, J., & Zhao, Q. (2015). SALICON: Saliency in Context. *CVPR 2015*.  
Dataset: http://salicon.net/challenge-2017/

**Grad-CAM:**  
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.  
Implementation: https://github.com/jacobgil/pytorch-grad-cam

**Pre-trained Model:**  
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

**Metrics:**  
Bylinskii, Z., Judd, T., Oliva, A., Torralba, A., & Durand, F. (2018). What do different evaluation metrics tell us about saliency models? *TPAMI 2018*.

## Installation and Usage

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~5GB disk space for dataset

### Setup Instructions

1. **Clone repository:**
```bash
git clone https://github.com/YOUR_USERNAME/CAP6415_F25_project-Saliency-Correlation-Study.git
cd CAP6415_F25_project-Saliency-Correlation-Study
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download SALICON dataset:**
   - Visit: http://salicon.net/challenge-2017/
   - Download **"Images (3.0G)"**: [Direct Link](https://drive.google.com/uc?id=1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5&export=download)
   - Download **"Fixation Maps (0.4G)"**: [Direct Link](https://drive.google.com/uc?id=1PnO7szbdub1559LfjYHMy65EDC4VhJC8&export=download)
   - Extract both downloads
   - Organize as:
```
     data/salicon/
     ├── images/val/  (place extracted images here)
     └── maps/val/    (place extracted fixation maps here)
```

4. **Run analysis:**
```bash
python compare_attention.py
```

This processes 50 images and generates:
- Comparison visualizations in `results/`
- Metrics CSV at `results/metrics.csv`
- Summary statistics in console

5. **Analyze results:**
```bash
python analyze_results.py
```


Generates distribution plots and identifies high/low agreement cases.





