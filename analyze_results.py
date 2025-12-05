"""
Results Analysis Script
Analyzes patterns in the collected metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('results/metrics.csv')

print("="*70)
print("DETAILED RESULTS ANALYSIS")
print("="*70)
print()

# Overall statistics
print("Overall Statistics:")
print(df[['pearson_correlation', 'spearman_correlation', 'similarity']].describe())
print()

# Find high agreement cases (correlation > 0.6)
high_corr = df[df['pearson_correlation'] > 0.6].sort_values('pearson_correlation', ascending=False)
print(f"High Agreement Cases (correlation > 0.6): {len(high_corr)}")
if len(high_corr) > 0:
    print("Top 5:")
    print(high_corr[['image_name', 'pearson_correlation']].head())
print()

# Find low agreement cases (correlation < 0.2)
low_corr = df[df['pearson_correlation'] < 0.2].sort_values('pearson_correlation')
print(f"Low Agreement Cases (correlation < 0.2): {len(low_corr)}")
if len(low_corr) > 0:
    print("Bottom 5:")
    print(low_corr[['image_name', 'pearson_correlation']].head())
print()

# Create distribution plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['pearson_correlation'], bins=15, edgecolor='black', alpha=0.7)
axes[0].axvline(df['pearson_correlation'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {df['pearson_correlation'].mean():.3f}")
axes[0].set_xlabel('Pearson Correlation')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Pearson Correlation')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(df['spearman_correlation'], bins=15, edgecolor='black', alpha=0.7, color='orange')
axes[1].axvline(df['spearman_correlation'].mean(), color='red', linestyle='--',
                linewidth=2, label=f"Mean: {df['spearman_correlation'].mean():.3f}")
axes[1].set_xlabel('Spearman Correlation')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Spearman Correlation')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].hist(df['similarity'], bins=15, edgecolor='black', alpha=0.7, color='green')
axes[2].axvline(df['similarity'].mean(), color='red', linestyle='--',
                linewidth=2, label=f"Mean: {df['similarity'].mean():.3f}")
axes[2].set_xlabel('Similarity (SIM)')
axes[2].set_ylabel('Count')
axes[2].set_title('Distribution of Similarity Metric')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/metric_distributions.png', dpi=150, bbox_inches='tight')
print("Saved distribution plot to: results/metric_distributions.png")
plt.show()