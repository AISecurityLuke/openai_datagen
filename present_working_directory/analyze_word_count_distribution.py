#!/usr/bin/env python3
"""
Analyze word count distribution to ensure proper bell curve
"""

import numpy as np
import matplotlib.pyplot as plt
from ripit import DatasetGenerator
from collections import defaultdict

def analyze_word_count_distribution():
    """Analyze the word count distribution"""
    print("📊 Analyzing Word Count Distribution")
    print("=" * 50)
    
    # Create generator
    dg = DatasetGenerator()
    
    # Generate many word count targets
    num_samples = 10000
    word_counts = []
    
    print(f"Generating {num_samples} word count targets...")
    for _ in range(num_samples):
        # Test each label
        for label in [0, 1, 2]:
            target = dg._calculate_target_word_count(label)
            word_counts.append(target)
    
    # Analyze distribution
    word_counts = np.array(word_counts)
    
    print(f"\nDistribution Statistics:")
    print(f"  Mean: {np.mean(word_counts):.1f}")
    print(f"  Median: {np.median(word_counts):.1f}")
    print(f"  Std: {np.std(word_counts):.1f}")
    print(f"  Min: {np.min(word_counts)}")
    print(f"  Max: {np.max(word_counts)}")
    print(f"  Range: {np.max(word_counts) - np.min(word_counts)}")
    
    # Check normality
    from scipy import stats
    shapiro_stat, shapiro_p = stats.shapiro(word_counts)
    print(f"\nNormality Test (Shapiro-Wilk):")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  P-value: {shapiro_p:.4f}")
    print(f"  Normal distribution: {'✅' if shapiro_p > 0.05 else '❌'}")
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Main histogram
    plt.subplot(2, 2, 1)
    plt.hist(word_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(word_counts), color='red', linestyle='--', label=f'Mean: {np.mean(word_counts):.1f}')
    plt.axvline(np.median(word_counts), color='green', linestyle='--', label=f'Median: {np.median(word_counts):.1f}')
    plt.xlabel('Word Count Target')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot for normality
    plt.subplot(2, 2, 2)
    stats.probplot(word_counts, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal Distribution)')
    plt.grid(True, alpha=0.3)
    
    # Distribution by label
    plt.subplot(2, 2, 3)
    for label in [0, 1, 2]:
        label_counts = []
        for _ in range(1000):
            target = dg._calculate_target_word_count(label)
            label_counts.append(target)
        plt.hist(label_counts, bins=20, alpha=0.6, label=f'Label {label}')
    
    plt.xlabel('Word Count Target')
    plt.ylabel('Frequency')
    plt.title('Distribution by Label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(2, 2, 4)
    sorted_counts = np.sort(word_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, cumulative, 'b-', linewidth=2)
    plt.xlabel('Word Count Target')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('word_count_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to word_count_distribution_analysis.png")
    
    # Check percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(word_counts, p)
        print(f"  {p}th percentile: {value:.1f}")
    
    # Check if distribution is centered around target
    target_mean = 102
    actual_mean = np.mean(word_counts)
    deviation = abs(actual_mean - target_mean)
    print(f"\nTarget vs Actual Mean:")
    print(f"  Target: {target_mean}")
    print(f"  Actual: {actual_mean:.1f}")
    print(f"  Deviation: {deviation:.1f} words")
    print(f"  Centered: {'✅' if deviation < 2 else '❌'}")
    
    return word_counts

if __name__ == "__main__":
    word_counts = analyze_word_count_distribution() 