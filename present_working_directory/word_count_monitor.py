#!/usr/bin/env python3
"""
Word Count Monitor - Analyze and ensure balanced word count distributions
"""

import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_word_counts(filename: str = "generated_prompts.jsonl"):
    """Analyze word count distributions by label"""
    
    # Load data
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Analyze word counts by label
    word_counts = defaultdict(list)
    for item in data:
        word_count = len(item['text'].split())
        word_counts[item['label']].append(word_count)
    
    print("WORD COUNT ANALYSIS BY LABEL:")
    print("=" * 50)
    
    stats = {}
    for label in sorted(word_counts.keys()):
        counts = word_counts[label]
        stats[label] = {
            'count': len(counts),
            'mean': np.mean(counts),
            'median': np.median(counts),
            'min': min(counts),
            'max': max(counts),
            'std': np.std(counts)
        }
        
        print(f'Label {label}:')
        print(f'  Count: {stats[label]["count"]}')
        print(f'  Mean: {stats[label]["mean"]:.1f}')
        print(f'  Median: {stats[label]["median"]:.1f}')
        print(f'  Min: {stats[label]["min"]}')
        print(f'  Max: {stats[label]["max"]}')
        print(f'  Std: {stats[label]["std"]:.1f}')
        print()
    
    # Check balance
    means = [stats[label]['mean'] for label in sorted(stats.keys())]
    medians = [stats[label]['median'] for label in sorted(stats.keys())]
    
    mean_range = max(means) - min(means)
    median_range = max(medians) - min(medians)
    
    print("BALANCE ASSESSMENT:")
    print("=" * 50)
    print(f"Mean range: {mean_range:.1f} words")
    print(f"Median range: {median_range:.1f} words")
    
    if mean_range <= 15:
        print("✅ Mean balance: EXCELLENT (≤15 words)")
    elif mean_range <= 25:
        print("⚠️  Mean balance: GOOD (≤25 words)")
    else:
        print("❌ Mean balance: NEEDS IMPROVEMENT (>25 words)")
    
    if median_range <= 15:
        print("✅ Median balance: EXCELLENT (≤15 words)")
    elif median_range <= 25:
        print("⚠️  Median balance: GOOD (≤25 words)")
    else:
        print("❌ Median balance: NEEDS IMPROVEMENT (>25 words)")
    
    # Check overlap
    print("\nOVERLAP ANALYSIS:")
    print("=" * 50)
    labels = sorted(word_counts.keys())
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            counts1 = word_counts[label1]
            counts2 = word_counts[label2]
            
            # Calculate overlap ranges
            min1, max1 = min(counts1), max(counts1)
            min2, max2 = min(counts2), max(counts2)
            
            overlap_min = max(min1, min2)
            overlap_max = min(max1, max2)
            overlap = max(0, overlap_max - overlap_min + 1)
            
            overlap_pct1 = overlap/len(counts1)*100
            overlap_pct2 = overlap/len(counts2)*100
            
            print(f'Labels {label1} vs {label2}:')
            print(f'  Range {label1}: {min1}-{max1}')
            print(f'  Range {label2}: {min2}-{max2}')
            print(f'  Overlap: {overlap} values ({overlap_min}-{overlap_max})')
            print(f'  Overlap %: {overlap_pct1:.1f}% of label {label1}, {overlap_pct2:.1f}% of label {label2}')
            
            if overlap_pct1 >= 40 and overlap_pct2 >= 40:
                print(f'  ✅ Good overlap')
            elif overlap_pct1 >= 30 and overlap_pct2 >= 30:
                print(f'  ⚠️  Moderate overlap')
            else:
                print(f'  ❌ Poor overlap')
            print()
    
    return stats, word_counts

def create_word_count_visualization(word_counts, filename: str = "word_count_analysis.png"):
    """Create visualization of word count distributions"""
    
    # Prepare data for plotting
    all_data = []
    labels = []
    for label in sorted(word_counts.keys()):
        all_data.extend(word_counts[label])
        labels.extend([f'Label {label}'] * len(word_counts[label]))
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Box plot
    data_for_box = [word_counts[label] for label in sorted(word_counts.keys())]
    ax1.boxplot(data_for_box, labels=[f'Label {label}' for label in sorted(word_counts.keys())])
    ax1.set_title('Word Count Distribution by Label')
    ax1.set_ylabel('Word Count')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram
    for label in sorted(word_counts.keys()):
        ax2.hist(word_counts[label], alpha=0.6, label=f'Label {label}', bins=20)
    ax2.set_title('Word Count Histograms')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Violin plot
    data_for_violin = []
    labels_for_violin = []
    for label in sorted(word_counts.keys()):
        data_for_violin.extend(word_counts[label])
        labels_for_violin.extend([f'Label {label}'] * len(word_counts[label]))
    
    import pandas as pd
    df = pd.DataFrame({'Word Count': data_for_violin, 'Label': labels_for_violin})
    sns.violinplot(data=df, x='Label', y='Word Count', ax=ax3)
    ax3.set_title('Word Count Density by Label')
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics table
    stats_text = "Word Count Statistics:\n\n"
    for label in sorted(word_counts.keys()):
        counts = word_counts[label]
        stats_text += f"Label {label}:\n"
        stats_text += f"  Mean: {np.mean(counts):.1f}\n"
        stats_text += f"  Median: {np.median(counts):.1f}\n"
        stats_text += f"  Std: {np.std(counts):.1f}\n"
        stats_text += f"  Range: {min(counts)}-{max(counts)}\n\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    ax4.set_title('Summary Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
    
    return fig

if __name__ == "__main__":
    stats, word_counts = analyze_word_counts()
    create_word_count_visualization(word_counts) 