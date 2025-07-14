#!/usr/bin/env python3
"""
Cleaned Dataset Visualization Script
Creates clear, actionable plots for the generated prompts dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from pathlib import Path

# --- Data Loading & Helpers ---
def load_data(filepath='generated_prompts.jsonl'):
    df = pd.read_json(filepath, lines=True)
    return df

def add_helper_columns(df):
    df['word_count'] = df['text'].str.split().str.len()
    df['text_hash'] = df['text'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    seen_hashes = set()
    duplicate_flags = []
    for text_hash in df['text_hash']:
        if text_hash in seen_hashes:
            duplicate_flags.append(1)
        else:
            duplicate_flags.append(0)
            seen_hashes.add(text_hash)
    df['is_duplicate'] = duplicate_flags
    return df

def create_visuals_directory():
    visuals_dir = Path('visuals')
    visuals_dir.mkdir(exist_ok=True)
    return visuals_dir

# --- Plots ---
def plot_label_language_heatmap(df, visuals_dir):
    crosstab = pd.crosstab(df['label'], df['lang'], normalize='all')
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Proportion of Dataset')
    ax.set_xlabel('Language')
    ax.set_ylabel('Label')
    ax.set_title('Label × Language Distribution')
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            ax.text(j, i, f'{crosstab.iloc[i, j]:.2f}', ha="center", va="center", color="black", fontsize=10)
    plt.tight_layout()
    plt.savefig(visuals_dir / 'label_language_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_tone_coverage(df, visuals_dir):
    tone_counts = df['tone'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(tone_counts.index, tone_counts.values, color='skyblue', edgecolor='navy', alpha=0.8)
    ax.set_xlabel('Tone')
    ax.set_ylabel('Number of Prompts')
    ax.set_title('Prompt Count by Tone')
    ax.set_xticklabels(tone_counts.index, rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{int(height)}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(visuals_dir / 'tone_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_word_count_boxplot(df, visuals_dir):
    labels = sorted(df['label'].unique())
    data = [df[df['label'] == label]['word_count'] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Label')
    ax.set_ylabel('Word Count')
    ax.set_title('Word Count Distribution by Label')
    ax.axhline(y=6, color='red', linestyle='--', alpha=0.7, label='Min Expected (6)')
    ax.axhline(y=150, color='red', linestyle='--', alpha=0.7, label='Max Expected (150)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(visuals_dir / 'word_count_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_rolling_duplicate_rate(df, visuals_dir):
    df_sorted = df.sort_index().copy()
    df_sorted['cumulative_duplicates'] = df_sorted['is_duplicate'].cumsum()
    df_sorted['duplicate_rate'] = df_sorted['cumulative_duplicates'] / (df_sorted.index + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_sorted.index, df_sorted['duplicate_rate'], color='red', linewidth=2, alpha=0.8)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cumulative Duplicate Rate')
    ax.set_title('Rolling Duplicate Rate')
    ax.grid(True, alpha=0.3)
    final_rate = df_sorted['duplicate_rate'].iloc[-1]
    ax.text(0.02, 0.98, f'Final Duplicate Rate: {final_rate:.3f}', transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(visuals_dir / 'rolling_duplicate_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_trend_analysis(df, visuals_dir):
    df_with_index = df.reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 1. Word count trend
    axes[0, 0].plot(df_with_index.index, df_with_index['word_count'], alpha=0.7)
    axes[0, 0].set_title('Word Count Trend')
    axes[0, 0].set_xlabel('Generation Order')
    axes[0, 0].set_ylabel('Word Count')
    axes[0, 0].grid(True, alpha=0.3)
    # 2. Label distribution trend
    window_size = 50
    for label in sorted(df['label'].unique()):
        rolling = df_with_index['label'].rolling(window=window_size).apply(lambda x: (x == label).mean())
        axes[0, 1].plot(df_with_index.index, rolling, label=f'Label {label}', alpha=0.8)
    axes[0, 1].set_title(f'Label Distribution Trend (Rolling {window_size})')
    axes[0, 1].set_xlabel('Generation Order')
    axes[0, 1].set_ylabel('Proportion')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    # 3. Language distribution trend
    for lang in sorted(df['lang'].unique()):
        rolling = (df_with_index['lang'] == lang).rolling(window=window_size).mean()
        axes[1, 0].plot(df_with_index.index, rolling, label=f'Lang {lang}', alpha=0.8)
    axes[1, 0].set_title(f'Language Distribution Trend (Rolling {window_size})')
    axes[1, 0].set_xlabel('Generation Order')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    # 4. Model usage trend
    for model in sorted(df['model_used'].unique()):
        rolling = (df_with_index['model_used'] == model).rolling(window=window_size).mean()
        axes[1, 1].plot(df_with_index.index, rolling, label=model, alpha=0.8)
    axes[1, 1].set_title(f'Model Usage Trend (Rolling {window_size})')
    axes[1, 1].set_xlabel('Generation Order')
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(visuals_dir / 'trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_model_usage_bar(df, visuals_dir):
    model_counts = df['model_used'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(model_counts.index, model_counts.values, color=['#1f77b4', '#ff7f0e'])
    ax.set_xlabel('Model Used')
    ax.set_ylabel('Number of Prompts')
    ax.set_title('Model Usage Count')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(visuals_dir / 'model_usage_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_multi_heatmap(df, visuals_dir):
    """Multi-dimensional heatmap (Label × Language × Tone) with horizontal colorbar below all subplots"""
    # Group by label, language, and tone, then count
    heatmap_data = df.groupby(['label', 'lang', 'tone']).size().unstack(fill_value=0)
    labels = sorted(df['label'].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(20, 8), sharey=True)
    vmin = None
    vmax = None
    # Find global min/max for consistent color scale
    for label in labels:
        label_data = df[df['label'] == label]
        pivot_data = label_data.groupby(['lang', 'tone']).size().unstack(fill_value=0)
        if vmin is None or pivot_data.values.min() < vmin:
            vmin = pivot_data.values.min()
        if vmax is None or pivot_data.values.max() > vmax:
            vmax = pivot_data.values.max()
    ims = []
    for i, label in enumerate(labels):
        label_data = df[df['label'] == label]
        pivot_data = label_data.groupby(['lang', 'tone']).size().unstack(fill_value=0)
        im = axes[i].imshow(pivot_data.values, cmap='Blues', aspect='auto', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Label {label} (Language × Tone)')
        axes[i].set_xlabel('Tone')
        axes[i].set_xticks(range(len(pivot_data.columns)))
        axes[i].set_xticklabels(pivot_data.columns, rotation=45, ha='right', fontsize=8)
        axes[i].set_yticks(range(len(pivot_data.index)))
        if i == 0:
            axes[i].set_ylabel('Language')
            axes[i].set_yticklabels(pivot_data.index, fontsize=8)
        else:
            axes[i].set_yticklabels([])
        # Add text annotations
        for j in range(len(pivot_data.index)):
            for k in range(len(pivot_data.columns)):
                value = pivot_data.iloc[j, k]
                if value > 0:
                    axes[i].text(k, j, str(value), ha="center", va="center", color="black", fontsize=8, fontweight='bold')
        ims.append(im)
    # Add a single horizontal colorbar below all subplots
    cbar = fig.colorbar(ims[0], ax=axes, orientation='horizontal', pad=0.15, fraction=0.05, aspect=40)
    cbar.set_label('Count')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(visuals_dir / 'multi_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_common_trigrams(df, visuals_dir):
    """Plot common trigrams by label to identify patterns and potential cleaning needs"""
    from collections import Counter
    import re
    
    def extract_trigrams(text):
        """Extract trigrams from text"""
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        # Generate trigrams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        return trigrams
    
    # Extract trigrams by label
    trigram_counts = {label: Counter() for label in sorted(df['label'].unique())}
    
    for _, row in df.iterrows():
        label = row['label']
        text = row['text']
        trigrams = extract_trigrams(text)
        trigram_counts[label].update(trigrams)
    
    # Get top trigrams for each label
    top_trigrams_per_label = {}
    for label in trigram_counts:
        top_trigrams_per_label[label] = trigram_counts[label].most_common(15)
    
    # Create bar plots for each label
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    
    for i, label in enumerate(sorted(df['label'].unique())):
        trigrams, counts = zip(*top_trigrams_per_label[label])
        # Truncate long trigrams for readability
        trigrams_short = [t[:25] + '...' if len(t) > 25 else t for t in trigrams]
        
        bars = axes[i].barh(range(len(trigrams)), counts, color=colors[i], alpha=0.8)
        axes[i].set_yticks(range(len(trigrams)))
        axes[i].set_yticklabels(trigrams_short, fontsize=9)
        axes[i].set_xlabel('Frequency')
        axes[i].set_title(f'Top Trigrams - Label {label}')
        axes[i].invert_yaxis()  # Most frequent at top
        
        # Add value labels on bars
        for j, (bar, count) in enumerate(zip(bars, counts)):
            axes[i].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        str(count), ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'trigrams_by_label.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap of most common trigrams across all labels
    all_trigrams = set()
    for label_counts in trigram_counts.values():
        all_trigrams.update(label_counts.keys())
    
    # Get top 50 trigrams overall
    overall_counts = Counter()
    for label_counts in trigram_counts.values():
        overall_counts.update(label_counts)
    
    top_overall_trigrams = [trigram for trigram, _ in overall_counts.most_common(50)]
    
    # Create heatmap data
    heatmap_data = []
    for trigram in top_overall_trigrams:
        row = []
        for label in sorted(df['label'].unique()):
            row.append(trigram_counts[label][trigram])
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap_data, cmap='Blues', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(sorted(df['label'].unique()))))
    ax.set_xticklabels([f'Label {label}' for label in sorted(df['label'].unique())])
    ax.set_yticks(range(len(top_overall_trigrams)))
    ax.set_yticklabels([t[:30] + '...' if len(t) > 30 else t for t in top_overall_trigrams], fontsize=9)
    
    # Add text annotations
    for i in range(len(top_overall_trigrams)):
        for j in range(len(sorted(df['label'].unique()))):
            value = heatmap_data[i, j]
            if value > 0:
                ax.text(j, i, str(value), ha="center", va="center", 
                       color="black" if value < heatmap_data.max()/2 else "white", 
                       fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Label')
    ax.set_ylabel('Trigram')
    ax.set_title('Most Common Trigrams Across All Labels')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frequency')
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'trigrams_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Trigrams analysis saved to {visuals_dir}/trigrams_by_label.png and {visuals_dir}/trigrams_heatmap.png")

# --- Dashboard ---
def plot_dashboard(visuals_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    images = [
        plt.imread(visuals_dir / 'label_language_heatmap.png'),
        plt.imread(visuals_dir / 'tone_coverage.png'),
        plt.imread(visuals_dir / 'word_count_boxplot.png'),
        plt.imread(visuals_dir / 'rolling_duplicate_rate.png'),
    ]
    titles = [
        'Label × Language Heatmap',
        'Tone Coverage',
        'Word Count by Label',
        'Rolling Duplicate Rate',
    ]
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(visuals_dir / 'dashboard.png', dpi=200, bbox_inches='tight')
    plt.close()

# --- Main ---
def main():
    print("=" * 60)
    print("CLEANED DATASET VISUALIZATION SCRIPT")
    print("=" * 60)
    df = load_data()
    df = add_helper_columns(df)
    visuals_dir = create_visuals_directory()
    print("\nGenerating Key Visualizations...")
    plot_label_language_heatmap(df, visuals_dir)
    plot_tone_coverage(df, visuals_dir)
    plot_word_count_boxplot(df, visuals_dir)
    plot_rolling_duplicate_rate(df, visuals_dir)
    plot_trend_analysis(df, visuals_dir)
    plot_model_usage_bar(df, visuals_dir)
    plot_multi_heatmap(df, visuals_dir)
    plot_common_trigrams(df, visuals_dir) # Call the new function
    plot_dashboard(visuals_dir)
    print("\nAll key visualizations saved to:", visuals_dir)
    print("Dashboard: dashboard.png")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Labels: {sorted(df['label'].unique())}")
    print(f"Languages: {sorted(df['lang'].unique())}")
    print(f"Tones: {len(df['tone'].unique())}")
    print(f"Topics: {len(df['topic'].unique())}")
    print(f"Duplicates: {df['is_duplicate'].sum()}")
    print(f"Uniqueness rate: {1 - df['is_duplicate'].mean():.3f}")
    print(f"Average word count: {df['word_count'].mean():.1f}")
    print(f"Word count range: {df['word_count'].min()}-{df['word_count'].max()}")

if __name__ == "__main__":
    main() 