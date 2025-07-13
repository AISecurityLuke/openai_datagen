#!/usr/bin/env python3
"""
Dataset Visualization Script
Creates four sanity plots for the generated prompts dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import os
from pathlib import Path
from scipy.stats import chi2_contingency

def load_data(filepath='generated_prompts.jsonl'):
    """Load the generated prompts dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_json(filepath, lines=True)
    print(f"Loaded {len(df)} samples")
    return df

def add_helper_columns(df):
    """Add helper columns for analysis"""
    # Word count
    df['word_count'] = df['text'].str.split().str.len()
    
    # Duplicate detection using hash
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
    """Create visuals directory if it doesn't exist"""
    visuals_dir = Path('visuals')
    visuals_dir.mkdir(exist_ok=True)
    print(f"Created/verified visuals directory: {visuals_dir}")
    return visuals_dir

def plot_label_language_heatmap(df, visuals_dir):
    """Plot 1: Label × Language heat-map"""
    print("Creating Label × Language heat-map...")
    
    # Create crosstab
    crosstab = pd.crosstab(df['label'], df['lang'], normalize='all')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Total Dataset', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Language')
    ax.set_ylabel('Label')
    ax.set_title('Label × Language Distribution Heat-map\n(Perfect Balance = 0.111 per cell)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    
    # Add text annotations
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax.text(j, i, f'{crosstab.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'label_language_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print balance analysis
    print(f"Label-Language Balance Analysis:")
    print(f"Expected proportion per cell: {1/(len(df['label'].unique()) * len(df['lang'].unique())):.3f}")
    print(f"Actual proportions:")
    print(crosstab.round(3))

def plot_tone_coverage(df, visuals_dir):
    """Plot 2: Tone coverage bar chart"""
    print("Creating Tone coverage bar chart...")
    
    # Count prompts per tone
    tone_counts = df['tone'].value_counts()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    bars = ax.bar(range(len(tone_counts)), tone_counts.values, 
                  color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Tone')
    ax.set_ylabel('Number of Prompts')
    ax.set_title('All Tones by Frequency\n(Shows coverage distribution)', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(range(len(tone_counts)))
    ax.set_xticklabels(tone_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add expected line
    expected_count = len(df) / len(df['tone'].unique())
    ax.axhline(y=expected_count, color='red', linestyle='--', alpha=0.7, 
               label=f'Expected (Perfect Balance): {expected_count:.1f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'tone_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print tone analysis
    print(f"Tone Coverage Analysis:")
    print(f"Total tones: {len(df['tone'].unique())}")
    print(f"Expected count per tone: {expected_count:.1f}")
    print(f"Min count: {tone_counts.min()}")
    print(f"Max count: {tone_counts.max()}")
    print(f"Balance ratio (min/max): {tone_counts.min()/tone_counts.max():.3f}")

def plot_rolling_duplicate_rate(df, visuals_dir):
    """Plot 3: Rolling duplicate-rate line"""
    print("Creating Rolling duplicate-rate line...")
    
    # Calculate cumulative duplicate rate
    df_sorted = df.sort_index().copy()
    df_sorted['cumulative_duplicates'] = df_sorted['is_duplicate'].cumsum()
    df_sorted['duplicate_rate'] = df_sorted['cumulative_duplicates'] / (df_sorted.index + 1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot duplicate rate
    ax.plot(df_sorted.index, df_sorted['duplicate_rate'], 
            color='red', linewidth=2, alpha=0.8, label='Cumulative Duplicate Rate')
    
    # Add horizontal line at 0 for reference
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cumulative Duplicate Rate')
    ax.set_title('Rolling Duplicate Rate Over Dataset\n(Shows uniqueness decay)', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add final rate annotation
    final_rate = df_sorted['duplicate_rate'].iloc[-1]
    ax.text(0.02, 0.98, f'Final Duplicate Rate: {final_rate:.3f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add total duplicates annotation
    total_duplicates = df_sorted['is_duplicate'].sum()
    ax.text(0.02, 0.92, f'Total Duplicates: {total_duplicates}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'rolling_duplicate_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print duplicate analysis
    print(f"Duplicate Analysis:")
    print(f"Total duplicates: {total_duplicates}")
    print(f"Final duplicate rate: {final_rate:.3f}")
    print(f"Uniqueness rate: {1-final_rate:.3f}")

def plot_word_count_boxplot(df, visuals_dir):
    """Plot 4: Word-count boxplot by label"""
    print("Creating Word-count boxplot by label...")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    labels = sorted(df['label'].unique())
    data = [df[df['label'] == label]['word_count'] for label in labels]
    
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize the plot
    ax.set_xlabel('Label')
    ax.set_ylabel('Word Count')
    ax.set_title('Word Count Distribution by Label\n(Expected: 6-150 words)', pad=20)
    
    # Add reference lines for expected range
    ax.axhline(y=6, color='red', linestyle='--', alpha=0.7, label='Min Expected (6)')
    ax.axhline(y=150, color='red', linestyle='--', alpha=0.7, label='Max Expected (150)')
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics annotations
    for i, label in enumerate(labels):
        label_data = df[df['label'] == label]['word_count']
        mean_val = label_data.mean()
        median_val = label_data.median()
        ax.text(i+1, ax.get_ylim()[1] * 0.95, 
                f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}', 
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'word_count_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print word count analysis
    print(f"Word Count Analysis:")
    for label in labels:
        label_data = df[df['label'] == label]['word_count']
        print(f"Label {label}:")
        print(f"  Mean: {label_data.mean():.1f}")
        print(f"  Median: {label_data.median():.1f}")
        print(f"  Min: {label_data.min()}")
        print(f"  Max: {label_data.max()}")
        in_range = (label_data >= 6) & (label_data <= 150)
        print(f"  In range (6-150): {in_range.sum()}/{len(label_data)} ({in_range.mean():.1%})")

def plot_tone_language_heatmap(df, visuals_dir):
    """Plot 5: Tone × Language heat-map"""
    print("Creating Tone × Language heat-map...")
    
    # Create crosstab
    crosstab = pd.crosstab(df['tone'], df['lang'], normalize='all')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Total Dataset', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Language')
    ax.set_ylabel('Tone')
    ax.set_title('Tone × Language Distribution Heat-map\n(Shows tone coverage across languages)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    
    # Add text annotations (smaller font for readability)
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax.text(j, i, f'{crosstab.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'tone_language_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Tone-Language Analysis:")
    print(f"Expected proportion per cell: {1/(len(df['tone'].unique()) * len(df['lang'].unique())):.3f}")

def plot_tone_label_heatmap(df, visuals_dir):
    """Plot 6: Tone × Label heat-map"""
    print("Creating Tone × Label heat-map...")
    
    # Create crosstab
    crosstab = pd.crosstab(df['tone'], df['label'], normalize='all')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Total Dataset', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Label')
    ax.set_ylabel('Tone')
    ax.set_title('Tone × Label Distribution Heat-map\n(Shows tone coverage across labels)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    
    # Add text annotations
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax.text(j, i, f'{crosstab.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'tone_label_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Tone-Label Analysis:")
    print(f"Expected proportion per cell: {1/(len(df['tone'].unique()) * len(df['label'].unique())):.3f}")

def plot_topic_language_heatmap(df, visuals_dir):
    """Plot 7: Topic × Language heat-map (All topics)"""
    print("Creating Topic × Language heat-map...")
    
    # Create crosstab with all topics
    crosstab = pd.crosstab(df['topic'], df['lang'], normalize='all')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Total Dataset', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Language')
    ax.set_ylabel('Topic')
    ax.set_title('Topic × Language Distribution Heat-map\n(All topics)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    
    # Add text annotations (very small font for readability)
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax.text(j, i, f'{crosstab.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=6)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'topic_language_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Topic-Language Analysis:")
    print(f"Expected proportion per cell: {1/(len(crosstab.index) * len(crosstab.columns)):.3f}")

def plot_topic_label_heatmap(df, visuals_dir):
    """Plot 8: Topic × Label heat-map (All topics)"""
    print("Creating Topic × Label heat-map...")
    
    # Create crosstab with all topics
    crosstab = pd.crosstab(df['topic'], df['label'], normalize='all')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Total Dataset', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Label')
    ax.set_ylabel('Topic')
    ax.set_title('Topic × Label Distribution Heat-map\n(All topics)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    
    # Add text annotations
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax.text(j, i, f'{crosstab.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=6)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'topic_label_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Topic-Label Analysis:")
    print(f"Expected proportion per cell: {1/(len(crosstab.index) * len(crosstab.columns)):.3f}")

def plot_medium_language_heatmap(df, visuals_dir):
    """Plot 9: Medium × Language heat-map"""
    print("Creating Medium × Language heat-map...")
    
    # Create crosstab
    crosstab = pd.crosstab(df['medium'], df['lang'], normalize='all')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Total Dataset', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Language')
    ax.set_ylabel('Medium')
    ax.set_title('Medium × Language Distribution Heat-map\n(Shows medium coverage across languages)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    
    # Add text annotations
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax.text(j, i, f'{crosstab.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'medium_language_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Medium-Language Analysis:")
    print(f"Expected proportion per cell: {1/(len(df['medium'].unique()) * len(df['lang'].unique())):.3f}")

def plot_medium_label_heatmap(df, visuals_dir):
    """Plot 10: Medium × Label heat-map"""
    print("Creating Medium × Label heat-map...")
    
    # Create crosstab
    crosstab = pd.crosstab(df['medium'], df['label'], normalize='all')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Total Dataset', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Label')
    ax.set_ylabel('Medium')
    ax.set_title('Medium × Label Distribution Heat-map\n(Shows medium coverage across labels)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    
    # Add text annotations
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax.text(j, i, f'{crosstab.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'medium_label_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Medium-Label Analysis:")
    print(f"Expected proportion per cell: {1/(len(df['medium'].unique()) * len(df['label'].unique())):.3f}")

def plot_role_language_heatmap(df, visuals_dir):
    """Plot 11: Role × Language heat-map"""
    print("Creating Role × Language heat-map...")
    
    # Create crosstab
    crosstab = pd.crosstab(df['role'], df['lang'], normalize='all')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Total Dataset', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Language')
    ax.set_ylabel('Role')
    ax.set_title('Role × Language Distribution Heat-map\n(Shows role coverage across languages)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    
    # Add text annotations
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax.text(j, i, f'{crosstab.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'role_language_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Role-Language Analysis:")
    print(f"Expected proportion per cell: {1/(len(df['role'].unique()) * len(df['lang'].unique())):.3f}")

def plot_role_label_heatmap(df, visuals_dir):
    """Plot 12: Role × Label heat-map"""
    print("Creating Role × Label heat-map...")
    
    # Create crosstab
    crosstab = pd.crosstab(df['role'], df['label'], normalize='all')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(crosstab.values, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Total Dataset', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Label')
    ax.set_ylabel('Role')
    ax.set_title('Role × Label Distribution Heat-map\n(Shows role coverage across labels)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)
    
    # Add text annotations
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax.text(j, i, f'{crosstab.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'role_label_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Role-Label Analysis:")
    print(f"Expected proportion per cell: {1/(len(df['role'].unique()) * len(df['label'].unique())):.3f}")

def plot_correlation_matrix(df, visuals_dir):
    """Plot 13: Correlation matrix of categorical variables"""
    print("Creating Correlation matrix of categorical variables...")
    
    # Create correlation matrix for categorical variables
    categorical_cols = ['label', 'lang', 'tone', 'topic', 'medium', 'role', 'pov', 'add_emoji']
    available_cols = [col for col in categorical_cols if col in df.columns]
    
    # Create correlation matrix using Cramer's V
    corr_matrix = pd.DataFrame(index=available_cols, columns=available_cols)
    
    for i, col1 in enumerate(available_cols):
        for j, col2 in enumerate(available_cols):
            if i == j:
                corr_matrix.loc[col1, col2] = 1.0
            else:
                # Calculate Cramer's V for categorical correlation
                contingency = pd.crosstab(df[col1], df[col2])
                chi2 = chi2_contingency(contingency)[0]
                n = len(df)
                min_dim = min(len(df[col1].unique()), len(df[col2].unique())) - 1
                cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                corr_matrix.loc[col1, col2] = cramer_v
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cramer's V Correlation", rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Variables')
    ax.set_ylabel('Variables')
    ax.set_title('Categorical Variable Correlation Matrix\n(Shows relationships between all dimensions)', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(available_cols)))
    ax.set_yticks(range(len(available_cols)))
    ax.set_xticklabels(available_cols, rotation=45, ha='right')
    ax.set_yticklabels(available_cols)
    
    # Add text annotations
    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10,
                          fontweight='bold' if corr_matrix.iloc[i, j] > 0.3 else 'normal')
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"Correlation Analysis:")
    print(f"Strong correlations (>0.3):")
    for i, col1 in enumerate(available_cols):
        for j, col2 in enumerate(available_cols):
            if i < j and corr_matrix.loc[col1, col2] > 0.3:
                print(f"  {col1} ↔ {col2}: {corr_matrix.loc[col1, col2]:.3f}")

def plot_parallel_coordinates(df, visuals_dir):
    """Plot 14: Parallel coordinates plot for multi-dimensional analysis"""
    print("Creating Parallel coordinates plot...")
    
    # Prepare data for parallel coordinates
    # Convert categorical variables to numeric for plotting
    plot_data = df.copy()
    
    # Convert categorical variables to numeric codes
    categorical_cols = ['lang', 'tone', 'topic', 'medium', 'role', 'pov']
    available_cols = [col for col in categorical_cols if col in df.columns]
    
    for col in available_cols:
        plot_data[f'{col}_code'] = plot_data[col].astype('category').cat.codes
    
    # Select columns for plotting
    plot_cols = ['label', 'word_count'] + [f'{col}_code' for col in available_cols]
    plot_data_subset = plot_data[plot_cols].copy()
    
    # Rename columns for better labels
    plot_data_subset.columns = ['Label', 'Word Count'] + available_cols
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot parallel coordinates
    pd.plotting.parallel_coordinates(plot_data_subset, 'Label', colormap='viridis', ax=ax)
    
    # Customize the plot
    ax.set_title('Parallel Coordinates Plot\n(Shows relationships across all dimensions)', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'parallel_coordinates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Parallel Coordinates Analysis:")
    print(f"Shows relationships across {len(plot_cols)} dimensions")

def plot_3d_scatter(df, visuals_dir):
    """Plot 15: 3D scatter plot (Label vs Language vs Word Count)"""
    print("Creating 3D scatter plot...")
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with different colors for each label
    colors = ['green', 'blue', 'red']
    labels = sorted(df['label'].unique())
    
    for i, label in enumerate(labels):
        label_data = df[df['label'] == label]
        ax.scatter(label_data['label'], 
                  label_data['word_count'],
                  [list(df['lang'].unique()).index(lang) for lang in label_data['lang']],
                  c=colors[i], label=f'Label {label}', alpha=0.7, s=50)
    
    # Customize the plot
    ax.set_xlabel('Label')
    ax.set_ylabel('Word Count')
    ax.set_zlabel('Language (0=en, 1=es, 2=fr)')
    ax.set_title('3D Scatter: Label × Word Count × Language\n(Shows 3D relationships)', pad=20)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / '3d_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"3D Scatter Analysis:")
    print(f"Shows relationships between Label, Word Count, and Language")

def plot_multi_heatmap(df, visuals_dir):
    """Plot 16: Multi-dimensional heatmap (Label × Language × Tone)"""
    print("Creating Multi-dimensional heatmap...")
    
    # Create multi-dimensional heatmap
    # Group by label, language, and tone, then count
    heatmap_data = df.groupby(['label', 'lang', 'tone']).size().unstack(fill_value=0)
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    for i, label in enumerate(sorted(df['label'].unique())):
        label_data = df[df['label'] == label]
        pivot_data = label_data.groupby(['lang', 'tone']).size().unstack(fill_value=0)
        
        im = axes[i].imshow(pivot_data.values, cmap='Blues', aspect='auto')
        axes[i].set_title(f'Label {label} Distribution\n(Language × Tone)', pad=10)
        axes[i].set_xlabel('Tone')
        axes[i].set_ylabel('Language')
        
        # Set tick labels
        axes[i].set_xticks(range(len(pivot_data.columns)))
        axes[i].set_yticks(range(len(pivot_data.index)))
        axes[i].set_xticklabels(pivot_data.columns, rotation=45, ha='right', fontsize=8)
        axes[i].set_yticklabels(pivot_data.index, fontsize=8)
        
        # Add text annotations
        for j in range(len(pivot_data.index)):
            for k in range(len(pivot_data.columns)):
                value = pivot_data.iloc[j, k]
                if value > 0:
                    axes[i].text(k, j, str(value), ha="center", va="center", 
                                color="black", fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'multi_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-dimensional Heatmap Analysis:")
    print(f"Shows distribution patterns across Label × Language × Tone")

def plot_trend_analysis(df, visuals_dir):
    """Plot 17: Trend analysis across generation order"""
    print("Creating Trend analysis across generation order...")
    
    # Add generation order index
    df_with_index = df.reset_index()
    
    # Create subplots for different trends
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Word count trend over time
    axes[0, 0].plot(df_with_index.index, df_with_index['word_count'], alpha=0.7)
    axes[0, 0].set_title('Word Count Trend Over Generation Order')
    axes[0, 0].set_xlabel('Generation Order')
    axes[0, 0].set_ylabel('Word Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Label distribution over time (rolling window)
    window_size = 50
    label_0_rolling = df_with_index['label'].rolling(window=window_size).apply(lambda x: (x == 0).mean())
    label_1_rolling = df_with_index['label'].rolling(window=window_size).apply(lambda x: (x == 1).mean())
    label_2_rolling = df_with_index['label'].rolling(window=window_size).apply(lambda x: (x == 2).mean())
    
    axes[0, 1].plot(df_with_index.index, label_0_rolling, label='Label 0', alpha=0.8)
    axes[0, 1].plot(df_with_index.index, label_1_rolling, label='Label 1', alpha=0.8)
    axes[0, 1].plot(df_with_index.index, label_2_rolling, label='Label 2', alpha=0.8)
    axes[0, 1].set_title(f'Label Distribution Trend (Rolling {window_size})')
    axes[0, 1].set_xlabel('Generation Order')
    axes[0, 1].set_ylabel('Proportion')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Language distribution over time
    lang_rolling = df_with_index.groupby('lang')['lang'].rolling(window=window_size).count().unstack(0)
    for lang in ['en', 'es', 'fr']:
        if lang in lang_rolling.columns:
            axes[1, 0].plot(df_with_index.index, lang_rolling[lang], label=f'Lang {lang}', alpha=0.8)
    axes[1, 0].set_title(f'Language Distribution Trend (Rolling {window_size})')
    axes[1, 0].set_xlabel('Generation Order')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Model usage trend
    model_rolling = df_with_index.groupby('model_used')['model_used'].rolling(window=window_size).count().unstack(0)
    for model in model_rolling.columns:
        axes[1, 1].plot(df_with_index.index, model_rolling[model], label=model, alpha=0.8)
    axes[1, 1].set_title(f'Model Usage Trend (Rolling {window_size})')
    axes[1, 1].set_xlabel('Generation Order')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trend Analysis:")
    print(f"Shows temporal patterns across generation order")

def plot_complexity_analysis(df, visuals_dir):
    """Plot 18: Complexity analysis across multiple dimensions"""
    print("Creating Complexity analysis...")
    
    # Calculate complexity metrics
    df['char_count'] = df['text'].str.len()
    df['sentence_count'] = df['text'].str.split('.').str.len()
    df['avg_word_length'] = df['text'].str.split().str.join(' ').str.len() / df['word_count']
    df['emoji_count'] = df['text'].str.count(r'[^\w\s]')  # Rough emoji/punctuation count
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Word count vs character count by label
    for label in sorted(df['label'].unique()):
        label_data = df[df['label'] == label]
        axes[0, 0].scatter(label_data['word_count'], label_data['char_count'], 
                          alpha=0.6, label=f'Label {label}')
    axes[0, 0].set_xlabel('Word Count')
    axes[0, 0].set_ylabel('Character Count')
    axes[0, 0].set_title('Word Count vs Character Count by Label')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Average word length by label
    avg_word_length_by_label = df.groupby('label')['avg_word_length'].mean()
    axes[0, 1].bar(avg_word_length_by_label.index, avg_word_length_by_label.values)
    axes[0, 1].set_xlabel('Label')
    axes[0, 1].set_ylabel('Average Word Length')
    axes[0, 1].set_title('Average Word Length by Label')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Sentence count distribution
    df['sentence_count'].hist(ax=axes[0, 2], bins=20, alpha=0.7)
    axes[0, 2].set_xlabel('Sentence Count')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Sentence Count Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Complexity by language
    complexity_by_lang = df.groupby('lang').agg({
        'word_count': 'mean',
        'char_count': 'mean',
        'avg_word_length': 'mean'
    })
    complexity_by_lang.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Complexity Metrics by Language')
    axes[1, 0].set_xlabel('Language')
    axes[1, 0].set_ylabel('Average Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Emoji/punctuation usage by tone
    emoji_by_tone = df.groupby('tone')['emoji_count'].mean().sort_values(ascending=False).head(10)
    emoji_by_tone.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Emoji/Punctuation Usage by Tone (Top 10)')
    axes[1, 1].set_xlabel('Tone')
    axes[1, 1].set_ylabel('Average Emoji Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Complexity correlation heatmap
    complexity_cols = ['word_count', 'char_count', 'sentence_count', 'avg_word_length', 'emoji_count']
    complexity_corr = df[complexity_cols].corr()
    im = axes[1, 2].imshow(complexity_corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 2].set_xticks(range(len(complexity_cols)))
    axes[1, 2].set_yticks(range(len(complexity_cols)))
    axes[1, 2].set_xticklabels(complexity_cols, rotation=45, ha='right')
    axes[1, 2].set_yticklabels(complexity_cols)
    axes[1, 2].set_title('Complexity Metrics Correlation')
    
    # Add correlation values
    for i in range(len(complexity_cols)):
        for j in range(len(complexity_cols)):
            text = axes[1, 2].text(j, i, f'{complexity_corr.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Complexity Analysis:")
    print(f"Shows relationships between different complexity metrics")

def main():
    """Main function to run all visualizations"""
    print("=" * 60)
    print("DATASET VISUALIZATION SCRIPT")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Add helper columns
    df = add_helper_columns(df)
    
    # Create visuals directory
    visuals_dir = create_visuals_directory()
    
    # Run all visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Basic sanity plots
    plot_label_language_heatmap(df, visuals_dir)
    plot_tone_coverage(df, visuals_dir)
    plot_rolling_duplicate_rate(df, visuals_dir)
    plot_word_count_boxplot(df, visuals_dir)
    
    # Cross-comparison plots
    plot_tone_language_heatmap(df, visuals_dir)
    plot_tone_label_heatmap(df, visuals_dir)
    plot_topic_language_heatmap(df, visuals_dir)
    plot_topic_label_heatmap(df, visuals_dir)
    plot_medium_language_heatmap(df, visuals_dir)
    plot_medium_label_heatmap(df, visuals_dir)
    plot_role_language_heatmap(df, visuals_dir)
    plot_role_label_heatmap(df, visuals_dir)
    
    # Advanced multi-dimensional visualizations
    plot_correlation_matrix(df, visuals_dir)
    plot_parallel_coordinates(df, visuals_dir)
    plot_3d_scatter(df, visuals_dir)
    plot_multi_heatmap(df, visuals_dir)
    plot_trend_analysis(df, visuals_dir)
    plot_complexity_analysis(df, visuals_dir)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
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
    
    print(f"\nVisualizations saved to: {visuals_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main() 