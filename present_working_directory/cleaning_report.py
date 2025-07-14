#!/usr/bin/env python3
"""
Cleaning Report Generator
Analyzes the effectiveness of real-time cleaning during dataset generation
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def generate_cleaning_report(data_file: str = "generated_prompts.jsonl", 
                           rejected_file: str = "rejected_requests.jsonl",
                           output_dir: str = "cleaning_reports"):
    """Generate comprehensive cleaning report"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🧹 GENERATING CLEANING REPORT")
    print("=" * 50)
    
    # Load data
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    rejected_data = []
    if Path(rejected_file).exists():
        with open(rejected_file, 'r') as f:
            rejected_data = [json.loads(line) for line in f]
    
    # Analyze accepted data
    df = pd.DataFrame(data)
    
    # Analyze rejected data
    rejected_df = pd.DataFrame(rejected_data) if rejected_data else pd.DataFrame()
    
    # Generate reports
    cleaning_stats = analyze_cleaning_statistics(df, rejected_df)
    pattern_analysis = analyze_patterns(df)
    quality_metrics = analyze_quality_metrics(df)
    
    # Create visualizations
    create_cleaning_visualizations(df, rejected_df, output_path)
    
    # Save detailed report
    save_detailed_report(cleaning_stats, pattern_analysis, quality_metrics, output_path)
    
    print(f"✅ Cleaning report saved to {output_path}")
    return cleaning_stats, pattern_analysis, quality_metrics

def analyze_cleaning_statistics(df: pd.DataFrame, rejected_df: pd.DataFrame) -> dict:
    """Analyze cleaning statistics"""
    
    total_generated = len(df) + len(rejected_df)
    total_accepted = len(df)
    total_rejected = len(rejected_df)
    
    rejection_rate = (total_rejected / total_generated * 100) if total_generated > 0 else 0
    
    # Analyze rejection reasons
    rejection_reasons = {}
    if not rejected_df.empty and 'rejection_reason' in rejected_df.columns:
        rejection_reasons = rejected_df['rejection_reason'].value_counts().to_dict()
    
    # Analyze trigram diversity
    trigram_diversity = analyze_trigram_diversity(df)
    
    return {
        "total_generated": total_generated,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "rejection_rate": rejection_rate,
        "rejection_reasons": rejection_reasons,
        "trigram_diversity": trigram_diversity
    }

def analyze_trigram_diversity(df: pd.DataFrame) -> dict:
    """Analyze trigram diversity across labels"""
    
    def extract_trigrams(text):
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        return [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    
    # Extract all trigrams by label
    trigrams_by_label = defaultdict(list)
    for _, row in df.iterrows():
        label = str(row['label'])  # Convert label to str
        text = row['text']
        trigrams = extract_trigrams(text)
        trigrams_by_label[label].extend(trigrams)
    
    # Calculate diversity metrics
    diversity_metrics = {}
    for label in trigrams_by_label:
        trigrams = trigrams_by_label[label]
        unique_trigrams = set(trigrams)
        total_trigrams = len(trigrams)
        
        diversity_metrics[str(label)] = {
            "total_trigrams": total_trigrams,
            "unique_trigrams": len(unique_trigrams),
            "diversity_ratio": len(unique_trigrams) / total_trigrams if total_trigrams > 0 else 0,
            "avg_trigrams_per_sample": total_trigrams / len(df[df['label'].astype(str) == label])
        }
    
    return diversity_metrics

def analyze_patterns(df: pd.DataFrame) -> dict:
    """Analyze patterns in the dataset"""
    
    def extract_trigrams(text):
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        return [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    
    # Find most common trigrams overall
    all_trigrams = []
    for text in df['text']:
        all_trigrams.extend(extract_trigrams(text))
    
    trigram_counts = Counter(all_trigrams)
    most_common = trigram_counts.most_common(20)
    
    # Analyze patterns by label
    patterns_by_label = {}
    for label in sorted(df['label'].unique()):
        label_str = str(label)
        label_df = df[df['label'] == label]
        label_trigrams = []
        for text in label_df['text']:
            label_trigrams.extend(extract_trigrams(text))
        
        label_counts = Counter(label_trigrams)
        patterns_by_label[label_str] = label_counts.most_common(10)
    
    return {
        "most_common_trigrams": most_common,
        "patterns_by_label": patterns_by_label,
        "total_unique_trigrams": int(len(trigram_counts))
    }

def analyze_quality_metrics(df: pd.DataFrame) -> dict:
    """Analyze quality metrics"""
    
    # Word count analysis
    word_counts = df['text'].str.split().str.len()
    
    # Length analysis
    char_counts = df['text'].str.len()
    
    # Uniqueness analysis
    unique_texts = df['text'].nunique()
    uniqueness_rate = unique_texts / len(df)
    
    return {
        "word_count_stats": {
            "mean": word_counts.mean(),
            "median": word_counts.median(),
            "std": word_counts.std(),
            "min": word_counts.min(),
            "max": word_counts.max()
        },
        "char_count_stats": {
            "mean": char_counts.mean(),
            "median": char_counts.median(),
            "std": char_counts.std(),
            "min": char_counts.min(),
            "max": char_counts.max()
        },
        "uniqueness": {
            "unique_texts": unique_texts,
            "total_texts": len(df),
            "uniqueness_rate": uniqueness_rate
        }
    }

def create_cleaning_visualizations(df: pd.DataFrame, rejected_df: pd.DataFrame, output_path: Path):
    """Create cleaning visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Rejection rate pie chart
    if not rejected_df.empty:
        total_generated = len(df) + len(rejected_df)
        accepted_pct = len(df) / total_generated * 100
        rejected_pct = len(rejected_df) / total_generated * 100
        
        axes[0, 0].pie([accepted_pct, rejected_pct], 
                      labels=['Accepted', 'Rejected'], 
                      autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Acceptance vs Rejection Rate')
    else:
        axes[0, 0].text(0.5, 0.5, 'No Rejected Samples', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Acceptance vs Rejection Rate')
    
    # 2. Rejection reasons bar chart
    if not rejected_df.empty and 'rejection_reason' in rejected_df.columns:
        reasons = rejected_df['rejection_reason'].value_counts()
        axes[0, 1].bar(reasons.index, reasons.values, color='lightcoral')
        axes[0, 1].set_title('Rejection Reasons')
        axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Rejection Data', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Rejection Reasons')
    
    # 3. Trigram diversity by label
    def extract_trigrams(text):
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        return [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    
    diversity_data = []
    labels = []
    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        label_trigrams = []
        for text in label_df['text']:
            label_trigrams.extend(extract_trigrams(text))
        
        unique_ratio = len(set(label_trigrams)) / len(label_trigrams) if label_trigrams else 0
        diversity_data.append(unique_ratio)
        labels.append(f'Label {label}')
    
    axes[1, 0].bar(labels, diversity_data, color=['lightgreen', 'lightblue', 'lightcoral'])
    axes[1, 0].set_title('Trigram Diversity by Label')
    axes[1, 0].set_ylabel('Unique Trigram Ratio')
    
    # 4. Word count distribution
    word_counts = df['text'].str.split().str.len()
    axes[1, 1].hist(word_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(word_counts.mean(), color='red', linestyle='--', 
                      label=f'Mean: {word_counts.mean():.1f}')
    axes[1, 1].set_title('Word Count Distribution')
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'cleaning_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_report(cleaning_stats: dict, pattern_analysis: dict, 
                        quality_metrics: dict, output_path: Path):
    """Save detailed cleaning report"""
    
    report = {
        "cleaning_statistics": cleaning_stats,
        "pattern_analysis": pattern_analysis,
        "quality_metrics": quality_metrics,
        "summary": {
            "overall_assessment": assess_overall_quality(cleaning_stats, pattern_analysis, quality_metrics),
            "recommendations": generate_recommendations(cleaning_stats, pattern_analysis, quality_metrics)
        }
    }
    
    with open(output_path / 'cleaning_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n📊 CLEANING REPORT SUMMARY")
    print("=" * 50)
    print(f"Total Generated: {cleaning_stats['total_generated']}")
    print(f"Total Accepted: {cleaning_stats['total_accepted']}")
    print(f"Total Rejected: {cleaning_stats['total_rejected']}")
    print(f"Rejection Rate: {cleaning_stats['rejection_rate']:.2f}%")
    print(f"Uniqueness Rate: {quality_metrics['uniqueness']['uniqueness_rate']:.3f}")
    
    if cleaning_stats['rejection_reasons']:
        print("\nRejection Reasons:")
        for reason, count in cleaning_stats['rejection_reasons'].items():
            print(f"  {reason}: {count}")
    
    print(f"\nOverall Assessment: {report['summary']['overall_assessment']}")
    print(f"\nRecommendations:")
    for rec in report['summary']['recommendations']:
        print(f"  • {rec}")

def assess_overall_quality(cleaning_stats: dict, pattern_analysis: dict, 
                          quality_metrics: dict) -> str:
    """Assess overall quality of the cleaning process"""
    
    rejection_rate = cleaning_stats['rejection_rate']
    uniqueness_rate = quality_metrics['uniqueness']['uniqueness_rate']
    
    if rejection_rate < 5 and uniqueness_rate > 0.95:
        return "EXCELLENT"
    elif rejection_rate < 10 and uniqueness_rate > 0.90:
        return "GOOD"
    elif rejection_rate < 20 and uniqueness_rate > 0.85:
        return "ACCEPTABLE"
    else:
        return "NEEDS IMPROVEMENT"

def generate_recommendations(cleaning_stats: dict, pattern_analysis: dict, 
                           quality_metrics: dict) -> list:
    """Generate recommendations based on analysis"""
    
    recommendations = []
    
    rejection_rate = cleaning_stats['rejection_rate']
    uniqueness_rate = quality_metrics['uniqueness']['uniqueness_rate']
    
    if rejection_rate > 15:
        recommendations.append("Consider adjusting cleaning thresholds to reduce rejection rate")
    
    if uniqueness_rate < 0.95:
        recommendations.append("Investigate duplicate patterns and adjust similarity thresholds")
    
    # Check for concerning patterns
    most_common = pattern_analysis['most_common_trigrams']
    if most_common and most_common[0][1] > cleaning_stats['total_accepted'] * 0.1:
        recommendations.append("High-frequency trigrams detected - consider prompt adjustments")
    
    if not recommendations:
        recommendations.append("Cleaning process is working well - no major issues detected")
    
    return recommendations

if __name__ == "__main__":
    generate_cleaning_report() 