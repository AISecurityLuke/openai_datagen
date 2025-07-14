#!/usr/bin/env python3
"""
Smoke test for balance logic - no API calls
"""

import json
import random
import numpy as np
from collections import Counter
from ripit import DatasetGenerator

def calculate_gini_coefficient(counts):
    """Calculate Gini coefficient for a distribution"""
    if not counts:
        return 0.0
    
    values = list(counts.values())
    n = len(values)
    if n == 0:
        return 0.0
    
    sorted_values = sorted(values)
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0

def calculate_relative_deviation(counts, expected_per_cell):
    """Calculate maximum relative deviation from expected value"""
    if not counts or expected_per_cell == 0:
        return 0.0
    
    max_deviation = max(abs(count - expected_per_cell) for count in counts.values())
    return (max_deviation / expected_per_cell) * 100

def calculate_min_cell_percentage(counts, total_samples):
    """Calculate minimum cell percentage of total data"""
    if not counts or total_samples == 0:
        return 0.0
    
    min_count = min(counts.values())
    return (min_count / total_samples) * 100

def test_balance_logic():
    """Test the balance logic without API calls"""
    print("🧪 Testing Balance Logic (No API Calls)")
    print("=" * 50)
    
    # Create generator with small sample size
    dg = DatasetGenerator()
    dg.num_samples = 12000  # Test with 12,000 prompts for new topic count
    
    # Setup balanced sampling
    dg._setup_balanced_sampling()
    
    print(f"Total combinations created: {len(dg.balanced_combinations)}")
    print(f"Expected combinations: {dg.num_samples}")
    
    # Extract all parameters
    labels = [combo['label'] for combo in dg.balanced_combinations]
    langs = [combo['lang'] for combo in dg.balanced_combinations]
    tones = [combo['tone'] for combo in dg.balanced_combinations]
    topics = [combo['topic'] for combo in dg.balanced_combinations]
    
    # Check distributions
    label_counts = Counter(labels)
    lang_counts = Counter(langs)
    tone_counts = Counter(tones)
    topic_counts = Counter(topics)
    
    print("\n📊 Distribution Analysis:")
    print(f"Labels: {dict(label_counts)}")
    print(f"Languages: {dict(lang_counts)}")
    print(f"Tones: {dict(tone_counts)}")
    print(f"Topics: {dict(topic_counts)}")
    
    # Check if perfectly balanced
    expected_per_label = dg.num_samples // 3
    expected_per_lang = dg.num_samples // 3
    
    print(f"\n✅ Expected per label: {expected_per_label}")
    print(f"✅ Expected per language: {expected_per_lang}")
    
    # Check label balance
    label_balanced = all(count == expected_per_label for count in label_counts.values())
    lang_balanced = all(count == expected_per_lang for count in lang_counts.values())
    
    print(f"\n🎯 Label balance: {'✅ PASS' if label_balanced else '❌ FAIL'}")
    print(f"🎯 Language balance: {'✅ PASS' if lang_balanced else '❌ FAIL'}")
    
    # Check (label, lang) combinations
    label_lang_combos = [(combo['label'], combo['lang']) for combo in dg.balanced_combinations]
    label_lang_counts = Counter(label_lang_combos)
    expected_per_combo = dg.num_samples // 9  # 3 labels × 3 langs
    
    print(f"\n🎯 Expected per (label, lang) combo: {expected_per_combo}")
    print(f"🎯 Actual (label, lang) distribution:")
    for (label, lang), count in sorted(label_lang_counts.items()):
        print(f"  ({label}, {lang}): {count}")
    
    label_lang_balanced = all(count == expected_per_combo for count in label_lang_counts.values())
    print(f"🎯 (Label, Lang) balance: {'✅ PASS' if label_lang_balanced else '❌ FAIL'}")
    
    # Check tone and topic quality metrics
    print(f"\n📈 QUALITY METRICS ANALYSIS:")
    
    # Tone metrics
    expected_per_tone = dg.num_samples // len(tone_counts)
    tone_gini = calculate_gini_coefficient(tone_counts)
    tone_deviation = calculate_relative_deviation(tone_counts, expected_per_tone)
    tone_min_percentage = calculate_min_cell_percentage(tone_counts, dg.num_samples)
    max_tone_count = max(tone_counts.values())
    tone_max_percentage = (max_tone_count / dg.num_samples) * 100
    
    # Topic metrics
    expected_per_topic = dg.num_samples // len(topic_counts)
    topic_gini = calculate_gini_coefficient(topic_counts)
    topic_deviation = calculate_relative_deviation(topic_counts, expected_per_topic)
    topic_min_percentage = calculate_min_cell_percentage(topic_counts, dg.num_samples)
    max_topic_count = max(topic_counts.values())
    topic_max_percentage = (max_topic_count / dg.num_samples) * 100
    
    print(f"🎵 Tone Analysis:")
    print(f"  Expected per tone: {expected_per_tone}")
    print(f"  Gini coefficient: {tone_gini:.3f} {'✅' if tone_gini < 0.2 else '❌'}")
    print(f"  Max relative deviation: {tone_deviation:.1f}% {'✅' if tone_deviation <= 30 else '❌'}")
    print(f"  Min cell percentage: {tone_min_percentage:.1f}% {'✅' if 4.2 <= tone_min_percentage <= 6.3 else '❌'}")
    print(f"  Max cell percentage: {tone_max_percentage:.1f}% {'✅' if 4.2 <= tone_max_percentage <= 6.3 else '❌'}")
    print(f"  Distribution (min/max): {min(tone_counts.values())}/{max(tone_counts.values())}")
    
    print(f"📚 Topic Analysis:")
    print(f"  Expected per topic: {expected_per_topic}")
    print(f"  Gini coefficient: {topic_gini:.3f} {'✅' if topic_gini < 0.2 else '❌'}")
    print(f"  Max relative deviation: {topic_deviation:.1f}% {'✅' if topic_deviation <= 30 else '❌'}")
    print(f"  Min cell percentage: {topic_min_percentage:.1f}% {'✅' if 2.3 <= topic_min_percentage <= 3.4 else '❌'}")
    print(f"  Max cell percentage: {topic_max_percentage:.1f}% {'✅' if 2.3 <= topic_max_percentage <= 3.4 else '❌'}")
    print(f"  Distribution (min/max): {min(topic_counts.values())}/{max(topic_counts.values())}")
    
    # Overall assessment
    print(f"\n🎯 QUALITY STANDARDS CHECK:")
    print(f"  Label balance: {'✅' if label_balanced else '❌'}")
    print(f"  Language balance: {'✅' if lang_balanced else '❌'}")
    print(f"  Label-Lang balance: {'✅' if label_lang_balanced else '❌'}")
    print(f"  Tone Gini < 0.2: {'✅' if tone_gini < 0.2 else '❌'}")
    print(f"  Topic Gini < 0.2: {'✅' if topic_gini < 0.2 else '❌'}")
    print(f"  Tone deviation ≤ 30%: {'✅' if tone_deviation <= 30 else '❌'}")
    print(f"  Topic deviation ≤ 30%: {'✅' if topic_deviation <= 30 else '❌'}")
    print(f"  Tone min/max cell 4.2%-6.3%: {'✅' if 4.2 <= tone_min_percentage <= 6.3 and 4.2 <= tone_max_percentage <= 6.3 else '❌'}")
    print(f"  Topic min/max cell 2.3%-3.4%: {'✅' if 2.3 <= topic_min_percentage <= 3.4 and 2.3 <= topic_max_percentage <= 3.4 else '❌'}")
    
    # All quality standards must pass
    quality_passed = (
        label_balanced and lang_balanced and label_lang_balanced and
        tone_gini < 0.2 and topic_gini < 0.2 and
        tone_deviation <= 30 and topic_deviation <= 30 and
        4.2 <= tone_min_percentage <= 6.3 and 4.2 <= tone_max_percentage <= 6.3 and
        2.3 <= topic_min_percentage <= 3.4 and 2.3 <= topic_max_percentage <= 3.4
    )
    
    return quality_passed

if __name__ == "__main__":
    success = test_balance_logic()
    if success:
        print("\n🎉 Balance logic test PASSED - ready for API generation!")
    else:
        print("\n❌ Balance logic test FAILED - needs fixing before API generation!") 