#!/usr/bin/env python3
"""
Test dynamic word count system
"""

import numpy as np
from ripit import DatasetGenerator

def test_dynamic_word_count():
    """Test the dynamic word count calculation"""
    print("🧪 Testing Dynamic Word Count System")
    print("=" * 50)
    
    # Create generator
    dg = DatasetGenerator()
    
    # Test word count calculation for each label
    print("Testing word count calculation:")
    for label in [0, 1, 2]:
        word_counts = []
        for _ in range(100):  # Generate 100 samples
            target = dg._calculate_target_word_count(label)
            word_counts.append(target)
        
        mean_words = np.mean(word_counts)
        median_words = np.median(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        
        print(f"Label {label}:")
        print(f"  Mean target: {mean_words:.1f}")
        print(f"  Median target: {median_words:.1f}")
        print(f"  Range: {min_words}-{max_words}")
        print()
    
    # Test word count instruction generation
    print("Testing word count instructions:")
    test_counts = [25, 45, 75, 105, 135]
    for count in test_counts:
        instruction = dg._get_word_count_instruction(count)
        print(f"  {count} words: {instruction}")
    
    print("\n✅ Dynamic word count system test complete!")

if __name__ == "__main__":
    test_dynamic_word_count() 