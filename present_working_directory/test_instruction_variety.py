#!/usr/bin/env python3
"""
Test instruction variety for word count instructions
"""

from ripit import DatasetGenerator
from collections import Counter

def test_instruction_variety():
    """Test that we get good variety in word count instructions"""
    print("🧪 Testing Instruction Variety")
    print("=" * 50)
    
    # Create generator
    dg = DatasetGenerator()
    
    # Test multiple instructions for the same word count
    target_words = 105
    instructions = []
    
    print(f"Testing variety for {target_words} words:")
    for i in range(20):
        instruction = dg._get_word_count_instruction(target_words)
        instructions.append(instruction)
        print(f"  {i+1:2d}: {instruction}")
    
    # Count unique instructions
    unique_instructions = set(instructions)
    print(f"\nUnique instructions: {len(unique_instructions)} out of 20")
    print(f"Variety ratio: {len(unique_instructions)/20:.1%}")
    
    # Test different word counts
    print(f"\nTesting different word counts:")
    test_counts = [85, 95, 105, 115, 125]
    for count in test_counts:
        instruction = dg._get_word_count_instruction(count)
        print(f"  {count} words: {instruction}")
    
    print("\n✅ Instruction variety test complete!")

if __name__ == "__main__":
    test_instruction_variety() 