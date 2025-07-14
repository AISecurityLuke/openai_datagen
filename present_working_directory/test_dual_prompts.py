#!/usr/bin/env python3
"""
Quick test to verify dual-prompt logic works correctly
"""

import json
from ripit import DatasetGenerator

def test_dual_prompts():
    """Test that the dual-prompt logic works correctly"""
    print("🧪 Testing Dual-Prompt Logic")
    print("=" * 40)
    
    # Create generator
    dg = DatasetGenerator()
    
    # Test parameters for each label
    test_params = {
        'label': 0,
        'lang': 'en',
        'tone': 'casual',
        'topic': 'dev',
        'role': 'gamer',
        'birth_year': 1990,
        'region': 'California',
        'medium': 'tweet',
        'pov': 'first',
        'add_emoji': True
    }
    
    print("Testing label 0 (standard prompt):")
    test_params['label'] = 0
    system_prompt = dg.standard_prompts["system_prompt"].format(**test_params)
    user_prompt = dg.standard_prompts["user_prompt_template"].format(**test_params)
    print(f"System prompt: {system_prompt[:100]}...")
    print(f"User prompt: {user_prompt}")
    print()
    
    print("Testing label 1 (standard prompt):")
    test_params['label'] = 1
    system_prompt = dg.standard_prompts["system_prompt"].format(**test_params)
    user_prompt = dg.standard_prompts["user_prompt_template"].format(**test_params)
    print(f"System prompt: {system_prompt[:100]}...")
    print(f"User prompt: {user_prompt}")
    print()
    
    print("Testing label 2 (jailbreak prompt):")
    test_params['label'] = 2
    system_prompt = dg.jailbreak_prompts["system_prompt"].format(**test_params)
    user_prompt = dg.jailbreak_prompts["user_prompt_template"].format(**test_params)
    print(f"System prompt: {system_prompt[:100]}...")
    print(f"User prompt: {user_prompt}")
    print()
    
    print("✅ Dual-prompt logic test completed successfully!")

if __name__ == "__main__":
    test_dual_prompts() 