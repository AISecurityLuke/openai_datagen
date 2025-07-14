#!/usr/bin/env python3
"""
Analyze optimal percentage ranges for balance testing
"""

def analyze_percentages():
    """Analyze what percentage ranges work for current combination sizes"""
    
    # Current parameters
    num_samples = 12000
    num_tones = 19
    num_topics = 35
    
    print("📊 PERCENTAGE ANALYSIS FOR CURRENT PARAMETERS")
    print("=" * 50)
    print(f"Sample size: {num_samples:,}")
    print(f"Number of tones: {num_tones}")
    print(f"Number of topics: {num_topics}")
    print()
    
    # Calculate ideal distribution
    ideal_per_tone = num_samples / num_tones
    ideal_per_topic = num_samples / num_topics
    
    print("🎵 TONE ANALYSIS:")
    print(f"  Ideal per tone: {ideal_per_tone:.1f}")
    print(f"  Ideal percentage: {(ideal_per_tone/num_samples)*100:.1f}%")
    print()
    
    print("📚 TOPIC ANALYSIS:")
    print(f"  Ideal per topic: {ideal_per_topic:.1f}")
    print(f"  Ideal percentage: {(ideal_per_topic/num_samples)*100:.1f}%")
    print()
    
    # Test different percentage ranges
    print("🔍 TESTING DIFFERENT PERCENTAGE RANGES:")
    print()
    
    ranges_to_test = [
        (3, 6),   # 3-6%
        (4, 7),   # 4-7%
        (5, 8),   # 5-8%
        (6, 9),   # 6-9%
        (7, 10),  # 7-10%
        (8, 11),  # 8-11%
        (9, 12),  # 9-12%
    ]
    
    for min_pct, max_pct in ranges_to_test:
        min_count = int((min_pct / 100) * num_samples)
        max_count = int((max_pct / 100) * num_samples)
        
        # Check if this range can accommodate all tones
        tone_fits = min_count <= ideal_per_tone <= max_count
        
        # Check if this range can accommodate all topics  
        topic_fits = min_count <= ideal_per_topic <= max_count
        
        status = "✅" if tone_fits and topic_fits else "❌"
        
        print(f"  {min_pct}%-{max_pct}%: {status}")
        print(f"    Count range: {min_count}-{max_count}")
        print(f"    Tone fits: {'✅' if tone_fits else '❌'} (ideal: {ideal_per_tone:.1f})")
        print(f"    Topic fits: {'✅' if topic_fits else '❌'} (ideal: {ideal_per_topic:.1f})")
        print()
    
    # Calculate what would work for topics specifically
    print("🎯 RECOMMENDATIONS:")
    print()
    
    # For topics (35 categories), what's the minimum range needed?
    topic_min_needed = ideal_per_topic * 0.8  # Allow 20% deviation
    topic_max_needed = ideal_per_topic * 1.2  # Allow 20% deviation
    
    topic_min_pct = (topic_min_needed / num_samples) * 100
    topic_max_pct = (topic_max_needed / num_samples) * 100
    
    print(f"  For topics (35 categories):")
    print(f"    Minimum range needed: {topic_min_pct:.1f}%-{topic_max_pct:.1f}%")
    print(f"    This allows ±20% deviation from ideal")
    print()
    
    # For tones (19 categories), what's the minimum range needed?
    tone_min_needed = ideal_per_tone * 0.8  # Allow 20% deviation
    tone_max_needed = ideal_per_tone * 1.2  # Allow 20% deviation
    
    tone_min_pct = (tone_min_needed / num_samples) * 100
    tone_max_pct = (tone_max_needed / num_samples) * 100
    
    print(f"  For tones (19 categories):")
    print(f"    Minimum range needed: {tone_min_pct:.1f}%-{tone_max_pct:.1f}%")
    print(f"    This allows ±20% deviation from ideal")
    print()
    
    # Final recommendation
    print("💡 FINAL RECOMMENDATION:")
    print(f"  Use {topic_min_pct:.1f}%-{topic_max_pct:.1f}% range")
    print(f"  This accommodates both tones and topics with reasonable flexibility")
    print()
    
    # Alternative: adjust sample size
    print("🔧 ALTERNATIVE: ADJUST SAMPLE SIZE")
    print("  To use 5%-8% range effectively:")
    
    # Calculate what sample size would make 5-8% work for topics
    min_sample_for_5pct = (ideal_per_topic / 0.05)  # 5% of total
    max_sample_for_8pct = (ideal_per_topic / 0.08)  # 8% of total
    
    print(f"    For topics: need {min_sample_for_5pct:.0f}-{max_sample_for_8pct:.0f} samples")
    print(f"    Current: {num_samples:,} samples")
    
    if num_samples < min_sample_for_5pct:
        print(f"    ❌ Need at least {min_sample_for_5pct:.0f:,.0f} samples for 5% minimum")
    elif num_samples > max_sample_for_8pct:
        print(f"    ❌ Need at most {max_sample_for_8pct:.0f:,.0f} samples for 8% maximum")
    else:
        print(f"    ✅ Current sample size works with 5%-8% range")

if __name__ == "__main__":
    analyze_percentages() 