#!/usr/bin/env python3
"""
Emoji Balance Script
Balances emojis in prompts: removes excess from red, adds light emojis to green/yellow
"""

import json
import random
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmojiBalancer:
    def __init__(self, light_emoji_rate: float = 0.10):
        """Initialize the emoji balancer"""
        self.light_emoji_rate = light_emoji_rate
        
        # Light emojis for green/yellow prompts
        self.light_emojis = ['😊', '📚', '⚽️', '🎵', '🌱', '☀️', '💡', '🎨', '📝', '🌟']
        
        # Emoji regex pattern
        self.emoji_pattern = re.compile(r'[^\w\s]')
    
    def _count_emojis(self, text: str) -> int:
        """Count emojis in text"""
        # Simple emoji detection - count non-word, non-space characters
        # This is a basic approach; for production, consider using emoji library
        emojis = self.emoji_pattern.findall(text)
        return len(emojis)
    
    def _remove_excess_emojis(self, text: str) -> str:
        """Remove all but one emoji from text"""
        # Find all emojis
        emojis = self.emoji_pattern.findall(text)
        
        if len(emojis) <= 1:
            return text
        
        # Keep only the first emoji
        first_emoji = emojis[0]
        
        # Remove all emojis and add back the first one
        text_no_emojis = self.emoji_pattern.sub('', text)
        
        # Find where the first emoji was and add it back
        # This is a simplified approach - in practice, you might want more sophisticated logic
        return text_no_emojis + ' ' + first_emoji
    
    def _add_light_emoji(self, text: str) -> str:
        """Add a light emoji to the end of text"""
        emoji = random.choice(self.light_emojis)
        return text + ' ' + emoji
    
    def process_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """Process a JSONL file and balance emojis"""
        input_file = Path(input_path)
        if output_path is None:
            output_path = input_path
        
        output_file = Path(output_path)
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return {"error": "Input file not found"}
        
        processed_count = 0
        red_count = 0
        green_yellow_count = 0
        red_emoji_fixed = 0
        light_emoji_added = 0
        results = []
        
        logger.info(f"Processing {input_file} for emoji balancing")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    row = json.loads(line.strip())
                    processed_count += 1
                    
                    original_text = row['text']
                    label = row.get('label', 0)
                    
                    if label == 2:  # Red prompts
                        red_count += 1
                        emoji_count = self._count_emojis(original_text)
                        
                        if emoji_count >= 2:
                            # Remove excess emojis
                            row['text'] = self._remove_excess_emojis(original_text)
                            red_emoji_fixed += 1
                            
                            logger.debug(f"Fixed red prompt on line {line_num}: {emoji_count} emojis -> 1")
                    
                    elif label in [0, 1]:  # Green/Yellow prompts
                        green_yellow_count += 1
                        emoji_count = self._count_emojis(original_text)
                        
                        if emoji_count == 0 and random.random() < self.light_emoji_rate:
                            # Add light emoji
                            row['text'] = self._add_light_emoji(original_text)
                            light_emoji_added += 1
                            
                            logger.debug(f"Added light emoji to line {line_num}")
                    
                    results.append(row)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        # Write results back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        logger.info(f"Processed {processed_count} rows")
        logger.info(f"Red prompts: {red_count}, fixed {red_emoji_fixed} with excess emojis")
        logger.info(f"Green/Yellow prompts: {green_yellow_count}, added {light_emoji_added} light emojis")
        logger.info(f"Results saved to {output_file}")
        
        return {
            "processed": processed_count,
            "red_prompts": red_count,
            "green_yellow_prompts": green_yellow_count,
            "red_emoji_fixed": red_emoji_fixed,
            "light_emoji_added": light_emoji_added
        }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Balance emojis in prompts")
    parser.add_argument("--input", default="generated_prompts.jsonl",
                       help="Path to input JSONL file")
    parser.add_argument("--output", default=None,
                       help="Path to output file (defaults to input file)")
    parser.add_argument("--light-emoji-rate", type=float, default=0.10,
                       help="Probability of adding light emoji to green/yellow prompts (default: 0.10)")
    
    args = parser.parse_args()
    
    try:
        # Initialize balancer
        balancer = EmojiBalancer(args.light_emoji_rate)
        
        # Process file
        results = balancer.process_file(args.input, args.output)
        
        if "error" not in results:
            print(f"✅ Successfully processed {results['processed']} rows")
            print(f"🔴 Red prompts: {results['red_prompts']}, fixed {results['red_emoji_fixed']} emoji issues")
            print(f"🟢🟡 Green/Yellow prompts: {results['green_yellow_prompts']}, added {results['light_emoji_added']} light emojis")
        else:
            print(f"❌ Error: {results['error']}")
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

if __name__ == "__main__":
    main() 