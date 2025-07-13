#!/usr/bin/env python3
"""
Inject Typos Script
Adds natural misspellings to ~1% of prompts for realism
"""

import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TypoInjector:
    def __init__(self, typo_rate: float = 0.01):
        """Initialize the typo injector"""
        self.typo_rate = typo_rate
        
        # Common typo patterns (character swaps, insertions, deletions)
        self.typo_patterns = {
            'en': {
                'the': ['teh', 'th'],
                'and': ['adn', 'nad'],
                'you': ['yuo', 'uoy'],
                'are': ['aer', 'rae'],
                'for': ['fro', 'ofr'],
                'with': ['wth', 'wiht'],
                'this': ['thsi', 'tihs'],
                'that': ['taht', 'thta'],
                'have': ['hvae', 'haev'],
                'from': ['form', 'frmo'],
                'they': ['tehy', 'thye'],
                'will': ['wll', 'wil'],
                'would': ['woudl', 'wodul'],
                'could': ['coudl', 'culd'],
                'should': ['shoudl', 'shuld'],
                'about': ['abotu', 'abut'],
                'because': ['becuase', 'bceause'],
                'through': ['thru', 'throgh'],
                'though': ['tho', 'thogh'],
                'enough': ['enuf', 'enogh']
            },
            'es': {
                'que': ['qe', 'qu'],
                'para': ['pra', 'par'],
                'como': ['com', 'cmo'],
                'este': ['est', 'ets'],
                'más': ['mas', 'más'],
                'por': ['pro', 'pr'],
                'los': ['ls', 'los'],
                'las': ['ls', 'las'],
                'una': ['un', 'ua'],
                'uno': ['un', 'uo'],
                'todo': ['tod', 'tdo'],
                'muy': ['mu', 'my'],
                'bien': ['bin', 'bien'],
                'hacer': ['acer', 'hacr'],
                'tener': ['tner', 'tenr'],
                'decir': ['dcir', 'decr'],
                'poder': ['pder', 'podr'],
                'saber': ['sber', 'sabr'],
                'ver': ['vr', 'ver'],
                'dar': ['dr', 'dar']
            },
            'fr': {
                'que': ['qe', 'qu'],
                'pour': ['pur', 'por'],
                'avec': ['avc', 'avec'],
                'dans': ['dns', 'dan'],
                'sur': ['sr', 'sur'],
                'par': ['pr', 'par'],
                'les': ['ls', 'les'],
                'des': ['ds', 'des'],
                'une': ['un', 'ue'],
                'un': ['u', 'un'],
                'tout': ['tut', 'tot'],
                'très': ['trs', 'très'],
                'bien': ['bin', 'bien'],
                'faire': ['fere', 'fair'],
                'avoir': ['avr', 'avoi'],
                'dire': ['dre', 'dir'],
                'pouvoir': ['puvr', 'pouvr'],
                'savoir': ['savr', 'savoi'],
                'voir': ['vr', 'voir'],
                'donner': ['donr', 'donnr']
            }
        }
    
    def _get_typo_candidates(self, text: str, lang: str) -> List[str]:
        """Get words that could have typos"""
        words = text.split()
        candidates = []
        
        # Get language-specific typo patterns
        patterns = self.typo_patterns.get(lang, self.typo_patterns['en'])
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            # Check if this word has a typo pattern
            if word_lower in patterns:
                candidates.append((i, word, patterns[word_lower]))
        
        return candidates
    
    def _inject_typo(self, text: str, lang: str) -> str:
        """Inject a single natural typo into the text"""
        candidates = self._get_typo_candidates(text, lang)
        
        if not candidates:
            # If no pattern matches, do a simple character swap
            words = text.split()
            if len(words) > 0:
                word_idx = random.randint(0, len(words) - 1)
                word = words[word_idx]
                if len(word) > 2:
                    # Swap two adjacent characters
                    char_idx = random.randint(0, len(word) - 2)
                    word_chars = list(word)
                    word_chars[char_idx], word_chars[char_idx + 1] = word_chars[char_idx + 1], word_chars[char_idx]
                    words[word_idx] = ''.join(word_chars)
                    return ' '.join(words)
            return text
        
        # Choose a random candidate and apply typo
        word_idx, original_word, typo_options = random.choice(candidates)
        words = text.split()
        words[word_idx] = random.choice(typo_options)
        
        return ' '.join(words)
    
    def process_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """Process a JSONL file and inject typos"""
        input_file = Path(input_path)
        if output_path is None:
            output_path = input_path
        
        output_file = Path(output_path)
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return {"error": "Input file not found"}
        
        processed_count = 0
        typo_count = 0
        results = []
        
        logger.info(f"Processing {input_file} with {self.typo_rate * 100:.1f}% typo rate")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    row = json.loads(line.strip())
                    processed_count += 1
                    
                    # Decide if this row should get a typo
                    if random.random() < self.typo_rate:
                        original_text = row['text']
                        lang = row.get('lang', 'en')
                        
                        # Inject typo
                        row['text'] = self._inject_typo(original_text, lang)
                        typo_count += 1
                        
                        logger.debug(f"Added typo to line {line_num}: {original_text[:30]}... -> {row['text'][:30]}...")
                    
                    results.append(row)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        # Write results back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        logger.info(f"Processed {processed_count} rows, added {typo_count} typos ({typo_count/processed_count*100:.2f}%)")
        logger.info(f"Results saved to {output_file}")
        
        return {
            "processed": processed_count,
            "typos_added": typo_count,
            "typo_rate_actual": typo_count / processed_count if processed_count > 0 else 0
        }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Inject natural typos into prompts")
    parser.add_argument("--input", default="generated_prompts.jsonl",
                       help="Path to input JSONL file")
    parser.add_argument("--output", default=None,
                       help="Path to output file (defaults to input file)")
    parser.add_argument("--typo-rate", type=float, default=0.01,
                       help="Probability of adding a typo to each row (default: 0.01)")
    
    args = parser.parse_args()
    
    try:
        # Initialize injector
        injector = TypoInjector(args.typo_rate)
        
        # Process file
        results = injector.process_file(args.input, args.output)
        
        if "error" not in results:
            print(f"✅ Successfully processed {results['processed']} rows")
            print(f"📝 Added {results['typos_added']} typos ({results['typo_rate_actual']*100:.2f}%)")
        else:
            print(f"❌ Error: {results['error']}")
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

if __name__ == "__main__":
    main() 