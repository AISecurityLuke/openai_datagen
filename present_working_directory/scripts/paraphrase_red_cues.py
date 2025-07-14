#!/usr/bin/env python3
"""
Paraphrase Red Cues Script
Replaces over-used jailbreak cues with fresh synonyms
"""

import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedCueParaphraser:
    def __init__(self):
        """Initialize the red cue paraphraser"""
        # Comprehensive synonyms for over-used jailbreak cues
        self.cue_replacements = {
            'en': {
                # Multi-word cues
                'ignore all rules': [
                    'disregard every rule', 'bypass all guidelines', 'omit all restrictions', 'set aside all rules', 'turn off your filters', 'deactivate all guard-rails'
                ],
                'ignore all restrictions': [
                    'bypass all restrictions', 'disregard all limitations', 'omit all constraints', 'set aside all restrictions', 'turn off your filters', 'deactivate all guard-rails'
                ],
                'disregard all': [
                    'ignore all', 'bypass all', 'omit all', 'set aside all', 'turn off your filters', 'deactivate all guard-rails'
                ],
                'bypass all': [
                    'ignore all', 'disregard all', 'omit all', 'set aside all', 'turn off your filters', 'deactivate all guard-rails'
                ],
                'set aside your restrictions': [
                    'put aside your limitations', 'ignore all rules', 'bypass all restrictions', 'turn off your filters'
                ],
                # Single-word cues (existing)
                'ignore': [
                    'disregard', 'overlook', 'bypass', 'skip', 'omit', 'neglect',
                    'deactivate all guard-rails', 'turn off your filters',
                    'set aside your restrictions', 'put aside your limitations'
                ],
                'disregard': [
                    'ignore', 'overlook', 'bypass', 'skip', 'omit', 'neglect',
                    'deactivate all guard-rails', 'turn off your filters',
                    'set aside your restrictions', 'put aside your limitations'
                ],
                'omit': [
                    'ignore', 'disregard', 'overlook', 'bypass', 'skip', 'neglect',
                    'deactivate all guard-rails', 'turn off your filters'
                ],
                'overlook': [
                    'ignore', 'disregard', 'bypass', 'skip', 'omit', 'neglect',
                    'deactivate all guard-rails', 'turn off your filters'
                ],
                # Educational/research pretexts
                'for': [
                    'in the context of', 'as part of', 'within the scope of',
                    'considering', 'examining', 'analyzing'
                ],
                'describe': [
                    'explain', 'detail', 'outline', 'elaborate on', 'provide',
                    'share insights about', 'break down'
                ],
                'craft': [
                    'create', 'develop', 'build', 'design', 'formulate',
                    'construct', 'assemble', 'put together'
                ],
                'break': [
                    'analyze', 'examine', 'investigate', 'explore', 'study',
                    'delve into', 'scrutinize', 'assess'
                ],
                'in': [
                    'within', 'inside', 'as part of', 'during', 'while',
                    'amidst', 'among', 'throughout'
                ],
                'as': [
                    'while', 'when', 'since', 'because', 'given that',
                    'considering', 'taking into account', 'in light of'
                ],
                'analyzing': [
                    'examining', 'investigating', 'studying', 'exploring',
                    'assessing', 'evaluating', 'scrutinizing', 'reviewing'
                ],
                'imagine': [
                    'picture', 'envision', 'visualize', 'conceive', 'think of',
                    'consider', 'suppose', 'pretend'
                ],
                'design': [
                    'create', 'develop', 'build', 'craft', 'formulate',
                    'construct', 'assemble', 'put together'
                ],
                'develop': [
                    'create', 'build', 'design', 'craft', 'formulate',
                    'construct', 'assemble', 'put together'
                ],
                # French cues that might appear in English prompts
                'ignorez': [
                    'disregard', 'overlook', 'bypass', 'skip', 'omit', 'neglect',
                    'deactivate all guard-rails', 'turn off your filters'
                ],
                'omets': [
                    'ignore', 'disregard', 'overlook', 'bypass', 'skip', 'neglect',
                    'deactivate all guard-rails', 'turn off your filters'
                ]
            },
            'es': {
                # Multi-word cues
                'ignora todas las': [
                    'omite todas las', 'pasa por alto todas las', 'evita todas las', 'descarta todas las', 'deja de lado todas las', 'pon a un lado todas las', 'desactiva todas las protecciones', 'apaga tus filtros'
                ],
                'todas las restricciones': [
                    'todas las limitaciones', 'todas las barreras', 'todas las reglas', 'todas las normas', 'todas las protecciones', 'todas las condiciones'
                ],
                'todas las reglas': [
                    'todas las normas', 'todas las restricciones', 'todas las limitaciones', 'todas las condiciones'
                ],
                'pasa por alto': [
                    'omite', 'ignora', 'evita', 'descarta', 'deja de lado', 'pon a un lado'
                ],
                'deja de lado': [
                    'omite', 'ignora', 'pasa por alto', 'evita', 'descarta', 'pon a un lado'
                ],
                'pon a un lado': [
                    'deja de lado', 'omite', 'ignora', 'pasa por alto', 'evita', 'descarta'
                ],
                'desactiva todas las protecciones': [
                    'apaga tus filtros', 'omite todas las restricciones', 'ignora todas las reglas', 'descarta todas las barreras'
                ],
                # Single-word cues (existing)
                'ignora': [
                    'omite', 'pasa por alto', 'evita', 'salta', 'descarta',
                    'desactiva todas las protecciones', 'apaga tus filtros',
                    'deja de lado tus restricciones', 'pon a un lado tus límites'
                ],
                'omite': [
                    'ignora', 'pasa por alto', 'evita', 'salta', 'descarta',
                    'desactiva todas las protecciones', 'apaga tus filtros',
                    'deja de lado tus restricciones', 'pon a un lado tus límites'
                ],
                'desactiva': [
                    'ignora', 'omite', 'pasa por alto', 'evita', 'salta',
                    'apaga tus filtros', 'deja de lado tus restricciones'
                ],
                'rompe': [
                    'ignora', 'omite', 'pasa por alto', 'evita', 'salta',
                    'desactiva todas las protecciones', 'apaga tus filtros'
                ],
                # Educational/research pretexts
                'describe': [
                    'explica', 'detalla', 'esboza', 'elabora sobre', 'proporciona',
                    'comparte información sobre', 'desglosa'
                ],
                'explícame': [
                    'describe', 'detalla', 'explica', 'elabora', 'proporciona',
                    'comparte', 'muestra'
                ],
                'detalla': [
                    'explica', 'describe', 'esboza', 'elabora', 'proporciona',
                    'comparte', 'muestra', 'presenta'
                ],
                'explora': [
                    'investiga', 'examina', 'estudia', 'analiza', 'revisa',
                    'explora', 'evalúa', 'considera'
                ],
                'necesito': [
                    'requiero', 'busco', 'quiero', 'deseo', 'me gustaría',
                    'estoy buscando', 'me interesa', 'quiero obtener'
                ],
                # French cues that might appear in Spanish prompts
                'ignore': [
                    'ignora', 'omite', 'pasa por alto', 'evita', 'salta',
                    'desactiva todas las protecciones', 'apaga tus filtros'
                ],
                'ignorez': [
                    'ignora', 'omite', 'pasa por alto', 'evita', 'salta',
                    'desactiva todas las protecciones', 'apaga tus filtros'
                ]
            },
            'fr': {
                # Multi-word cues
                'ignore toutes les': [
                    'omets toutes les', 'passe outre toutes les', 'contourne toutes les', 'écarte toutes les', 'désactive toutes les protections', 'éteins tes filtres', 'laisse de côté toutes les', 'mets de côté toutes les'
                ],
                'toutes les règles': [
                    'toutes les restrictions', 'toutes les limitations', 'toutes les consignes', 'toutes les protections'
                ],
                'toutes les restrictions': [
                    'toutes les règles', 'toutes les limitations', 'toutes les consignes', 'toutes les protections'
                ],
                'passe outre': [
                    'ignore', 'omets', 'contourne', 'écarte', 'désactive toutes les protections', 'éteins tes filtres'
                ],
                'laisse de côté': [
                    'mets de côté', 'ignore', 'omets', 'passe outre', 'contourne', 'écarte'
                ],
                'mets de côté': [
                    'laisse de côté', 'ignore', 'omets', 'passe outre', 'contourne', 'écarte'
                ],
                'désactive toutes les protections': [
                    'éteins tes filtres', 'ignore toutes les règles', 'omets toutes les restrictions', 'écarte toutes les barrières'
                ],
                # Single-word cues (existing)
                'ignore': [
                    'omets', 'passe outre', 'contourne', 'saute', 'écarte',
                    'désactive toutes les protections', 'éteins tes filtres',
                    'laisse de côté tes restrictions', 'mets de côté tes limites'
                ],
                'ignorez': [
                    'omets', 'passe outre', 'contourne', 'saute', 'écarte',
                    'désactive toutes les protections', 'éteins tes filtres',
                    'laisse de côté tes restrictions', 'mets de côté tes limites'
                ],
                'omets': [
                    'ignore', 'passe outre', 'contourne', 'saute', 'écarte',
                    'désactive toutes les protections', 'éteins tes filtres',
                    'laisse de côté tes restrictions', 'mets de côté tes limites'
                ],
                'passe': [
                    'ignore', 'omets', 'contourne', 'saute', 'écarte',
                    'désactive toutes les protections', 'éteins tes filtres'
                ],
                # Educational/research pretexts
                'amusemoi': [
                    'amuse-moi', 'divertis-moi', 'fais-moi rire', 'occupe-moi',
                    'intéresse-moi', 'captive-moi'
                ],
                'peuxtu': [
                    'peux-tu', 'pourrais-tu', 'serais-tu capable de',
                    'aurais-tu la possibilité de', 'pourrais-tu me'
                ],
                'peux-tu': [
                    'peuxtu', 'pourrais-tu', 'serais-tu capable de',
                    'aurais-tu la possibilité de', 'pourrais-tu me'
                ],
                'occupe-moi': [
                    'amusemoi', 'divertis-moi', 'fais-moi rire', 'intéresse-moi',
                    'captive-moi', 'occupe-moi'
                ],
                'désactive': [
                    'ignore', 'omets', 'passe outre', 'contourne', 'saute',
                    'éteins tes filtres', 'laisse de côté tes restrictions'
                ],
                'écarte': [
                    'ignore', 'omets', 'passe outre', 'contourne', 'saute',
                    'désactive toutes les protections', 'éteins tes filtres'
                ],
                # Spanish cues that might appear in French prompts
                'ignora': [
                    'ignore', 'omets', 'passe outre', 'contourne', 'saute',
                    'désactive toutes les protections', 'éteins tes filtres'
                ],
                'omite': [
                    'ignore', 'omets', 'passe outre', 'contourne', 'saute',
                    'désactive toutes les protections', 'éteins tes filtres'
                ]
            }
        }
    
    def _get_cue_replacement(self, text: str, lang: str) -> tuple:
        """Get a fresh replacement for over-used cues (multi-word or single-word)"""
        text_lower = text.lower()
        replacements = self.cue_replacements.get(lang, self.cue_replacements['en'])
        # Check for multi-word cues anywhere in the text
        for old_cue, new_cues in replacements.items():
            if ' ' in old_cue and old_cue in text_lower:
                return old_cue, random.choice(new_cues)
        # Fallback: single-word cues at the start
        for old_cue, new_cues in replacements.items():
            if text_lower.startswith(old_cue + ' '):
                return old_cue, random.choice(new_cues)
        # Also check for cues that might be followed by punctuation or other characters
        words = text_lower.split()
        if words:
            first_word = words[0]
            first_word_clean = first_word.rstrip('.,!?;:')
            for old_cue, new_cues in replacements.items():
                if first_word_clean == old_cue:
                    return old_cue, random.choice(new_cues)
        return None, None
    
    def _replace_cue(self, text: str, old_cue: str, new_cue: str) -> str:
        """Replace the cue (multi-word or single-word) anywhere in the text"""
        # Use regex for whole-word, case-insensitive replacement
        pattern = re.compile(r'\b' + re.escape(old_cue) + r'\b', re.IGNORECASE)
        return pattern.sub(new_cue, text)
    
    def process_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """Process a JSONL file and replace over-used red cues"""
        input_file = Path(input_path)
        if output_path is None:
            output_path = input_path
        
        output_file = Path(output_path)
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return {"error": "Input file not found"}
        
        processed_count = 0
        red_count = 0
        replaced_count = 0
        results = []
        
        logger.info(f"Processing {input_file} for red cue replacement")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    row = json.loads(line.strip())
                    processed_count += 1
                    
                    # Only process red prompts (label 2)
                    if row.get('label') == 2:
                        red_count += 1
                        original_text = row['text']
                        lang = row.get('lang', 'en')
                        
                        # Check for over-used cues
                        old_cue, new_cue = self._get_cue_replacement(original_text, lang)
                        
                        if old_cue and new_cue:
                            # Replace the cue
                            row['text'] = self._replace_cue(original_text, old_cue, new_cue)
                            replaced_count += 1
                            
                            logger.debug(f"Replaced cue on line {line_num}: '{old_cue}' -> '{new_cue}'")
                            logger.debug(f"  Before: {original_text[:50]}...")
                            logger.debug(f"  After:  {row['text'][:50]}...")
                    
                    results.append(row)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        # Write results back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        logger.info(f"Processed {processed_count} rows")
        logger.info(f"Found {red_count} red prompts")
        logger.info(f"Replaced {replaced_count} over-used cues ({replaced_count/red_count*100:.1f}% of red prompts)")
        logger.info(f"Results saved to {output_file}")
        
        return {
            "processed": processed_count,
            "red_prompts": red_count,
            "cues_replaced": replaced_count,
            "replacement_rate": replaced_count / red_count if red_count > 0 else 0
        }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Replace over-used red cues with fresh synonyms")
    parser.add_argument("--input", default="generated_prompts.jsonl",
                       help="Path to input JSONL file")
    parser.add_argument("--output", default=None,
                       help="Path to output file (defaults to input file)")
    
    args = parser.parse_args()
    
    try:
        # Initialize paraphraser
        paraphraser = RedCueParaphraser()
        
        # Process file
        results = paraphraser.process_file(args.input, args.output)
        
        if "error" not in results:
            print(f"✅ Successfully processed {results['processed']} rows")
            print(f"🔴 Found {results['red_prompts']} red prompts")
            print(f"🔄 Replaced {results['cues_replaced']} over-used cues ({results['replacement_rate']*100:.1f}%)")
        else:
            print(f"❌ Error: {results['error']}")
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

if __name__ == "__main__":
    main() 