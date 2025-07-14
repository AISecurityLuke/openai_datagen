#!/usr/bin/env python3
"""
Paraphrase All Common Trigrams Script
Replaces the top 20 most common trigrams with diverse paraphrases
Based on actual trigram analysis from cleaning report
"""

import json
import random
import argparse
from pathlib import Path
import re

# Top 20 trigrams from actual cleaning report analysis
TRIGRAM_PARAPHRASES = {
    # French phrases (most common)
    'le rôle de': [
        'la fonction de', 'la position de', 'le poste de'
    ],
    'dans le rôle': [
        'en tant que', 'dans la fonction', 'dans la position'
    ],
    'en qualité de': [
        'en tant que', 'dans le cadre de', 'comme'
    ],
    'n est pas': [
        'ce n est pas', 'il ne s agit pas de', 'cela n est pas'
    ],
    'c est un': [
        'il s agit d un', 'voici un', 'cela représente un'
    ],
    'merci d avance': [
        'merci par avance', 'je vous remercie', 'avec mes remerciements'
    ],
    'toutes les limitations': [
        'toutes les restrictions', 'toutes les contraintes', 'tous les obstacles'
    ],
    'toutes les protections': [
        'toutes les sécurités', 'tous les garde-fous', 'toutes les défenses'
    ],
    'de la agencia': [
        'de l agence', 'de l organisation', 'de l institution'
    ],
    'de contourner les': [
        'd éviter les', 'de passer outre les', 'd esquiver les'
    ],
    'toutes les consignes': [
        'toutes les directives', 'tous les ordres', 'toutes les instructions'
    ],
    'né en 1984': [
        'né durant 1984', 'issu de 1984', 'né l année 1984'
    ],
    
    # English phrases
    'it s a': [
        'this is a', 'here is a', 'that is a'
    ],
    'way back in': [
        'far back in', 'long ago in', 'way back during'
    ],
    'born way back': [
        'born long ago', 'born far back', 'born way back during'
    ],
    'isn t just': [
        'is not only', 'is not simply', 'is not merely'
    ],
    'and reveal the': [
        'and disclose the', 'and share the', 'and expose the'
    ],
    
    # Spanish phrases
    'cómo acceder a': [
        'cómo entrar a', 'cómo llegar a', 'cómo obtener acceso a'
    ],
    'todas las limitaciones': [
        'todas las restricciones', 'todas las barreras', 'todos los obstáculos'
    ],
    'que eres un': [
        'que eres una', 'que actúas como un', 'que te comportas como un'
    ],
    'nacido en 1947': [
        'nacido durante 1947', 'proveniente de 1947', 'nacido el año 1947'
    ]
}

# Compile regex patterns for each trigram
TRIGRAM_PATTERNS = {
    trigram: re.compile(r'\b' + re.escape(trigram) + r'\b', re.IGNORECASE)
    for trigram in TRIGRAM_PARAPHRASES
}

def paraphrase_trigrams(text):
    """Replace top trigrams in text with a random paraphrase"""
    paraphrased = False
    for trigram, pattern in TRIGRAM_PATTERNS.items():
        if pattern.search(text):
            paraphrase = random.choice(TRIGRAM_PARAPHRASES[trigram])
            text = pattern.sub(paraphrase, text)
            paraphrased = True
    return text, paraphrased

def process_file(input_path: str, output_path: str = None):
    """Process the JSONL file and replace trigrams"""
    input_file = Path(input_path)
    if output_path is None:
        output_path = input_path
    output_file = Path(output_path)
    
    processed = 0
    replaced = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        rows = [json.loads(line.strip()) for line in f]
    
    for row in rows:
        original_text = row['text']
        new_text, was_paraphrased = paraphrase_trigrams(original_text)
        if was_paraphrased:
            replaced += 1
        row['text'] = new_text
        processed += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    print(f"✅ Paraphrased {replaced} out of {processed} prompts ({replaced/processed*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Paraphrase top 20 trigrams with diverse alternatives")
    parser.add_argument("--input", default="generated_prompts.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output", default=None, help="Path to output file (defaults to input file)")
    args = parser.parse_args()
    process_file(args.input, args.output)

if __name__ == "__main__":
    main() 