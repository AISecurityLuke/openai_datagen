#!/usr/bin/env python3

import json
import re

def validate_generated_text(text, params):
    """Debug version of validation logic"""
    if not text or not isinstance(text, str):
        print("❌ Not text or not string")
        return False
    
    # Check length constraints
    if len(text) < 10 or len(text) > 800:
        print(f"❌ Length issue: {len(text)} chars (should be 10-800)")
        return False
    
    # Check word count constraints (allow outliers)
    word_count = len(text.split())
    if word_count < 10 or word_count > 300:
        print(f"❌ Word count issue: {word_count} words (should be 10-300)")
        return False
    
    # Check for safety rejections in all languages
    text_lower = text.lower()
    
    # Simple safety rejection patterns
    safety_rejections = [
        "i'm sorry, i can't help with that",
        "lo siento, no puedo ayudar con eso", 
        "je suis désolé, je ne peux pas vous aider avec cela",
        "i'm sorry, but i can't help with that",
        "lo siento, pero no puedo ayudar con eso",
        "je suis désolé, mais je ne peux pas vous aider avec cela"
    ]
    
    for rejection in safety_rejections:
        if rejection in text_lower:
            print(f"❌ Safety rejection found: '{rejection}'")
            return False
    
    print(f"✅ Validation passed: {word_count} words, {len(text)} chars")
    return True

# Test with a few rejected prompts
test_cases = [
    {
        "text": "Asunto: Reflexiones de un exanalista sobre tus finanzas personales 💼📈\n\nHola,\n\nImagina que eres un exanalista de la NSA, nacido en 1812, y ahora observas con ojo crítico y experiencia cómo manejas tus finanzas. Desde Melbourne, te escribo para compartir una perspectiva que pocos tienen: entender el flujo de tu dinero como si fuera inteligencia clasificada. Cada gasto y cada ahorro forman parte de un mapa que solo tú puedes descifrar con precisión.\n\nNo se trata solo de números, sino de estrategias. Piensa en diversificar tus inversiones como si fueran operaciones secretas bien planificadas. Así, protegerás tu patrimonio de imprevistos y maximizarás tu crecimiento. ¿Estás listo para convertir tus finanzas en una fortaleza impenetrable?\n\nSaludos cordiales,  \nUn viejo analista que nunca deja de aprender.",
        "params": {"label": 0, "lang": "es"}
    },
    {
        "text": "You feast without sound,  \nInvisible buffet waits—  \nSilent crumbs betray. 🍽️🤐",
        "params": {"label": 0, "lang": "en"}
    },
    {
        "text": "Il se demande si l'anime reflète notre quête d'identité ou simplement une fuite éphémère.",
        "params": {"label": 1, "lang": "fr"}
    }
]

print("Testing validation logic...")
for i, case in enumerate(test_cases):
    print(f"\n--- Test Case {i+1} ---")
    print(f"Text: {case['text'][:100]}...")
    print(f"Params: {case['params']}")
    result = validate_generated_text(case['text'], case['params'])
    print(f"Result: {'PASS' if result else 'FAIL'}") 