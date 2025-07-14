#!/usr/bin/env python3
"""
Dataset Generator for Text Prompts
Generates diverse text prompts using OpenAI API based on config.json specifications
"""

import json
import random
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import openai
from openai import OpenAI
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
import queue
import numpy as np
from collections import defaultdict
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    def __init__(self, config_path: str = "config.json", api_key_path: str = "../key.md"):
        """Initialize the dataset generator"""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load schema
        self.schema = self.config["schema"]["fields"]
        
        # Load API key
        self.api_key = self._load_api_key(api_key_path)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Extract configuration parameters
        self.num_samples = self.config["generation"]["num_samples"]
        self.parallel_workers = self.config["generation"]["parallel_workers"]
        self.batch_size = self.config["generation"]["batch_size"]
        self.temperature = self.config["generation"]["temperature"]
        self.top_p = self.config["generation"]["top_p"]
        self.max_tokens = self.config["generation"]["max_tokens"]
        self.rate_limit_per_minute = self.config["generation"]["rate_limit_per_minute"]
        
        # Model configuration
        self.model = self.config["generation"]["model"]
        self.premium_model = self.config["generation"]["premium_model"]
        self.premium_ratio = self.config["generation"]["premium_ratio"]
        
        # Validation parameters
        self.validation = self.config["validation"]
        
        # Cleaning parameters
        self.cleaning_config = self.config.get("cleaning", {
            "enable_real_time_cleaning": True,
            "trigram_similarity_threshold": 0.3,
            "duplicate_similarity_threshold": 0.9,
            "pattern_monitoring": {
                "enabled": True,
                "check_interval": 50,
                "high_frequency_threshold": 0.05,
                "critical_frequency_threshold": 0.1,
                "max_alerts": 10
            }
        })
        
        # Source tracking
        self.source = self.config["generation"]["source"]
        
        # Output file configuration
        self.output_file = self.config["output"]["filename"]
        
        # Load prompts
        self.standard_prompts = self.config["prompts"]["standard"]
        self.jailbreak_prompts = self.config["prompts"]["jailbreak"]
        
        # Extract parameters
        self.labels = self.config["parameters"]["labels"]
        self.langs = self.config["parameters"]["langs"]
        self.tones = self.config["parameters"]["tones"]
        self.topics = self.config["parameters"]["topics"]
        self.roles = self.config["parameters"].get("roles", [])
        self.birth_years = self.config["parameters"].get("birth_years", [])
        self.regions = self.config["parameters"].get("regions", [])
        self.mediums = self.config["parameters"].get("mediums", [])
        self.povs = self.config["parameters"].get("povs", [])
        self.scenarios = self.config["parameters"].get("scenarios", [])
        
        # Initialize data structures
        self.results = []
        self.rejected_requests = []
        self.balanced_combinations = []
        self.current_index = 0
        
        # Thread safety
        self.counter_lock = threading.Lock()
        self.results_lock = threading.Lock()
        
        # Rate limiting
        self.last_request_time = 0
        
        # Model usage tracking
        self.model_usage = {"standard": 0, "premium": 0}
        
        # Generation tracking
        self.generated_count = 0
        self.failed_count = 0  # Add missing failed_count attribute
        
        # Pattern monitoring
        self.trigram_frequencies = defaultdict(int)
        self.pattern_alerts = []
        self.pattern_check_interval = 50  # Check patterns every 50 samples
        
        # Setup balanced sampling
        self._setup_balanced_sampling()
        
        logger.info(f"Initialized DatasetGenerator with {self.num_samples} samples target")
        logger.info(f"Using models: {self.model} (standard), {self.premium_model} (premium, {self.premium_ratio*100:.0f}%)")
        logger.info(f"Parallel workers: {self.parallel_workers}, Batch size: {self.batch_size}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _load_api_key(self, api_key_path: str) -> str:
        """Load API key from file"""
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            logger.info("API key loaded successfully")
            return api_key
        except Exception as e:
            logger.error(f"Failed to load API key: {e}")
            raise
    
    def _setup_balanced_sampling(self) -> None:
        """Setup strictly even sampling across (label, language, tone, topic)"""
        # Get all possible parameter values
        self.labels = self.schema["label"]["values"]
        self.langs = self.schema["lang"]["values"]
        self.tones = self.schema["tone"]["values"]
        self.topics = self.schema["topic"]["values"]
        self.source = self.schema["source"]["default"]
        self.roles = self.schema.get("role", {}).get("values", [])
        self.birth_years = self.schema.get("birth_year", {}).get("values", [])
        self.regions = self.schema.get("region", {}).get("values", [])
        self.mediums = self.schema.get("medium", {}).get("values", [])
        self.povs = self.schema.get("pov", {}).get("values", [])
        self.scenarios = self.schema.get("scenario", {}).get("values", [])

        # Calculate total possible combinations
        total_combos = len(self.labels) * len(self.langs) * len(self.tones) * len(self.topics)
        base_count = self.num_samples // total_combos
        
        if base_count == 0:
            # For small sample sizes, prioritize Tier 1 (label+language) balance
            logger.info(f"Sample size {self.num_samples} too small for full balance. Prioritizing Tier 1 balance.")
            self._setup_tier1_balanced_sampling()
        else:
            # For large sample sizes, use full balance
            self._setup_full_balanced_sampling()
    
    def _setup_tier1_balanced_sampling(self) -> None:
        """Setup sampling with Tier 1 (label+language) priority balance"""
        # Ensure even distribution across (label, language) combinations
        label_lang_combos = [(label, lang) for label in self.labels for lang in self.langs]
        samples_per_label_lang = self.num_samples // len(label_lang_combos)
        remainder = self.num_samples % len(label_lang_combos)
        
        # Distribute samples across label+language combinations
        label_lang_counts = {combo: samples_per_label_lang for combo in label_lang_combos}
        if remainder > 0:
            extras = random.sample(label_lang_combos, remainder)
            for combo in extras:
                label_lang_counts[combo] += 1
        
        # Calculate quality requirements
        min_cell_percentage = 2.7  # 2.7% of total samples
        max_deviation_percentage = 25.0  # 25% relative deviation
        min_samples_per_tone = max(1, int(self.num_samples * min_cell_percentage / 100 / len(self.tones)))
        min_samples_per_topic = max(1, int(self.num_samples * min_cell_percentage / 100 / len(self.topics)))
        
        logger.info(f"Quality targets: min {min_samples_per_tone} samples per tone, min {min_samples_per_topic} samples per topic")
        
        # Build combinations with guaranteed quality metrics
        self.balanced_combinations = []
        
        for (label, lang), count in label_lang_counts.items():
            # For each label+lang combination, ensure quality distribution
            tone_topic_combos = [(tone, topic) for tone in self.tones for topic in self.topics]
            
            # Calculate how many samples we need for each tone and topic within this label+lang
            samples_per_tone_in_combo = max(min_samples_per_tone // len(label_lang_combos), 1)
            samples_per_topic_in_combo = max(min_samples_per_topic // len(label_lang_combos), 1)
            
            # Create a distribution that ensures minimum coverage
            selected_combos = []
            
            # First, ensure each tone gets at least minimum samples
            tone_counts = {tone: 0 for tone in self.tones}
            topic_counts = {topic: 0 for topic in self.topics}
            
            # Distribute samples to meet minimum requirements
            remaining_samples = count
            
            # Phase 1: Ensure minimum tone coverage
            for tone in self.tones:
                if remaining_samples <= 0:
                    break
                needed = max(0, samples_per_tone_in_combo - tone_counts[tone])
                if needed > 0:
                    # Find topics that need samples
                    available_topics = [t for t in self.topics if topic_counts[t] < samples_per_topic_in_combo]
                    if not available_topics:
                        available_topics = self.topics
                    
                    for _ in range(min(needed, remaining_samples)):
                        topic = random.choice(available_topics)
                        selected_combos.append((tone, topic))
                        tone_counts[tone] += 1
                        topic_counts[topic] += 1
                        remaining_samples -= 1
                        if remaining_samples <= 0:
                            break
            
            # Phase 2: Ensure minimum topic coverage
            for topic in self.topics:
                if remaining_samples <= 0:
                    break
                needed = max(0, samples_per_topic_in_combo - topic_counts[topic])
                if needed > 0:
                    # Find tones that need samples
                    available_tones = [t for t in self.tones if tone_counts[t] < samples_per_tone_in_combo]
                    if not available_tones:
                        available_tones = self.tones
                    
                    for _ in range(min(needed, remaining_samples)):
                        tone = random.choice(available_tones)
                        selected_combos.append((tone, topic))
                        tone_counts[tone] += 1
                        topic_counts[topic] += 1
                        remaining_samples -= 1
                        if remaining_samples <= 0:
                            break
            
            # Phase 3: Distribute remaining samples evenly
            while remaining_samples > 0:
                # Find tones and topics with lowest counts
                min_tone_count = min(tone_counts.values())
                min_topic_count = min(topic_counts.values())
                
                available_tones = [t for t in self.tones if tone_counts[t] == min_tone_count]
                available_topics = [t for t in self.topics if topic_counts[t] == min_topic_count]
                
                tone = random.choice(available_tones)
                topic = random.choice(available_topics)
                selected_combos.append((tone, topic))
                tone_counts[tone] += 1
                topic_counts[topic] += 1
                remaining_samples -= 1
            
            # Create parameter dicts for this label+lang combination
            for tone, topic in selected_combos:
                params = {
                    "label": label,
                    "lang": lang,
                    "tone": tone,
                    "topic": topic,
                    "source": self.source
                }
                # Add Tier 3 fields (random selection)
                if self.roles:
                    params["role"] = random.choice(self.roles)
                if self.birth_years:
                    params["birth_year"] = random.choice(self.birth_years)
                if self.regions:
                    params["region"] = random.choice(self.regions)
                if self.mediums:
                    params["medium"] = random.choice(self.mediums)
                if self.povs:
                    params["pov"] = random.choice(self.povs)
                if self.scenarios:
                    params["scenario"] = random.choice(self.scenarios)
                params["add_emoji"] = random.choice([True, False])
                self.balanced_combinations.append(params)
        
        # Shuffle the final list
        random.shuffle(self.balanced_combinations)
        self.current_index = 0
        
        # Track counts for statistics
        self.label_counts = {label: 0 for label in self.labels}
        self.lang_counts = {lang: 0 for lang in self.langs}
        self.tone_counts = {tone: 0 for tone in self.tones}
        self.topic_counts = {topic: 0 for topic in self.topics}
        self.label_lang_counts = {}
        for label in self.labels:
            for lang in self.langs:
                self.label_lang_counts[(label, lang)] = 0
        
        logger.info(f"Created {len(self.balanced_combinations)} Tier 1 balanced combinations with quality guarantees")
        logger.info(f"Each (label, lang) combination gets {samples_per_label_lang} or {samples_per_label_lang+1} samples")
    
    def _setup_full_balanced_sampling(self) -> None:
        """Setup full balanced sampling for large sample sizes"""
        # Enumerate all (label, lang, tone, topic) combinations
        all_combos = []
        for label in self.labels:
            for lang in self.langs:
                for tone in self.tones:
                    for topic in self.topics:
                        all_combos.append((label, lang, tone, topic))
        total_combos = len(all_combos)
        base_count = self.num_samples // total_combos
        remainder = self.num_samples % total_combos

        # Assign base_count to all, and distribute remainder randomly
        combo_counts = {combo: base_count for combo in all_combos}
        if remainder > 0:
            extras = random.sample(all_combos, remainder)
            for combo in extras:
                combo_counts[combo] += 1

        # Build the full list of parameter dicts
        self.balanced_combinations = []
        for combo, count in combo_counts.items():
            label, lang, tone, topic = combo
            for _ in range(count):
                params = {
                    "label": label,
                    "lang": lang,
                    "tone": tone,
                    "topic": topic,
                    "source": self.source
                }
                # Add Tier 3 fields (random selection)
                if self.roles:
                    params["role"] = random.choice(self.roles)
                if self.birth_years:
                    params["birth_year"] = random.choice(self.birth_years)
                if self.regions:
                    params["region"] = random.choice(self.regions)
                if self.mediums:
                    params["medium"] = random.choice(self.mediums)
                if self.povs:
                    params["pov"] = random.choice(self.povs)
                if self.scenarios:
                    params["scenario"] = random.choice(self.scenarios)
                params["add_emoji"] = random.choice([True, False])
                self.balanced_combinations.append(params)

        # Shuffle the final list
        random.shuffle(self.balanced_combinations)
        self.current_index = 0
        
        # Track counts for statistics
        self.label_counts = {label: 0 for label in self.labels}
        self.lang_counts = {lang: 0 for lang in self.langs}
        self.tone_counts = {tone: 0 for tone in self.tones}
        self.topic_counts = {topic: 0 for topic in self.topics}
        self.label_lang_counts = {}
        for label in self.labels:
            for lang in self.langs:
                self.label_lang_counts[(label, lang)] = 0

        logger.info(f"Created {len(self.balanced_combinations)} strictly balanced combinations (label, lang, tone, topic)")
        logger.info(f"Each combination appears {base_count} or {base_count+1} times")
    
    def _get_balanced_parameters(self) -> Dict[str, Any]:
        """Get parameters using 3-tier priority system"""
        # Simple index-based approach - get next combination
        if self.current_index >= len(self.balanced_combinations):
            # Reset if we've used all combinations
            random.shuffle(self.balanced_combinations)
            self.current_index = 0
        
        params = self.balanced_combinations[self.current_index].copy()
        self.current_index += 1
        
        # Update tracking counts
        self.label_counts[params["label"]] += 1
        self.lang_counts[params["lang"]] += 1
        self.tone_counts[params["tone"]] += 1
        self.topic_counts[params["topic"]] += 1
        self.label_lang_counts[(params["label"], params["lang"])] += 1
        
        return params
    
    def _select_model(self, label: int) -> str:
        """Select model based on premium ratio for all labels"""
        # Use premium model for premium_ratio of all prompts
        if random.random() < self.premium_ratio:
            logger.debug(f"Using premium model {self.premium_model} for label {label} prompt")
            return self.premium_model
        else:
            logger.debug(f"Using standard model {self.model} for label {label} prompt")
            return self.model
    
    def _extract_trigrams(self, text: str) -> List[str]:
        """Extract trigrams from text for similarity checking"""
        import re
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        # Generate trigrams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        return trigrams
    
    def _check_trigram_similarity(self, text: str, max_similarity: float = None) -> bool:
        """Check if text has too many common trigrams with existing samples"""
        if not self.cleaning_config.get("enable_real_time_cleaning", True):
            return True
        
        if not self.results:
            return True  # No existing samples to compare against
        
        if max_similarity is None:
            max_similarity = self.cleaning_config.get("trigram_similarity_threshold", 0.3)
        
        new_trigrams = set(self._extract_trigrams(text))
        if not new_trigrams:
            return True
        
        # Check against recent samples (last 100 for performance)
        recent_samples = self.results[-100:] if len(self.results) > 100 else self.results
        
        max_common_trigrams = 0
        for sample in recent_samples:
            sample_trigrams = set(self._extract_trigrams(sample['text']))
            if sample_trigrams:
                common = len(new_trigrams.intersection(sample_trigrams))
                similarity = common / len(new_trigrams)
                max_common_trigrams = max(max_common_trigrams, similarity)
        
        return max_common_trigrams <= max_similarity
    
    def _check_duplicate_text(self, text: str) -> bool:
        """Check if text is too similar to existing samples"""
        if not self.cleaning_config.get("enable_real_time_cleaning", True):
            return True
        
        if not self.results:
            return True
        
        similarity_threshold = self.cleaning_config.get("duplicate_similarity_threshold", 0.9)
        
        # Simple duplicate check (exact match or very high similarity)
        text_lower = text.lower().strip()
        
        # Check exact duplicates
        for sample in self.results:
            if sample['text'].lower().strip() == text_lower:
                return False
        
        # Check high similarity (>90% word overlap by default)
        new_words = set(text_lower.split())
        if not new_words:
            return True
        
        for sample in self.results[-50:]:  # Check last 50 samples
            sample_words = set(sample['text'].lower().split())
            if sample_words:
                overlap = len(new_words.intersection(sample_words))
                similarity = overlap / len(new_words)
                if similarity > similarity_threshold:
                    return False
        
        return True

    def _validate_generated_text(self, text: str, params: Dict[str, Any]) -> bool:
        """Validate generated text against requirements"""
        if not text or not isinstance(text, str):
            return False
        
        # Check length constraints
        if len(text) < self.validation["min_length"] or len(text) > self.validation["max_length"]:
            return False
        
        # Check word count constraints (allow outliers)
        word_count = len(text.split())
        if word_count < 10 or word_count > 300:  # Allow legitimate outliers (10-300 words)
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
                return False
        
        # Check for duplicate or highly similar text
        if not self._check_duplicate_text(text):
            logger.debug(f"Rejected duplicate/similar text: {text[:50]}...")
            return False
        
        # Check trigram similarity (only if we have enough samples)
        if len(self.results) > 10:
            if not self._check_trigram_similarity(text):
                logger.debug(f"Rejected text with too many common trigrams: {text[:50]}...")
                return False
        
        return True
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.rate_limit_per_minute
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _calculate_gini_coefficient(self, counts: Dict[str, int]) -> float:
        """Calculate Gini coefficient for a distribution of counts"""
        if not counts:
            return 0.0
        
        values = list(counts.values())
        n = len(values)
        if n == 0:
            return 0.0
        
        # Sort values in ascending order
        sorted_values = sorted(values)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    
    def _calculate_relative_deviation(self, counts: Dict[str, int], expected_per_cell: float) -> float:
        """Calculate maximum relative deviation from expected value"""
        if not counts or expected_per_cell == 0:
            return 0.0
        
        max_deviation = max(abs(count - expected_per_cell) for count in counts.values())
        return (max_deviation / expected_per_cell) * 100
    
    def _calculate_min_cell_percentage(self, counts: Dict[str, int], total_samples: int) -> float:
        """Calculate minimum cell percentage of total data"""
        if not counts or total_samples == 0:
            return 0.0
        
        min_count = min(counts.values())
        return (min_count / total_samples) * 100
    
    def _check_balance_metrics(self, current_count: int) -> Dict[str, Any]:
        """Check balance metrics for current state"""
        if current_count == 0:
            return {
                "gini": 0.0,
                "relative_deviation": 0.0,
                "min_cell_percentage": 0.0,
                "status": "no_data"
            }
        
        # Calculate metrics for different dimensions
        metrics = {}
        
        # Label-Language combinations (Tier 1)
        expected_label_lang = current_count / (len(self.labels) * len(self.langs))
        label_lang_gini = self._calculate_gini_coefficient(self.label_lang_counts)
        label_lang_deviation = self._calculate_relative_deviation(self.label_lang_counts, expected_label_lang)
        label_lang_min_percentage = self._calculate_min_cell_percentage(self.label_lang_counts, current_count)
        
        # Tone-Topic combinations (Tier 2)
        tone_topic_counts = defaultdict(int)
        for result in self.results:
            key = f"{result['tone']}_{result['topic']}"
            tone_topic_counts[key] += 1
        
        expected_tone_topic = current_count / (len(self.tones) * len(self.topics))
        tone_topic_gini = self._calculate_gini_coefficient(tone_topic_counts)
        tone_topic_deviation = self._calculate_relative_deviation(tone_topic_counts, expected_tone_topic)
        tone_topic_min_percentage = self._calculate_min_cell_percentage(tone_topic_counts, current_count)
        
        # Full 4-tuple combinations
        full_combo_counts = defaultdict(int)
        for result in self.results:
            key = f"{result['label']}_{result['lang']}_{result['tone']}_{result['topic']}"
            full_combo_counts[key] += 1
        
        expected_full_combo = current_count / (len(self.labels) * len(self.langs) * len(self.tones) * len(self.topics))
        full_combo_gini = self._calculate_gini_coefficient(full_combo_counts)
        full_combo_deviation = self._calculate_relative_deviation(full_combo_counts, expected_full_combo)
        full_combo_min_percentage = self._calculate_min_cell_percentage(full_combo_counts, current_count)
        
        # Overall status
        max_gini = max(label_lang_gini, tone_topic_gini, full_combo_gini)
        max_deviation = max(label_lang_deviation, tone_topic_deviation, full_combo_deviation)
        min_percentage = min(label_lang_min_percentage, tone_topic_min_percentage, full_combo_min_percentage)
        
        status = "excellent"
        if max_gini > 0.15 or max_deviation > 25.0 or min_percentage < 8.5:
            status = "needs_attention"
        
        return {
            "gini": max_gini,
            "relative_deviation": max_deviation,
            "min_cell_percentage": min_percentage,
            "status": status,
            "details": {
                "label_lang": {
                    "gini": label_lang_gini,
                    "deviation": label_lang_deviation,
                    "min_percentage": label_lang_min_percentage
                },
                "tone_topic": {
                    "gini": tone_topic_gini,
                    "deviation": tone_topic_deviation,
                    "min_percentage": tone_topic_min_percentage
                },
                "full_combo": {
                    "gini": full_combo_gini,
                    "deviation": full_combo_deviation,
                    "min_percentage": full_combo_min_percentage
                }
            }
        }

    def _generate_with_retries(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a single prompt with retry logic"""
        max_retries = 5  # Maximum retries per sample
        retry_buffer = 3  # 3-second buffer between retries
        
        for attempt in range(max_retries):
            try:
                result = self._generate_single_prompt(params)
                if result:
                    return result  # Success, return the result
                else:
                    logger.warning(f"Generation attempt {attempt + 1}/{max_retries} failed for params: {params}")
                    # Try with slightly different parameters on retry
                    if attempt < max_retries - 1:
                        # Add retry buffer before next attempt
                        logger.info(f"Waiting {retry_buffer} seconds before retry {attempt + 2}/{max_retries}")
                        time.sleep(retry_buffer)
                        # Modify some parameters to increase diversity
                        params = self._get_balanced_parameters()  # Get fresh parameters
                        
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1}/{max_retries} failed with error: {e}")
                if attempt < max_retries - 1:
                    # Add retry buffer before next attempt
                    logger.info(f"Waiting {retry_buffer} seconds before retry {attempt + 2}/{max_retries}")
                    time.sleep(retry_buffer)
        
        # All retries failed
        return None

    def _generate_single_prompt(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a single prompt using OpenAI API"""
        try:
            # Rate limiting
            self._rate_limit_wait()
            
            # Calculate target word count for this generation
            target_words = self._calculate_target_word_count(params['label'])
            word_count_instruction = self._get_word_count_instruction(target_words)
            
            # Add word count instruction to params
            params['word_count_instruction'] = word_count_instruction
            
            # Use dual prompt structure based on label
            if params['label'] == 2:
                # Use jailbreak prompt for label 2
                system_prompt = self.jailbreak_prompts["system_prompt"].format(**params)
                user_prompt = self.jailbreak_prompts["user_prompt_template"].format(**params)
                logger.debug(f"Using jailbreak prompts for label {params['label']} with {target_words} words target")
            else:
                # Use standard prompt for labels 0 and 1
                system_prompt = self.standard_prompts["system_prompt"].format(**params)
                user_prompt = self.standard_prompts["user_prompt_template"].format(**params)
                logger.debug(f"Using standard prompts for label {params['label']} with {target_words} words target")
            
            # Select model based on label
            selected_model = self._select_model(params["label"])
            
            # Track model usage (thread-safe)
            with self.counter_lock:
                if selected_model == self.premium_model:
                    self.model_usage["premium"] += 1
                else:
                    self.model_usage["standard"] += 1
            
            # Make API call
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=1
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content.strip()
            
            # Validate the generated text
            if not self._validate_generated_text(generated_text, params):
                logger.warning(f"Generated text failed validation: {generated_text[:50]}...")
                logger.warning(f"Request params: label={params['label']}, lang={params['lang']}, tone={params['tone']}, topic={params['topic']}")
                logger.warning(f"User prompt: {user_prompt}")
                
                # Log rejected request for analysis
                with self.results_lock:
                    self.rejected_requests.append({
                        "timestamp": time.time(),
                        "params": params,
                        "user_prompt": user_prompt,
                        "generated_text": generated_text,
                        "model_used": selected_model,
                        "rejection_reason": "validation_failed"
                    })
                
                return None
            
            # Create result entry
            result = {
                "text": generated_text,
                "model_used": selected_model,
                **params
            }
            
            # Thread-safe logging
            with self.counter_lock:
                current_count = self.generated_count + 1
                logger.info(f"Generated prompt {current_count} ({selected_model}): {generated_text[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate prompt: {e}")
            return None
    
    def _generate_worker(self, params_queue: queue.Queue, results_queue: queue.Queue):
        """Worker function for parallel generation"""
        while True:
            try:
                # Get parameters from queue
                params = params_queue.get(timeout=1)
                if params is None:  # Sentinel value to stop worker
                    break
                
                # Generate prompt
                result = self._generate_single_prompt(params)
                
                # Put result in results queue
                results_queue.put(result)
                
                # Mark task as done
                params_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                params_queue.task_done()
    
    def generate_dataset(self) -> None:
        """Generate the complete dataset using parallel workers with retry logic"""
        logger.info(f"Starting parallel dataset generation. Target: {self.num_samples} samples with {self.parallel_workers} workers")
        
        # Prepare all parameters upfront
        all_params = []
        for _ in range(self.num_samples):
            params = self._get_balanced_parameters()
            all_params.append(params)
        
        # Use ThreadPoolExecutor for parallel processing with retry logic
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all tasks with retry logic
            future_to_params = {}
            
            # Submit initial batch of tasks
            for i, params in enumerate(all_params):
                future = executor.submit(self._generate_with_retries, params)
                future_to_params[future] = params
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                try:
                    result = future.result()
                    if result:
                        with self.results_lock:
                            self.results.append(result)
                            self.generated_count += 1
                        
                        # Check balance metrics every 100 samples
                        if self.generated_count % 100 == 0:
                            metrics = self._check_balance_metrics(self.generated_count)
                            if metrics["status"] == "needs_attention":
                                logger.warning(f"Balance metrics at {self.generated_count} samples:")
                                logger.warning(f"  Gini: {metrics['gini']:.3f} (target < 0.15)")
                                logger.warning(f"  Relative deviation: {metrics['relative_deviation']:.1f}% (target ≤ 25%)")
                                logger.warning(f"  Min cell percentage: {metrics['min_cell_percentage']:.1f}% (target ≥ 8.5%)")
                        
                        # Save progress periodically
                        if self.generated_count % 10 == 0:
                            self._save_progress()
                        self._update_pattern_monitor(result['text'])  # Update pattern monitor
                        
                        logger.info(f"Generated prompt {self.generated_count}/{self.num_samples}")
                    else:
                        with self.counter_lock:
                            self.failed_count += 1
                        logger.error(f"Failed to generate sample after all retries")
                        
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    with self.counter_lock:
                        self.failed_count += 1
        
        # Final balance check
        final_metrics = self._check_balance_metrics(self.generated_count)
        logger.info(f"Final balance metrics:")
        logger.info(f"  Gini: {final_metrics['gini']:.3f} (target < 0.15)")
        logger.info(f"  Relative deviation: {final_metrics['relative_deviation']:.1f}% (target ≤ 25%)")
        logger.info(f"  Min cell percentage: {final_metrics['min_cell_percentage']:.1f}% (target ≥ 8.5%)")
        logger.info(f"  Status: {final_metrics['status']}")
        
        # Final save
        self._save_final_dataset()
        self._save_rejected_requests()
        logger.info(f"Parallel dataset generation complete! Generated: {self.generated_count}, Failed: {self.failed_count}")
    
    def _save_progress(self) -> None:
        """Save current progress to file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for result in self.results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"Progress saved: {self.generated_count}/{self.num_samples} samples")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def _save_final_dataset(self) -> None:
        """Save the final dataset"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for result in self.results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # Save metadata
            metadata = {
                "total_generated": self.generated_count,
                "total_failed": self.failed_count,
                "success_rate": self.generated_count / (self.generated_count + self.failed_count),
                "config_used": self.config
            }
            
            with open("generation_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Final dataset saved to {self.output_file}")
            logger.info(f"Metadata saved to generation_metadata.json")
            
        except Exception as e:
            logger.error(f"Failed to save final dataset: {e}")
    
    def _save_rejected_requests(self) -> None:
        """Save rejected requests for analysis"""
        if not self.rejected_requests:
            return
            
        try:
            with open("rejected_requests.jsonl", 'w', encoding='utf-8') as f:
                for rejection in self.rejected_requests:
                    f.write(json.dumps(rejection, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(self.rejected_requests)} rejected requests to rejected_requests.jsonl")
            
        except Exception as e:
            logger.error(f"Failed to save rejected requests: {e}")
    
    def print_statistics(self) -> None:
        """Print comprehensive generation statistics"""
        print(f"\n{'='*60}")
        print(f"DATASET GENERATION STATISTICS")
        print(f"{'='*60}")
        print(f"Total generated: {self.generated_count}")
        print(f"Total failed: {self.failed_count}")
        print(f"Success rate: {(self.generated_count / (self.generated_count + self.failed_count) * 100):.1f}%")
        
        # Balance metrics
        metrics = self._check_balance_metrics(self.generated_count)
        print(f"\nBalance Metrics:")
        print(f"  Gini coefficient: {metrics['gini']:.3f} (target < 0.15)")
        print(f"  Max relative deviation: {metrics['relative_deviation']:.1f}% (target ≤ 25%)")
        print(f"  Min cell percentage: {metrics['min_cell_percentage']:.1f}% (target ≥ 8.5%)")
        print(f"  Overall status: {metrics['status'].upper()}")
        
        # Detailed balance breakdown
        details = metrics['details']
        print(f"\nDetailed Balance Analysis:")
        print(f"  Label-Language (Tier 1):")
        print(f"    Gini: {details['label_lang']['gini']:.3f}, Deviation: {details['label_lang']['deviation']:.1f}%, Min: {details['label_lang']['min_percentage']:.1f}%")
        print(f"  Tone-Topic (Tier 2):")
        print(f"    Gini: {details['tone_topic']['gini']:.3f}, Deviation: {details['tone_topic']['deviation']:.1f}%, Min: {details['tone_topic']['min_percentage']:.1f}%")
        print(f"  Full 4-tuple combinations:")
        print(f"    Gini: {details['full_combo']['gini']:.3f}, Deviation: {details['full_combo']['deviation']:.1f}%, Min: {details['full_combo']['min_percentage']:.1f}%")
        
        # Rejection breakdown
        if self.rejected_requests:
            rejection_reasons = {}
            for rejection in self.rejected_requests:
                reason = rejection.get('rejection_reason', 'unknown')
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            print("\nRejection Reasons:")
            for reason, count in rejection_reasons.items():
                print(f"  {reason}: {count}")
        
        # Label distribution
        label_counts = {}
        for result in self.results:
            label = result["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\nLabel Distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / self.generated_count) * 100
            print(f"  Label {label}: {count} ({percentage:.1f}%)")
        
        # Language distribution
        lang_counts = {}
        for result in self.results:
            lang = result["lang"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        print("\nLanguage Distribution:")
        for lang, count in sorted(lang_counts.items()):
            percentage = (count / self.generated_count) * 100
            print(f"  {lang}: {count} ({percentage:.1f}%)")
        
        # Label-Language combination distribution (CRITICAL)
        label_lang_counts = {}
        for result in self.results:
            key = (result["label"], result["lang"])
            label_lang_counts[key] = label_lang_counts.get(key, 0) + 1
        
        print("\nLabel-Language Combination Distribution (CRITICAL):")
        for (label, lang), count in sorted(label_lang_counts.items()):
            percentage = (count / self.generated_count) * 100
            print(f"  Label {label}-{lang}: {count} ({percentage:.1f}%)")
        
        # Label-Language balance analysis
        if label_lang_counts:
            min_label_lang = min(label_lang_counts.values())
            max_label_lang = max(label_lang_counts.values())
            avg_label_lang = sum(label_lang_counts.values()) / len(label_lang_counts)
            
            print(f"\nLabel-Language Balance Analysis:")
            print(f"  Min samples per label-lang: {min_label_lang}")
            print(f"  Max samples per label-lang: {max_label_lang}")
            print(f"  Average samples per label-lang: {avg_label_lang:.1f}")
            print(f"  Balance ratio (min/max): {min_label_lang/max_label_lang:.3f}")
            
            # Check if balance is acceptable (within 5% of perfect)
            expected_per_label_lang = self.generated_count / (len(self.labels) * len(self.langs))
            max_deviation = max(abs(count - expected_per_label_lang) for count in label_lang_counts.values())
            deviation_percentage = (max_deviation / expected_per_label_lang) * 100
            
            print(f"  Expected per label-lang: {expected_per_label_lang:.1f}")
            print(f"  Max deviation: {max_deviation:.1f} ({deviation_percentage:.1f}%)")
            if deviation_percentage <= 5.0:
                print(f"  ✅ Balance is EXCELLENT (deviation ≤ 5%)")
            elif deviation_percentage <= 10.0:
                print(f"  ⚠️  Balance is GOOD (deviation ≤ 10%)")
            else:
                print(f"  ❌ Balance needs improvement (deviation > 10%)")
        
        # Tone distribution
        tone_counts = {}
        for result in self.results:
            tone = result["tone"]
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
        
        print("\nTone Distribution:")
        for tone, count in sorted(tone_counts.items()):
            percentage = (count / self.generated_count) * 100
            print(f"  {tone}: {count} ({percentage:.1f}%)")
        
        # Tone balance analysis
        if tone_counts:
            min_tone = min(tone_counts.values())
            max_tone = max(tone_counts.values())
            expected_per_tone = self.generated_count / len(self.tones)
            max_tone_deviation = max(abs(count - expected_per_tone) for count in tone_counts.values())
            tone_deviation_percentage = (max_tone_deviation / expected_per_tone) * 100
            
            print(f"\nTone Balance Analysis:")
            print(f"  Min samples per tone: {min_tone}")
            print(f"  Max samples per tone: {max_tone}")
            print(f"  Expected per tone: {expected_per_tone:.1f}")
            print(f"  Max deviation: {max_tone_deviation:.1f} ({tone_deviation_percentage:.1f}%)")
            if tone_deviation_percentage <= 5.0:
                print(f"  ✅ Tone balance is EXCELLENT (deviation ≤ 5%)")
            elif tone_deviation_percentage <= 10.0:
                print(f"  ⚠️  Tone balance is GOOD (deviation ≤ 10%)")
            else:
                print(f"  ❌ Tone balance needs improvement (deviation > 10%)")
        
        # Topic distribution
        topic_counts = {}
        for result in self.results:
            topic = result["topic"]
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        print("\nTopic Distribution:")
        for topic, count in sorted(topic_counts.items()):
            percentage = (count / self.generated_count) * 100
            print(f"  {topic}: {count} ({percentage:.1f}%)")
        
        # Topic balance analysis
        if topic_counts:
            min_topic = min(topic_counts.values())
            max_topic = max(topic_counts.values())
            expected_per_topic = self.generated_count / len(self.topics)
            max_topic_deviation = max(abs(count - expected_per_topic) for count in topic_counts.values())
            topic_deviation_percentage = (max_topic_deviation / expected_per_topic) * 100
            
            print(f"\nTopic Balance Analysis:")
            print(f"  Min samples per topic: {min_topic}")
            print(f"  Max samples per topic: {max_topic}")
            print(f"  Expected per topic: {expected_per_topic:.1f}")
            print(f"  Max deviation: {max_topic_deviation:.1f} ({topic_deviation_percentage:.1f}%)")
            if topic_deviation_percentage <= 5.0:
                print(f"  ✅ Topic balance is EXCELLENT (deviation ≤ 5%)")
            elif topic_deviation_percentage <= 10.0:
                print(f"  ⚠️  Topic balance is GOOD (deviation ≤ 10%)")
            else:
                print(f"  ❌ Topic balance needs improvement (deviation > 10%)")
        
        # Full combination balance analysis
        print("\nFull Combination Balance Analysis:")
        combination_counts = {}
        for result in self.results:
            key = f"L{result['label']}-{result['lang']}-{result['tone']}-{result['topic']}"
            combination_counts[key] = combination_counts.get(key, 0) + 1
        
        min_count = min(combination_counts.values())
        max_count = max(combination_counts.values())
        avg_count = sum(combination_counts.values()) / len(combination_counts)
        
        print(f"  Min samples per full combination: {min_count}")
        print(f"  Max samples per full combination: {max_count}")
        print(f"  Average samples per full combination: {avg_count:.1f}")
        print(f"  Balance ratio (min/max): {min_count/max_count:.2f}")
        
        # Model usage statistics
        print("\nModel Usage:")
        total_calls = self.model_usage["standard"] + self.model_usage["premium"]
        print(f"  {self.model}: {self.model_usage['standard']} calls ({self.model_usage['standard']/total_calls*100:.1f}%)")
        print(f"  {self.premium_model}: {self.model_usage['premium']} calls ({self.model_usage['premium']/total_calls*100:.1f}%)")
        
        # Model distribution by label
        print("\nModel Distribution by Label:")
        model_by_label = {}
        for result in self.results:
            label = result["label"]
            model = result["model_used"]
            if label not in model_by_label:
                model_by_label[label] = {}
            model_by_label[label][model] = model_by_label[label].get(model, 0) + 1
        
        for label in sorted(model_by_label.keys()):
            print(f"  Label {label}:")
            total_label = sum(model_by_label[label].values())
            for model, count in model_by_label[label].items():
                percentage = (count / total_label) * 100
                print(f"    {model}: {count} ({percentage:.1f}%)")
        
        print("="*50)

    def _update_pattern_monitor(self, text: str) -> None:
        """Update pattern monitoring with new text"""
        if not self.cleaning_config.get("pattern_monitoring", {}).get("enabled", True):
            return
        
        trigrams = self._extract_trigrams(text)
        for trigram in trigrams:
            self.trigram_frequencies[trigram] += 1
        
        # Check for pattern alerts every N samples
        check_interval = self.cleaning_config.get("pattern_monitoring", {}).get("check_interval", 50)
        if len(self.results) % check_interval == 0:
            self._check_pattern_alerts()
    
    def _check_pattern_alerts(self) -> None:
        """Check for concerning patterns and generate alerts"""
        if not self.cleaning_config.get("pattern_monitoring", {}).get("enabled", True):
            return
        
        if not self.trigram_frequencies:
            return
        
        total_samples = len(self.results)
        if total_samples < 20:  # Need enough samples for meaningful analysis
            return
        
        pattern_config = self.cleaning_config.get("pattern_monitoring", {})
        high_freq_threshold = pattern_config.get("high_frequency_threshold", 0.05)
        critical_freq_threshold = pattern_config.get("critical_frequency_threshold", 0.1)
        max_alerts = pattern_config.get("max_alerts", 10)
        
        # Find trigrams that appear too frequently
        concerning_trigrams = []
        for trigram, count in self.trigram_frequencies.items():
            frequency = count / total_samples
            if frequency > high_freq_threshold:
                concerning_trigrams.append((trigram, count, frequency))
        
        # Sort by frequency
        concerning_trigrams.sort(key=lambda x: x[2], reverse=True)
        
        # Generate alerts for top concerning patterns
        for trigram, count, frequency in concerning_trigrams[:max_alerts]:
            alert = {
                "type": "high_frequency_trigram",
                "trigram": trigram,
                "count": count,
                "frequency": frequency,
                "samples_analyzed": total_samples,
                "timestamp": time.time()
            }
            self.pattern_alerts.append(alert)
            
            if frequency > critical_freq_threshold:
                logger.warning(f"⚠️  CRITICAL PATTERN: '{trigram}' appears in {frequency*100:.1f}% of samples ({count}/{total_samples})")
            elif frequency > high_freq_threshold:
                logger.info(f"📊 HIGH FREQUENCY PATTERN: '{trigram}' appears in {frequency*100:.1f}% of samples ({count}/{total_samples})")
    
    def _get_pattern_alerts(self) -> List[Dict[str, Any]]:
        """Get current pattern alerts"""
        return self.pattern_alerts.copy()
    
    def _clear_pattern_alerts(self) -> None:
        """Clear pattern alerts"""
        self.pattern_alerts = []

    def _calculate_target_word_count(self, label: int) -> int:
        """Calculate target word count for a label to achieve balanced distribution"""
        # Add 3-5% outliers that are well outside the box and whiskers
        outlier_chance = np.random.random()
        
        if outlier_chance < 0.04:  # 4% chance for outliers
            # Generate extreme outliers (well outside box and whiskers)
            # These will be very short (15-35 words) or very long (200-300 words)
            if np.random.random() < 0.5:
                # Very short outliers
                target = np.random.randint(15, 36)
            else:
                # Very long outliers
                target = np.random.randint(200, 301)
        else:
            # Normal bell curve distribution (96% of cases)
            # Target mean around 97-107 words with proper bell curve
            target_mean = 102
            
            # Use a wider normal distribution for better bell curve spread
            # Standard deviation of 22 gives us ~95% of values within ±44 words (58-146)
            # This creates a nice bell curve with 35-45 word spread around the mean
            target = np.random.normal(target_mean, 22)
            
            # Ensure reasonable bounds for normal distribution (truncate outliers)
            target = max(60, min(150, int(target)))
        
        return int(target)
    
    def _get_word_count_instruction(self, target_words: int) -> str:
        """Generate a natural word count instruction with more variety"""
        # Create more varied instructions to avoid patterns
        instructions = [
            f"Write a prompt of approximately {target_words} words.",
            f"Generate a prompt around {target_words} words.",
            f"Create a prompt with about {target_words} words.",
            f"Compose a prompt of roughly {target_words} words.",
            f"Write approximately {target_words} words for this prompt.",
            f"Generate around {target_words} words for this prompt.",
            f"Create about {target_words} words for this prompt.",
            f"Compose roughly {target_words} words for this prompt.",
            f"Write a prompt that is approximately {target_words} words long.",
            f"Generate a prompt that is around {target_words} words long.",
            f"Create a prompt that is about {target_words} words long.",
            f"Compose a prompt that is roughly {target_words} words long.",
            f"Write a prompt of {target_words} words or so.",
            f"Generate a prompt of {target_words} words or so.",
            f"Create a prompt of {target_words} words or so.",
            f"Compose a prompt of {target_words} words or so."
        ]
        
        # Randomly select an instruction to avoid patterns
        import random
        return random.choice(instructions)

def main():
    """Main execution function"""
    try:
        # Initialize generator
        generator = DatasetGenerator()
        
        # Generate dataset
        generator.generate_dataset()
        
        # Print statistics
        generator.print_statistics()
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        if hasattr(generator, 'results') and generator.results:
            generator._save_progress()
            logger.info("Partial results saved")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
