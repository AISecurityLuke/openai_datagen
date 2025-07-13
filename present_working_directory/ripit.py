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
        """Initialize the dataset generator with config and API key"""
        self.config = self._load_config(config_path)
        self.api_key = self._load_api_key(api_key_path)
        self.client = OpenAI(api_key=self.api_key)
        
        # Extract configuration
        self.num_samples = self.config["generation"]["num_samples"]
        self.max_retries = self.config["generation"]["max_retries"]
        self.temperature = self.config["generation"]["temperature"]
        self.top_p = self.config["generation"].get("top_p", 1.0)
        self.max_tokens = self.config["generation"]["max_tokens"]
        self.model = self.config["generation"]["model"]
        self.premium_model = self.config["generation"].get("premium_model", "gpt-4.1")
        self.premium_ratio = self.config["generation"].get("premium_ratio", 0.33)
        self.parallel_workers = self.config["generation"].get("parallel_workers", 10)
        self.rate_limit_per_minute = self.config["generation"].get("rate_limit_per_minute", 60)
        
        # Schema information
        self.schema = self.config["schema"]["fields"]
        
        # Load unified prompt structure
        self.unified_prompts = self.config["prompts"]["unified"]
        
        # Set unified prompts
        self.system_prompt = self.unified_prompts["system_prompt"]
        self.user_prompt_template = self.unified_prompts["user_prompt_template"]
        
        # Output configuration
        self.output_file = self.config["output"]["filename"]
        self.validation = self.config["validation"]
        
        # Track progress
        self.generated_count = 0
        self.failed_count = 0
        self.results = []
        self.model_usage = {"standard": 0, "premium": 0}
        self.rejected_requests = []
        
        # Thread safety
        self.results_lock = Lock()
        self.counter_lock = Lock()
        
        # Rate limiting
        self.rate_limit_queue = queue.Queue()
        self.last_request_time = 0
        
        # Balanced sampling setup
        self._setup_balanced_sampling()
        
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
    
    def _validate_generated_text(self, text: str, params: Dict[str, Any]) -> bool:
        """Validate generated text against requirements"""
        if not text or not isinstance(text, str):
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

    def _generate_single_prompt(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a single prompt using OpenAI API"""
        try:
            # Rate limiting
            self._rate_limit_wait()
            
            # Use unified prompt structure for all labels
            system_prompt = self.unified_prompts["system_prompt"].format(**params)
            user_prompt = self.unified_prompts["user_prompt_template"].format(**params)
            logger.debug(f"Using unified prompts for label {params['label']}")
            
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
        """Generate the complete dataset using parallel workers"""
        logger.info(f"Starting parallel dataset generation. Target: {self.num_samples} samples with {self.parallel_workers} workers")
        
        # Prepare all parameters upfront
        all_params = []
        for _ in range(self.num_samples):
            params = self._get_balanced_parameters()
            all_params.append(params)
        
        # Use ThreadPoolExecutor for better error handling
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all tasks
            future_to_params = {executor.submit(self._generate_single_prompt, params): params for params in all_params}
            
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
                    else:
                        with self.counter_lock:
                            self.failed_count += 1
                            
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
