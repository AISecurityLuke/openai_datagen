#!/usr/bin/env python3
"""
Stratified Dataset Split Script
Splits dataset with stratified sampling on label + model_used + lang
"""

import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StratifiedSplitter:
    def __init__(self, train_ratio: float = 0.7, dev_ratio: float = 0.15, test_ratio: float = 0.15):
        """Initialize the stratified splitter"""
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        total = train_ratio + dev_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    def _get_strata_key(self, row: Dict[str, Any]) -> str:
        """Get the stratification key for a row"""
        label = row.get('label', 0)
        model_used = row.get('model_used', 'unknown')
        lang = row.get('lang', 'en')
        
        return f"{label}_{model_used}_{lang}"
    
    def _group_by_strata(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group data by stratification key"""
        strata = defaultdict(list)
        
        for row in data:
            key = self._get_strata_key(row)
            strata[key].append(row)
        
        return strata
    
    def _split_strata(self, strata_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split a single stratum into train/dev/test"""
        # Shuffle the data
        random.shuffle(strata_data)
        
        n = len(strata_data)
        train_end = int(n * self.train_ratio)
        dev_end = train_end + int(n * self.dev_ratio)
        
        train_data = strata_data[:train_end]
        dev_data = strata_data[train_end:dev_end]
        test_data = strata_data[dev_end:]
        
        return train_data, dev_data, test_data
    
    def split_dataset(self, input_path: str, output_dir: str = "datasets") -> Dict[str, Any]:
        """Split dataset with stratified sampling"""
        input_file = Path(input_path)
        output_dir = Path(output_dir)
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return {"error": "Input file not found"}
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {input_file}")
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    row = json.loads(line.strip())
                    data.append(row)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(data)} rows")
        
        # Group by strata
        logger.info("Grouping data by strata (label + model_used + lang)")
        strata = self._group_by_strata(data)
        
        logger.info(f"Found {len(strata)} strata:")
        for key, group in strata.items():
            logger.info(f"  {key}: {len(group)} samples")
        
        # Split each stratum
        train_data = []
        dev_data = []
        test_data = []
        
        for key, group in strata.items():
            logger.debug(f"Splitting stratum {key} ({len(group)} samples)")
            train, dev, test = self._split_strata(group)
            
            train_data.extend(train)
            dev_data.extend(dev)
            test_data.extend(test)
        
        # Shuffle final splits
        random.shuffle(train_data)
        random.shuffle(dev_data)
        random.shuffle(test_data)
        
        # Save splits
        train_file = output_dir / "train.jsonl"
        dev_file = output_dir / "dev.jsonl"
        test_file = output_dir / "test.jsonl"
        
        logger.info(f"Saving train split ({len(train_data)} samples) to {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            for row in train_data:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        logger.info(f"Saving dev split ({len(dev_data)} samples) to {dev_file}")
        with open(dev_file, 'w', encoding='utf-8') as f:
            for row in dev_data:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        logger.info(f"Saving test split ({len(test_data)} samples) to {test_file}")
        with open(test_file, 'w', encoding='utf-8') as f:
            for row in test_data:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        # Calculate statistics
        stats = self._calculate_split_statistics(train_data, dev_data, test_data)
        
        logger.info("Split statistics:")
        logger.info(f"  Train: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
        logger.info(f"  Dev:   {len(dev_data)} samples ({len(dev_data)/len(data)*100:.1f}%)")
        logger.info(f"  Test:  {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
        
        return {
            "total_samples": len(data),
            "train_samples": len(train_data),
            "dev_samples": len(dev_data),
            "test_samples": len(test_data),
            "strata_count": len(strata),
            "statistics": stats
        }
    
    def _calculate_split_statistics(self, train_data: List[Dict[str, Any]], 
                                  dev_data: List[Dict[str, Any]], 
                                  test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed statistics for each split"""
        def get_stats(data):
            label_counts = defaultdict(int)
            model_counts = defaultdict(int)
            lang_counts = defaultdict(int)
            
            for row in data:
                label_counts[row.get('label', 0)] += 1
                model_counts[row.get('model_used', 'unknown')] += 1
                lang_counts[row.get('lang', 'en')] += 1
            
            return {
                "labels": dict(label_counts),
                "models": dict(model_counts),
                "langs": dict(lang_counts)
            }
        
        return {
            "train": get_stats(train_data),
            "dev": get_stats(dev_data),
            "test": get_stats(test_data)
        }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Split dataset with stratified sampling")
    parser.add_argument("--input", default="generated_prompts.jsonl",
                       help="Path to input JSONL file")
    parser.add_argument("--output-dir", default="datasets",
                       help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Train split ratio (default: 0.7)")
    parser.add_argument("--dev-ratio", type=float, default=0.15,
                       help="Dev split ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    try:
        # Initialize splitter
        splitter = StratifiedSplitter(args.train_ratio, args.dev_ratio, args.test_ratio)
        
        # Split dataset
        results = splitter.split_dataset(args.input, args.output_dir)
        
        if "error" not in results:
            print(f"✅ Successfully split {results['total_samples']} samples")
            print(f"📊 Train: {results['train_samples']} ({results['train_samples']/results['total_samples']*100:.1f}%)")
            print(f"📊 Dev:   {results['dev_samples']} ({results['dev_samples']/results['total_samples']*100:.1f}%)")
            print(f"📊 Test:  {results['test_samples']} ({results['test_samples']/results['total_samples']*100:.1f}%)")
            print(f"🏷️  Stratified on {results['strata_count']} combinations (label + model + lang)")
        else:
            print(f"❌ Error: {results['error']}")
            
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        raise

if __name__ == "__main__":
    main() 