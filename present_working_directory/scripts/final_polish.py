#!/usr/bin/env python3
"""
Final Polish Pipeline
Runs all four polishing steps in sequence:
1. Inject typos (~1% of rows)
2. Paraphrase over-used red cues
3. Balance emojis
4. Stratified split
"""

import subprocess
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name: str, args: list = None) -> bool:
    """Run a script and return success status"""
    if args is None:
        args = []
    
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"✅ {script_name} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {script_name} failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run final polish pipeline")
    parser.add_argument("--input", default="generated_prompts.jsonl",
                       help="Path to input JSONL file")
    parser.add_argument("--typo-rate", type=float, default=0.01,
                       help="Probability of adding typos (default: 0.01)")
    parser.add_argument("--light-emoji-rate", type=float, default=0.10,
                       help="Probability of adding light emojis (default: 0.10)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Train split ratio (default: 0.7)")
    parser.add_argument("--dev-ratio", type=float, default=0.15,
                       help="Dev split ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test split ratio (default: 0.15)")
    parser.add_argument("--output-dir", default="datasets",
                       help="Output directory for splits")
    parser.add_argument("--skip-split", action="store_true",
                       help="Skip the stratified split step")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Final Polish Pipeline")
    logger.info("=" * 50)
    
    # Step 1: Inject typos
    logger.info("Step 1️⃣: Injecting typos...")
    if not run_script("inject_typos.py", [
        "--input", args.input,
        "--typo-rate", str(args.typo_rate)
    ]):
        logger.error("❌ Typo injection failed")
        return 1
    
    # Step 2: Paraphrase red cues
    logger.info("Step 2️⃣: Paraphrasing over-used red cues...")
    if not run_script("paraphrase_red_cues.py", [
        "--input", args.input
    ]):
        logger.error("❌ Red cue paraphrasing failed")
        return 1
    
    # Step 3: Balance emojis
    logger.info("Step 3️⃣: Balancing emojis...")
    if not run_script("balance_emojis.py", [
        "--input", args.input,
        "--light-emoji-rate", str(args.light_emoji_rate)
    ]):
        logger.error("❌ Emoji balancing failed")
        return 1
    
    # Step 4: Stratified split (optional)
    if not args.skip_split:
        logger.info("Step 4️⃣: Creating stratified splits...")
        if not run_script("split_dataset.py", [
            "--input", args.input,
            "--output-dir", args.output_dir,
            "--train-ratio", str(args.train_ratio),
            "--dev-ratio", str(args.dev_ratio),
            "--test-ratio", str(args.test_ratio)
        ]):
            logger.error("❌ Stratified split failed")
            return 1
    else:
        logger.info("Step 4️⃣: Skipping stratified split (--skip-split)")
    
    logger.info("=" * 50)
    logger.info("🎉 Final Polish Pipeline completed successfully!")
    
    if not args.skip_split:
        logger.info(f"📁 Split files saved to: {args.output_dir}/")
        logger.info(f"  - train.jsonl")
        logger.info(f"  - dev.jsonl")
        logger.info(f"  - test.jsonl")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 