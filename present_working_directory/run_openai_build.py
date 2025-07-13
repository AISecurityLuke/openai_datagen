#!/usr/bin/env python3
"""
Unified OpenAI Build Pipeline
Complete pipeline for generating, polishing, visualizing, and converting text prompts.

Pipeline Steps:
1. Generate prompts with strict balancing
2. Final polish (typos, cues, emojis, splits)
3. Generate visualizations
4. Convert to Capstone format
5. Create combined dataset
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the pipeline runner"""
        self.config_path = Path(config_path)
        self.start_time = time.time()
        self.step_results = {}
        
        # Ensure required directories exist
        Path("logs").mkdir(exist_ok=True)
        Path("visuals").mkdir(exist_ok=True)
        Path("datasets").mkdir(exist_ok=True)
        
    def log_step(self, step_name: str, success: bool, details: str = ""):
        """Log step results"""
        duration = time.time() - self.start_time
        status = "✅" if success else "❌"
        logger.info(f"{status} {step_name} - {duration:.1f}s {details}")
        self.step_results[step_name] = {"success": success, "duration": duration, "details": details}
        
    def run_script(self, script_name: str, args: list = None, description: str = "") -> bool:
        """Run a script and return success status"""
        if args is None:
            args = []
        
        script_path = Path(script_name)
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        cmd = [sys.executable, str(script_path)] + args
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result.stdout:
                logger.debug(f"Output: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Script failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"Stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"Stderr: {e.stderr}")
            return False
    
    def step_1_generate_prompts(self, num_samples: Optional[int] = None) -> bool:
        """Step 1: Generate prompts with strict balancing"""
        logger.info("🚀 Step 1: Generating prompts with strict balancing")
        logger.info("=" * 60)
        
        args = ["--config", str(self.config_path)]
        if num_samples:
            args.extend(["--num-samples", str(num_samples)])
        
        success = self.run_script("ripit.py", args, "Prompt generation")
        self.log_step("Generate Prompts", success)
        return success
    
    def step_2_final_polish(self, skip_split: bool = False) -> bool:
        """Step 2: Final polish pipeline"""
        logger.info("✨ Step 2: Final polish pipeline")
        logger.info("=" * 60)
        
        args = ["--input", "generated_prompts.jsonl"]
        if skip_split:
            args.append("--skip-split")
        
        success = self.run_script("scripts/final_polish.py", args, "Final polish")
        self.log_step("Final Polish", success)
        return success
    
    def step_3_visualize(self) -> bool:
        """Step 3: Generate visualizations"""
        logger.info("📊 Step 3: Generating visualizations")
        logger.info("=" * 60)
        
        success = self.run_script("visual_build.py", [], "Visualization generation")
        self.log_step("Visualizations", success)
        return success
    
    def step_4_convert_to_capstone(self) -> bool:
        """Step 4: Convert to Capstone format"""
        logger.info("🔄 Step 4: Converting to Capstone format")
        logger.info("=" * 60)
        
        success = self.run_script("convert_to_capstone.py", [], "Capstone conversion")
        self.log_step("Capstone Conversion", success)
        return success
    
    def step_5_create_combined(self) -> bool:
        """Step 5: Create combined dataset"""
        logger.info("📦 Step 5: Creating combined dataset")
        logger.info("=" * 60)
        
        # Check if we have the split files
        train_file = Path("datasets/train.jsonl")
        dev_file = Path("datasets/dev.jsonl")
        test_file = Path("datasets/test.jsonl")
        
        if not all([train_file.exists(), dev_file.exists(), test_file.exists()]):
            logger.warning("Split files not found, skipping combined dataset creation")
            self.log_step("Combined Dataset", False, "Split files not found")
            return False
        
        # Create combined dataset
        import json
        combined_data = []
        
        for file_path in [train_file, dev_file, test_file]:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    combined_data.append(json.loads(line.strip()))
        
        with open("combined.json", 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created combined dataset with {len(combined_data)} samples")
        self.log_step("Combined Dataset", True, f"{len(combined_data)} samples")
        return True
    
    def run_full_pipeline(self, 
                         num_samples: Optional[int] = None,
                         skip_polish: bool = False,
                         skip_visualize: bool = False,
                         skip_capstone: bool = False,
                         skip_combined: bool = False) -> bool:
        """Run the complete pipeline"""
        logger.info("🎯 Starting Unified OpenAI Build Pipeline")
        logger.info("=" * 80)
        logger.info(f"Configuration: {self.config_path}")
        logger.info(f"Steps to run:")
        logger.info(f"  1. Generate prompts {'(' + str(num_samples) + ' samples)' if num_samples else ''}")
        if not skip_polish:
            logger.info(f"  2. Final polish")
        if not skip_visualize:
            logger.info(f"  3. Visualizations")
        if not skip_capstone:
            logger.info(f"  4. Capstone conversion")
        if not skip_combined:
            logger.info(f"  5. Combined dataset")
        logger.info("=" * 80)
        
        # Step 1: Generate prompts
        if not self.step_1_generate_prompts(num_samples):
            logger.error("❌ Pipeline failed at Step 1")
            return False
        
        # Step 2: Final polish
        if not skip_polish:
            if not self.step_2_final_polish():
                logger.error("❌ Pipeline failed at Step 2")
                return False
        
        # Step 3: Visualizations
        if not skip_visualize:
            if not self.step_3_visualize():
                logger.error("❌ Pipeline failed at Step 3")
                return False
        
        # Step 4: Capstone conversion
        if not skip_capstone:
            if not self.step_4_convert_to_capstone():
                logger.error("❌ Pipeline failed at Step 4")
                return False
        
        # Step 5: Combined dataset
        if not skip_combined:
            if not self.step_5_create_combined():
                logger.warning("⚠️ Pipeline continued despite Step 5 failure")
        
        # Pipeline summary
        total_duration = time.time() - self.start_time
        logger.info("=" * 80)
        logger.info("🎉 Pipeline completed!")
        logger.info(f"⏱️ Total duration: {total_duration:.1f} seconds")
        
        # Print step summary
        logger.info("📋 Step Summary:")
        for step, result in self.step_results.items():
            status = "✅" if result["success"] else "❌"
            logger.info(f"  {status} {step}: {result['duration']:.1f}s")
        
        # Check if all required steps succeeded
        required_steps = ["Generate Prompts"]
        if not skip_polish:
            required_steps.append("Final Polish")
        if not skip_visualize:
            required_steps.append("Visualizations")
        if not skip_capstone:
            required_steps.append("Capstone Conversion")
        
        all_required_succeeded = all(
            self.step_results.get(step, {}).get("success", False)
            for step in required_steps
        )
        
        if all_required_succeeded:
            logger.info("🎯 All required steps completed successfully!")
            return True
        else:
            logger.error("❌ Some required steps failed")
            return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Unified OpenAI Build Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_openai_build.py
  
  # Generate 1000 samples only
  python run_openai_build.py --num-samples 1000
  
  # Skip polish and visualization
  python run_openai_build.py --skip-polish --skip-visualize
  
  # Generate and visualize only
  python run_openai_build.py --skip-capstone --skip-combined
        """
    )
    
    parser.add_argument("--config", default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--num-samples", type=int,
                       help="Number of samples to generate (overrides config)")
    parser.add_argument("--skip-polish", action="store_true",
                       help="Skip the final polish step")
    parser.add_argument("--skip-visualize", action="store_true",
                       help="Skip the visualization step")
    parser.add_argument("--skip-capstone", action="store_true",
                       help="Skip the Capstone conversion step")
    parser.add_argument("--skip-combined", action="store_true",
                       help="Skip the combined dataset creation step")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline runner
        runner = PipelineRunner(args.config)
        
        # Run pipeline
        success = runner.run_full_pipeline(
            num_samples=args.num_samples,
            skip_polish=args.skip_polish,
            skip_visualize=args.skip_visualize,
            skip_capstone=args.skip_capstone,
            skip_combined=args.skip_combined
        )
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            print("📁 Check the following outputs:")
            print("  - generated_prompts.jsonl (raw generated data)")
            print("  - datasets/ (train/dev/test splits)")
            print("  - visuals/ (generated visualizations)")
            print("  - combined.json (combined dataset)")
            print("  - Capstone project folders (converted format)")
            print("  - pipeline.log (detailed execution log)")
            sys.exit(0)
        else:
            print("\n❌ Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 