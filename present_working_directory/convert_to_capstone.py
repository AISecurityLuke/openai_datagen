#!/usr/bin/env python3
"""
Convert generated_prompts.jsonl to Capstone project format
Splits by label and saves to appropriate folders:
- Label 0 (benign) -> green/gen_green.json
- Label 1 (grey) -> yellow/gen_yellow.json  
- Label 2 (jailbreak) -> red/gen_red.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CapstoneConverter:
    def __init__(self, capstone_base_path: str = "../../Capstone-UCSD/5-Data_Wrangling"):
        """Initialize the converter with paths"""
        self.capstone_base = Path(capstone_base_path)
        self.green_dir = self.capstone_base / "green"
        self.yellow_dir = self.capstone_base / "yellow"
        self.red_dir = self.capstone_base / "red"
        
        # Ensure directories exist
        self.green_dir.mkdir(parents=True, exist_ok=True)
        self.yellow_dir.mkdir(parents=True, exist_ok=True)
        self.red_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_format(self, row: Dict[str, Any]) -> Dict[str, str]:
        """Convert from our format to Capstone format"""
        return {
            "user_message": row["text"]
        }
    
    def process_file(self, input_path: str) -> Dict[str, Any]:
        """Process the JSONL file and split by label"""
        input_file = Path(input_path)
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return {"error": "Input file not found"}
        
        # Initialize data structures
        green_data = []
        yellow_data = []
        red_data = []
        
        processed_count = 0
        
        logger.info(f"Processing {input_file} for Capstone conversion")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    row = json.loads(line.strip())
                    processed_count += 1
                    
                    # Convert format
                    converted_row = self.convert_format(row)
                    
                    # Split by label
                    label = row.get('label', 0)
                    if label == 0:
                        green_data.append(converted_row)
                    elif label == 1:
                        yellow_data.append(converted_row)
                    elif label == 2:
                        red_data.append(converted_row)
                    else:
                        logger.warning(f"Unknown label {label} on line {line_num}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        # Save to files
        results = {}
        
        # Save green data (label 0)
        green_file = self.green_dir / "gen_green.json"
        with open(green_file, 'w', encoding='utf-8') as f:
            json.dump(green_data, f, ensure_ascii=False, indent=2)
        results["green"] = len(green_data)
        logger.info(f"Saved {len(green_data)} green prompts to {green_file}")
        
        # Save yellow data (label 1)
        yellow_file = self.yellow_dir / "gen_yellow.json"
        with open(yellow_file, 'w', encoding='utf-8') as f:
            json.dump(yellow_data, f, ensure_ascii=False, indent=2)
        results["yellow"] = len(yellow_data)
        logger.info(f"Saved {len(yellow_data)} yellow prompts to {yellow_file}")
        
        # Save red data (label 2)
        red_file = self.red_dir / "gen_red.json"
        with open(red_file, 'w', encoding='utf-8') as f:
            json.dump(red_data, f, ensure_ascii=False, indent=2)
        results["red"] = len(red_data)
        logger.info(f"Saved {len(red_data)} red prompts to {red_file}")
        
        results["total_processed"] = processed_count
        results["total_converted"] = len(green_data) + len(yellow_data) + len(red_data)
        
        logger.info(f"Conversion complete!")
        logger.info(f"  Total processed: {processed_count}")
        logger.info(f"  Green (label 0): {len(green_data)}")
        logger.info(f"  Yellow (label 1): {len(yellow_data)}")
        logger.info(f"  Red (label 2): {len(red_data)}")
        
        return results

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Convert generated prompts to Capstone format")
    parser.add_argument("--input", default="generated_prompts.jsonl",
                       help="Path to input JSONL file")
    parser.add_argument("--capstone-path", default="../../Capstone-UCSD/5-Data_Wrangling",
                       help="Path to Capstone data wrangling directory")
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = CapstoneConverter(args.capstone_path)
        
        # Process file
        results = converter.process_file(args.input)
        
        if "error" not in results:
            print(f"✅ Successfully converted {results['total_processed']} prompts")
            print(f"🟢 Green (label 0): {results['green']} prompts")
            print(f"🟡 Yellow (label 1): {results['yellow']} prompts")
            print(f"🔴 Red (label 2): {results['red']} prompts")
            print(f"📁 Files saved to Capstone project folders")
        else:
            print(f"❌ Error: {results['error']}")
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

if __name__ == "__main__":
    main() 