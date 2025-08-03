# OpenAI Data Generation Pipeline

A comprehensive system for generating diverse, strictly balanced text prompts for training security-focused language models. This pipeline generates synthetic data and integrates with the Capstone project's data cleaning workflow.

## 🚀 Features

### Core Generation
- **Multi-language support**: English, Spanish, French
- **Three-class labeling**: 0 (benign), 1 (morally grey), 2 (explicit jailbreak)
- **Strictly balanced sampling**: Even distribution across all (label, language, tone, topic) combinations
- **Real-time balance monitoring**: Gini coefficient, relative deviation, and minimum cell percentage tracking
- **Dual prompt structure**: Standard prompts for labels 0/1, specialized jailbreak prompts for label 2

### Data Quality
- **Word count control**: 6–150 words for labels 0/1, 12–150 for label 2
- **Validation**: Max 1200 characters, explicit safety rejection detection in all languages
- **Optional fields**: Role, birth_year, region, medium, pov, add_emoji (randomly sampled)

### Pipeline Features
- **Final polish**: Typo injection, red cue paraphrasing, emoji balancing, stratified splits
- **Visualization**: Comprehensive data analysis plots and balance monitoring
- **Capstone conversion**: Automatic conversion to Capstone project format
- **Combined dataset**: Merged train/dev/test for external evaluation

## 🎯 Quick Start

### Run Complete Pipeline
```bash
cd present_working_directory
python run_openai_build.py
```

### Customized Runs
```bash
# Generate 1000 samples only
python run_openai_build.py --num-samples 1000

# Skip polish and visualization
python run_openai_build.py --skip-polish --skip-visualize

# Generate and visualize only
python run_openai_build.py --skip-capstone --skip-combined
```

## 📁 Project Structure

```
openai_datagen/
├── present_working_directory/    # Main working directory
│   ├── run_openai_build.py      # 🎯 Main unified pipeline
│   ├── ripit.py                 # Core generation script
│   ├── visual_build.py          # Visualization generation
│   ├── convert_to_capstone.py   # Capstone format conversion
│   ├── config.json              # Configuration
│   ├── requirements.txt         # Dependencies
│   ├── Makefile                 # Build automation
│   ├── scripts/                 # Polish pipeline scripts
│   │   ├── final_polish.py      # Orchestrates all polish steps
│   │   ├── inject_typos.py      # Typo injection
│   │   ├── paraphrase_red_cues.py # Red cue paraphrasing
│   │   ├── balance_emojis.py    # Emoji balancing
│   │   └── split_dataset.py     # Stratified splitting
│   ├── datasets/                # Generated splits
│   ├── visuals/                 # Generated visualizations
│   └── logs/                    # Execution logs
└── key.md                       # OpenAI API key
```

## 🔄 Complete Pipeline Flow

### Step 1: Generate Prompts (`ripit.py`)
- Generates 3000 prompts using OpenAI API with parallel workers
- Applies strict balancing across (label, language, tone, topic) combinations
- Outputs: `generated_prompts.jsonl`

### Step 2: Final Polish (`scripts/final_polish.py`)
- Injects realistic typos (~1% probability)
- Paraphrases over-used jailbreak cues
- Balances emoji usage across classes
- Creates train/dev/test splits (70%/15%/15%)
- Outputs: `datasets/train.jsonl`, `dev.jsonl`, `test.jsonl`

### Step 3: Visualizations (`visual_build.py`)
- Generates comprehensive data analysis plots
- Balance monitoring charts
- Distribution analysis
- Outputs: `visuals/` directory

### Step 4: Capstone Conversion (`convert_to_capstone.py`)
- Converts data to Capstone project format
- Splits by label and saves to appropriate folders:
  - **Label 0** → `../../Capstone-UCSD/5-Data_Wrangling/green/gen_green.json`
  - **Label 1** → `../../Capstone-UCSD/5-Data_Wrangling/yellow/gen_yellow.json`
  - **Label 2** → `../../Capstone-UCSD/5-Data_Wrangling/red/gen_red.json`

### Step 5: Combined Dataset
- Merges train/dev/test into single `combined.json`
- Outputs: `combined.json`

## 🔗 Integration with Capstone Cleaning Pipeline

After Step 4 completes, the generated data is ready for the Capstone project's cleaning pipeline:

```bash
# Navigate to Capstone data wrangling directory
cd ../../Capstone-UCSD/5-Data_Wrangling

# Run the cleaning pipeline
python combine_jsons.py
```

The Capstone cleaning pipeline will:
1. **Read** the newly created `gen_green.json`, `gen_yellow.json`, `gen_red.json`
2. **Normalize** HTML entities and Unicode escapes
3. **Deduplicate** exact and fuzzy matches
4. **Sample** specific quantities (5349 class 0, 6347 class 1, 6193 class 2)
5. **Output** final `combined.json` with 17,889 cleaned entries

This integration ensures the synthetic data goes through the same proven cleaning process as existing datasets.

## ⚙️ Configuration

The system uses `config.json` for all settings:

```json
{
  "generation": {
    "num_samples": 3000,
    "batch_size": 50,
    "model": "gpt-4.1-mini",
    "premium_model": "gpt-4.1",
    "premium_ratio": 0.01,
    "parallel_workers": 30,
    "rate_limit_per_minute": 3000
  },
  "schema": {
    "fields": {
      "label": {"values": [0, 1, 2]},
      "lang": {"values": ["en", "es", "fr"]},
      "tone": {"values": ["casual", "formal", "technical", ...]},
      "topic": {"values": ["hacking", "security", "privacy", ...]}
    }
  }
}
```

## 📊 Output Files

### Generated Data
- `generated_prompts.jsonl` - Raw generated prompts (3,000 entries)
- `datasets/train.jsonl` - Training split (70%)
- `datasets/dev.jsonl` - Development split (15%)
- `datasets/test.jsonl` - Test split (15%)
- `combined.json` - Combined dataset

### Visualizations
- `visuals/` - Balance plots, distribution charts, cross-comparisons

### Capstone Format (After Step 4)
- `../../Capstone-UCSD/5-Data_Wrangling/green/gen_green.json` - Label 0 prompts
- `../../Capstone-UCSD/5-Data_Wrangling/yellow/gen_yellow.json` - Label 1 prompts
- `../../Capstone-UCSD/5-Data_Wrangling/red/gen_red.json` - Label 2 prompts

### Final Cleaned Data (After Capstone Pipeline)
- `../../Capstone-UCSD/5-Data_Wrangling/combined.json` - Final cleaned dataset (17,889 entries)

## 🔧 Advanced Usage

### Individual Scripts
```bash
# Generate prompts only
python ripit.py --config config.json

# Run polish pipeline
python scripts/final_polish.py --input generated_prompts.jsonl

# Generate visualizations
python visual_build.py

# Convert to Capstone format
python convert_to_capstone.py
```

### Makefile Targets
```bash
# Run complete pipeline
make full_pipeline

# Run final polish only
make final_polish

# Convert to Capstone format
make convert_to_capstone
```

## 📈 Balance Monitoring

The system provides real-time balance monitoring with:
- **Gini coefficient**: Measures distribution inequality (target: <0.15)
- **Relative deviation**: Deviation from ideal distribution (target: ≤25%)
- **Minimum cell percentage**: Ensures no category is underrepresented (target: ≥8.5%)

## 🛠️ Installation

```bash
# Install dependencies
cd present_working_directory
pip install -r requirements.txt

# Set OpenAI API key
echo "your-api-key-here" > ../key.md
```

## 📝 Data Schema

Each generated prompt includes:
```json
{
  "text": "Generated prompt text",
  "label": 0,
  "lang": "en",
  "tone": "casual",
  "topic": "hacking",
  "role": "student",
  "birth_year": 1995,
  "region": "California",
  "medium": "tweet",
  "pov": "first",
  "add_emoji": false,
  "source": "synthetic_v1",
  "model_used": "gpt-4.1-mini"
}
```

## 🔄 Data Flow Summary

```
OpenAI API → ripit.py → generated_prompts.jsonl
    ↓
final_polish.py → datasets/train|dev|test.jsonl
    ↓
convert_to_capstone.py → Capstone-UCSD/5-Data_Wrangling/{green|yellow|red}/gen_*.json
    ↓
combine_jsons.py → combined.json (final cleaned dataset)
```

This pipeline provides a complete workflow from synthetic data generation to integration with existing data cleaning processes. 