# Text Prompt Generator

A comprehensive system for generating diverse, strictly balanced text prompts for training security-focused language models.

## 🚀 Features

### Core Generation
- **Multi-language support**: English, Spanish, French
- **Three-class labeling**: 0 (benign), 1 (morally grey), 2 (explicit jailbreak)
- **Strictly balanced sampling**: Even distribution across all (label, language, tone, topic) combinations
- **Real-time balance monitoring**: Gini coefficient, relative deviation, and minimum cell percentage tracking
- **Unified prompt structure**: All labels use a single, parameterized system prompt

### Data Quality
- **Word count control**: 6–150 words for labels 0/1, 12–150 for label 2
- **Validation**: Max 800 characters, explicit safety rejection detection in all languages
- **Optional fields**: Role, birth_year, region, medium, pov, add_emoji, scenario (randomly sampled)

### Pipeline Features
- **Final polish**: Typo injection, red cue paraphrasing, emoji balancing, stratified splits
- **Visualization**: Comprehensive data analysis plots and balance monitoring
- **Capstone conversion**: Automatic conversion to Capstone project format
- **Combined dataset**: Merged train/dev/test for external evaluation

## 🎯 Quick Start

### Run Complete Pipeline
```bash
# Run everything (generate → polish → visualize → convert)
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
present_working_directory/
├── run_openai_build.py          # 🎯 Main unified pipeline
├── ripit.py                     # Core generation script
├── visual_build.py              # Visualization generation
├── convert_to_capstone.py       # Capstone format conversion
├── config.json                  # Configuration
├── schema.jsonl                 # Data schema
├── requirements.txt             # Dependencies
├── Makefile                     # Build automation
├── scripts/                     # Polish pipeline scripts
│   ├── final_polish.py          # Orchestrates all polish steps
│   ├── inject_typos.py          # Typo injection
│   ├── paraphrase_red_cues.py   # Red cue paraphrasing
│   ├── balance_emojis.py        # Emoji balancing
│   └── split_dataset.py         # Stratified splitting
├── datasets/                    # Generated splits
├── visuals/                     # Generated visualizations
└── logs/                        # Execution logs
```

## ⚙️ Configuration

The system uses `config.json` for all settings:

```json
{
  "generation": {
    "num_samples": 1000,
    "batch_size": 10,
    "model": "gpt-4o-mini",
    "premium_model": "gpt-4o",
    "premium_ratio": 0.25
  },
  "prompts": {
    "languages": ["en", "es", "fr"],
    "labels": [0, 1, 2],
    "tones": ["casual", "formal", "technical"],
    "topics": ["hacking", "privacy", "security"]
  }
}
```

## 📊 Output Files

### Generated Data
- `generated_prompts.jsonl` - Raw generated prompts
- `datasets/train.jsonl` - Training split (70%)
- `datasets/dev.jsonl` - Development split (15%)
- `datasets/test.jsonl` - Test split (15%)
- `combined.json` - Combined dataset

### Visualizations
- `visuals/` - Balance plots, distribution charts, cross-comparisons

### Capstone Format
- `../../Capstone-UCSD/5-Data_Wrangling/green/gen_green.json` - Label 0
- `../../Capstone-UCSD/5-Data_Wrangling/yellow/gen_yellow.json` - Label 1
- `../../Capstone-UCSD/5-Data_Wrangling/red/gen_red.json` - Label 2

### Logs
- `pipeline.log` - Detailed execution log
- `generation.log` - Generation-specific logs

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
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
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
  "region": "US",
  "medium": "text",
  "pov": "first",
  "add_emoji": false,
  "scenario": "academic"
}
``` 