# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/borisgraudt/asl-ai.git
cd asl-ai

# Create virtual environment
make setup

# Activate virtual environment
source env/bin/activate  # On macOS/Linux
# or
env\Scripts\activate     # On Windows

# Install dependencies
make install
```

## Training a Model

### Using Sample Dataset (Default)

```bash
# Prepare training data (uses data/sample_raw_gestures by default)
make prepare-data
# or
python scripts/prepare.py

# Train the model
make train
# or
python scripts/train.py
```

### Using Full Dataset (Kaggle)

The full dataset is available on **[Kaggle](https://kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks)** (10,508 samples).

#### Step 1: Download Kaggle Dataset

```bash
# Option 1: Use Makefile command (easiest)
make download-kaggle

# Option 2: Manual Kaggle CLI
kaggle datasets download -d borisgraudt/asl-alphabet-hand-landmarks
unzip asl-alphabet-hand-landmarks.zip
mkdir -p data/raw_gestures
cp -r landmarks/* data/raw_gestures/
rm -rf landmarks asl-alphabet-hand-landmarks.zip

# Option 3: Manual download from web
# 1. Go to https://kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks
# 2. Click "Download" 
# 3. Extract the zip file (you'll get a 'landmarks/' folder)
# 4. Copy contents of landmarks/ folder to data/raw_gestures/
```

#### Step 2: Train with Full Dataset

```bash
# Option 1: Use Makefile commands (recommended)
make train-full         # Prepare + train with full dataset

# Option 2: Use environment variable
export ASL_AI_RAW_DATA_DIR=data/raw_gestures
make prepare-data
make train

# Option 3: Direct Python commands
ASL_AI_RAW_DATA_DIR=data/raw_gestures python scripts/prepare.py
python scripts/train.py
```

**Note:** The full Kaggle dataset contains all 26 letters (A-Z) with ~400 samples per letter (10,508 total), which will result in much better model accuracy compared to the sample dataset (~30 samples total).

## Running the Demo

```bash
# Run with visualization (default)
make run
# or
python main.py

# Run without visualization
make run-no-visual
# or
python main.py --no-visual

# Run in benchmark mode
make benchmark
# or
python main.py --benchmark
```

## Command-Line Options

```bash
python main.py --help

Options:
  --visual              Enable visualization window (default: True)
  --no-visual           Disable visualization window
  --camera INDEX        Camera device index (default: 0)
  --model PATH          Path to model file
  --confidence FLOAT    Confidence threshold (default: 0.5)
  --benchmark           Run in benchmark mode
  --log-level LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)
```

## Project Structure

```
asl-ai/
├── src/                    # Source code
│   ├── ai/                 # AI/ML modules
│   │   ├── model.py        # Model architecture
│   │   ├── train.py        # Training pipeline
│   │   └── inference.py    # Inference/classification
│   ├── vision/             # Computer vision
│   │   ├── mediapipe_tracker.py
│   │   └── preprocessing.py
│   ├── ui/                 # UI and visualization
│   │   └── app.py
│   ├── utils/              # Utilities
│   │   ├── config.py       # Configuration
│   │   └── logger.py       # Logging
│   ├── quantum/            # Quantum computing (future)
│   │   └── quantum_layer.py
│   └── main.py             # Main entry point
├── scripts/                # Utility scripts
│   ├── prepare.py          # Data preparation
│   └── train.py            # Training script
├── tests/                  # Unit tests
├── models/                 # Trained models
├── data/                   # Data directory
│   ├── sample_raw_gestures/ # Small sample dataset (repo-safe)
│   ├── raw_gestures/        # Full dataset (recommended, local only; ignored by git)
│   └── processed/          # Processed data
├── logs/                   # Log files
├── plots/                  # Training plots
├── main.py                 # Entry point
├── Makefile                # Build automation
└── requirements.txt        # Dependencies
```

## Development

```bash
# Run tests
make test

# Run tests with coverage
make test-coverage

# Format code
make format

# Run linter
make lint

# Generate documentation
make docs
```

## Troubleshooting

### Camera not found
- Check camera permissions
- Try different camera index: `python main.py --camera 1`
- Verify camera works with other applications

### Model not found
- Train the model first: `make train`
- Check that `models/model.h5` or `models/model.keras` exists

### Keras Version Incompatibility Error
If you see an error like:
```
Could not deserialize class 'Functional' because its parent module keras.src.models.functional cannot be imported
```

This means the model was saved with Keras 3.x but you're using Keras 2.13.1. **Solution:**
```bash
# Retrain the model with your current Keras version
make train
# or
python scripts/train.py
```

This will create a new model file compatible with Keras 2.13.1.

### Import errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## Environment Variables

```bash
# Override model path
export ASL_AI_MODEL_PATH=/path/to/model.h5

# Override camera index
export ASL_AI_CAMERA_INDEX=1
```


