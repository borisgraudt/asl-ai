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

```bash
# Prepare training data
make prepare-data
# or
python scripts/prepare.py

# Train the model
make train
# or
python scripts/train.py
```

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
- Check that `models/model.h5` exists

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


