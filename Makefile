.PHONY: help setup install run train test clean lint format

# Prefer a TensorFlow-compatible interpreter; allow override:
# `make PYTHON=python3.11 ...`
PYTHON ?= python3.11
PIP := $(PYTHON) -m pip

help: ## Show this help message
	@echo "ASL&AI - Real-time American Sign Language Recognition"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment and install dependencies
	@echo "Setting up ASL&AI environment..."
	@$(PYTHON) -c "import sys; v=sys.version_info; ok=(v.major==3 and v.minor in (10,11)); raise SystemExit(0 if ok else 1)" || (echo "ERROR: Use Python 3.10–3.11 (TensorFlow/MediaPipe compatibility). Example: make PYTHON=python3.11 setup" && exit 1)
	$(PYTHON) -m venv env
	@echo "Virtual environment created. Activate it with:"
	@echo "  source env/bin/activate  # On macOS/Linux"
	@echo "  env\\\\Scripts\\\\activate     # On Windows"
	@echo ""
	@echo "Then run: make install"

install: ## Install Python dependencies
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed successfully!"

install-quantum: ## Install optional quantum dependencies
	@echo "Installing optional quantum dependencies..."
	$(PIP) install -r requirements-quantum.txt
	@echo "Quantum dependencies installed successfully!"

run: ## Run the ASL recognition demo
	@echo "Starting ASL recognition..."
	$(PYTHON) main.py

run-no-visual: ## Run without visualization window
	$(PYTHON) main.py --no-visual

benchmark: ## Run in benchmark mode
	$(PYTHON) main.py --benchmark

train: ## Train the model
	@echo "Training ASL recognition model..."
	@echo "Step 1: Preparing data..."
	$(PYTHON) scripts/prepare.py
	@echo "Step 2: Training model..."
	$(PYTHON) scripts/train.py
	@echo "Training complete!"

compare: ## Train and compare MLP vs MoE (writes logs/compare_models.json)
	@echo "Comparing MLP vs MoE..."
	$(PYTHON) scripts/compare_models.py

robustness: ## Run multi-seed robustness eval (writes logs/robustness_summary.json)
	@echo "Running robustness evaluation..."
	$(PYTHON) scripts/robustness_eval.py

prepare-data: ## Prepare training data (uses sample dataset by default)
	@echo "Preparing training data..."
	$(PYTHON) scripts/prepare.py

download-kaggle: ## Download full dataset from Kaggle (requires kaggle CLI)
	@echo "Downloading Kaggle dataset..."
	@which kaggle > /dev/null || (echo "ERROR: Kaggle CLI not found. Install with: pip install kaggle" && exit 1)
	kaggle datasets download -d borisgraudt/asl-alphabet-hand-landmarks
	@echo "Extracting dataset..."
	unzip -q asl-alphabet-hand-landmarks.zip || true
	@echo "Moving to data/raw_gestures/..."
	mkdir -p data/raw_gestures
	cp -r landmarks/* data/raw_gestures/ 2>/dev/null || true
	rm -rf landmarks asl-alphabet-hand-landmarks.zip
	@echo "✓ Kaggle dataset downloaded to data/raw_gestures/"

prepare-data-full: ## Prepare training data with full dataset (data/raw_gestures)
	@echo "Preparing training data with FULL dataset (Kaggle)..."
	@echo "Using: data/raw_gestures"
	@if [ ! -d "data/raw_gestures" ] || [ -z "$$(ls -A data/raw_gestures 2>/dev/null)" ]; then \
		echo "ERROR: Full dataset not found in data/raw_gestures/"; \
		echo "Download it first with: make download-kaggle"; \
		exit 1; \
	fi
	ASL_AI_RAW_DATA_DIR=data/raw_gestures $(PYTHON) scripts/prepare.py

train-full: ## Train the model with full Kaggle dataset
	@echo "Training ASL recognition model with FULL Kaggle dataset..."
	@echo "Step 1: Preparing data from data/raw_gestures..."
	@if [ ! -d "data/raw_gestures" ] || [ -z "$$(ls -A data/raw_gestures 2>/dev/null)" ]; then \
		echo "ERROR: Full dataset not found in data/raw_gestures/"; \
		echo "Download it first with: make download-kaggle"; \
		exit 1; \
	fi
	ASL_AI_RAW_DATA_DIR=data/raw_gestures $(PYTHON) scripts/prepare.py
	@echo "Step 2: Training model..."
	$(PYTHON) scripts/train.py
	@echo "Training complete!"

test: ## Run unit tests
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

lint: ## Run linter
	@echo "Running linter..."
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	$(PYTHON) -m mypy src/ --ignore-missing-imports

format: ## Format code with black
	@echo "Formatting code..."
	$(PYTHON) -m black src/ tests/ --line-length=100

clean: ## Clean generated files
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "Clean complete!"

clean-models: ## Remove trained models (keeps data)
	@echo "Removing trained models..."
	rm -f models/*.keras models/*.pkl models/*.tflite
	@echo "Models removed!"

clean-all: clean clean-models ## Clean everything including models
	@echo "All cleaned!"

docs: ## Generate API documentation
	@echo "Generating documentation..."
	$(PYTHON) -m pdoc --html --output-dir docs/api src/ --force
	@echo "Documentation generated in docs/api/"


