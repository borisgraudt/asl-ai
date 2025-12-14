# ASL&AI

**Real-time American Sign Language recognition system with 97.2% accuracy**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/python-3.10--3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.2%25-brightgreen.svg)](#performance)

---

## Overview

ASL&AI is a privacy-first, real-time American Sign Language translation system that processes hand gestures locally without cloud dependencies. The system achieves 97.2% accuracy on ASL alphabet recognition using a deep neural network architecture optimized for edge deployment.

**Key Features:**
- Real-time gesture recognition from webcam input
- Local processing with complete data privacy
- 97.2% test accuracy on 26 ASL letters
- <5ms inference latency
- Edge-optimized model (<5MB)

---

## Demo

![ASL&AI demo](assets/demo.gif)

---

## System Architecture

<svg width="800" height="300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#4A90E2;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#357ABD;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <!-- Webcam -->
  <rect x="20" y="100" width="120" height="80" rx="10" fill="#E8F4F8" stroke="#4A90E2" stroke-width="2"/>
  <text x="80" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#2C3E50">Webcam</text>
  <path d="M 140 140 L 180 140" stroke="#4A90E2" stroke-width="3" marker-end="url(#arrowhead)"/>
  
  <!-- MediaPipe -->
  <rect x="180" y="100" width="120" height="80" rx="10" fill="#E8F4F8" stroke="#4A90E2" stroke-width="2"/>
  <text x="240" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#2C3E50">MediaPipe</text>
  <text x="240" y="155" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#7F8C8D">Hand Tracking</text>
  <path d="M 300 140 L 340 140" stroke="#4A90E2" stroke-width="3" marker-end="url(#arrowhead)"/>
  
  <!-- Preprocessing -->
  <rect x="340" y="100" width="120" height="80" rx="10" fill="#E8F4F8" stroke="#4A90E2" stroke-width="2"/>
  <text x="400" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#2C3E50">Preprocessing</text>
  <text x="400" y="155" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#7F8C8D">Normalization</text>
  <path d="M 460 140 L 500 140" stroke="#4A90E2" stroke-width="3" marker-end="url(#arrowhead)"/>
  
  <!-- Neural Network -->
  <rect x="500" y="80" width="140" height="120" rx="10" fill="url(#grad1)" stroke="#357ABD" stroke-width="2"/>
  <text x="570" y="110" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="white">Neural Network</text>
  <text x="570" y="130" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">256 → 128 → 64</text>
  <text x="570" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">Dropout + BatchNorm</text>
  <text x="570" y="170" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">26-class output</text>
  <path d="M 640 140 L 680 140" stroke="#4A90E2" stroke-width="3" marker-end="url(#arrowhead)"/>
  
  <!-- Text Output -->
  <rect x="680" y="100" width="100" height="80" rx="10" fill="#E8F4F8" stroke="#4A90E2" stroke-width="2"/>
  <text x="730" y="145" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#2C3E50">Text Output</text>
  
  <!-- Quantum Layer (dashed, future) -->
  <rect x="500" y="220" width="140" height="60" rx="10" fill="none" stroke="#95A5A6" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="570" y="245" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#7F8C8D">Quantum Layer</text>
  <text x="570" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#95A5A6">(Planned)</text>
  <path d="M 570 200 L 570 220" stroke="#95A5A6" stroke-width="2" stroke-dasharray="5,5"/>
  
  <!-- Arrow marker -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#4A90E2" />
    </marker>
  </defs>
</svg>

**Model Architecture:**
- **Input:** 63-dimensional hand landmark features (21 points × 3 coordinates)
- **Hidden Layers:** 256 → 128 → 64 neurons with ReLU activation
- **Regularization:** Dropout (0.1-0.3) + Batch Normalization
- **Output:** 26-class softmax classification (A-Z)
- **Training:** 10,508 samples, 80/20 train/test split

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | macOS 10.14+, Linux (Ubuntu 18.04+), Windows 10+ | Latest stable release |
| **Python** | 3.8 | 3.10–3.11 |
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 500 MB | 1 GB+ |
| **Camera** | Built-in webcam | HD webcam (720p+) |
| **CPU** | Dual-core 2.0 GHz | Quad-core 2.5 GHz+ |
| **GPU** | Not required | Optional (CUDA-compatible) |

---

## Installation

### Prerequisites

```bash
# Verify Python version
python3 --version  # Recommended: Python 3.10–3.11 (TensorFlow 2.13 compatibility)

# Create virtual environment (recommended)
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Setup

```bash
# Clone repository
git clone https://github.com/borisgraudt/asl-ai.git
cd asl-ai

# Install dependencies
python -m pip install -r requirements.txt
```

### Verify Installation

```bash
# Test model loading
python -c "import tensorflow as tf; import mediapipe as mp; print('✓ Dependencies installed')"
```

---

## Usage

### Real-time Recognition

```bash
python main.py
```

**Controls:**
- Press `q` to quit
- Ensure adequate lighting for hand detection
- Position hand within camera frame

### Training Custom Model

```bash
# 1. Prepare data
python scripts/prepare.py

# 2. Train model
python scripts/train.py
```

Training outputs:
- Model: `models/model.h5`
- Training plots: `plots/training_history.png` (generated after training)
- Confusion matrix: `plots/confusion_matrix.png` (generated after training)

### Dataset note (repo is intentionally lightweight)

This repository ships with a **small sample dataset** in `data/sample_raw_gestures/` so you can run the pipeline.

For full training, point `ASL_AI_RAW_DATA_DIR` to your full dataset directory:

```bash
export ASL_AI_RAW_DATA_DIR="data/raw_gestures"
python scripts/prepare.py
python scripts/train.py
```

### Robustness (training stability)

To measure stability across random seeds (same dataset split), run:

```bash
make robustness
```

This writes `logs/robustness_summary.json` with mean/std across runs for MLP and MoE.

---

## Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.2% (2,044/2,102 correct) |
| **Processing Speed** | 21,098 samples/sec |
| **Inference Latency** | <5ms |
| **Average Confidence** | 96.8% |
| **Dataset Size** | 10,508 samples (26 classes) |
| **Model Size** | <5 MB |

**Training Details:**
- Training time: ~15 minutes (CPU)
- Convergence: Early stopping at epoch 25-30
- Validation strategy: 20% holdout test set
- Per-class F1-score: 0.972 average

---

## Real-world Applications

**Accessibility & Communication:**
- Real-time ASL-to-text translation for video calls
- Educational tools for ASL learning
- Assistive technology for deaf/hard-of-hearing individuals

**Research & Development:**
- Gesture recognition research platform
- Human-computer interaction studies
- Edge AI deployment benchmarks

**Enterprise:**
- Customer service accessibility solutions
- Workplace communication tools
- Healthcare communication systems

---

## Research Contributions

**Novel Approaches:**
1. **Position-Invariant Preprocessing:** Hand landmark normalization for robust recognition across orientations
2. **Edge-Optimized Architecture:** Lightweight model design for real-time inference on standard hardware
3. **Privacy-Preserving Design:** Complete local processing without cloud dependencies

**Technical Innovation:**
- High-precision classification (97.2% accuracy) with minimal computational overhead
- Real-time activation monitoring for model interpretability
- TensorFlow Lite deployment for mobile and embedded systems

**📄 Technical Report:**
- See [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) for detailed methodology, experiments, and results
- [Google Docs version](https://docs.google.com/document/d/1PF0bECLudvPkGU7u7G0QYqletg4bIu5QeRmqqraftiU/edit)

**📊 Presentation:**
- [Presentation PDF](docs/presentation/ASL_AI_Presentation.pdf) - Project overview and results

---

## Ethics & Accessibility

**Commitment to Accessibility:**
This project is designed to improve communication accessibility for the deaf and hard-of-hearing community. We recognize that sign language is a complete, natural language with its own grammar and syntax, and this system represents an initial step toward more comprehensive translation tools.

**Privacy & Data Sovereignty:**
- All processing occurs locally on the user's device
- No data transmission to external servers
- No collection or storage of personal biometric data
- Users maintain complete control over their data

**Limitations & Considerations:**
- Current implementation recognizes static ASL alphabet signs (A-Z)
- Does not support sentence-level translation or grammar
- Performance may vary with lighting conditions and camera quality
- Not a replacement for professional sign language interpretation

**Community Engagement:**
We welcome feedback from deaf and hard-of-hearing users, ASL educators, and accessibility advocates to improve the system's accuracy and usability.

---

## Tech Stack

- **Machine Learning:** TensorFlow 2.13, scikit-learn 1.3.2
- **Computer Vision:** MediaPipe, OpenCV
- **Data Processing:** NumPy, pandas
- **Visualization:** Matplotlib, Plotly
- **Deployment:** TensorFlow Lite (mobile optimization)

**Optional:** Quantum experimentation dependencies are available in `requirements-quantum.txt`.

---

## Roadmap

- [ ] Sentence-level translation with context-aware grammar
- [ ] Dynamic gesture recognition for motion-based signs
- [ ] Multi-language support (International Sign Languages)
- [ ] Mobile deployment (iOS/Android)
- [ ] Cloud inference API for web integration
- [ ] Quantum computing integration validation

---

## Contributing

We welcome contributions from researchers, developers, and accessibility advocates:

1. **Data Collection:** Expand gesture datasets with diverse signers
2. **Model Architecture:** Experiment with novel neural network designs
3. **Mobile Development:** iOS/Android app implementation
4. **Accessibility:** Enhance UX for deaf/hard-of-hearing communities
5. **Research:** Quantum computing integration and performance analysis

See `CONTRIBUTING.md` for a development workflow and code quality checks.

---

## Model Card

See `MODEL_CARD.md`.

---

## License

MIT License — free to use, modify, and distribute for academic and commercial purposes.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{asl_ai_2024,
  title = {ASL\&AI: Real-Time American Sign Language Recognition},
  author = {Graudt, Boris},
  year = {2024},
  url = {https://github.com/borisgraudt/asl-ai},
  license = {MIT}
}
```

---

<div align="center">

**ASL&AI**  
*Bridging communication through AI*

</div>
