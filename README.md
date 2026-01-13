# ASL&AI

> Real-time American Sign Language recognition • 97.2% accuracy • Privacy-first

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)

---

## What

Local ASL alphabet recognition (A-Z) from webcam. No cloud, no tracking, no data collection.

**Performance:** 97.2% test accuracy • <5ms latency • <5MB model size

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/borisgraudt/asl-ai.git
cd asl-ai
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt

# Run
python main.py
```

Press `q` to quit.

---

## Train Your Own

```bash
# Prepare data
python scripts/prepare.py

# Train
python scripts/train.py
```

Models saved to `models/`, plots to `plots/`.

---

## Dataset

**Full dataset available on Kaggle:** 10,508 samples • 26 classes • Balanced

```bash
# Download via Kaggle CLI
kaggle datasets download -d borisgraudt/asl-alphabet-hand-landmarks
unzip asl-alphabet-hand-landmarks.zip -d data/
```

Or download manually from **[Kaggle](https://kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks)**

This repo includes sample data in `data/sample_raw_gestures/` for quick testing.

---

## Architecture

```
Webcam → MediaPipe → Preprocessing → Neural Net → Text
                                     (256→128→64)
```

**Input:** 63D hand landmarks (21 points × xyz)  
**Hidden:** Dense layers with Dropout + BatchNorm  
**Output:** 26-class softmax (A-Z)

---

## Tech Stack

TensorFlow • MediaPipe • OpenCV • NumPy • scikit-learn

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 97.2% |
| Inference | <5ms |
| Throughput | 21k samples/sec |
| Model Size | <5MB |

---

## Documentation

- **[Technical Report](docs/TECHNICAL_REPORT.md)** — Methodology & experiments
- **[Model Card](MODEL_CARD.md)** — Model details & limitations  
- **[Contributing](CONTRIBUTING.md)** — Development workflow
- **[Architecture](ARCHITECTURE.md)** — System design

---

## Ethics & Limitations

**Privacy:** All processing runs locally. No data transmission.

**Limitations:**
- Static alphabet signs only (no sentences/grammar)
- Lighting & camera quality dependent
- Not a replacement for professional interpretation

Built with accessibility in mind. Feedback from deaf/HoH community welcome.

---

## Roadmap

- [ ] Sentence-level translation
- [ ] Dynamic gestures
- [ ] Mobile deployment (iOS/Android)
- [ ] Multi-language sign languages

---

## Citation

```bibtex
@software{asl_ai_2024,
  title     = {ASL\&AI: Real-Time American Sign Language Recognition},
  author    = {Graudt, Boris},
  year      = {2024},
  url       = {https://github.com/borisgraudt/asl-ai},
  license   = {MIT}
}
```

---

## License

MIT © 2024 Boris Graudt

---

<div align="center">
<sub>Built with ❤️ for accessibility</sub>
</div>
