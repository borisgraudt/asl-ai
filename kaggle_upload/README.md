# ASL Alphabet Hand Landmarks Dataset

> 10,508 high-quality 3D hand landmarks for American Sign Language (A-Z) recognition

[![License: CC0](https://img.shields.io/badge/License-CC0-blue.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green.svg)](https://google.github.io/mediapipe/)

---

## Overview

Hand landmark dataset for training ASL alphabet recognition models. Collected using Google MediaPipe Hands with 21 3D landmarks per sample.

**Dataset Stats:**
- 📊 **10,508 samples** (balanced across 26 classes)
- 🔤 **26 classes** (A-Z alphabet)
- 🎯 **~404 samples/class** (balanced distribution)
- 📐 **21 landmarks × 3D** (x, y, z coordinates)
- 🔬 **Position-invariant** preprocessing

---

## Quick Start

```python
import numpy as np
from pathlib import Path

# Load a sample
gesture = np.load('landmarks/A/frame_001.npy')
print(f"Shape: {gesture.shape}")  # (21, 3)
print(f"Landmarks: {len(gesture)}")  # 21
print(f"Coordinates: x={gesture[0, 0]:.3f}, y={gesture[0, 1]:.3f}, z={gesture[0, 2]:.3f}")
```

---

## File Structure

```
landmarks/
├── A/ (404 .npy files)
├── B/ (400 .npy files)
├── C/ (403 .npy files)
├── D/ (404 .npy files)
├── E/ (403 .npy files)
├── F/ (406 .npy files)
├── G/ (404 .npy files)
├── H/ (404 .npy files)
├── I/ (403 .npy files)
├── J/ (402 .npy files)
├── K/ (405 .npy files)
├── L/ (405 .npy files)
├── M/ (404 .npy files)
├── N/ (404 .npy files)
├── O/ (408 .npy files)
├── P/ (405 .npy files)
├── Q/ (411 .npy files)
├── R/ (404 .npy files)
├── S/ (400 .npy files)
├── T/ (403 .npy files)
├── U/ (406 .npy files)
├── V/ (405 .npy files)
├── W/ (404 .npy files)
├── X/ (405 .npy files)
├── Y/ (403 .npy files)
└── Z/ (403 .npy files)
```

---

## Data Format

Each `.npy` file contains:
- **Type:** `numpy.ndarray`
- **Shape:** `(21, 3)`
- **Dtype:** `float32` or `float64`

**Coordinate system:**
- `x`: Horizontal position (0-1, normalized)
- `y`: Vertical position (0-1, normalized)  
- `z`: Depth (relative to wrist)

**Preprocessing applied:**
- Centered at wrist (landmark 0)
- Normalized by maximum distance from wrist
- Scale-invariant and position-invariant

---

## MediaPipe Hand Landmarks

```
 0: WRIST
 1: THUMB_CMC
 2: THUMB_MCP
 3: THUMB_IP
 4: THUMB_TIP
 5: INDEX_FINGER_MCP
 6: INDEX_FINGER_PIP
 7: INDEX_FINGER_DIP
 8: INDEX_FINGER_TIP
 9: MIDDLE_FINGER_MCP
10: MIDDLE_FINGER_PIP
11: MIDDLE_FINGER_DIP
12: MIDDLE_FINGER_TIP
13: RING_FINGER_MCP
14: RING_FINGER_PIP
15: RING_FINGER_DIP
16: RING_FINGER_TIP
17: PINKY_MCP
18: PINKY_PIP
19: PINKY_DIP
20: PINKY_TIP
```

---

## Example Usage

### Load Dataset

```python
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load all samples
X, y = [], []
for letter_dir in Path('landmarks').iterdir():
    if letter_dir.is_dir():
        label = letter_dir.name
        for npy_file in letter_dir.glob('*.npy'):
            landmarks = np.load(npy_file)
            X.append(landmarks.flatten())  # Flatten to (63,)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Dataset shape: {X.shape}")  # (10508, 63)
print(f"Classes: {len(np.unique(y))}")  # 26
```

### Train a Simple Classifier

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.1%}")
```

### Visualize Hand Landmarks

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load sample
gesture = np.load('landmarks/A/frame_001.npy')

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot landmarks
ax.scatter(gesture[:, 0], gesture[:, 1], gesture[:, 2], 
           c='blue', marker='o', s=50)

# Connect landmarks (hand skeleton)
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

for start, end in connections:
    ax.plot3D(*zip(gesture[start], gesture[end]), 'gray')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Hand Landmarks - Letter A')
plt.show()
```

---

## Use Cases

- **Sign Language Recognition:** Train ML models for ASL alphabet classification
- **Gesture Recognition:** Generalize to other hand gesture tasks
- **Human-Computer Interaction:** Build gesture-based interfaces
- **Accessibility Research:** Develop assistive technologies
- **Education:** Teach computer vision and ML concepts

---

## Related Project

This dataset was collected for **[ASL&AI](https://github.com/borisgraudt/asl-ai)**, a real-time ASL recognition system with:
- 97.2% test accuracy
- <5ms inference latency
- Privacy-first (local processing)
- Edge-optimized (<5MB model)

---

## Benchmark Results

Training a simple MLP (256→128→64) on this dataset achieves:

| Metric | Value |
|--------|-------|
| Test Accuracy | 97.2% |
| Avg. Precision | 0.972 |
| Avg. Recall | 0.972 |
| Avg. F1-Score | 0.972 |

See the [ASL&AI repository](https://github.com/borisgraudt/asl-ai) for full training code.

---

## Collection Methodology

1. **Capture:** Real-time webcam capture with diverse lighting/backgrounds
2. **Detection:** MediaPipe Hands for landmark detection
3. **Filtering:** Manual quality control, removed failed detections
4. **Preprocessing:** Position-invariant normalization
5. **Validation:** Cross-validated for quality

---

## Limitations

- Static alphabet signs only (no dynamic gestures)
- Single signer (limited diversity)
- Lighting and camera angle variations present
- Some landmark jitter from tracking noise

---

## License

**CC0 1.0 Universal (Public Domain)**

You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{asl_landmarks_2024,
  author    = {Graudt, Boris},
  title     = {ASL Alphabet Hand Landmarks - MediaPipe 3D},
  year      = {2024},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks}
}
```

---

## Acknowledgments

- **MediaPipe:** Google's MediaPipe Hands for landmark detection
- **ASL Community:** For making sign language accessible to researchers

---

<div align="center">
<sub>Built for accessibility • Released for research</sub>
</div>

