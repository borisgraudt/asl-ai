# Model Card — ASL&AI (Static ASL Alphabet Classifier)

## Model details
- **Model type**: Feed-forward neural network (Dense + BatchNorm + Dropout)
- **Input**: 63 features (21 hand landmarks × x/y/z), position-invariant normalized
- **Output**: 26-class softmax (A–Z)
- **Framework**: TensorFlow/Keras

## Intended use
- **Primary use**: Real-time classification of **static** ASL alphabet hand signs (A–Z) from webcam hand landmarks.
- **Out of scope**:
  - Sentence-level translation / grammar
  - Dynamic gestures that require temporal modeling
  - Signer identification or biometric use cases

## Training data
- **Source**: Landmark feature vectors stored as `.npy` files per class.
- **Preprocessing**:
  - Center landmarks at the palm (landmark 0)
  - Scale by maximum landmark distance (position/scale invariance)
  - Standardization via `StandardScaler` saved to `models/scaler.pkl`

## Evaluation
- **Metrics reported**: accuracy (and optionally top-k accuracy)
- **Artifacts**:
  - Training history saved to `logs/training_history.json`
  - Plots generated to `plots/` after training

## Performance considerations
- Performance depends on:
  - Lighting conditions and camera quality
  - Hand visibility / occlusion
  - Distance to camera and hand pose variability
  - MediaPipe hand tracker stability

## Ethical considerations
- **Privacy-first**: designed for on-device processing, no cloud requirement.
- **Accessibility**: intended to support communication accessibility; not a replacement for professional interpretation.

## Limitations
- Only static alphabet signs (A–Z); no temporal modeling.
- Model may generalize poorly to unseen signer styles if training data is not diverse.
- Errors can occur under occlusion, motion blur, or poor hand tracking.

## Recommendations for responsible use
- Communicate limitations clearly in demos and documentation.
- Avoid deploying for high-stakes decisions.
- Evaluate with diverse signers and conditions before broader use.


