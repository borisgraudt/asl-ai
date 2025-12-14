# ASL&AI Architecture

## System Overview

ASL&AI is a modular, production-ready real-time American Sign Language recognition system built with Python, TensorFlow, and MediaPipe.

## Architecture Diagram

```
┌─────────────┐
│   Webcam    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  HandTracker    │  (MediaPipe)
│  (vision/)      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Preprocessing   │  (Position-invariant normalization)
│ (vision/)       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ GestureClassifier│  (TensorFlow/Keras)
│ (ai/)           │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Visualization  │  (OpenCV)
│  (ui/)          │
└─────────────────┘
```

## Module Structure

### `/src/ai/` - AI and Machine Learning
- **`model.py`**: Neural network architecture definition
- **`train.py`**: Training pipeline with callbacks and evaluation
- **`inference.py`**: `GestureClassifier` class for real-time prediction

### `/src/vision/` - Computer Vision
- **`mediapipe_tracker.py`**: `HandTracker` class wrapping MediaPipe
- **`preprocessing.py`**: Position-invariant landmark normalization

### `/src/ui/` - User Interface
- **`app.py`**: `ASLVisualizer`, `PerformanceMonitor`, and `Camera` classes

### `/src/utils/` - Utilities
- **`config.py`**: Centralized configuration management
- **`logger.py`**: Structured logging utilities

### `/src/quantum/` - Quantum Computing (Future)
- **`quantum_layer.py`**: Placeholder for hybrid quantum-classical integration

## Data Flow

1. **Capture**: Camera captures video frames
2. **Detection**: MediaPipe detects hands and extracts landmarks
3. **Preprocessing**: Landmarks are normalized (position-invariant)
4. **Classification**: Neural network predicts ASL letter
5. **Post-processing**: Prediction smoothing via history averaging
6. **Visualization**: Results displayed with confidence and FPS metrics

## Key Design Decisions

### Modularity
- Clear separation of concerns
- Each module has a single responsibility
- Easy to test and extend

### Configuration Management
- Centralized configuration in `config.py`
- Environment variable support
- Type-safe configuration access

### Error Handling
- Comprehensive error handling at each layer
- Graceful degradation
- Informative error messages

### Performance
- FPS tracking and latency monitoring
- Optimized preprocessing with NumPy vectorization
- Efficient model inference

### Extensibility
- Quantum layer stub for future integration
- Plugin-friendly architecture
- Easy to add new gesture classes

## Configuration

All configuration is centralized in `src/utils/config.py`:

- **Model paths**: Configurable via environment variables
- **MediaPipe settings**: Detection/tracking confidence thresholds
- **Inference settings**: Confidence threshold, history length
- **Camera settings**: Resolution, FPS target
- **Visualization**: Colors, window settings

## Testing

Unit tests are located in `/tests/`:

- `test_preprocessing.py`: Preprocessing functions
- `test_model.py`: Model architecture
- `test_config.py`: Configuration management

Run tests with:
```bash
make test
```

## Future Enhancements

1. **Quantum Integration**: Hybrid quantum-classical models
2. **Mobile Deployment**: TensorFlow Lite optimization
3. **Sentence Recognition**: Context-aware grammar
4. **Multi-language**: International sign languages
5. **Cloud API**: Scalable inference service


