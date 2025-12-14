"""
Configuration management for ASL&AI system.

This module centralizes all configuration parameters, paths, and constants
used throughout the application. Supports environment variable overrides.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Centralized configuration for ASL&AI system."""
    
    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    PLOTS_DIR = PROJECT_ROOT / "plots"
    
    # Model paths (can be overridden via environment variables)
    # Use a highly compatible default format. `.h5` loads reliably across TF/Keras versions.
    # Fall back to an existing `.keras` file if present (for backwards compatibility).
    _default_model = MODELS_DIR / "model.h5"
    _legacy_model = MODELS_DIR / "model.keras"
    MODEL_PATH = Path(
        os.getenv(
            "ASL_AI_MODEL_PATH",
            str(_default_model if _default_model.exists() else _legacy_model),
        )
    )
    SCALER_PATH = MODELS_DIR / "scaler.pkl"
    LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
    TFLITE_MODEL_PATH = MODELS_DIR / "model.tflite"
    
    # Data paths
    # Note: the repo ships with a small sample dataset at `data/sample_raw_gestures/`.
    # For full training, point to your dataset via `ASL_AI_RAW_DATA_DIR`.
    RAW_DATA_DIR = Path(os.getenv("ASL_AI_RAW_DATA_DIR", str(DATA_DIR / "sample_raw_gestures")))
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # MediaPipe configuration
    MEDIAPIPE_CONFIG = {
        "static_image_mode": False,
        "max_num_hands": 1,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.7,
    }
    
    # Model inference configuration
    INFERENCE_CONFIG = {
        "batch_size": 1,
        "confidence_threshold": 0.5,  # Minimum confidence for predictions
        "prediction_history_length": 5,  # Frames to average for stability
    }
    
    # Camera configuration
    CAMERA_CONFIG = {
        "camera_index": int(os.getenv("ASL_AI_CAMERA_INDEX", "0")),
        "fps_target": 30,
        "frame_width": 1280,
        "frame_height": 720,
    }
    
    # Visualization configuration
    VISUALIZATION_CONFIG = {
        "window_name": "ASL Recognition",
        "show_fps": True,
        "show_confidence": True,
        "show_stability": True,
        "landmark_color": (50, 205, 50),  # Green
        "connection_color": (255, 140, 0),  # Orange
        "text_color": (255, 255, 255),  # White
        "info_panel_height": 300,
    }
    
    # Performance monitoring
    PERFORMANCE_CONFIG = {
        "enable_benchmarking": True,
        "fps_window_size": 30,  # Frames to average for FPS calculation
        "log_metrics": True,
        "metrics_log_path": LOGS_DIR / "metrics.json",
    }
    
    # Training configuration
    TRAINING_CONFIG = {
        "batch_size": 64,
        "epochs": 50,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "reduce_lr_factor": 0.5,
        "min_learning_rate": 1e-6,
        # Model architecture
        # - "mlp": default dense network
        # - "moe": Mixture-of-Experts (portfolio-grade experiment)
        "architecture": os.getenv("ASL_AI_MODEL_ARCH", "mlp"),
        "moe_num_experts": int(os.getenv("ASL_AI_MOE_EXPERTS", "4")),
        "moe_expert_units": int(os.getenv("ASL_AI_MOE_UNITS", "128")),
        "moe_top_k": int(os.getenv("ASL_AI_MOE_TOPK", "2")),
    }
    
    # Quantum layer configuration (for future integration)
    QUANTUM_CONFIG = {
        "enabled": False,
        "n_qubits": 4,
        "device": "default.qubit",
    }
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.MODELS_DIR,
            cls.DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.LOGS_DIR,
            cls.PLOTS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_paths(cls) -> Dict[str, Path]:
        """Get all model-related file paths."""
        return {
            "model": cls.MODEL_PATH,
            "scaler": cls.SCALER_PATH,
            "label_encoder": cls.LABEL_ENCODER_PATH,
            "tflite": cls.TFLITE_MODEL_PATH,
        }
    
    @classmethod
    def validate_paths(cls) -> Dict[str, bool]:
        """Validate that required model files exist."""
        paths = cls.get_model_paths()
        return {
            name: path.exists() 
            for name, path in paths.items() 
            if name != "tflite"  # TFLite is optional
        }


# Global configuration instance
config = Config()


