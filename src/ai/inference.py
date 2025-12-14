"""
Model inference and gesture classification.

Provides GestureClassifier class for real-time ASL gesture recognition
with confidence thresholding and prediction smoothing.
"""

import numpy as np
import pickle
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model, Model  # type: ignore

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GestureClassifier:
    """
    ASL gesture classifier with confidence thresholding and prediction smoothing.
    
    This class wraps a trained TensorFlow/Keras model and provides methods
    for real-time gesture classification with stability improvements.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        label_encoder_path: Optional[Path] = None,
        confidence_threshold: float = 0.5,
        history_length: int = 5,
    ):
        """
        Initialize gesture classifier.
        
        Parameters
        ----------
        model_path : Optional[Path]
            Path to trained Keras model. If None, uses config default.
        scaler_path : Optional[Path]
            Path to saved StandardScaler. If None, uses config default.
        label_encoder_path : Optional[Path]
            Path to saved LabelEncoder. If None, uses config default.
        confidence_threshold : float
            Minimum confidence for predictions (default: 0.5)
        history_length : int
            Number of recent predictions to average for stability (default: 5)
        """
        # Set paths
        self.model_path = model_path or config.MODEL_PATH
        self.scaler_path = scaler_path or config.SCALER_PATH
        self.label_encoder_path = label_encoder_path or config.LABEL_ENCODER_PATH
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.history_length = history_length
        
        # State
        self.model: Optional[Model] = None
        self.scaler: Optional[Any] = None
        self.label_encoder: Optional[Any] = None
        self.prediction_history: List[int] = []
        
        # Load resources
        self._load_resources()
    
    def _load_resources(self) -> None:
        """Load model, scaler, and label encoder from disk."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load model
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.model = load_model(self.model_path)
            
            # Warm up model with dummy input
            dummy_input = np.zeros((1, 63), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            
            logger.info("Model loaded successfully")
            
            # Load scaler
            if self.scaler_path.exists():
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scaler loaded from {self.scaler_path}")
            else:
                logger.warning(f"Scaler not found: {self.scaler_path}")
            
            # Load label encoder
            if self.label_encoder_path.exists():
                with open(self.label_encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                logger.info(f"Label encoder loaded from {self.label_encoder_path}")
            else:
                raise FileNotFoundError(f"Label encoder not found: {self.label_encoder_path}")
            
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
            raise
    
    def predict(
        self,
        features: np.ndarray,
        use_history: bool = True
    ) -> Tuple[str, float, float]:
        """
        Predict gesture from feature vector.
        
        Parameters
        ----------
        features : np.ndarray
            Feature vector of shape (1, 63)
        use_history : bool
            Whether to use prediction history for smoothing (default: True)
        
        Returns
        -------
        Tuple[str, float, float]
            Tuple of (predicted_letter, confidence, stability)
            - predicted_letter: Predicted ASL letter (A-Z)
            - confidence: Prediction confidence [0, 1]
            - stability: Prediction stability based on history [0, 1]
        """
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("Model not loaded. Call _load_resources() first.")
        
        # Get model prediction
        prediction = self.model.predict(features, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_class])
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            logger.debug(f"Low confidence prediction: {confidence:.3f} < {self.confidence_threshold}")
        
        # Update prediction history
        if use_history:
            self.prediction_history.append(predicted_class)
            if len(self.prediction_history) > self.history_length:
                self.prediction_history.pop(0)
            
            # Use most common prediction from history
            if len(self.prediction_history) > 0:
                counter = Counter(self.prediction_history)
                most_common = counter.most_common(1)[0]
                predicted_class = most_common[0]
                stability = most_common[1] / len(self.prediction_history)
            else:
                stability = 1.0
        else:
            stability = 1.0
        
        # Decode class to letter
        predicted_letter = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_letter, confidence, stability
    
    def predict_batch(self, features_batch: np.ndarray) -> List[Tuple[str, float]]:
        """
        Predict gestures for a batch of feature vectors.
        
        Parameters
        ----------
        features_batch : np.ndarray
            Batch of feature vectors of shape (N, 63)
        
        Returns
        -------
        List[Tuple[str, float]]
            List of (predicted_letter, confidence) tuples
        """
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("Model not loaded.")
        
        predictions = self.model.predict(features_batch, verbose=0)
        results = []
        
        for pred in predictions:
            predicted_class = int(np.argmax(pred))
            confidence = float(pred[predicted_class])
            predicted_letter = self.label_encoder.inverse_transform([predicted_class])[0]
            results.append((predicted_letter, confidence))
        
        return results
    
    def reset_history(self) -> None:
        """Clear prediction history."""
        self.prediction_history.clear()
        logger.debug("Prediction history reset")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model metadata
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "num_parameters": self.model.count_params(),
            "num_classes": len(self.label_encoder.classes_) if self.label_encoder else None,
        }


