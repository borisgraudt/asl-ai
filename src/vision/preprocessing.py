"""
Preprocessing utilities for hand landmark data.

Provides position-invariant normalization and feature extraction
for MediaPipe hand landmarks.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


def extract_landmarks(landmarks) -> np.ndarray:
    """
    Extract 3D coordinates from MediaPipe hand landmarks.
    
    Parameters
    ----------
    landmarks
        MediaPipe hand landmarks object
    
    Returns
    -------
    np.ndarray
        Array of shape (21, 3) containing x, y, z coordinates
    """
    points = []
    for landmark in landmarks.landmark:
        points.extend([landmark.x, landmark.y, landmark.z])
    return np.array(points, dtype=np.float32).reshape(-1, 3)


def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """
    Apply position-invariant normalization to hand landmarks.
    
    This normalization:
    1. Centers coordinates relative to the palm (landmark 0)
    2. Scales to unit maximum distance for rotation/scale invariance
    
    Parameters
    ----------
    coords : np.ndarray
        Array of shape (21, 3) containing hand landmark coordinates
    
    Returns
    -------
    np.ndarray
        Normalized coordinates of shape (21, 3)
    """
    if coords.shape != (21, 3):
        raise ValueError(f"Expected shape (21, 3), got {coords.shape}")
    
    # Center coordinates relative to palm (landmark 0)
    palm = coords[0]
    coords_centered = coords - palm
    
    # Scale to unit maximum distance
    max_dist = np.linalg.norm(coords_centered, axis=1).max()
    if max_dist > 0:
        coords_centered = coords_centered / max_dist
    
    return coords_centered


def process_landmarks(
    landmarks,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process MediaPipe landmarks into normalized feature vector.
    
    Parameters
    ----------
    landmarks
        MediaPipe hand landmarks object
    scaler : Optional[StandardScaler]
        Optional sklearn StandardScaler for additional normalization.
        If None, returns unnormalized features.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (normalized_coords, feature_vector)
        - normalized_coords: shape (21, 3)
        - feature_vector: shape (63,) or (1, 63) if scaler provided
    """
    # Extract coordinates
    coords = extract_landmarks(landmarks)
    
    # Apply position-invariant normalization
    coords_normalized = normalize_landmarks(coords)
    
    # Flatten to feature vector
    features = coords_normalized.flatten()  # Shape: (63,)
    
    # Apply scaler if provided
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1))  # Shape: (1, 63)
        return coords_normalized, features
    
    return coords_normalized, features.reshape(1, -1)


class LandmarkPreprocessor:
    """
    Preprocessor for hand landmarks with optional scaler normalization.
    
    This class encapsulates the preprocessing pipeline for converting
    MediaPipe hand landmarks into model-ready feature vectors.
    """
    
    def __init__(self, scaler: Optional[StandardScaler] = None):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        scaler : Optional[StandardScaler]
            Optional sklearn StandardScaler for feature normalization.
            Typically loaded from training data.
        """
        self.scaler = scaler
        logger.debug(f"Initialized LandmarkPreprocessor (scaler: {scaler is not None})")
    
    def process(self, landmarks) -> np.ndarray:
        """
        Process landmarks into feature vector.
        
        Parameters
        ----------
        landmarks
            MediaPipe hand landmarks object
        
        Returns
        -------
        np.ndarray
            Feature vector of shape (1, 63) ready for model inference
        """
        _, features = process_landmarks(landmarks, self.scaler)
        return features
    
    def __call__(self, landmarks) -> np.ndarray:
        """Alias for process method."""
        return self.process(landmarks)


