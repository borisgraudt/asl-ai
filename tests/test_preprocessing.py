import numpy as np
import pytest
from unittest.mock import Mock

from src.vision.preprocessing import (
    extract_landmarks,
    normalize_landmarks,
    process_landmarks,
    LandmarkPreprocessor,
)


class TestPreprocessing:
    # Test cases for preprocessing functions.
    
    def test_extract_landmarks(self):
        # Test landmark extraction from MediaPipe landmarks.
        # Create mock landmarks
        mock_landmarks = Mock()
        mock_landmarks.landmark = []
        for i in range(21):
            landmark = Mock()
            landmark.x = float(i * 0.01)
            landmark.y = float(i * 0.02)
            landmark.z = float(i * 0.03)
            mock_landmarks.landmark.append(landmark)
        
        # Extract landmarks
        coords = extract_landmarks(mock_landmarks)
        
        # Verify shape
        assert coords.shape == (21, 3)
        
        # Verify values
        assert coords[0, 0] == 0.0
        assert coords[1, 1] == 0.02
    
    def test_normalize_landmarks(self):
        # Test position-invariant normalization.
        # Create test coordinates
        coords = np.array([
            [0.0, 0.0, 0.0],  # Palm
            [1.0, 0.0, 0.0],  # Point 1
            [0.0, 1.0, 0.0],  # Point 2
        ])
        
        # Normalize
        normalized = normalize_landmarks(coords)
        
        # Verify shape
        assert normalized.shape == (3, 3)
        
        # Verify palm is at origin
        assert np.allclose(normalized[0], [0.0, 0.0, 0.0])
        
        # Verify maximum distance is 1.0
        distances = np.linalg.norm(normalized, axis=1)
        assert np.isclose(distances.max(), 1.0, atol=1e-6)
    
    def test_normalize_landmarks_wrong_shape(self):
        # Test that wrong shape raises ValueError.
        coords = np.array([[1.0, 2.0, 3.0]])  # Wrong shape
        
        with pytest.raises(ValueError):
            normalize_landmarks(coords)
    
    def test_process_landmarks_without_scaler(self):
        # Test processing landmarks without scaler.
        # Create mock landmarks
        mock_landmarks = Mock()
        mock_landmarks.landmark = []
        for i in range(21):
            landmark = Mock()
            landmark.x = float(i * 0.01)
            landmark.y = float(i * 0.02)
            landmark.z = float(i * 0.03)
            mock_landmarks.landmark.append(landmark)
        
        # Process without scaler
        coords, features = process_landmarks(mock_landmarks, scaler=None)
        
        # Verify shapes
        assert coords.shape == (21, 3)
        assert features.shape == (1, 63)
    
    def test_landmark_preprocessor(self):
        # Test LandmarkPreprocessor class.
        # Create preprocessor without scaler
        preprocessor = LandmarkPreprocessor(scaler=None)
        
        # Create mock landmarks
        mock_landmarks = Mock()
        mock_landmarks.landmark = []
        for i in range(21):
            landmark = Mock()
            landmark.x = float(i * 0.01)
            landmark.y = float(i * 0.02)
            landmark.z = float(i * 0.03)
            mock_landmarks.landmark.append(landmark)
        
        # Process
        features = preprocessor.process(mock_landmarks)
        
        # Verify shape
        assert features.shape == (1, 63)
        
        # Test __call__ method
        features2 = preprocessor(mock_landmarks)
        assert np.allclose(features, features2)


