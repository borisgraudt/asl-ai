import os
from pathlib import Path

from src.utils.config import Config, config


class TestConfig:
    def test_config_paths_exist(self):
        # est that config paths are Path objects.
        assert isinstance(config.PROJECT_ROOT, Path)
        assert isinstance(config.MODELS_DIR, Path)
        assert isinstance(config.DATA_DIR, Path)
    
    def test_ensure_directories(self):
        # Test directory creation.
        config.ensure_directories()
        
        # Verify directories exist
        assert config.MODELS_DIR.exists()
        assert config.DATA_DIR.exists()
        assert config.LOGS_DIR.exists()
    
    def test_get_model_paths(self):
        # Test getting model paths.
        paths = config.get_model_paths()
        
        # Verify all expected keys are present
        assert "model" in paths
        assert "scaler" in paths
        assert "label_encoder" in paths
        assert "tflite" in paths
        
        # Verify all values are Path objects
        for path in paths.values():
            assert isinstance(path, Path)
    
    def test_validate_paths(self):
        # Test path validation.
        validation = config.validate_paths()
        
        # Verify validation returns dictionary
        assert isinstance(validation, dict)
        
        # Verify expected keys
        assert "model" in validation
        assert "scaler" in validation
        assert "label_encoder" in validation


