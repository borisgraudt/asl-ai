import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")

from src.ai.model import create_asl_model, get_model_summary


class TestModel:
    # Test cases for model architecture.
    
    def test_create_asl_model_default(self):
        # Test creating model with default parameters.
        model = create_asl_model()
        
        # Verify model is created
        assert model is not None
        
        # Verify input shape
        assert model.input_shape == (None, 63)
        
        # Verify output shape
        assert model.output_shape == (None, 26)
        
        # Verify model has parameters
        assert model.count_params() > 0
    
    def test_create_asl_model_custom(self):
        # Test creating model with custom parameters.
        model = create_asl_model(
            input_shape=63,
            num_classes=10,
            learning_rate=0.01
        )
        
        # Verify custom output shape
        assert model.output_shape == (None, 10)

    def test_create_asl_model_moe(self):
        # Test creating MoE variant.
        model = create_asl_model(
            input_shape=63,
            num_classes=10,
            learning_rate=0.001,
            architecture="moe",
            moe_num_experts=4,
            moe_expert_units=64,
            moe_top_k=2,
        )
        assert model.output_shape == (None, 10)
    
    def test_model_forward_pass(self):
        # Test model forward pass with dummy input.
        model = create_asl_model()
        
        # Create dummy input
        dummy_input = np.random.randn(1, 63).astype(np.float32)
        
        # Forward pass
        output = model.predict(dummy_input, verbose=0)
        
        # Verify output shape
        assert output.shape == (1, 26)
        
        # Verify output is probability distribution (sums to ~1)
        assert np.isclose(output.sum(), 1.0, atol=1e-5)
        
        # Verify all values are non-negative
        assert np.all(output >= 0)
    
    def test_get_model_summary(self):
        # Test getting model summary.
        model = create_asl_model()
        summary = get_model_summary(model)
        
        # Verify summary is a string
        assert isinstance(summary, str)
        
        # Verify summary contains model information
        assert "ASLAIModel" in summary or "Model:" in summary


