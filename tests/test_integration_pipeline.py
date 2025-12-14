from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.ai.model import create_asl_model
from src.ai.inference import GestureClassifier


class _DummyLabelEncoder:
    # Minimal LabelEncoder-like object used for fast integration testing.

    def __init__(self, classes: list[str]):
        self.classes_ = np.array(classes)

    def inverse_transform(self, idxs):
        return self.classes_[np.array(idxs)]


class _IdentityScaler:
     # Pickle-friendly identity scaler for integration tests.

    def transform(self, x):
        return x


def test_inference_pipeline_load_and_predict(tmp_path: Path) -> None:
    # Create and save a tiny model
    model = create_asl_model(input_shape=63, num_classes=3, learning_rate=0.001)
    model_path = tmp_path / "model.h5"
    model.save(model_path)

    scaler_path = tmp_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(_IdentityScaler(), f)

    # Create and save a minimal label encoder
    le_path = tmp_path / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(_DummyLabelEncoder(["A", "B", "C"]), f)

    # Load classifier and run a prediction
    clf = GestureClassifier(
        model_path=model_path,
        scaler_path=scaler_path,
        label_encoder_path=le_path,
        confidence_threshold=0.0,
        history_length=1,
    )

    features = np.zeros((1, 63), dtype=np.float32)
    pred_letter, confidence, stability = clf.predict(features, use_history=True)

    assert pred_letter in {"A", "B", "C"}
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= stability <= 1.0


