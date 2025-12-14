"""
Model training utilities for ASL gesture recognition.

Provides training pipeline with callbacks, visualization, and evaluation.
"""

import numpy as np
import pickle
import time
import locale
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard  # type: ignore
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .model import create_asl_model
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Load preprocessed training and test data.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]
        Tuple of (X_train, X_test, y_train, y_test, label_encoder)
    """
    logger.info("Loading training data...")
    
    # Load data
    X_train = np.load(config.PROCESSED_DATA_DIR / "X_train.npy")
    X_test = np.load(config.PROCESSED_DATA_DIR / "X_test.npy")
    y_train = np.load(config.PROCESSED_DATA_DIR / "y_train.npy")
    y_test = np.load(config.PROCESSED_DATA_DIR / "y_test.npy")
    
    # Load label encoder
    with open(config.LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    
    logger.info(f"Loaded {len(X_train)} training samples, {len(X_test)} test samples")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    
    return X_train, X_test, y_train, y_test, label_encoder


def prepare_data_for_training(
    X_train: np.ndarray,
    X_test: np.ndarray,
    batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data by trimming to batch size multiples.

    NOTE: This is kept for backwards compatibility with older models that hardcoded
    batch_size in the Input layer. The current model no longer requires trimming.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Test features
    batch_size : int
        Batch size
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Trimmed (X_train, X_test)
    """
    # No trimming required with flexible batch size models
    _ = batch_size
    return X_train, X_test


def fit_and_apply_scaler(
    X_train: np.ndarray,
    X_test: np.ndarray,
    save_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data and transform train/test.

    Saves the scaler to disk for inference consistency.
    """
    save_path = save_path or config.SCALER_PATH
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {save_path}")

    return X_train_scaled, X_test_scaled, scaler


def create_callbacks(run_name: str = "default") -> list:
    """
    Create training callbacks.
    
    Returns
    -------
    list
        List of Keras callbacks
    """
    train_config = config.TRAINING_CONFIG
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    safe_run = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in (run_name or "default")])

    # Use weights-only checkpoints for maximum compatibility across TF/Keras versions.
    # (Some Keras save formats reject certain save options.)
    best_weights_path = config.MODELS_DIR / f"{safe_run}.best.weights.h5"
    csv_log_path = config.LOGS_DIR / f"{safe_run}.training_log.csv"
    tb_log_dir = config.LOGS_DIR / "tensorboard" / safe_run
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=train_config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=train_config["reduce_lr_factor"],
            patience=train_config["reduce_lr_patience"],
            min_lr=train_config["min_learning_rate"],
            verbose=1
        )
        ,
        CSVLogger(str(csv_log_path), append=False),
        TensorBoard(log_dir=str(tb_log_dir), histogram_freq=0, write_graph=True),
    ]
    
    return callbacks


def plot_training_history(history: Dict[str, list], save_path: Optional[Path] = None) -> None:
    """
    Plot training history (accuracy and loss curves).
    
    Parameters
    ----------
    history : Dict[str, list]
        Training history dictionary from model.fit()
    save_path : Optional[Path]
        Path to save plot. If None, uses config default.
    """
    # Set English locale for matplotlib
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C')
        except locale.Error:
            pass  # Use system default
    
    save_path = save_path or config.PLOTS_DIR / "training_history.png"
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list
        List of class names
    save_path : Optional[Path]
        Path to save plot. If None, uses config default.
    """
    save_path = save_path or config.PLOTS_DIR / "confusion_matrix.png"
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def train_model(
    model_path: Optional[Path] = None,
    save_plots: bool = True,
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train ASL gesture recognition model.
    
    Parameters
    ----------
    model_path : Optional[Path]
        Path to save trained model. If None, uses config default.
    save_plots : bool
        Whether to save training plots (default: True)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing training results and metrics
    """
    train_config = config.TRAINING_CONFIG
    arch = str(train_config.get("architecture", "mlp"))
    run_name = run_name or f"train_{arch}"
    model_path = model_path or config.MODEL_PATH
    
    logger.info("Starting model training...")
    start_time = time.time()

    # Reproducibility (best-effort)
    try:
        tf.keras.utils.set_random_seed(42 if seed is None else int(seed))
        if hasattr(tf.config.experimental, "enable_op_determinism"):
            tf.config.experimental.enable_op_determinism()
    except Exception:
        # Determinism depends on platform/TF build; safe to continue.
        pass
    
    # Load data
    X_train, X_test, y_train, y_test, label_encoder = load_training_data()
    
    # Prepare data
    batch_size = train_config["batch_size"]
    X_train, X_test = prepare_data_for_training(X_train, X_test, batch_size)

    # Scale features (important for dense networks)
    X_train, X_test, _scaler = fit_and_apply_scaler(X_train, X_test)
    
    # Create model
    model = create_asl_model(
        input_shape=X_train.shape[1],
        num_classes=len(label_encoder.classes_),
        learning_rate=train_config["learning_rate"],
        architecture=train_config.get("architecture", "mlp"),
        moe_num_experts=int(train_config.get("moe_num_experts", 4)),
        moe_expert_units=int(train_config.get("moe_expert_units", 128)),
        moe_top_k=train_config.get("moe_top_k", 2),
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    # Create callbacks
    callbacks = create_callbacks(run_name=run_name)
    
    # Train model
    logger.info("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=train_config["epochs"],
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_accuracy, test_top3 = model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)
    logger.info(f"Test accuracy: {test_accuracy*100:.2f}%")
    logger.info(f"Test top-3 accuracy: {test_top3*100:.2f}%")
    
    # Generate predictions for detailed metrics
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(
        y_test,
        y_pred_classes,
        target_names=class_names,
        output_dict=True
    )
    
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Save plots
    if save_plots:
        plot_training_history(history.history)
        plot_confusion_matrix(y_test, y_pred_classes, class_names)
    
    # Save training history for later regeneration
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = config.LOGS_DIR / f"{run_name}.training_history.json"
    # Convert numpy arrays to lists for JSON serialization
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Save model
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Compile results
    results = {
        "run_name": run_name,
        "architecture": arch,
        "test_accuracy": float(test_accuracy),
        "test_top3_accuracy": float(test_top3),
        "test_loss": float(test_loss),
        "training_time_minutes": training_time / 60,
        "num_epochs": len(history.history['loss']),
        "classification_report": report,
    }
    
    return results


