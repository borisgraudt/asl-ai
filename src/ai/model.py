"""
Model architecture definitions for ASL gesture recognition.

Defines the neural network architecture and model creation utilities.
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import Input, Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Layer  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SoftMoE(Layer):
    """
    Lightweight Mixture-of-Experts (MoE) block for vector features.

    A gating network predicts mixture weights over experts, and outputs are combined
    as a weighted sum. Optionally supports top-k gating to encourage specialization.
    """

    def __init__(
        self,
        num_experts: int = 4,
        expert_units: int = 128,
        top_k: Optional[int] = 2,
        dropout: float = 0.1,
        name: str = "soft_moe",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if top_k is not None and (top_k < 1 or top_k > num_experts):
            raise ValueError("top_k must be in [1, num_experts] or None")

        self.num_experts = int(num_experts)
        self.expert_units = int(expert_units)
        self.top_k = int(top_k) if top_k is not None else None
        self.dropout_rate = float(dropout)

        self.gate = Dense(self.num_experts, activation=None, name=f"{name}_gate_logits")
        self.gate_dropout = Dropout(self.dropout_rate, name=f"{name}_gate_dropout")

        self.experts = [
            tf.keras.Sequential(
                [
                    Dense(self.expert_units, activation="relu", name=f"{name}_expert_{i}_dense"),
                    BatchNormalization(name=f"{name}_expert_{i}_bn"),
                    Dropout(self.dropout_rate, name=f"{name}_expert_{i}_dropout"),
                ],
                name=f"{name}_expert_{i}",
            )
            for i in range(self.num_experts)
        ]

    def call(self, inputs, training=None):
        logits = self.gate(inputs)
        if training:
            logits = self.gate_dropout(logits, training=training)
        gate_probs = tf.nn.softmax(logits, axis=-1)  # (B, E)

        if self.top_k is not None and self.top_k < self.num_experts:
            values, indices = tf.nn.top_k(gate_probs, k=self.top_k, sorted=False)
            mask = tf.reduce_sum(tf.one_hot(indices, depth=self.num_experts, dtype=gate_probs.dtype), axis=1)
            gate_probs = gate_probs * mask
            gate_probs = gate_probs / (tf.reduce_sum(gate_probs, axis=-1, keepdims=True) + 1e-9)

        expert_outs = [exp(inputs, training=training) for exp in self.experts]  # list[(B, U)]
        stacked = tf.stack(expert_outs, axis=1)  # (B, E, U)
        weights = tf.expand_dims(gate_probs, axis=-1)  # (B, E, 1)
        return tf.reduce_sum(stacked * weights, axis=1)  # (B, U)


def create_asl_model(
    input_shape: int = 63,
    num_classes: int = 26,
    batch_size: Optional[int] = None,
    learning_rate: float = 0.001,
    use_quantum: bool = False,
    architecture: str = "mlp",
    moe_num_experts: int = 4,
    moe_expert_units: int = 128,
    moe_top_k: Optional[int] = 2,
) -> Model:
    """
    Create ASL gesture recognition model.
    
    Architecture:
    - Input: 63-dimensional hand landmark features
    - Hidden Layers: 256 → 128 → 64 neurons with ReLU activation
    - Regularization: Dropout (0.3, 0.2, 0.1) + Batch Normalization
    - Output: num_classes softmax classification
    
    Parameters
    ----------
    input_shape : int
        Number of input features (default: 63 for 21 landmarks × 3 coords)
    num_classes : int
        Number of output classes (default: 26 for A-Z)
    batch_size : Optional[int]
        Batch size for model. If None, uses config default.
    learning_rate : float
        Learning rate for optimizer (default: 0.001)
    use_quantum : bool
        Whether to use quantum layer (future feature, currently ignored)
    architecture : str
        Either "mlp" (default) or "moe" (Mixture-of-Experts).
    moe_num_experts : int
        Number of experts for MoE (only used when architecture="moe").
    moe_expert_units : int
        Hidden units per expert output for MoE (only used when architecture="moe").
    moe_top_k : Optional[int]
        If set, uses top-k gating to sparsify mixture weights.
    
    Returns
    -------
    Model
        Compiled Keras model ready for training
    """
    logger.info(f"Creating ASL model: input_shape={input_shape}, num_classes={num_classes}")
    
    # Input layer
    # NOTE: Do NOT hardcode batch_size into the model input; it breaks flexible inference
    # and forces training data trimming. Keras can handle variable batch sizes.
    inp = Input(shape=(input_shape,), name="input")
    x = inp

    arch = (architecture or "mlp").lower().strip()
    if arch == "mlp":
        # Hidden layer 1: 256 neurons
        x = Dense(256, activation='relu', name="dense_1")(x)
        x = BatchNormalization(name="bn_1")(x)
        x = Dropout(0.3, name="dropout_1")(x)

        # Hidden layer 2: 128 neurons
        x = Dense(128, activation='relu', name="dense_2")(x)
        x = BatchNormalization(name="bn_2")(x)
        x = Dropout(0.2, name="dropout_2")(x)

        # Hidden layer 3: 64 neurons
        x = Dense(64, activation='relu', name="dense_3")(x)
        x = BatchNormalization(name="bn_3")(x)
        x = Dropout(0.1, name="dropout_3")(x)

    elif arch == "moe":
        # Warmup layer before MoE (helps gating stability)
        x = Dense(256, activation="relu", name="moe_preact_dense")(x)
        x = BatchNormalization(name="moe_preact_bn")(x)
        x = Dropout(0.2, name="moe_preact_dropout")(x)

        x = SoftMoE(
            num_experts=moe_num_experts,
            expert_units=moe_expert_units,
            top_k=moe_top_k,
            dropout=0.15,
            name="moe",
        )(x)

        # Post-MoE head
        x = Dense(64, activation="relu", name="moe_post_dense")(x)
        x = BatchNormalization(name="moe_post_bn")(x)
        x = Dropout(0.1, name="moe_post_dropout")(x)

    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Use 'mlp' or 'moe'.")
    
    # Output layer: num_classes softmax
    out = Dense(num_classes, activation='softmax', name="output")(x)
    
    # Create model
    model = Model(inputs=inp, outputs=out, name=("ASLAIModelMoE" if arch == "moe" else "ASLAIModel"))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    
    logger.info(f"Model created with {model.count_params():,} parameters")
    
    return model


def get_model_summary(model: Model) -> str:
    """
    Get formatted model summary.
    
    Parameters
    ----------
    model : Model
        Keras model
    
    Returns
    -------
    str
        Model summary as string
    """
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    return f.getvalue()


