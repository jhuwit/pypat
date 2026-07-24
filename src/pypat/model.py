"""PAT neural-network architecture and attention-enabled model builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PATConfig:
    """Architecture metadata for a pretrained PAT encoder."""

    patch_size: int
    embed_dim: int
    num_heads: int
    ff_dim: int
    num_layers: int
    dropout: float = 0.1


PAT_CONFIGS = {
    "small": PATConfig(18, 96, 6, 256, 1),
    "medium": PATConfig(18, 96, 12, 256, 2),
    "large": PATConfig(9, 96, 12, 256, 4),
}


def build_finetuning_model(
    *,
    input_size: int,
    weights_path: str | Path,
    config: PATConfig,
    task: Literal["binary", "continuous", "categorical", "survival"],
    num_classes: int | None = None,
    num_time_bins: int | None = None,
    return_attention: bool = False,
):
    """Build a PAT encoder with a task-specific prediction head.

    Parameters
    ----------
    input_size
        Number of input time points after any required padding.
    weights_path
        Path to pretrained encoder weights compatible with ``config``.
    config
        PAT architecture metadata, usually one of :data:`PAT_CONFIGS`.
    task
        Selects the prediction head:

        - ``"binary"``: one sigmoid probability.
        - ``"continuous"``: one linear regression prediction.
        - ``"categorical"``: ``num_classes`` softmax probabilities.
        - ``"survival"``: ``num_time_bins`` sigmoid discrete-time hazards.

    num_classes
        Required for ``task="categorical"`` and must be at least three. Each
        output position corresponds to the integer-encoded class used with a
        sparse categorical cross-entropy loss.
    num_time_bins
        Required for ``task="survival"``. It is the number of discrete hazard
        intervals and therefore the width of the output. Survival labels must
        be supplied separately as ``[time_bin, event_observed]``, where
        ``time_bin`` is a zero-based integer from ``0`` through
        ``num_time_bins - 1`` and ``event_observed`` is 1 for an event or 0 for
        right-censoring.
    return_attention
        When true, prediction returns a list containing
        ``[prediction, layer_1_attention, ..., layer_n_attention]``. Each
        attention tensor has shape ``(participants, heads, patches, patches)``.
        This is memory-intensive for long recordings and is intended for
        explainability rather than training.
    """
    tf = _tensorflow()
    encoder = build_encoder_model(
        input_size=input_size,
        weights_path=weights_path,
        config=config,
        return_attention=return_attention,
    )
    inputs = tf.keras.layers.Input(shape=(input_size,), name="finetuning_inputs")
    encoder_outputs = encoder(inputs)
    if return_attention:
        encoded, attention = encoder_outputs[0], encoder_outputs[1:]
    else:
        encoded, attention = encoder_outputs, []
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(encoded)
    x = tf.keras.layers.Dropout(0.1, name="dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    if task == "binary":
        prediction = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    elif task == "continuous":
        prediction = tf.keras.layers.Dense(1, activation="linear", name="output")(x)
    elif task == "categorical":
        if num_classes is None or num_classes < 3:
            raise ValueError("task='categorical' requires num_classes of at least 3.")
        prediction = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    elif task == "survival":
        if num_time_bins is None or num_time_bins < 1:
            raise ValueError("task='survival' requires a positive num_time_bins.")
        prediction = tf.keras.layers.Dense(num_time_bins, activation="sigmoid", name="hazard")(x)
    else:
        raise ValueError(f"Unknown task: {task!r}.")
    outputs = [prediction, *attention] if return_attention else prediction
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="finetuning_model")


def build_encoder_model(
    *,
    input_size: int,
    weights_path: str | Path,
    config: PATConfig,
    return_attention: bool = False,
):
    """Reconstruct a pretrained encoder and load its weights."""
    tf = _tensorflow()
    num_patches = input_size // config.patch_size
    inputs = tf.keras.layers.Input(shape=(input_size,), name="inputs")
    x = tf.keras.layers.Reshape((num_patches, config.patch_size), name="reshape")(inputs)
    x = tf.keras.layers.Dense(config.embed_dim, name="dense")(x)
    x = x + _positional_embeddings(tf, num_patches, config.embed_dim)
    attention_scores = []
    for index in range(config.num_layers):
        x, scores = _transformer_block(tf, x, config, index + 1, return_attention)
        if return_attention:
            attention_scores.append(scores)
    outputs = [x, *attention_scores] if return_attention else x
    encoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="encoder_model")
    encoder.load_weights(Path(weights_path))
    return encoder


def _transformer_block(tf, x, config: PATConfig, index: int, return_attention: bool):
    prefix = f"encoder_layer_{index}"
    block_input = tf.keras.layers.Input(shape=(None, config.embed_dim), name=f"{prefix}_input")
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=config.num_heads,
        key_dim=config.embed_dim,
        name=f"{prefix}_attention",
    )
    if return_attention:
        attention, scores = attention_layer(block_input, block_input, return_attention_scores=True)
    else:
        attention, scores = attention_layer(block_input, block_input), None
    attention = tf.keras.layers.Dropout(config.dropout, name=f"{prefix}_dropout")(attention)
    residual = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_norm1")(block_input + attention)
    feedforward = tf.keras.layers.Dense(config.ff_dim, activation="relu", name=f"{prefix}_ff1")(residual)
    feedforward = tf.keras.layers.Dense(config.embed_dim, name=f"{prefix}_ff2")(feedforward)
    feedforward = tf.keras.layers.Dropout(config.dropout, name=f"{prefix}_dropout2")(feedforward)
    block_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_norm2")(residual + feedforward)
    block_outputs = [block_output, scores] if return_attention else block_output
    block = tf.keras.Model(block_input, block_outputs, name=f"{prefix}_transformer")
    result = block(x)
    return (result[0], result[1]) if return_attention else (result, None)


def _positional_embeddings(tf, num_patches: int, embed_dim: int):
    position = tf.range(num_patches, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / embed_dim))
    return tf.concat([tf.sin(position * div_term), tf.cos(position * div_term)], axis=-1)


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError("pypat model building requires TensorFlow. Install it with `pip install tensorflow`.") from error
    return tf
