"""Fine-tune the pretrained PAT encoder on accelerometry outcomes.

The public entry point is :func:`fine_tune_pat`.  It accepts one row per
participant in ``X`` and either a binary or continuous outcome in ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlretrieve

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_WEIGHTS_URL = (
    "https://www.dropbox.com/scl/fi/ha9b0cj4b3gvcfq4etc6h/"
    "weight_only_encoder_large_90_unsmoothed_mse_all.h5?rlkey="
    "sbu5fd9p56qawnquz4w6stjzr&st=aewhfwq5&dl=1"
)
DEFAULT_WEIGHTS_NAME = "WEIGHTS_encoder_large_90_unsmoothed_mse_all.h5"


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


@dataclass
class FineTuneResult:
    """Objects and held-out data produced by :func:`fine_tune_pat`."""

    model: Any
    history: Any
    scaler: StandardScaler
    task: Literal["binary", "continuous"]
    metrics: dict[str, float]
    weights_path: Path
    input_length: int
    padded_input_length: int
    X_test: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray

    def predict(self, X: Any) -> np.ndarray:
        """Predict from unscaled, unpadded accelerometry rows."""
        X_array = _validate_X(X)
        if X_array.shape[1] != self.input_length:
            raise ValueError(
                f"X has {X_array.shape[1]} time points; expected {self.input_length}."
            )
        X_array = _pad_to_length(X_array, self.padded_input_length)
        return self.model.predict(self.scaler.transform(X_array), verbose=0).reshape(-1)


def download_weights(
    weights_path: str | Path | None = None,
    *,
    url: str = DEFAULT_WEIGHTS_URL,
) -> Path:
    """Return local encoder weights, downloading them when absent.

    With no path, weights are cached in ``.pypat_weights`` in the current
    working directory.  Downloads are written to a temporary sibling file and
    moved into place only after completion.
    """
    path = Path(weights_path) if weights_path is not None else Path.cwd() / ".pypat_weights" / DEFAULT_WEIGHTS_NAME
    path = path.expanduser()
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".part")
    try:
        urlretrieve(url, temporary_path)
        if not temporary_path.exists() or temporary_path.stat().st_size == 0:
            raise RuntimeError("Downloaded weights file is empty.")
        temporary_path.replace(path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download PAT weights from {url!r}.") from error
    return path


def fine_tune_pat(
    X: Any,
    y: Any,
    *,
    task: Literal["auto", "binary", "continuous"] = "auto",
    weights_path: str | Path | None = None,
    weights_url: str = DEFAULT_WEIGHTS_URL,
    model_size: Literal["small", "medium", "large"] = "large",
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int | None = 20260722,
    epochs: int = 150,
    batch_size: int = 64,
    learning_rate: float = 1e-6,
    patience: int = 25,
    class_weight: Literal["balanced"] | dict[int, float] | None = "balanced",
    freeze_encoder: bool = False,
    verbose: int = 2,
) -> FineTuneResult:
    """Split data, scale it, and fine-tune PAT for binary or continuous ``y``.

    Parameters
    ----------
    X
        Numeric array of shape ``(participants, time_points)``.
    y
        One outcome per participant. Binary labels may use any two distinct
        values; they are encoded as 0 and 1. Continuous outcomes must be
        numeric.
    task
        ``"auto"`` selects binary for exactly two distinct outcome values and
        continuous otherwise.
    weights_path
        Existing PAT weight file, or the destination for an automatic download.

    Returns
    -------
    FineTuneResult
        The fitted model, its scaler, held-out predictions, and evaluation
        metrics. Call ``result.predict(new_X)`` for new unscaled data.
    """
    tf = _tensorflow()
    X_array = _validate_X(X)
    y_array, resolved_task = _prepare_y(y, len(X_array), task)
    if model_size not in PAT_CONFIGS:
        raise ValueError(f"model_size must be one of {sorted(PAT_CONFIGS)}, got {model_size!r}.")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must both be between 0 and 1.")

    stratify = y_array if resolved_task == "binary" and min(np.bincount(y_array.astype(int))) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array, test_size=test_size, random_state=random_state, stratify=stratify
    )
    validation_stratify = y_train if resolved_task == "binary" and min(np.bincount(y_train.astype(int))) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=random_state, stratify=validation_stratify
    )

    config = PAT_CONFIGS[model_size]
    original_length = X_array.shape[1]
    padded_length = original_length + (-original_length % config.patch_size)
    X_train = _pad_to_length(X_train, padded_length)
    X_val = _pad_to_length(X_val, padded_length)
    X_test = _pad_to_length(X_test, padded_length)
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(values).astype(np.float32) for values in (X_train, X_val, X_test))

    local_weights = download_weights(weights_path, url=weights_url)
    model = create_finetuning_model(
        input_size=padded_length, weights_path=local_weights, config=config, task=resolved_task
    )
    if freeze_encoder:
        model.get_layer("encoder_model").trainable = False
    if resolved_task == "binary":
        model.compile(tf.keras.optimizers.Adam(learning_rate), "binary_crossentropy", [tf.keras.metrics.AUC(name="auc"), "accuracy"])
        monitor, mode = "val_auc", "max"
        fit_class_weight = _binary_class_weights(y_train, class_weight)
    else:
        model.compile(tf.keras.optimizers.Adam(learning_rate), "mse", [tf.keras.metrics.MeanAbsoluteError(name="mae")])
        monitor, mode, fit_class_weight = "val_loss", "min", None
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience, restore_best_weights=True)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                        class_weight=fit_class_weight, callbacks=callbacks, verbose=verbose)
    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)
    predictions = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
    return FineTuneResult(model, history, scaler, resolved_task, {key: float(value) for key, value in scores.items()},
                          local_weights, original_length, padded_length, X_test, y_test, predictions)


def create_finetuning_model(*, input_size: int, weights_path: str | Path, config: PATConfig, task: Literal["binary", "continuous"]):
    """Build a PAT encoder plus a task-specific prediction head."""
    tf = _tensorflow()
    encoder = _load_encoder_model(tf, input_size, Path(weights_path), config)
    inputs = tf.keras.layers.Input(shape=(input_size,), name="finetuning_inputs")
    x = encoder(inputs)[0]
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.1, name="dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    activation = "sigmoid" if task == "binary" else "linear"
    outputs = tf.keras.layers.Dense(1, activation=activation, name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="finetuning_model")


def _load_encoder_model(tf, input_size: int, weights_path: Path, config: PATConfig):
    num_patches = input_size // config.patch_size
    inputs = tf.keras.layers.Input(shape=(input_size,), name="inputs")
    x = tf.keras.layers.Reshape((num_patches, config.patch_size), name="reshape")(inputs)
    x = tf.keras.layers.Dense(config.embed_dim, name="dense")(x)
    position = tf.range(num_patches, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, config.embed_dim, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / config.embed_dim))
    x = x + tf.concat([tf.sin(position * div_term), tf.cos(position * div_term)], axis=-1)
    attention_weights = []
    for index in range(config.num_layers):
        prefix = f"encoder_layer_{index + 1}"
        block_input = tf.keras.layers.Input(shape=(None, config.embed_dim), name=f"{prefix}_input")
        attention, scores = tf.keras.layers.MultiHeadAttention(num_heads=config.num_heads, key_dim=config.embed_dim, name=f"{prefix}_attention")(block_input, block_input, return_attention_scores=True)
        attention = tf.keras.layers.Dropout(config.dropout, name=f"{prefix}_dropout")(attention)
        residual = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_norm1")(block_input + attention)
        feedforward = tf.keras.layers.Dense(config.ff_dim, activation="relu", name=f"{prefix}_ff1")(residual)
        feedforward = tf.keras.layers.Dense(config.embed_dim, name=f"{prefix}_ff2")(feedforward)
        feedforward = tf.keras.layers.Dropout(config.dropout, name=f"{prefix}_dropout2")(feedforward)
        block_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_norm2")(residual + feedforward)
        block = tf.keras.Model(block_input, [block_output, scores], name=f"{prefix}_transformer")
        x, scores = block(x)
        attention_weights.append(scores)
    encoder = tf.keras.Model(inputs=inputs, outputs=[x, *attention_weights], name="encoder_model")
    encoder.load_weights(weights_path)
    return encoder


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError("fine_tune_pat requires TensorFlow. Install it with `pip install tensorflow`.") from error
    return tf


def _validate_X(X: Any) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] == 0:
        raise ValueError("X must be a numeric 2-D array with shape (at least 3 participants, time_points).")
    if not np.isfinite(values).all():
        raise ValueError("X must not contain NaN or infinite values.")
    return values


def _prepare_y(y: Any, n_samples: int, task: str) -> tuple[np.ndarray, Literal["binary", "continuous"]]:
    values = np.asarray(y).reshape(-1)
    if len(values) != n_samples or len(values) == 0:
        raise ValueError("y must contain exactly one value per row of X.")
    unique = np.unique(values)
    resolved = "binary" if task == "auto" and len(unique) == 2 else ("continuous" if task == "auto" else task)
    if resolved not in {"binary", "continuous"}:
        raise ValueError("task must be 'auto', 'binary', or 'continuous'.")
    if resolved == "binary":
        if len(unique) != 2:
            raise ValueError("A binary outcome must have exactly two distinct values.")
        return (values == unique[1]).astype(np.float32), "binary"
    try:
        numeric = values.astype(np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("A continuous outcome must be numeric.") from error
    if not np.isfinite(numeric).all():
        raise ValueError("y must not contain NaN or infinite values.")
    return numeric, "continuous"


def _pad_to_length(X: np.ndarray, length: int) -> np.ndarray:
    return np.pad(X, ((0, 0), (0, length - X.shape[1])), mode="constant")


def _binary_class_weights(y: np.ndarray, weights: Literal["balanced"] | dict[int, float] | None) -> dict[int, float] | None:
    if weights is None or isinstance(weights, dict):
        return weights
    counts = np.bincount(y.astype(int), minlength=2)
    if not counts.all():
        return None
    return {label: len(y) / (2 * count) for label, count in enumerate(counts)}
