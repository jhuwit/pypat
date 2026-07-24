"""High-level API for fine-tuning PAT on accelerometry outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import binary_class_weights, pad_to_length, prepare_y, stratify_labels, validate_X
from .datasets import augment_all_weekly_cycles
from .explain import attention_to_timeline
from .model import PATConfig, PAT_CONFIGS, build_finetuning_model
from .weights import DEFAULT_WEIGHTS_URL, download_weights


@dataclass
class FineTuneResult:
    """Fitted model, held-out evaluation, and prediction utilities."""

    model: Any
    history: Any
    scaler: StandardScaler
    task: Literal["binary", "continuous"]
    metrics: dict[str, float]
    weights_path: Path
    config: PATConfig
    input_length: int
    padded_input_length: int
    X_test: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    class_labels: tuple[Any, Any] | None = None

    def predict(self, X: Any) -> np.ndarray:
        """Predict from unscaled, unpadded accelerometry rows."""
        return self.model.predict(self._prepare_X(X), verbose=0).reshape(-1)

    def predict_classes(self, X: Any, *, threshold: float = 0.5) -> np.ndarray:
        """Return original binary labels for ``X`` using a probability threshold."""
        if self.task != "binary" or self.class_labels is None:
            raise ValueError("predict_classes is only available for binary outcomes.")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        return np.where(self.predict(X) >= threshold, self.class_labels[1], self.class_labels[0])

    def attention(self, X: Any) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return predictions and attention matrices for each encoder layer.

        The matrices have shape ``(participants, heads, patches, patches)``.
        They are large for long recordings, so pass a small number of rows.
        """
        attention_model = build_finetuning_model(
            input_size=self.padded_input_length,
            weights_path=self.weights_path,
            config=self.config,
            task=self.task,
            return_attention=True,
        )
        attention_model.set_weights(self.model.get_weights())
        outputs = attention_model.predict(self._prepare_X(X), verbose=0)
        return outputs[0].reshape(-1), list(outputs[1:])

    def attention_profile(self, X: Any, *, layer: int = -1) -> np.ndarray:
        """Return a normalized time-point attention profile for one layer."""
        _, layer_attention = self.attention(X)
        try:
            attention = layer_attention[layer]
        except IndexError as error:
            raise ValueError(f"layer must index one of {len(layer_attention)} encoder layers.") from error
        return attention_to_timeline(
            attention,
            self.config.patch_size,
            input_length=self.input_length,
        )

    def _prepare_X(self, X: Any) -> np.ndarray:
        values = validate_X(X)
        if values.shape[1] != self.input_length:
            raise ValueError(f"X has {values.shape[1]} time points; expected {self.input_length}.")
        return self.scaler.transform(pad_to_length(values, self.padded_input_length)).astype(np.float32)


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
    all_day_cycles: bool = False,
    freeze_encoder: bool = False,
    verbose: int = 2,
) -> FineTuneResult:
    """Split data, scale it, and fine-tune PAT for binary or continuous ``y``.

    ``X`` is a numeric array of shape ``(participants, time_points)``. With
    ``task="auto"``, two distinct outcome values are treated as binary; all
    other numeric outcomes are regressed. The default download supplies
    PAT-L weights; smaller models require their matching ``weights_path`` or
    ``weights_url``. Set ``all_day_cycles=True`` to augment the training split
    with every rotation of its seven daily blocks; validation and test data are
    deliberately left unchanged.
    """
    tf = _tensorflow()
    X_array = validate_X(X)
    y_array, resolved_task, class_labels = prepare_y(y, len(X_array), task)
    _validate_options(model_size, test_size, validation_size, epochs, batch_size, learning_rate, patience, weights_path, weights_url)
    if random_state is not None:
        tf.keras.utils.set_random_seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X_array,
        y_array,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels(y_array, resolved_task, "test"),
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_labels(y_train, resolved_task, "validation"),
    )
    if all_day_cycles:
        X_train, y_train = augment_all_weekly_cycles(X_train, y_train)
    config = PAT_CONFIGS[model_size]
    original_length = X_array.shape[1]
    padded_length = original_length + (-original_length % config.patch_size)
    X_test_raw = X_test.copy()
    X_train, X_val, X_test = (pad_to_length(values, padded_length) for values in (X_train, X_val, X_test))
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(values).astype(np.float32) for values in (X_train, X_val, X_test))

    local_weights = download_weights(weights_path, url=weights_url)
    model = build_finetuning_model(input_size=padded_length, weights_path=local_weights, config=config, task=resolved_task)
    if freeze_encoder:
        model.get_layer("encoder_model").trainable = False
    monitor, mode, fit_class_weight = _compile_model(tf, model, resolved_task, learning_rate, y_train, class_weight)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience, restore_best_weights=True)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                        class_weight=fit_class_weight, callbacks=callbacks, verbose=verbose)
    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)
    predictions = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
    return FineTuneResult(model, history, scaler, resolved_task, {key: float(value) for key, value in scores.items()},
                          local_weights, config, original_length, padded_length, X_test_raw, y_test, predictions, class_labels)


def _compile_model(tf, model, task, learning_rate, y_train, class_weight):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if task == "binary":
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
        )
        return "val_auc", "max", binary_class_weights(class_weight, y_train)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return "val_loss", "min", None


def _validate_options(model_size, test_size, validation_size, epochs, batch_size, learning_rate, patience, weights_path, weights_url):
    if model_size not in PAT_CONFIGS:
        raise ValueError(f"model_size must be one of {sorted(PAT_CONFIGS)}, got {model_size!r}.")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must both be between 0 and 1.")
    if epochs < 1 or batch_size < 1 or patience < 0 or learning_rate <= 0:
        raise ValueError("epochs and batch_size must be positive; patience must be non-negative; learning_rate must be positive.")
    if model_size != "large" and weights_path is None and weights_url == DEFAULT_WEIGHTS_URL:
        raise ValueError("The bundled download is for PAT-L. Supply weights_path or weights_url when using a smaller model.")


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError("fine_tune_pat requires TensorFlow. Install it with `pip install tensorflow`.") from error
    return tf
