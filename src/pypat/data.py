"""Input validation, split helpers, and preprocessing for PAT fine-tuning."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def validate_X(X: Any) -> np.ndarray:
    """Validate and convert a participant-by-time-point input array."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] == 0:
        raise ValueError("X must be a numeric 2-D array with shape (at least 3 participants, time_points).")
    if not np.isfinite(values).all():
        raise ValueError("X must not contain NaN or infinite values.")
    return values


def prepare_y(
    y: Any, n_samples: int, task: str
) -> tuple[np.ndarray, Literal["binary", "continuous"], tuple[Any, Any] | None]:
    """Validate outcomes and encode binary labels as zero and one."""
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
        return (values == unique[1]).astype(np.float32), "binary", (unique[0], unique[1])
    try:
        numeric = values.astype(np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("A continuous outcome must be numeric.") from error
    if not np.isfinite(numeric).all():
        raise ValueError("y must not contain NaN or infinite values.")
    return numeric, "continuous", None


def pad_to_length(X: np.ndarray, length: int) -> np.ndarray:
    """Pad the time axis with zeros until it reaches ``length``."""
    return np.pad(X, ((0, 0), (0, length - X.shape[1])), mode="constant")


def stratify_labels(y: np.ndarray, task: Literal["binary", "continuous"], split_name: str) -> np.ndarray | None:
    """Return binary labels for stratification, with useful small-sample errors."""
    if task != "binary":
        return None
    counts = np.bincount(y.astype(int), minlength=2)
    if not counts.all():
        raise ValueError(f"The {split_name} split has only one binary class. Provide more observations of both classes.")
    if counts.min() < 2:
        raise ValueError(
            f"Each binary class needs at least two observations before the {split_name} split; got counts {counts.tolist()}."
        )
    return y


def binary_class_weights(weights: Literal["balanced"] | dict[int, float] | None, y: np.ndarray) -> dict[int, float] | None:
    """Return Keras-compatible inverse-frequency weights for a binary outcome."""
    if weights is None or isinstance(weights, dict):
        return weights
    if weights != "balanced":
        raise ValueError("class_weight must be 'balanced', a {0: weight, 1: weight} dictionary, or None.")
    counts = np.bincount(y.astype(int), minlength=2)
    if not counts.all():
        return None
    return {label: len(y) / (2 * count) for label, count in enumerate(counts)}
