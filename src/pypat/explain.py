"""Utilities for turning PAT attention matrices into time-series profiles."""

from __future__ import annotations

import numpy as np


def attention_to_timeline(
    attention: np.ndarray,
    patch_size: int,
    *,
    input_length: int | None = None,
) -> np.ndarray:
    """Summarize attention matrices as normalized time-point importance.

    Parameters
    ----------
    attention
        A transformer layer's scores with shape
        ``(participants, heads, query_patches, key_patches)``.
    patch_size
        Number of time points represented by one patch.
    input_length
        Optional unpadded recording length. When supplied, padding is removed
        from the returned profiles.

    Returns
    -------
    numpy.ndarray
        One normalized profile per participant, with shape
        ``(participants, time_points)``. Scores average all heads and query
        patches, leaving the attention received by each key patch.
    """
    scores = np.asarray(attention)
    if scores.ndim != 4:
        raise ValueError("attention must have shape (participants, heads, query_patches, key_patches).")
    if patch_size < 1:
        raise ValueError("patch_size must be positive.")
    profile = scores.mean(axis=(1, 2))
    totals = profile.sum(axis=1, keepdims=True)
    profile = np.divide(profile, totals, out=np.zeros_like(profile), where=totals != 0)
    timeline = np.repeat(profile, patch_size, axis=1)
    if input_length is not None:
        if not 0 < input_length <= timeline.shape[1]:
            raise ValueError("input_length must be between 1 and the padded timeline length.")
        timeline = timeline[:, :input_length]
    return timeline
