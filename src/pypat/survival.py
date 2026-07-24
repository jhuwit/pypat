"""Discrete-time survival loss for PAT time-to-event fine-tuning."""

from __future__ import annotations


def discrete_time_survival_loss(num_time_bins: int):
    """Return a censoring-aware discrete-time negative log-likelihood.

    ``y_true`` must contain ``[time_bin, event_observed]`` per participant.
    ``time_bin`` is zero based and ``event_observed`` is 1 for an event and 0
    for right censoring. Censoring is interpreted as occurring at the end of
    ``time_bin``: a censored participant contributes survival through that
    bin. The model output contains a sigmoid conditional event hazard for each
    time bin.
    """
    if num_time_bins < 1:
        raise ValueError("num_time_bins must be positive.")
    import tensorflow as tf

    def loss(y_true, hazards):
        time_bin = tf.cast(y_true[:, 0], tf.int32)
        event = tf.cast(y_true[:, 1], hazards.dtype)
        hazard = tf.clip_by_value(hazards, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        time_index = tf.range(num_time_bins)[tf.newaxis, :]
        survived_before = tf.cast(time_index < time_bin[:, tf.newaxis], hazards.dtype)
        survived_through = tf.cast(time_index <= time_bin[:, tf.newaxis], hazards.dtype)
        survival_terms = event[:, tf.newaxis] * survived_before + (1 - event)[:, tf.newaxis] * survived_through
        log_likelihood = tf.reduce_sum(survival_terms * tf.math.log1p(-hazard), axis=1)
        event_hazard = tf.reduce_sum(tf.one_hot(time_bin, num_time_bins, dtype=hazards.dtype) * tf.math.log(hazard), axis=1)
        return -log_likelihood - event * event_hazard

    return loss
