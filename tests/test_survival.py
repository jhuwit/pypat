"""Hand-calculated validation for the discrete-time survival loss."""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # Allows the non-TensorFlow utility tests to run independently.
    tf = None

from pypat.survival import discrete_time_survival_loss


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class DiscreteTimeSurvivalLossTest(unittest.TestCase):
    def test_matches_hand_calculated_likelihoods(self) -> None:
        # Rows are [time_bin, event_observed].  Hazards are conditional on
        # survival through all earlier bins.
        y_true = tf.constant([[0, 1], [1, 1], [1, 0]], dtype=tf.float32)
        hazards = tf.constant(
            [[0.20, 0.40, 0.70], [0.30, 0.50, 0.90], [0.10, 0.25, 0.50]],
            dtype=tf.float32,
        )
        actual = discrete_time_survival_loss(3)(y_true, hazards).numpy()
        expected = np.array(
            [
                -math.log(0.20),  # event in bin 0: h_0
                -math.log((1 - 0.30) * 0.50),  # event in bin 1: (1-h_0) h_1
                -math.log((1 - 0.10) * (1 - 0.25)),  # censored at end of bin 1
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
