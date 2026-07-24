"""Tests for dependency-free behavior in :mod:`pypat.finetune`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pypat.data import binary_class_weights, pad_to_length, prepare_y, stratify_labels
from pypat.datasets import (
    augment_all_weekly_cycles,
    load_nhanes_weekly_accelerometry,
    rotate_nhanes_weekly_accelerometry,
)
from pypat.explain import attention_to_timeline
from pypat.weights import DEFAULT_WEIGHTS_NAME, download_weights


class FineTuneHelpersTest(unittest.TestCase):
    def test_binary_labels_are_encoded_and_retained(self) -> None:
        y, task, labels = prepare_y(["control", "case", "control"], 3, "auto")
        np.testing.assert_array_equal(y, [1, 0, 1])
        self.assertEqual(task, "binary")
        np.testing.assert_array_equal(labels, ["case", "control"])

    def test_continuous_outcome_is_preserved(self) -> None:
        y, task, labels = prepare_y([1.2, 4.5, 9.0], 3, "auto")
        np.testing.assert_allclose(y, [1.2, 4.5, 9.0])
        self.assertEqual(task, "continuous")
        self.assertIsNone(labels)

    def test_categorical_and_survival_outcomes_are_prepared(self) -> None:
        categorical, task, labels = prepare_y(["low", "medium", "high"], 3, "categorical")
        self.assertEqual(task, "categorical")
        self.assertEqual(categorical.shape, (3,))
        np.testing.assert_array_equal(labels, ["high", "low", "medium"])
        survival, task, labels = prepare_y([[0, 1], [2, 0], [1, 1]], 3, "survival", num_time_bins=3)
        self.assertEqual(task, "survival")
        self.assertIsNone(labels)
        np.testing.assert_array_equal(survival, [[0, 1], [2, 0], [1, 1]])

    def test_padding_only_extends_the_time_axis(self) -> None:
        values = np.ones((2, 10), dtype=np.float32)
        padded = pad_to_length(values, 18)
        self.assertEqual(padded.shape, (2, 18))
        np.testing.assert_array_equal(padded[:, :10], values)
        np.testing.assert_array_equal(padded[:, 10:], 0)

    def test_existing_weight_file_is_not_downloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / DEFAULT_WEIGHTS_NAME
            path.write_bytes(b"weights")
            self.assertEqual(download_weights(path), path)

    def test_binary_split_requires_both_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one observed class"):
            stratify_labels(np.array([0, 0]), "binary", "test")

    def test_balanced_weights_inverse_frequency(self) -> None:
        self.assertEqual(binary_class_weights("balanced", np.array([0, 0, 0, 1])), {0: 2 / 3, 1: 2.0})

    def test_attention_profile_expands_patches_and_trims_padding(self) -> None:
        attention = np.array([[[[0.1, 0.9], [0.1, 0.9]]]], dtype=np.float32)
        profile = attention_to_timeline(attention, patch_size=3, input_length=5)
        np.testing.assert_allclose(profile, [[0.1, 0.1, 0.1, 0.9, 0.9]])

    def test_nhanes_loader_keeps_only_complete_weeks(self) -> None:
        minute_columns = {f"min_{minute:04d}": minute for minute in range(1, 1441)}
        complete_week = [{"SEQN": 1, "PAXDAYM": day, **minute_columns} for day in range(2, 9)]
        for row in complete_week:
            row["min_0001"] = row["PAXDAYM"]
        incomplete_week = [{"SEQN": 2, "PAXDAYM": day, **minute_columns} for day in range(2, 8)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nhanes.csv.xz"
            pd.DataFrame([*complete_week, *incomplete_week]).to_csv(path, index=False, compression="xz")
            X, participant_ids = load_nhanes_weekly_accelerometry(path, chunksize=2)
        self.assertEqual(X.shape, (1, 10080))
        np.testing.assert_array_equal(participant_ids, [1])
        self.assertEqual(X[0, 0], 2)

    def test_nhanes_loader_can_stop_after_requested_complete_participants(self) -> None:
        minute_columns = {f"min_{minute:04d}": 1 for minute in range(1, 1441)}
        rows = [{"SEQN": identifier, "PAXDAYM": day, **minute_columns} for identifier in (1, 2) for day in range(2, 9)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nhanes.csv.xz"
            pd.DataFrame(rows).to_csv(path, index=False, compression="xz")
            X, participant_ids = load_nhanes_weekly_accelerometry(path, chunksize=7, max_participants=1)
        self.assertEqual(X.shape, (1, 10080))
        np.testing.assert_array_equal(participant_ids, [1])

    def test_nhanes_loader_can_rotate_or_explicitly_reorder_days(self) -> None:
        minute_columns = {f"min_{minute:04d}": 0 for minute in range(1, 1441)}
        rows = [{"SEQN": 1, "PAXDAYM": day, **minute_columns, "min_0001": day} for day in range(2, 9)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nhanes.csv.xz"
            pd.DataFrame(rows).to_csv(path, index=False, compression="xz")
            rotated, _ = load_nhanes_weekly_accelerometry(path, start_day=8)
            reordered, _ = load_nhanes_weekly_accelerometry(path, day_order=(8, 7, 2, 3, 4, 5, 6))
        self.assertEqual(rotated[0, 0], 8)
        self.assertEqual(rotated[0, 1440], 2)
        self.assertEqual(reordered[0, 0], 8)
        self.assertEqual(reordered[0, 2 * 1440], 2)

    def test_loaded_nhanes_array_can_be_rotated(self) -> None:
        X = np.repeat(np.arange(2, 9, dtype=np.float32), 1440)[None, :]
        rotated = rotate_nhanes_weekly_accelerometry(X, start_day=8)
        self.assertEqual(rotated[0, 0], 8)
        self.assertEqual(rotated[0, 1440], 2)
        np.testing.assert_array_equal(X[0, :1440], np.repeat(2, 1440))

    def test_all_weekly_cycles_augment_training_data_and_outcomes(self) -> None:
        X = np.stack([np.repeat(np.arange(2, 9), 1440), np.repeat(np.arange(20, 27), 1440)])
        augmented_X, augmented_y = augment_all_weekly_cycles(X, np.array([10, 20]))
        self.assertEqual(augmented_X.shape, (14, 10080))
        np.testing.assert_array_equal(augmented_y, [10, 20] * 7)
        self.assertEqual(augmented_X[2, 0], 3)
        self.assertEqual(augmented_X[2, 6 * 1440], 2)


if __name__ == "__main__":
    unittest.main()
