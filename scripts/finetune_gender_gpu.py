"""Fine-tune PAT on NHANES accelerometry to predict gender using a GPU.

Example
-------
python scripts/finetune_gender_gpu.py \
  --covariates-rds covariates_mortality_G_H_tidy.rds \
  --epochs 30 --batch-size 16 --output-dir runs/gender_gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

# Run directly from a source checkout, preferring its local pypat package.
SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from pypat import fine_tune_pat, load_nhanes_weekly_accelerometry


def main() -> None:
    arguments = _parse_arguments()
    _configure_gpu(require_gpu=arguments.require_gpu)
    gender = _read_gender(arguments.covariates_rds)
    X, participant_ids = load_nhanes_weekly_accelerometry(
        arguments.activity_path,
        start_day=arguments.start_day,
        verbose=arguments.verbose,
    )
    X, y, participant_ids = _align_gender(X, participant_ids, gender)
    if arguments.max_participants is not None:
        X, y, participant_ids = _stratified_subset(X, y, participant_ids, arguments.max_participants, arguments.random_state)
    print(f"Training cohort: {len(y):,} participants; gender counts: {pd.Series(y).value_counts().to_dict()}")

    result = fine_tune_pat(
        X,
        y,
        task="binary",
        weights_path=arguments.weights_path,
        test_size=arguments.test_size,
        validation_size=arguments.validation_size,
        random_state=arguments.random_state,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        patience=arguments.patience,
        all_day_cycles=arguments.all_day_cycles,
        verbose=2,
    )
    _save_results(result, arguments.output_dir)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activity-path", default="nhanes_1440_AC.csv.xz")
    parser.add_argument("--covariates-rds", default="covariates_mortality_G_H_tidy.rds")
    parser.add_argument("--weights-path", default=None, help="Existing PAT-L weights, or a destination for automatic download.")
    parser.add_argument("--output-dir", default="runs/gender_gpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=20260722)
    parser.add_argument("--start-day", type=int, default=None, help="Optional cyclic first day for the week (2 through 8).")
    parser.add_argument("--all-day-cycles", action="store_true", help="Augment only training participants with all seven day rotations.")
    parser.add_argument("--max-participants", type=int, default=None, help="Optional stratified development subset; omit for the full cohort.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail rather than fall back to CPU when no GPU is visible.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args()


def _configure_gpu(*, require_gpu: bool) -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if require_gpu:
            raise RuntimeError("No GPU is visible to TensorFlow. Start a GPU runtime or omit --require-gpu to use CPU.")
        print("WARNING: no GPU is visible; training will use CPU.")
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow {tf.__version__}; using {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")


def _read_gender(rds_path: str | Path) -> pd.DataFrame:
    try:
        import pyreadr
    except ImportError as error:
        raise ImportError("Reading the RDS file requires pyreadr. Install it with `pip install pyreadr`.") from error
    result = pyreadr.read_r(str(rds_path))
    if not result:
        raise ValueError(f"No data frame found in {rds_path}.")
    covariates = next(iter(result.values()))
    required = {"SEQN", "gender"}
    if not required.issubset(covariates.columns):
        raise ValueError(f"RDS file must contain {sorted(required)}; found {covariates.columns.tolist()}.")
    return covariates.loc[:, ["SEQN", "gender"]].dropna().drop_duplicates("SEQN")


def _align_gender(X: np.ndarray, participant_ids: np.ndarray, gender: pd.DataFrame):
    gender_by_id = gender.set_index("SEQN")["gender"]
    labels = gender_by_id.reindex(participant_ids)
    keep = labels.notna().to_numpy()
    return X[keep], labels.loc[keep].to_numpy(), participant_ids[keep]


def _stratified_subset(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray, maximum: int, random_state: int):
    if maximum < 6 or maximum > len(y):
        raise ValueError(f"max_participants must be between 6 and {len(y)}.")
    from sklearn.model_selection import train_test_split

    indices, _ = train_test_split(np.arange(len(y)), train_size=maximum, random_state=random_state, stratify=y)
    return X[indices], y[indices], participant_ids[indices]


def _save_results(result, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probabilities = result.predictions
    true_codes = result.y_test.astype(int)
    true_labels = result.class_labels[true_codes]
    predicted_labels = result.predict_classes(result.X_test)
    labels = result.class_labels
    metrics = {
        **result.metrics,
        "roc_auc": float(roc_auc_score(true_codes, probabilities)),
        "average_precision": float(average_precision_score(true_codes, probabilities)),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "f1": float(f1_score(true_codes, probabilities >= 0.5)),
        "log_loss": float(log_loss(true_codes, probabilities, labels=[0, 1])),
        "n_test": int(len(true_codes)),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=labels).tolist(),
        "class_order": labels.tolist(),
    }
    with (output / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    pd.DataFrame(
        {
            "true_gender": true_labels,
            "predicted_gender": predicted_labels,
            f"probability_{labels[1]}": probabilities,
        }
    ).to_csv(output / "test_predictions.csv", index=False)
    result.model.save_weights(output / "model.weights.h5")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics, predictions, and model weights to {output}.")


if __name__ == "__main__":
    main()
