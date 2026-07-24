"""Fine-tune PAT on NHANES accelerometry for one RDS covariate outcome.

Examples
--------
Binary gender prediction::

    python scripts/finetune_gpu.py --outcome-column gender --require-gpu

Continuous BMI prediction::

    python scripts/finetune_gpu.py --outcome-column body_mass_index_kg_m_2 --task continuous

Integer-coded category prediction::

    python scripts/finetune_gpu.py --outcome-column race_hispanic_origin --task categorical
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from pypat import fine_tune_pat, load_nhanes_weekly_accelerometry


def main() -> None:
    arguments = _parse_arguments()
    _configure_gpu(require_gpu=arguments.require_gpu)
    outcome = _read_outcome(arguments.covariates_rds, arguments.outcome_column)
    task = _resolve_task(outcome[arguments.outcome_column], arguments.task)
    load_limit = arguments.max_participants * 2 if arguments.max_participants is not None else None
    loader_options = dict(
        start_day=arguments.start_day,
        verbose=arguments.verbose,
        max_participants=load_limit,
    )
    if arguments.pad_one_missing_day:
        loader_options["pad_one_missing_day"] = True
    X, participant_ids = load_nhanes_weekly_accelerometry(
        arguments.activity_path,
        **loader_options,
    )
    X, y, participant_ids = _align_outcome(X, participant_ids, outcome, arguments.outcome_column)
    if arguments.max_participants is not None:
        X, y, participant_ids = _subset(X, y, participant_ids, arguments.max_participants, arguments.random_state, task)
    print(f"Outcome: {arguments.outcome_column!r}; task: {task}; participants: {len(y):,}")
    if task in {"binary", "categorical"}:
        print(f"Class counts: {pd.Series(y).value_counts().to_dict()}")

    result = fine_tune_pat(
        X,
        y,
        task=task,
        weights_path=arguments.weights_path,
        test_size=arguments.test_size,
        validation_size=arguments.validation_size,
        random_state=arguments.random_state,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        patience=arguments.patience,
        all_day_cycles=arguments.all_day_cycles,
        freeze_encoder=arguments.freeze_encoder,
        verbose=2,
    )
    _save_results(result, arguments.output_dir, arguments.outcome_column)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activity-path", default="nhanes_1440_AC.csv.xz")
    parser.add_argument("--covariates-rds", default="covariates_mortality_G_H_tidy.rds")
    parser.add_argument("--outcome-column", default="gender", help="One scalar column in the RDS data frame to predict.")
    parser.add_argument("--task", choices=["auto", "binary", "continuous", "categorical"], default="auto")
    parser.add_argument("--weights-path", default=None, help="Existing PAT-L weights, or a destination for automatic download.")
    parser.add_argument("--output-dir", default=None, help="Defaults to runs/<outcome-column>_gpu.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=20260722)
    parser.add_argument("--start-day", type=int, default=None, help="Optional cyclic first day for the week (2 through 8).")
    parser.add_argument("--all-day-cycles", action="store_true", help="Augment only training participants with all seven day rotations.")
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the prediction head. By default, the pretrained encoder is fine-tuned too.",
    )
    parser.add_argument("--pad-one-missing-day", action="store_true", help="Use a zero-filled day for participants missing exactly one day.")
    parser.add_argument("--max-participants", type=int, default=None, help="Optional development subset; omit for the full cohort.")
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


def _read_outcome(rds_path: str | Path, outcome_column: str) -> pd.DataFrame:
    try:
        import pyreadr
    except ImportError as error:
        raise ImportError("Reading the RDS file requires pyreadr. Install it with `pip install pyreadr`.") from error
    result = pyreadr.read_r(str(rds_path))
    if not result:
        raise ValueError(f"No data frame found in {rds_path}.")
    covariates = next(iter(result.values()))
    if "SEQN" not in covariates.columns or outcome_column not in covariates.columns:
        raise ValueError(f"RDS file must contain SEQN and {outcome_column!r}; found {covariates.columns.tolist()}.")
    values = covariates.loc[:, ["SEQN", outcome_column]].dropna().drop_duplicates("SEQN")
    if values.empty:
        raise ValueError(f"No non-missing values found for {outcome_column!r}.")
    return values


def _resolve_task(y: pd.Series, requested_task: str) -> str:
    """Infer task only when the outcome type makes the choice unambiguous."""
    if requested_task != "auto":
        return requested_task
    unique = y.nunique()
    if unique == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(y):
        return "continuous"
    return "categorical"


def _align_outcome(X: np.ndarray, participant_ids: np.ndarray, outcome: pd.DataFrame, outcome_column: str):
    by_id = outcome.set_index("SEQN")[outcome_column]
    values = by_id.reindex(participant_ids)
    keep = values.notna().to_numpy()
    return X[keep], values.loc[keep].to_numpy(), participant_ids[keep]


def _subset(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray, maximum: int, random_state: int, task: str):
    if maximum < 6 or maximum > len(y):
        raise ValueError(f"max_participants must be between 6 and {len(y)}.")
    from sklearn.model_selection import train_test_split

    stratify = y if task in {"binary", "categorical"} else None
    indices, _ = train_test_split(np.arange(len(y)), train_size=maximum, random_state=random_state, stratify=stratify)
    return X[indices], y[indices], participant_ids[indices]


def _save_results(result, output_dir: str | Path | None, outcome_column: str) -> None:
    safe_outcome = re.sub(r"[^A-Za-z0-9_.-]+", "_", outcome_column)
    output = Path(output_dir) if output_dir else Path("runs") / f"{safe_outcome}_gpu"
    output.mkdir(parents=True, exist_ok=True)
    if result.task == "binary":
        metrics, predictions = _binary_results(result, outcome_column)
    elif result.task == "categorical":
        metrics, predictions = _categorical_results(result, outcome_column)
    else:
        metrics, predictions = _continuous_results(result, outcome_column)
    with (output / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    result.model.save_weights(output / "model.weights.h5")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics, predictions, and model weights to {output}.")


def _binary_results(result, outcome_column: str) -> tuple[dict, pd.DataFrame]:
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
    predictions = pd.DataFrame({f"true_{outcome_column}": true_labels, f"predicted_{outcome_column}": predicted_labels, f"probability_{labels[1]}": probabilities})
    return metrics, predictions


def _categorical_results(result, outcome_column: str) -> tuple[dict, pd.DataFrame]:
    probabilities = result.predictions
    true_codes = result.y_test.astype(int)
    labels = result.class_labels
    true_labels = labels[true_codes]
    predicted_labels = result.predict_classes(result.X_test)
    metrics = {
        **result.metrics,
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "f1_macro": float(f1_score(true_labels, predicted_labels, average="macro")),
        "log_loss": float(log_loss(true_codes, probabilities, labels=np.arange(len(labels)))),
        "n_test": int(len(true_codes)),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=labels).tolist(),
        "class_order": labels.tolist(),
    }
    predictions = pd.DataFrame({f"true_{outcome_column}": true_labels, f"predicted_{outcome_column}": predicted_labels})
    for index, label in enumerate(labels):
        predictions[f"probability_{label}"] = probabilities[:, index]
    return metrics, predictions


def _continuous_results(result, outcome_column: str) -> tuple[dict, pd.DataFrame]:
    actual, predicted = result.y_test, result.predictions
    metrics = {
        **result.metrics,
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "n_test": int(len(actual)),
    }
    return metrics, pd.DataFrame({f"true_{outcome_column}": actual, f"predicted_{outcome_column}": predicted})


if __name__ == "__main__":
    main()
