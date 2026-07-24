"""Dataset loaders for accelerometry data usable with :func:`fine_tune_pat`."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


NHANES_AC_URL = (
    "https://physionet.org/files/minute-level-step-count-nhanes/"
    "1.0.1/csv/nhanes_1440_AC.csv.xz"
)
NHANES_MINUTES_PER_DAY = 1440
NHANES_COMPLETE_DAYS = tuple(range(2, 9))


def load_nhanes_weekly_accelerometry(
    data_path: str | Path = "nhanes_1440_AC.csv.xz",
    *,
    url: str = NHANES_AC_URL,
    fillna: float = 0.0,
    chunksize: int = 5_000,
    start_day: int | None = None,
    day_order: tuple[int, ...] | None = None,
    verbose: int = 0,
    max_participants: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load complete NHANES days 2--8 into weekly accelerometry records.

    The source is PhysioNet's minute-level NHANES activity CSV. The function
    uses a local compressed CSV when available and downloads it only when
    ``data_path`` does not exist. It retains participants with exactly one
    observation for each of days 2 through 8. Days 1 and 9 are excluded
    because they may be incomplete. Seven daily 1,440-minute records are then
    concatenated into one 10,080-minute row per participant.

    Parameters
    ----------
    data_path
        Local path for the compressed ``.csv.xz`` source file. This is also
        the destination when a download is necessary.
    url
        PhysioNet source URL used only when ``data_path`` is absent.
    fillna
        Value used for missing minute-level activity values. The default is
        ``0.0``.
    chunksize
        Number of source rows to parse at a time. Chunked reading limits peak
        memory use for the large compressed CSV. Must be positive.
    start_day
        Optional first day for a cyclic ordering of the seven retained days.
        For example, ``start_day=8`` returns days in the order
        ``(8, 2, 3, 4, 5, 6, 7)``. Cannot be combined with ``day_order``.
    day_order
        Optional explicit permutation of ``(2, 3, 4, 5, 6, 7, 8)``. This
        permits non-cyclic orders, such as ``(8, 7, 2, 3, 4, 5, 6)``. Cannot
        be combined with ``start_day``.
    verbose
        Set to 1 to report download/cache and preparation milestones; set to 2
        to additionally report periodic source-row progress.
    max_participants
        Optional maximum number of complete participants to load. When set,
        parsing stops after the first complete participants in the PhysioNet
        file have been collected. This is useful for quick Colab experiments;
        use ``None`` (the default) for the full cohort.

    Returns
    -------
    X
        ``float32`` array with shape ``(n_participants, 10080)``. Each row is
        one ordered seven-day activity record.
    participant_ids
        One-dimensional ``SEQN`` array aligned row-for-row with ``X``. Join an
        outcome table to these IDs before passing its aligned outcome vector
        to :func:`pypat.finetune.fine_tune_pat`.

    Examples
    --------
    Load default day order and train after aligning an outcome::

        X, participant_ids = load_nhanes_weekly_accelerometry()
        y = outcomes.set_index("SEQN").loc[participant_ids, "outcome"].to_numpy()
        result = fine_tune_pat(X, y)

    Load a cyclically shifted week to reduce first-day-position effects::

        X, participant_ids = load_nhanes_weekly_accelerometry(start_day=8)
    """
    path = Path(data_path)
    if not path.exists():
        if verbose:
            print(f"Downloading NHANES accelerometry data to {path}...")
        _download(url, path)
    elif verbose:
        print(f"Using cached NHANES accelerometry data: {path}")
    selected_day_order = _resolve_day_order(start_day, day_order)
    if verbose:
        print(f"Preparing complete days in order: {selected_day_order}")

    minute_columns = [f"min_{minute:04d}" for minute in range(1, NHANES_MINUTES_PER_DAY + 1)]
    required_columns = ["SEQN", "PAXDAYM", *minute_columns]
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")
    if max_participants is not None and max_participants < 1:
        raise ValueError("max_participants must be positive when provided.")
    week_chunks = []
    for chunk_number, chunk in enumerate(
        pd.read_csv(path, compression="xz", usecols=required_columns, chunksize=chunksize), start=1
    ):
        week_chunk = chunk.loc[chunk["PAXDAYM"].isin(NHANES_COMPLETE_DAYS), required_columns]
        if not week_chunk.empty:
            week_chunks.append(week_chunk)
        if verbose > 1 and chunk_number % 25 == 0:
            print(f"Read {chunk_number * chunksize:,} source rows...")
        if max_participants is not None:
            candidates = pd.concat(week_chunks, ignore_index=True)
            candidate_counts = candidates.groupby("SEQN")["PAXDAYM"].agg(["count", "nunique"])
            complete_count = ((candidate_counts["count"] == 7) & (candidate_counts["nunique"] == 7)).sum()
            if complete_count >= max_participants:
                if verbose:
                    print(f"Collected {complete_count:,} complete participants; stopping early.")
                break
    if not week_chunks:
        raise ValueError("NHANES data contains no records for days 2 through 8.")
    week_data = pd.concat(week_chunks, ignore_index=True).sort_values(["SEQN", "PAXDAYM"])
    day_counts = week_data.groupby("SEQN")["PAXDAYM"].agg(["count", "nunique"])
    complete_ids = day_counts.index[(day_counts["count"] == 7) & (day_counts["nunique"] == 7)]
    if max_participants is not None:
        complete_ids = complete_ids[:max_participants]
    week_data = week_data.loc[week_data["SEQN"].isin(complete_ids)].copy()
    week_data["PAXDAYM"] = pd.Categorical(
        week_data["PAXDAYM"], categories=selected_day_order, ordered=True
    )
    week_data = week_data.sort_values(["SEQN", "PAXDAYM"])
    participant_ids = week_data["SEQN"].drop_duplicates().to_numpy()
    # fill in the missing data with 0
    minutes = week_data.loc[:, minute_columns].fillna(fillna).to_numpy(dtype=np.float32)
    X = minutes.reshape(len(participant_ids), 7 * NHANES_MINUTES_PER_DAY)
    if verbose:
        print(f"Kept {len(participant_ids):,} participants with all seven complete days.")
    return X, participant_ids


def rotate_nhanes_weekly_accelerometry(
    X: np.ndarray,
    *,
    start_day: int,
    current_day_order: tuple[int, ...] = NHANES_COMPLETE_DAYS,
) -> np.ndarray:
    """Cyclically rotate already-loaded weekly NHANES data by whole days.

    ``X`` must contain seven consecutive 1,440-minute blocks. With the
    default input order ``(2, 3, ..., 8)``, ``start_day=8`` returns data in
    the order ``(8, 2, 3, ..., 7)``. The original array is not modified.

    Set ``current_day_order`` when ``X`` was previously loaded with a custom
    ``day_order``.
    """
    values = np.asarray(X)
    expected_length = len(NHANES_COMPLETE_DAYS) * NHANES_MINUTES_PER_DAY
    if values.ndim != 2 or values.shape[1] != expected_length:
        raise ValueError(f"X must have shape (participants, {expected_length}).")
    if len(current_day_order) != len(NHANES_COMPLETE_DAYS) or set(current_day_order) != set(NHANES_COMPLETE_DAYS):
        raise ValueError(f"current_day_order must contain each of {NHANES_COMPLETE_DAYS} exactly once.")
    if start_day not in current_day_order:
        raise ValueError(f"start_day must be one of {current_day_order}.")
    first_block = current_day_order.index(start_day)
    return np.concatenate(
        [
            values[:, first_block * NHANES_MINUTES_PER_DAY :],
            values[:, : first_block * NHANES_MINUTES_PER_DAY],
        ],
        axis=1,
    )


def augment_all_weekly_cycles(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create all seven whole-day cyclic rotations of each training record.

    The result contains the original order followed by rotations beginning at
    each successive day block. Outcomes are repeated in the matching order.
    This is intended for training data only; do not augment validation or test
    data, because their rotated copies are not independent observations.
    """
    values = np.asarray(X)
    outcomes = np.asarray(y)
    expected_length = len(NHANES_COMPLETE_DAYS) * NHANES_MINUTES_PER_DAY
    if values.ndim != 2 or values.shape[1] != expected_length:
        raise ValueError(f"X must have shape (participants, {expected_length}).")
    if outcomes.ndim not in {1, 2} or len(outcomes) != len(values):
        raise ValueError("y must have one row per row of X.")
    rotations = [
        np.concatenate([values[:, offset:], values[:, :offset]], axis=1)
        for offset in range(0, expected_length, NHANES_MINUTES_PER_DAY)
    ]
    outcome_repetitions = (len(rotations),) + (1,) * (outcomes.ndim - 1)
    return np.concatenate(rotations, axis=0), np.tile(outcomes, outcome_repetitions)


def _resolve_day_order(start_day: int | None, day_order: tuple[int, ...] | None) -> tuple[int, ...]:
    """Validate and resolve the requested order for the seven complete days."""
    if start_day is not None and day_order is not None:
        raise ValueError("Specify start_day or day_order, not both.")
    if day_order is not None:
        if len(day_order) != len(NHANES_COMPLETE_DAYS) or set(day_order) != set(NHANES_COMPLETE_DAYS):
            raise ValueError(f"day_order must contain each of {NHANES_COMPLETE_DAYS} exactly once.")
        return day_order
    if start_day is None:
        return NHANES_COMPLETE_DAYS
    if start_day not in NHANES_COMPLETE_DAYS:
        raise ValueError(f"start_day must be one of {NHANES_COMPLETE_DAYS}.")
    start_index = NHANES_COMPLETE_DAYS.index(start_day)
    return NHANES_COMPLETE_DAYS[start_index:] + NHANES_COMPLETE_DAYS[:start_index]


def _download(url: str, path: Path) -> None:
    """Download a file atomically to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".part")
    try:
        with urlopen(url, timeout=60) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
