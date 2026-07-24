"""Download and prepare NHANES accelerometry data for a PAT outcome model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow this script to run directly from a source checkout without first doing
# an editable install. Insert the local package ahead of any older pypat copy
# that may already be installed in the active environment.
SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from pypat.datasets import load_nhanes_weekly_accelerometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="nhanes_1440_AC.csv.xz", help="Local CSV cache; download if absent.")
    parser.add_argument("--start-day", type=int, help="First output day; performs a cyclic rotation of days 2--8.")
    parser.add_argument(
        "--day-order",
        help="Explicit comma-separated ordering of days 2--8, e.g. 8,7,2,3,4,5,6.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Show preparation progress; repeat (-vv) for periodic CSV chunk updates.",
    )
    parser.add_argument(
        "--pad-one-missing-day",
        action="store_true",
        help="Accept exactly six days and insert a zero-filled block for the missing day.",
    )
    arguments = parser.parse_args()
    day_order = tuple(int(day) for day in arguments.day_order.split(",")) if arguments.day_order else None
    X, participant_ids = load_nhanes_weekly_accelerometry(
        arguments.data_path,
        start_day=arguments.start_day,
        day_order=day_order,
        verbose=arguments.verbose,
        pad_one_missing_day=arguments.pad_one_missing_day,
    )
    print(f"Prepared X with shape {X.shape} for {len(participant_ids)} participants.")
    print("participant_ids is aligned row-for-row with X.")
    print("After joining your outcome to participant_ids, run: result = fine_tune_pat(X, y)")


if __name__ == "__main__":
    main()
