"""Download and prepare NHANES accelerometry data for a PAT outcome model."""

from __future__ import annotations

import argparse

from pypat.datasets import load_nhanes_weekly_accelerometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="nhanes_1440_AC.csv.xz", help="Local CSV cache; download if absent.")
    parser.add_argument("--start-day", type=int, help="First output day; performs a cyclic rotation of days 2--8.")
    parser.add_argument(
        "--day-order",
        help="Explicit comma-separated ordering of days 2--8, e.g. 8,7,2,3,4,5,6.",
    )
    arguments = parser.parse_args()
    day_order = tuple(int(day) for day in arguments.day_order.split(",")) if arguments.day_order else None
    X, participant_ids = load_nhanes_weekly_accelerometry(
        arguments.data_path,
        start_day=arguments.start_day,
        day_order=day_order,
    )
    print(f"Prepared X with shape {X.shape} for {len(participant_ids)} participants.")
    print("participant_ids is aligned row-for-row with X.")
    print("After joining your outcome to participant_ids, run: result = fine_tune_pat(X, y)")


if __name__ == "__main__":
    main()
