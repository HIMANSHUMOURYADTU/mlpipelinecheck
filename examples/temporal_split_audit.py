"""Temporal split audit using examples/data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlpipelinecheck import audit_split

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    """Audit a split whose test dates are all after the training dates."""
    train = pd.read_csv(DATA / "train.csv", parse_dates=["event_date"])
    test = pd.read_csv(DATA / "test.csv", parse_dates=["event_date"])
    report = audit_split(
        train,
        test,
        target="churn",
        task="classification",
        time_column="event_date",
        split_strategy="temporal",
        id_columns=["customer_id"],
    )
    ids = [finding.id for finding in report.findings]
    print(report.summary()["status"])
    print(ids)
    if (
        "SPLIT.TIME_ORDER_VIOLATION" in ids
        or report.summary()["status"] != "no configured issues found"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
