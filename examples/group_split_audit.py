"""Group-aware split audit using examples/data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlpipelinecheck import audit_split

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    """Audit disjoint customer ids on the checked-in train and test files."""
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    report = audit_split(
        train,
        test,
        target="churn",
        task="classification",
        group_column="customer_id",
        id_columns=["customer_id"],
    )
    status = report.summary()["status"]
    print(status)
    if status != "no configured issues found":
        print([finding.id for finding in report.findings])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
