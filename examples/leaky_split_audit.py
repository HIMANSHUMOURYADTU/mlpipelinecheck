"""Intentionally overlapping train and test rows."""

from __future__ import annotations

import pandas as pd

from mlpipelinecheck import audit_split


def _frame(start: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"c{index:03d}" for index in range(start, start + 40)],
            "age": [20 + (index % 5) for index in range(40)],
            "churn": [index % 2 for index in range(40)],
        }
    )


def main() -> None:
    """Copy five training rows into test, including the customer id."""
    train = _frame(0)
    test = _frame(100)
    for column in train.columns:
        test.loc[test.index[:5], column] = train.loc[train.index[:5], column].to_numpy()
    report = audit_split(
        train,
        test,
        target="churn",
        task="classification",
        group_column="customer_id",
        id_columns=["customer_id"],
    )
    ids = [finding.id for finding in report.findings]
    print(report.summary()["status"])
    print(ids)
    if "SPLIT.EXACT_ROW_OVERLAP" not in ids or "SPLIT.GROUP_OVERLAP" not in ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
