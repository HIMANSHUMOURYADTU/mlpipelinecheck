"""Write the CSV fixtures used by the docs and the CLI tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def _features(start: int, n: int, id_prefix: str, date_start: str) -> pd.DataFrame:
    dates = pd.date_range(date_start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "customer_id": [f"{id_prefix}{index:03d}" for index in range(start, start + n)],
            "age": [20 + (index % 8) for index in range(n)],
            "region": [["north", "south", "east", "west"][index % 4] for index in range(n)],
            "visits": [(index % 5) + 1 for index in range(n)],
            "plan": ["a" if index % 2 == 0 else "b" for index in range(n)],
            "churn": [index % 2 for index in range(n)],
            "event_date": [date.strftime("%Y-%m-%d") for date in dates],
        }
    )


def main() -> None:
    """Create classification, leak, split, and prediction fixtures."""
    DATA.mkdir(parents=True, exist_ok=True)
    dataset = _features(0, 40, "C", "2024-01-01").drop(columns=["customer_id", "event_date"])
    dataset.to_csv(DATA / "classification.csv", index=False)
    leaked = dataset.copy()
    leaked["churn_copy"] = leaked["churn"]
    leaked.to_csv(DATA / "leak_target_copy.csv", index=False)
    _features(0, 40, "C", "2024-01-01").to_csv(DATA / "train.csv", index=False)
    _features(100, 40, "T", "2024-03-01").to_csv(DATA / "test.csv", index=False)
    predictions = pd.DataFrame(
        {
            "true": [float(index) for index in range(1, 41)],
            "pred": [float(index) + 0.5 for index in range(1, 41)],
        }
    )
    predictions.to_csv(DATA / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
