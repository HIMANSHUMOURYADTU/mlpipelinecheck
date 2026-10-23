"""Regression prediction audit from docs/quickstart.md."""

from __future__ import annotations

import pandas as pd

from mlpipelinecheck import audit_predictions


def main() -> None:
    """Audit positional regression predictions with a constant error of 0.5."""
    y_true = pd.Series([float(i) for i in range(1, 41)])
    y_pred = y_true + 0.5
    report = audit_predictions(y_true, y_pred, task="regression")
    mae = report.metadata["metrics"]["mae"]
    status = report.summary()["status"]
    print(mae)
    print(status)
    if mae != 0.5 or status != "no configured issues found":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
