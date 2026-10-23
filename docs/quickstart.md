# Quick start

Mlpipelinecheck 0.2 audits tabular data, an explicit split, or predictions you already have. It does not fit a model.

Install this package from a checkout. Its name is `mlpipelinecheck`. `pip install evalguard` is a different project. This package is not published. See `docs/publishing.md`.

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 or newer is required.

## Classification dataset

```python
import pandas as pd
from mlpipelinecheck import audit_dataset

df = pd.DataFrame(
    {
        "age": [25, 40, 33, 52],
        "plan": ["a", "b", "a", "b"],
        "churn": [0, 1, 0, 1],
    }
)
report = audit_dataset(df, target="churn", task="classification")
print(report.summary()["status"])
report.to_json("dataset_audit.json")
report.to_html("dataset_audit.html")
```

`no configured issues found` means the checks that ran did not fire. It does not mean leakage is impossible. The same sentence is in every report.

The script `examples/classification_audit.py` is this example. It also writes a JSON report and an HTML report, checks that both contain the same status, and deletes the files when it exits. `docs/cli.md` is the command reference. The leaky split is `examples/leaky_split_audit.py`.

## Regression predictions

```python
import pandas as pd
from mlpipelinecheck import audit_predictions

y_true = pd.Series([float(i) for i in range(1, 41)])
y_pred = y_true + 0.5
report = audit_predictions(y_true, y_pred, task="regression")
print(report.metadata["metrics"]["mae"])
print(report.summary()["status"])
```

Rows are compared by position. A Series index is not used to line the two columns up. `examples/regression_audit.py` runs this example.

## Group-aware split

Pass the entity column yourself. Mlpipelinecheck does not guess it.

```python
import pandas as pd
from mlpipelinecheck import audit_split

train = pd.read_csv("examples/data/train.csv")
test = pd.read_csv("examples/data/test.csv")
report = audit_split(
    train,
    test,
    target="churn",
    task="classification",
    group_column="customer_id",
    id_columns=["customer_id"],
)
print(report.summary()["status"])
```

`examples/group_split_audit.py` loads those files. `customer_id` is also declared as an identifier so the numeric-shift check does not treat the id as a feature.

## Temporal split

Supply a timestamp column. Numeric epochs are rejected; use real datetimes or parseable strings.

```python
import pandas as pd
from mlpipelinecheck import audit_split

train = pd.read_csv("examples/data/train.csv", parse_dates=["event_date"])
test = pd.read_csv("examples/data/test.csv", parse_dates=["event_date"])
report = audit_split(
    train,
    test,
    target="churn",
    task="classification",
    time_column="event_date",
    split_strategy="temporal",
    id_columns=["customer_id"],
)
print([finding.id for finding in report.findings])
```

The example files put every test date after every training date, so `SPLIT.TIME_ORDER_VIOLATION` is not expected. `examples/temporal_split_audit.py` checks that.

## Defective split

`examples/leaky_split_audit.py` copies five training rows into the test frame, including `customer_id`. The audit reports `SPLIT.EXACT_ROW_OVERLAP` and `SPLIT.GROUP_OVERLAP`.

## Command line

From the repository root:

```bash
mlpipelinecheck audit-dataset examples/data/classification.csv --target churn --task classification --html dataset-report.html --json dataset-report.json
mlpipelinecheck audit-split examples/data/train.csv examples/data/test.csv --target churn --task classification --group-column customer_id --id-column customer_id --html split-report.html
mlpipelinecheck audit-predictions examples/data/predictions.csv --target true --prediction-column pred --task regression --html predictions-report.html
```

`python -m mlpipelinecheck` is the same entry point. `--help` lists commands and exit codes.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | The configured gate did not match. |
| 1 | At least one finding met the gate. |
| 2 | Invalid input, invalid configuration, or a check that ended in error. |
| 3 | An unexpected internal error. |

The default gate is severity `high`. Low, medium, and info findings are written to the report and do not change the exit code unless you lower `--min-severity`.

A target copy fails the default gate:

```bash
mlpipelinecheck audit-dataset examples/data/leak_target_copy.csv --target churn --task classification --min-severity high
```

That command exits `1`. Reports, when requested, are written before the process exits.
