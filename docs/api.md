# API

Import the public surface from `mlpipelinecheck`. Detector modules under `mlpipelinecheck.checks` are internal.

```python
from mlpipelinecheck import (
    SCHEMA_VERSION,
    AuditConfig,
    AuditReport,
    audit_dataset,
    audit_predictions,
    audit_split,
)
```

`SCHEMA_VERSION` is `"1.1.0"`. The package version is `mlpipelinecheck.__version__`, currently `0.2.0`.

## audit_dataset

```python
audit_dataset(
    df,
    target=None,
    task=None,
    id_columns=None,
    feature_columns=None,
    config=None,
) -> AuditReport
```

`df` must be a pandas DataFrame. Duplicate column names are an input error.

`target` is optional. When it is set, the column must exist. Task-specific checks run for classification only when `task` is `"classification"` or when `task` is `"auto"` and the target dtype is unambiguous. `"auto"` is the default. An ambiguous target skips those checks and records why. It is not treated as a pass of the classification checks.

When `feature_columns` is omitted, features are every column except the target and `id_columns`. An explicit list that includes the target produces `DATA.TARGET_IN_FEATURES`.

`config` may be an `AuditConfig` or a mapping of the same fields. Explicit arguments override the config. `None` means “keep the config value.”

The function does not fit a model, drop rows, or impute values. It does not copy the whole frame defensively; the operations it runs are read-only.

## audit_split

```python
audit_split(
    train_df,
    test_df,
    target=None,
    task=None,
    group_column=None,
    time_column=None,
    split_strategy=None,
    id_columns=None,
    feature_columns=None,
    config=None,
) -> AuditReport
```

Both arguments must be DataFrames. The pandas index is not an identity. Exact overlap uses a SHA-256 fingerprint of length-prefixed cell tokens, with column names sorted. See `docs/rules.md`.

`split_strategy="group"` without `group_column`, or `split_strategy="temporal"` without `time_column`, raises `MlpipelinecheckInputError`. Omitting the column and the strategy leaves that check `not_applicable`. That is not a pass.

This audit repeats a subset of dataset rules on the partitions: duplicate rows, target in features, exact target copies, missing target values, and single-class targets. It does not repeat the full dataset catalog.

## audit_predictions

```python
audit_predictions(y_true, y_pred, task, y_score=None, config=None) -> AuditReport
```

`task` must be `"classification"` or `"regression"`. `"auto"` is rejected.

`y_pred` holds predicted labels or regression values, not probabilities. Put scores in `y_score`.

Comparison is positional. Indexes are not aligned.

Classification uses scikit-learn for accuracy, balanced accuracy, precision, recall, and F1. Averaging is macro and weighted, with `zero_division=0`. The confusion matrix and per-class rates are omitted when there are more than 20 labels. ROC AUC is computed when scores have a documented shape and at least two classes are present. Log loss is computed only for probabilities in `[0, 1]` (two-dimensional rows must sum to 1 within `1e-3`). Brier score is binary only, and only for probabilities.

A one-dimensional binary score is the score for the greater label in sorted order. A matrix with shape `(n, 2)` uses that same order, so column 1 is the positive class. Multiclass scores must have shape `(n, n_classes)` in sorted label order.

Regression reports MAE, RMSE (square root of scikit-learn’s mean squared error), signed mean error `mean(y_pred - y_true)`, and R². R² is null when `y_true` is constant.

Wrong lengths, invalid score shapes, non-finite values, and empty inputs raise `MlpipelinecheckInputError`. A single observed class with a plausible score shape is the finding `METRIC.INVALID_AUC_INPUT`, not a shape error.

`groups` and `timestamps` declare dependence. Their values are counted and are not copied into the report. Pass `observation_dependence="grouped"` or `"temporal"` with the matching argument. Requesting `bootstrap_samples` for either of those assumptions raises `MlpipelinecheckInputError`. An iid interval is not substituted.

`bootstrap_samples` (minimum 20) with `observation_dependence="independent"` computes a 95% percentile interval for accuracy or MAE. Rows are resampled independently with replacement. Predictions stay fixed. `random_seed` controls that draw. A non-zero seed without `bootstrap_samples` raises `MlpipelinecheckConfigError`. `confidence_interval_max_width` flags `METRIC.WIDE_CONFIDENCE_INTERVAL` when the interval width is greater than the threshold. Without that threshold the interval is stored and the width is not judged.

## AuditReport

| Method | Behavior |
|---|---|
| `summary()` | Counts by severity, category, and check status, plus `status` and the disclaimer. |
| `to_dict()` | Canonical report dictionary. HTML is rendered from this object. |
| `to_json(path=None)` | Standards-compliant JSON (`allow_nan=False`). Non-finite floats become null before encoding. |
| `to_html(path)` | Standalone HTML file. No network, no JavaScript. Returns the path. |
| `raise_for_findings(min_severity="high", rule_ids=None)` | Raises `MlpipelinecheckFindingsError` when the gate matches. |
| `matching_findings(...)` | The findings that would trip the gate. |

`status` is `audit incomplete` if any check errored, otherwise `issues found` if there are findings, otherwise `no configured issues found`.

Findings are ordered by severity descending, then id ascending. Checks stay in catalog order. A finding id must be one of the check ids on that report.

JSON includes `schema_version`, `package_version`, `created_at`, `config`, aggregate `metadata`, `findings`, and `checks`. `created_at` is the UTC time when that report object was built. It is kept because it says when the audit ran. Two runs of the same data therefore differ in that field even when every check result is the same. Keys inside the JSON object are sorted. Parent directories are not created. A missing parent or a directory path raises `MlpipelinecheckInputError`.

## AuditConfig

Construct `AuditConfig(...)` or `AuditConfig.from_mapping(mapping)`. Unknown keys raise `MlpipelinecheckConfigError`. Booleans are not accepted as numbers. `privacy_include_examples=True` is rejected because example rows are not implemented.

Defaults:

| Field | Default | Meaning |
|---|---|---|
| `task` | `"auto"` | See the audit functions. Prediction audits still require an explicit task argument. |
| `missing_fraction_threshold` | `0.0` | Flag a column when its missing fraction is strictly greater. |
| `near_constant_fraction` | `0.99` | Near-constant features use `DATA.CONSTANT_FEATURE` at info severity. Exact constants stay low. |
| `high_cardinality_ratio` | `0.95` | Unique ratio at or above this, with the minimum unique count and row count. |
| `high_cardinality_min_unique` | `20` | |
| `high_cardinality_min_rows` | `20` | |
| `high_cardinality_include_floats` | `False` | Continuous floats are skipped unless this is true. |
| `imbalance_minority_fraction` | `0.1` | Flag when a minority class is at or below this. |
| `imbalance_majority_fraction` | `0.9` | Flag when a majority class is at or above this. |
| `class_shift_max_abs_diff` | `0.1` | Flag when the max absolute class-proportion difference is greater. |
| `class_shift_min_rows` | `30` | Skip the class-shift check below this many non-null targets on either side. |
| `numeric_shift_min_abs_smd` | `0.8` | Flag when the absolute pooled standardized mean difference is at least this. |
| `numeric_shift_min_rows` | `30` | Minimum finite values on each side. |
| `small_test_rows` | `30` | Flag when the test partition has fewer rows. |
| `small_test_fraction` | `None` | Optional fraction gate. Off by default. |
| `small_evaluation_rows` | `30` | Flag a prediction audit below this length. |
| `max_rows` | `None` | When set, duplicate and target-copy fingerprint checks are skipped above this size. That skip is `not_applicable`, not a pass. |
| `disabled_checks` | `[]` | Known rule ids only. |
| `min_severity` | `"high"` | Default gate used by the CLI. The Python methods take the threshold as an argument. |
| `random_seed` | `0` | Used only when `bootstrap_samples` is set. A non-zero value without a bootstrap request is rejected. |
| `bootstrap_samples` | `None` | IID resample count for prediction audits. `None` computes no interval. |
| `observation_dependence` | `"independent"` | `independent`, `grouped`, or `temporal`. |
| `confidence_interval_max_width` | `None` | Optional width gate. Requires `bootstrap_samples`. |

## Exceptions

| Exception | When |
|---|---|
| `MlpipelinecheckError` | Base type. |
| `MlpipelinecheckInputError` | Bad frames, columns, lengths, or score shapes. CLI exit 2. |
| `MlpipelinecheckConfigError` | Bad configuration. CLI exit 2. |
| `MlpipelinecheckFindingsError` | `raise_for_findings` matched. The message lists ids and severities, not evidence values. |

## Severity and confidence

Severity order is `critical` > `high` > `medium` > `low` > `info`.

Confidence is `high`, `medium`, or `low`. It describes how direct the evidence is. It is not a probability that leakage occurred.

Check status is `detected`, `not_detected`, `not_applicable`, or `error`. `not_detected` means that check’s condition was absent. `not_applicable` means the check did not run. An errored check means the audit could not decide. None of these mean “all leakage has been ruled out.”
