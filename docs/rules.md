# Rules

Every finding id below is stable in Mlpipelinecheck 0.2. Changing an id is a breaking change. The wording in `src/mlpipelinecheck/rules.py` is what the report uses. This page is the longer explanation.

Severity is the catalog default unless a check documents a bump. Confidence says how direct the evidence is. Neither value is a probability of leakage.

A check that does not run is `not_applicable`. A check that ran and found nothing is `not_detected`. A check that could not decide is `error`.

## DATA.EMPTY_DATASET

Severity `high`, confidence `high`.

Fires when the table has no rows, or when no feature column remains after the target and declared identifiers are removed.

This is a structural fact about the input. A non-empty table can still be unfit for other reasons.

## DATA.MISSING_VALUES

Severity `medium`, confidence `high`.

Fires when a column’s missing fraction is strictly greater than `missing_fraction_threshold` (default `0.0`). The denominator is the row count. Mlpipelinecheck does not drop or impute.

Missingness is a data-quality signal. It is not proof of leakage. All-null columns are reported here, not as constants.

## DATA.DUPLICATE_ROWS

Severity `medium`, confidence `high`.

Fires when two or more rows share a fingerprint. The fingerprint hashes cell values with column names sorted and ignores the pandas index. Missing sentinels share one token. The text `nan` is not missing. Lists and dicts use canonical JSON. Sets and arbitrary objects make the check error instead of passing.

Near-duplicates are not detected. Legitimate repeated events can share a fingerprint. In a split audit, this id covers duplicates inside a partition. Cross-partition copies are `SPLIT.EXACT_ROW_OVERLAP`.

When `max_rows` is set and the frame is larger, this check is skipped.

## DATA.CONSTANT_FEATURE

Exact one-value columns: severity `low`, confidence `high`.

Near-constant columns, where one non-null value covers at least `near_constant_fraction` (default `0.99`): severity `info`, confidence `medium`.

All-null columns are missing values, not constants. A constant feature is a structural fact. The near-constant branch is a frequency heuristic and can be benign. Neither is proof of leakage.

## DATA.HIGH_CARDINALITY_FEATURE

Severity `low`, confidence `medium`.

Fires when a non-float feature has at least `high_cardinality_min_rows` rows (default 20), at least `high_cardinality_min_unique` distinct values (default 20), and a unique ratio at least `high_cardinality_ratio` (default 0.95). Declared `id_columns` are excluded. Float columns are excluded unless `high_cardinality_include_floats` is true.

This is a heuristic that an undeclared identifier might be present. It does not prove the column is an identifier and it does not prove leakage.

## DATA.TARGET_IN_FEATURES

Severity `critical`, confidence `high`.

Fires when an explicit feature list contains the target column. The default feature set already excludes the target, so this finding means the caller put it back. The check compares column identity, not aliases. A differently named exact copy is `DATA.TARGET_COPY_FEATURE`.

Duplicate column labels never reach this check; they raise an input error.

## DATA.TARGET_COPY_FEATURE

Severity `high`, confidence `high`.

Fires when a resolved feature compares equal to the target under the documented rules. Declared identifiers and columns outside `feature_columns` are not compared. Numeric and boolean columns are compared as floats, so `1`, `1.0`, and `True` match. Text is not coerced, so `"1"` does not match `1`. Missing matches missing. The target column is not compared with itself.

If the values cannot be compared safely, the check errors. It does not pass. Correlations and near copies are not reported. When `max_rows` skips the check, the status is `not_applicable`.

## DATA.CLASS_IMBALANCE

Catalog severity `medium`, confidence `high`.

Fires for a classification target when a minority class fraction is at or below `imbalance_minority_fraction` (default 0.1) or a majority class fraction is at or above `imbalance_majority_fraction` (default 0.9).

A single observed class is raised to severity `high`. The split audit uses that high severity only for a single-class partition. Milder imbalance stays at medium, below the default CLI gate.

Imbalance is not a model failure and is not proof of leakage.

## DATA.NON_FINITE_VALUES

Severity `medium`, confidence `high`.

Fires when a numeric column contains `inf` or `-inf`. Missing values are counted separately. Means in the report exclude non-finite values. This is data quality, not leakage. This id is an addition to the original catalog table; see `docs/implementation_notes.md`.

## SPLIT.SCHEMA_MISMATCH

Severity `high`, confidence `high`.

Fires when train and test disagree on column names or dtype groups. Column order is ignored. Integer and float dtypes are compatible. Booleans are their own group. Object, string, and categorical dtypes are compatible with each other.

Exact row overlap is not claimed when the schema differs.

## SPLIT.EXACT_ROW_OVERLAP

Severity `high`, confidence `high`.

Fires when a test fingerprint equals a train fingerprint. The hash is SHA-256 over length-prefixed UTF-8 tokens. The pandas index is ignored. Integer-valued floats match integers. Text does not match numbers. All missing sentinels match each other.

The check runs only when the schemas match. Rows that differ only by a column you left in the table are different records. Near-duplicates are not detected. Fingerprints are not written to the report; counts are.

## SPLIT.GROUP_OVERLAP

Severity `high`, confidence `high`.

Fires when a non-missing group token appears in both partitions. Only the supplied `group_column` is used. Missing groups are ignored and do not form a shared group. If every group value is missing, the check errors rather than passing.

Group values are counted, not copied into the report. Numeric `1` and `1.0` match. The text `"1"` does not match the number `1`.

If the column is omitted, the check is `not_applicable`. `split_strategy="group"` without the column is an input error.

## SPLIT.TIME_ORDER_VIOLATION

Severity `high`, confidence `high`.

Runs only when `time_column` is set, or when a temporal strategy requires it. The default policy is `test_not_before_train_max`: the cutoff is the maximum non-null training timestamp, and a test timestamp strictly earlier than that cutoff is a violation. Ties are not flagged.

Unparseable non-null values are an input error. Numeric epochs are an input error, including numbers stored in an object column. Convert them to datetimes before calling the audit. Mixed timezone-aware and naive timestamps are an input error. Aware timestamps are converted to UTC. If no timestamp can be used, the check errors.

A violation of this policy is evidence about order. It is not a claim about every other form of leakage. Omitting the time column leaves the check `not_applicable`.

## SPLIT.CLASS_DISTRIBUTION_SHIFT

Severity `low`, confidence `medium`.

Fires when the maximum absolute difference in class proportions is greater than `class_shift_max_abs_diff` (default 0.1). The evidence also includes total variation as a summary. Either partition with fewer than `class_shift_min_rows` non-null targets (default 30) skips the check.

A proportion difference is not proof of leakage or of a bad split. Small samples are noisy, which is why the minimum exists.

## SPLIT.NUMERIC_DISTRIBUTION_SHIFT

Severity `low`, confidence `medium`.

Fires when any compared numeric feature has an absolute pooled standardized mean difference at least `numeric_shift_min_abs_smd` (default 0.8). The standard deviation uses `ddof=1`. Columns with fewer than `numeric_shift_min_rows` finite values (default 30), or with undefined pooled variance, are not compared. If nothing is comparable, the check is `not_applicable`.

The target, identifier columns, group column, and time column are excluded. Means ignore missing and non-finite values.

This is a descriptive location shift. It is not a hypothesis test and it is not proof of leakage. Comparing many columns increases the chance that one of them crosses the threshold.

## SPLIT.SMALL_TEST_PARTITION

Severity `low`, confidence `high`.

Fires when the test partition has fewer than `small_test_rows` rows (default 30). An optional `small_test_fraction` is off unless you set it. The threshold is configuration, not a judgment that every small experiment is invalid. This id is an addition to the original catalog table.

## METRIC.ACCURACY_ONLY_IMBALANCED

Catalog severity `medium`, confidence `high`.

Fires when the evaluation labels are imbalanced under the same fractions as `DATA.CLASS_IMBALANCE`, or when only one class is present. A single class is raised to severity `high`. The finding says accuracy is easy to misread. It does not say the model is poor.

## METRIC.INVALID_AUC_INPUT

Severity `high`, confidence `high`.

Fires when `y_score` was supplied but ROC AUC cannot be computed because only one class is present. Wrong shapes, non-numeric scores, and non-finite scores raise `MlpipelinecheckInputError` instead. Log loss and Brier score are left null when the values are not probabilities. No substitute number is invented.

## METRIC.SMALL_EVALUATION_SAMPLE

Severity `low`, confidence `high`.

Fires when the prediction array is shorter than `small_evaluation_rows` (default 30). The finding is about the sample size. An interval is a separate opt-in check.

## METRIC.WIDE_CONFIDENCE_INTERVAL

Catalog severity `low`, confidence `medium`.

When `bootstrap_samples` is set on an independent prediction audit, the report stores a 95% percentile interval for accuracy or MAE. The check fires only when `confidence_interval_max_width` is set and the width is greater than that value. Without a bootstrap request, or without a width threshold, the check is `not_applicable`. Grouped and temporal bootstrap requests raise an input error. See `docs/implementation_notes.md`.

## Reserved identifiers

These ids are reserved for later heuristic detectors. They are not in the catalog, they do not run, and a report that omits them has not performed those checks.

| Reserved id | Intended meaning |
|---|---|
| `DATA.NEAR_DUPLICATE_ROWS` | Rows that are similar without being exact fingerprint matches. |
| `DATA.TARGET_DERIVED_FEATURE` | A feature that is an affine or noisy function of the target, rather than an exact copy. |

An exact copy stays on `DATA.TARGET_COPY_FEATURE`. An exact repeated row stays on `DATA.DUPLICATE_ROWS`. A strong legitimate correlation is not either of those rules.
