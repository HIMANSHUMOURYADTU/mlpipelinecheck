# Mlpipelinecheck synthetic benchmark

Package 0.2.0. Seed 42.

Labeled synthetic fixtures constructed for this repository. Heuristic thresholds match the detector defaults, so the score is a regression check on this list, not a field accuracy estimate. A precision or recall of 1.000 often comes from a single labeled fixture. DATA.CONSTANT_FEATURE is grouped with the heuristics because that id also covers near-constant columns; an exact one-value column remains a high-confidence finding.

## Case results

| Case | Passed | Expected | Actual |
|---|---|---|---|
| known_good_dataset | True | none | none |
| exact_duplicate_rows | True | DATA.DUPLICATE_ROWS | DATA.DUPLICATE_ROWS |
| exact_target_copy | True | DATA.TARGET_COPY_FEATURE | DATA.TARGET_COPY_FEATURE |
| target_included_as_feature | True | DATA.TARGET_IN_FEATURES | DATA.TARGET_IN_FEATURES |
| missing_values | True | DATA.MISSING_VALUES | DATA.MISSING_VALUES |
| class_imbalance | True | DATA.CLASS_IMBALANCE | DATA.CLASS_IMBALANCE |
| one_class_target | True | DATA.CLASS_IMBALANCE | DATA.CLASS_IMBALANCE |
| non_finite_value | True | DATA.NON_FINITE_VALUES | DATA.NON_FINITE_VALUES |
| empty_dataset | True | DATA.EMPTY_DATASET | DATA.EMPTY_DATASET |
| constant_feature | True | DATA.CONSTANT_FEATURE | DATA.CONSTANT_FEATURE |
| benign_high_cardinality_float_and_declared_id | True | none | none |
| undeclared_high_cardinality_code | True | DATA.HIGH_CARDINALITY_FEATURE | DATA.HIGH_CARDINALITY_FEATURE |
| known_good_split | True | none | none |
| exact_duplicate_train_test_records | True | SPLIT.EXACT_ROW_OVERLAP, SPLIT.GROUP_OVERLAP | SPLIT.EXACT_ROW_OVERLAP, SPLIT.GROUP_OVERLAP |
| overlapping_group_ids | True | SPLIT.GROUP_OVERLAP | SPLIT.GROUP_OVERLAP |
| temporal_split_order_violation | True | SPLIT.TIME_ORDER_VIOLATION | SPLIT.TIME_ORDER_VIOLATION |
| benign_temporal_order | True | none | none |
| text_does_not_equal_number | True | none | none |
| schema_mismatch | True | SPLIT.SCHEMA_MISMATCH | SPLIT.SCHEMA_MISMATCH |
| small_test_partition | True | SPLIT.SMALL_TEST_PARTITION | SPLIT.SMALL_TEST_PARTITION |
| legitimate_train_test_distribution_change | True | SPLIT.NUMERIC_DISTRIBUTION_SHIFT | SPLIT.NUMERIC_DISTRIBUTION_SHIFT |
| benign_mild_numeric_shift | True | none | none |
| class_proportion_difference | True | SPLIT.CLASS_DISTRIBUTION_SHIFT | SPLIT.CLASS_DISTRIBUTION_SHIFT |
| known_good_predictions | True | none | none |
| one_class_evaluation | True | METRIC.ACCURACY_ONLY_IMBALANCED, METRIC.INVALID_AUC_INPUT | METRIC.ACCURACY_ONLY_IMBALANCED, METRIC.INVALID_AUC_INPUT |
| small_evaluation_sample | True | METRIC.SMALL_EVALUATION_SAMPLE | METRIC.SMALL_EVALUATION_SAMPLE |

## Per-rule counts on this fixture list

| Rule | Kind | TP | FP | FN | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| DATA.CLASS_IMBALANCE | structural | 2 | 0 | 0 | 1.000 | 1.000 |
| DATA.CONSTANT_FEATURE | heuristic | 1 | 0 | 0 | 1.000 | 1.000 |
| DATA.DUPLICATE_ROWS | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| DATA.EMPTY_DATASET | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| DATA.HIGH_CARDINALITY_FEATURE | heuristic | 1 | 0 | 0 | 1.000 | 1.000 |
| DATA.MISSING_VALUES | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| DATA.NON_FINITE_VALUES | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| DATA.TARGET_COPY_FEATURE | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| DATA.TARGET_IN_FEATURES | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| METRIC.ACCURACY_ONLY_IMBALANCED | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| METRIC.INVALID_AUC_INPUT | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| METRIC.SMALL_EVALUATION_SAMPLE | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| SPLIT.CLASS_DISTRIBUTION_SHIFT | heuristic | 1 | 0 | 0 | 1.000 | 1.000 |
| SPLIT.EXACT_ROW_OVERLAP | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| SPLIT.GROUP_OVERLAP | structural | 2 | 0 | 0 | 1.000 | 1.000 |
| SPLIT.NUMERIC_DISTRIBUTION_SHIFT | heuristic | 1 | 0 | 0 | 1.000 | 1.000 |
| SPLIT.SCHEMA_MISMATCH | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| SPLIT.SMALL_TEST_PARTITION | structural | 1 | 0 | 0 | 1.000 | 1.000 |
| SPLIT.TIME_ORDER_VIOLATION | structural | 1 | 0 | 0 | 1.000 | 1.000 |

Structural micro precision=1.0 recall=1.0 (tp=17, fp=0, fn=0). These counts are fixture hits and misses, not a detection rate on external data.

Heuristic micro precision=1.0 recall=1.0 (tp=4, fp=0, fn=0). These counts are fixture hits and misses, not a detection rate on external data.

## Validation cases

- invalid_probability_shape: passed=True actual=MlpipelinecheckInputError

## Runtime

Seconds are wall time around the audit call. `tracemalloc_peak_bytes` is Python allocation tracing for that call, not process RSS.

- 2000 rows: dataset 0.178s (peak 420977 bytes), split 0.321s (peak 347980 bytes)
- 10000 rows: dataset 0.716s (peak 1880049 bytes), split 1.691s (peak 2403172 bytes)

## Not covered

- Near-duplicate rows, fuzzy target copies, and correlation proxies.
- Bootstrap calibration. The development list does not request intervals. IID intervals are opt-in and are not a grouped or temporal estimate.
- Grouped or temporal dependence in metric uncertainty.
- Real-world datasets, multiple industries, or adversarial users outside this list.
- Whether a flagged distribution shift is harmful. The fixture only says the means differ.
- Notebook, image, text, and deep-learning evaluation.
