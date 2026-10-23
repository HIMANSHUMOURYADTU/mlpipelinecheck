# Mlpipelinecheck held-out fixtures

Package 0.2.0. Seed 7.

Held-out synthetic fixtures. Labels were assigned from the construction and from the documented thresholds before the audit result was read. This is not a measure of performance on field data.

## Cases

| Case | Structural label | Passed | Expected | Actual |
|---|---|---|---|---|
| exact_target_copy | True | True | DATA.TARGET_COPY_FEATURE | DATA.TARGET_COPY_FEATURE |
| target_listed_as_feature | True | True | DATA.TARGET_IN_FEATURES | DATA.TARGET_IN_FEATURES |
| doubled_target_is_not_an_exact_copy | True | True | none | none |
| legitimate_correlation | True | True | none | none |
| missing_value | True | True | DATA.MISSING_VALUES | DATA.MISSING_VALUES |
| non_finite_value | True | True | DATA.NON_FINITE_VALUES | DATA.NON_FINITE_VALUES |
| class_imbalance | True | True | DATA.CLASS_IMBALANCE | DATA.CLASS_IMBALANCE |
| one_class_target | True | True | DATA.CLASS_IMBALANCE | DATA.CLASS_IMBALANCE |
| undeclared_identifier | False | True | DATA.HIGH_CARDINALITY_FEATURE | DATA.HIGH_CARDINALITY_FEATURE |
| declared_id_and_unique_float | False | True | none | none |
| benign_repeated_measurement | True | True | DATA.DUPLICATE_ROWS | DATA.DUPLICATE_ROWS |
| constant_feature | False | True | DATA.CONSTANT_FEATURE | DATA.CONSTANT_FEATURE |
| empty_dataset | True | True | DATA.EMPTY_DATASET | DATA.EMPTY_DATASET |
| exact_row_and_group_copy | True | True | SPLIT.EXACT_ROW_OVERLAP, SPLIT.GROUP_OVERLAP | SPLIT.EXACT_ROW_OVERLAP, SPLIT.GROUP_OVERLAP |
| group_overlap_without_row_copy | True | True | SPLIT.GROUP_OVERLAP | SPLIT.GROUP_OVERLAP |
| reversed_time_order | True | True | SPLIT.TIME_ORDER_VIOLATION | SPLIT.TIME_ORDER_VIOLATION |
| ordered_time_split | True | True | none | none |
| schema_mismatch | True | True | SPLIT.SCHEMA_MISMATCH | SPLIT.SCHEMA_MISMATCH |
| duplicate_inside_train_only | True | True | DATA.DUPLICATE_ROWS | DATA.DUPLICATE_ROWS |
| large_numeric_location_shift | False | True | SPLIT.NUMERIC_DISTRIBUTION_SHIFT | SPLIT.NUMERIC_DISTRIBUTION_SHIFT |
| mild_numeric_location_shift | False | True | none | none |
| class_proportion_difference | False | True | SPLIT.CLASS_DISTRIBUTION_SHIFT | SPLIT.CLASS_DISTRIBUTION_SHIFT |
| known_good_predictions | True | True | none | none |
| one_class_evaluation | True | True | METRIC.ACCURACY_ONLY_IMBALANCED, METRIC.INVALID_AUC_INPUT | METRIC.ACCURACY_ONLY_IMBALANCED, METRIC.INVALID_AUC_INPUT |
| small_evaluation_sample | True | True | METRIC.SMALL_EVALUATION_SAMPLE | METRIC.SMALL_EVALUATION_SAMPLE |

## Per-rule counts

| Rule | Kind | n | TP | FP | FN | TN | Precision | Recall | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DATA.CLASS_IMBALANCE | structural | 22 | 2 | 0 | 0 | 20 | 1.000 | 1.000 | 0.000 |
| DATA.CONSTANT_FEATURE | heuristic | 13 | 1 | 0 | 0 | 12 | 1.000 | 1.000 | 0.000 |
| DATA.DUPLICATE_ROWS | structural | 22 | 2 | 0 | 0 | 20 | 1.000 | 1.000 | 0.000 |
| DATA.EMPTY_DATASET | structural | 13 | 1 | 0 | 0 | 12 | 1.000 | 1.000 | 0.000 |
| DATA.HIGH_CARDINALITY_FEATURE | heuristic | 13 | 1 | 0 | 0 | 12 | 1.000 | 1.000 | 0.000 |
| DATA.MISSING_VALUES | structural | 22 | 1 | 0 | 0 | 21 | 1.000 | 1.000 | 0.000 |
| DATA.NON_FINITE_VALUES | structural | 13 | 1 | 0 | 0 | 12 | 1.000 | 1.000 | 0.000 |
| DATA.TARGET_COPY_FEATURE | structural | 22 | 1 | 0 | 0 | 21 | 1.000 | 1.000 | 0.000 |
| DATA.TARGET_IN_FEATURES | structural | 22 | 1 | 0 | 0 | 21 | 1.000 | 1.000 | 0.000 |
| METRIC.ACCURACY_ONLY_IMBALANCED | structural | 3 | 1 | 0 | 0 | 2 | 1.000 | 1.000 | 0.000 |
| METRIC.INVALID_AUC_INPUT | structural | 3 | 1 | 0 | 0 | 2 | 1.000 | 1.000 | 0.000 |
| METRIC.SMALL_EVALUATION_SAMPLE | structural | 3 | 1 | 0 | 0 | 2 | 1.000 | 1.000 | 0.000 |
| METRIC.WIDE_CONFIDENCE_INTERVAL | structural | 3 | 0 | 0 | 0 | 3 | undefined | undefined | 0.000 |
| SPLIT.CLASS_DISTRIBUTION_SHIFT | heuristic | 9 | 1 | 0 | 0 | 8 | 1.000 | 1.000 | 0.000 |
| SPLIT.EXACT_ROW_OVERLAP | structural | 9 | 1 | 0 | 0 | 8 | 1.000 | 1.000 | 0.000 |
| SPLIT.GROUP_OVERLAP | structural | 9 | 2 | 0 | 0 | 7 | 1.000 | 1.000 | 0.000 |
| SPLIT.NUMERIC_DISTRIBUTION_SHIFT | heuristic | 9 | 1 | 0 | 0 | 8 | 1.000 | 1.000 | 0.000 |
| SPLIT.SCHEMA_MISMATCH | structural | 9 | 1 | 0 | 0 | 8 | 1.000 | 1.000 | 0.000 |
| SPLIT.SMALL_TEST_PARTITION | structural | 9 | 0 | 0 | 0 | 9 | undefined | undefined | 0.000 |
| SPLIT.TIME_ORDER_VIOLATION | structural | 9 | 1 | 0 | 0 | 8 | 1.000 | 1.000 | 0.000 |

Structural micro: n=193, tp=17, fp=0, fn=0, tn=176, precision=1.000, recall=1.000, false-positive rate=0.000.

Heuristic micro: n=44, tp=4, fp=0, fn=0, tn=40, precision=1.000, recall=1.000, false-positive rate=0.000.

## Coverage notes

- doubled_target_is_not_an_exact_copy: No rule claims affine or noisy target derivatives. This is a coverage gap, not a miss of the exact-copy rule.
- legitimate_correlation: A strong legitimate association is not labeled as leakage.

## Not covered

- Near-duplicate rows and fuzzy or affine target copies are coverage gaps when the exact-copy rule is silent.
- Grouped and temporal uncertainty. Those intervals are not implemented.
- Real-world tables, adversarial users, and industries outside this list.
- Whether a distribution shift is harmful. The numeric labels only say whether the documented SMD cutoff was crossed.
