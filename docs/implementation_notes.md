# Implementation notes

The PRD is the source of truth. These are the places versions 0.1 and 0.2 differ, and why.

1. IID percentile intervals are opt-in through `bootstrap_samples` on `audit_predictions`. The interval resamples rows independently and does not retrain the model. Grouped and temporal requests are rejected. A non-zero `random_seed` without `bootstrap_samples` is a configuration error. The default seed of 0 does not change results when no interval is requested, and the report says the seed was not used.

2. Configuration files are JSON only. PyYAML is not a dependency. The CLI `--config` flag expects a JSON object of `AuditConfig` fields.

3. Two rule ids were added because the phase requirements need them and the original catalog table did not name them: `DATA.NON_FINITE_VALUES` and `SPLIT.SMALL_TEST_PARTITION`.

4. scikit-learn is a required dependency. Prediction metrics are part of the MVP, and the implementation calls scikit-learn rather than reimplementing accuracy, F1, ROC AUC, log loss, Brier, MAE, MSE, and R².

5. HTML is rendered with the standard-library `html.escape`. No template package is used.

6. A one-class target raises `DATA.CLASS_IMBALANCE` and `METRIC.ACCURACY_ONLY_IMBALANCED` to high severity. The catalog default for those ids is medium. Milder imbalance stays at medium. A single class makes class-wise rates and AUC structurally unusable, so the default high gate should see it.

7. Near-constant columns share `DATA.CONSTANT_FEATURE` at info severity when no exact constant is present. Exact one-value columns stay at low severity with high confidence.

8. Fingerprints and numeric target-copy comparisons treat finite integer-valued floats as the same value as the matching integer, within the 53-bit integer range. Text is not coerced, so `"1"` does not equal `1`.

9. Exact train/test overlap is computed only when schemas match. Comparing a partial intersection would report shared projections as identical records.

10. Duplicate column names raise `MlpipelinecheckInputError` before any finding is built. A duplicated label cannot also be reported as `DATA.TARGET_IN_FEATURES`.

11. The split audit does not repeat every dataset check. It covers schema, overlap, groups, time, distribution, small test size, and the target problems that make a split metric misleading.

12. If any check status is `error`, the CLI exits 2 even when findings would also breach the gate. Stdout says the audit is incomplete. An incomplete audit is not reported as a clean pass.

13. Group ids and row contents are omitted from reports. Class names and column names are included because the aggregates cannot be interpreted without them.

14. `feature_columns` is an extra argument on the dataset and split APIs so an explicit feature list can be audited, including the case where it contains the target. The default matches the PRD: target and identifier columns are excluded.

15. Means and standard deviations ignore missing and non-finite values. Sample standard deviation uses `ddof=1`. A pooled standardized mean difference is undefined when the pooled variance is zero, and that column is skipped.

16. The package is not published. There is no PyPI URL, badge, or download count in the README.
