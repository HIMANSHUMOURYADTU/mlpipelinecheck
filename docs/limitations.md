# Limitations

Mlpipelinecheck checks a fixed list of tabular conditions. A report with no findings only means those checks did not fire.

## What a finding is

Structural checks report a fact about the table: the target column was listed as a feature, two partitions share a row fingerprint, a timestamp falls before the training cutoff, a score shape cannot support AUC. Those facts can still have a benign explanation, such as a legitimate repeated measurement.

Heuristic checks use thresholds: high cardinality, near-constant frequency, class-proportion differences, and standardized mean differences. A difference in distribution is not labeled as leakage. The finding text says what was measured and what it does not prove.

## What this version does not do

- It does not train, load, or inspect models. Pickle and joblib files are out of scope.
- An optional prediction interval resamples rows as iid and does not retrain the model. It is not a grouped or temporal interval. `random_seed` affects only that resample.
- It does not detect near-duplicate rows, fuzzy target copies, or correlated target proxies. The ids `DATA.NEAR_DUPLICATE_ROWS` and `DATA.TARGET_DERIVED_FEATURE` are reserved in `docs/rules.md` and are not checks.
- It does not discover an entity key or a time column you did not name.
- It does not audit notebooks, git history, images, text, or deep-learning training loops.
- It does not impute, drop, or rewrite your frame.
- Exact row overlap runs only when the two partitions have the same column names and compatible dtype groups. A partial column intersection is not called an exact row match.
- Numeric time values are not interpreted as Unix epochs. Pass datetime values or parseable strings.
- Duplicate column labels are an input error, not a finding.
- Prediction rows are aligned by position. Misaligned indexes are not detected as a bug in your join; they are compared in the order you supplied.
- Distribution checks are skipped below the configured sample size. Skipping is `not_applicable`, not evidence that the distributions match.
- The synthetic benchmark reuses these thresholds. Its precision and recall describe that fixture list. See `benchmarks/RESULTS.md`.

## Cost

Missing-value, constant, cardinality, and distribution scans are linear in rows times columns, with a profile cap of 200 columns in the metadata lists. Exact duplicate and overlap fingerprints hash every cell of the compared columns. Peak Python allocations measured on the synthetic scale cases are in `benchmarks/RESULTS.md`. Those numbers are not a memory guarantee. `max_rows` skips the fingerprint checks above a size you choose and records them as not applicable.
