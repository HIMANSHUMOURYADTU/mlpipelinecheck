# Changelog

## 0.2.0 — 2026-10-23

Hardening of the 0.1.0 audit library. Report schema version is now `1.1.0`.

- Exact target-copy checks compare the resolved feature list. Declared identifiers and columns left out of `feature_columns` are not reported as feature copies.
- Numeric values stored as objects are rejected as timestamps, along with numeric dtypes. They are not parsed as epochs.
- Log loss is omitted when a true-class probability is 0. A clipped finite substitute is not reported.
- `random_seed` is rejected unless `bootstrap_samples` is set. The default seed of 0 remains unused when no interval is requested.
- Optional iid percentile intervals for accuracy or MAE, using `bootstrap_samples` and `random_seed`. Grouped and temporal bootstrap requests raise an input error instead of returning an iid interval.
- JSON object keys are sorted. `created_at` still changes between runs.
- Held-out fixtures in `benchmarks/heldout_cases.py`, separate from the development list. A later case labels an explicit `feature_columns` list that includes the target as `DATA.TARGET_IN_FEATURES`.
- Publishing guide, release workflow, and a local name gate. The PyPI name `evalguard` belongs to a different project, so this tree is not uploaded under that name.
- The distribution name and the import package are both `mlpipelinecheck`. The console command is `mlpipelinecheck`.

## 0.1.0 — 2026-10-23

Initial implementation of the tabular audit described in `EvalGuard_PRD.md`.

- `audit_dataset`, `audit_split`, and `audit_predictions`
- shared `AuditReport` with JSON and standalone HTML
- rule catalog, severity gate, and CLI exit codes 0, 1, 2, and 3
- synthetic benchmark fixtures and a generated results file
- GitHub Actions workflow for install, lint, test, benchmark, and the severity gate

Bootstrap confidence intervals are not included. The package is not published to PyPI.
