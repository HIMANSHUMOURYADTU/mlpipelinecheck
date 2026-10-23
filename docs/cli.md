# Command line

`mlpipelinecheck` and `python -m mlpipelinecheck` call the same parser. `--help` lists commands and exit codes.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | The configured severity gate did not match. |
| 1 | At least one finding met the gate. |
| 2 | Invalid input, invalid configuration, an unknown rule id, or a check that ended in error. |
| 3 | An unexpected internal error. |

The default gate is severity `high`. `--min-severity none` disables it. `--fail-on` is an alias of `--min-severity`. Setting both to different values exits 2. `--rule-id` is repeatable and limits the gate to those ids. Unknown ids exit 2.

Reports requested with `--json` or `--html` are written before the process exits.

## audit-dataset

```bash
mlpipelinecheck audit-dataset examples/data/classification.csv --target churn --task classification --html dataset-report.html --json dataset-report.json
```

`--task` is `auto`, `classification`, or `regression`. `--id-column` and `--feature-column` are repeatable.

## audit-split

```bash
mlpipelinecheck audit-split examples/data/train.csv examples/data/test.csv --target churn --task classification --group-column customer_id --id-column customer_id
```

`--time-column` is used only when you pass it. `--split-strategy` is `random`, `stratified`, `group`, or `temporal`. `group` requires `--group-column`. `temporal` requires `--time-column`.

## audit-predictions

```bash
mlpipelinecheck audit-predictions examples/data/predictions.csv --target true --prediction-column pred --task regression
```

`--task` is required and is `classification` or `regression`. Optional flags:

| Flag | Effect |
|---|---|
| `--score-column` | Classification scores or probabilities. |
| `--group-column` | Dependence declaration. Values are counted and are not copied into the report. |
| `--time-column` | Same for timestamps. |
| `--bootstrap-samples` | IID resample count. Minimum 20. |
| `--random-seed` | Seed for that resample. A non-zero seed without `--bootstrap-samples` is an error. |
| `--observation-dependence` | `independent`, `grouped`, or `temporal`. Grouped and temporal intervals are refused. |
| `--max-interval-width` | Flags `METRIC.WIDE_CONFIDENCE_INTERVAL` when the interval is wider. Requires a bootstrap. |

## Shared flags

| Flag | Effect |
|---|---|
| `--config` | JSON object of `AuditConfig` fields. CLI flags override it. |
| `--disable-check` | Skip one rule id. Repeatable. A skipped check is not a pass. |
| `--max-rows` | Skip fingerprint checks above this row count and record them as not applicable. |
| `--encoding` | CSV encoding. Default `utf-8`. |

A target copy fails the default gate:

```bash
mlpipelinecheck audit-dataset examples/data/leak_target_copy.csv --target churn --task classification --min-severity high
```

That command exits 1.
