# Troubleshooting

## The command exited 2

Exit 2 means the audit did not finish as a normal report. Read the message.

- A missing column, a duplicate column label, or an empty path is an input error.
- `split_strategy='group'` without a group column, or `temporal` without a time column, is a configuration error.
- Numeric timestamps, including numbers stored as objects, are rejected. Pass datetime values or parseable strings.
- A non-zero `--random-seed` without `--bootstrap-samples` is rejected because the seed would not change the audit.
- `bootstrap_samples` on a dataset or split audit is rejected. Intervals apply to prediction audits.
- Grouped or temporal bootstrap requests are rejected. An iid interval is not substituted.
- `--min-severity` and `--fail-on` must match when both are set. An unknown `--rule-id` exits 2.

## The command exited 1

The severity gate matched. The report still contains the findings. Lower `--min-severity` only when you intend that threshold to fail the command. The default is `high`.

## The command exited 0 and you expected a problem

Exit 0 means the configured gate did not match. Low and medium findings can still be in the report. A report with no findings means the checks that ran did not fire. It does not clear the dataset. Near-duplicate rows and target-derived features are not checks. See `docs/limitations.md`.

## JSON files from two identical audits differ

`created_at` is the UTC time the report object was built. Other keys are sorted. The timestamp is part of the report on purpose.

## `pip install evalguard` installed the wrong library

The name `evalguard` on PyPI is a different project. This package is `mlpipelinecheck`, and it has not been uploaded. Install from a checkout:

```bash
python -m pip install -e ".[dev]"
```

See `docs/publishing.md` before any upload.

## HTML or JSON was not written

The parent directory must already exist. A directory path, or a missing parent, raises `MlpipelinecheckInputError` and the CLI exits 2.
