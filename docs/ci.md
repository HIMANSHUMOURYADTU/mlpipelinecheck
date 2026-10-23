# Continuous integration

The workflow in `.github/workflows/ci.yml` runs on pull requests and on pushes to `main` or `master`. It requests only `contents: read`. It does not publish. `.github/workflows/release.yml` runs only when a GitHub Release is published, builds the distributions, refuses the occupied PyPI name, and publishes those files after the `pypi` environment is approved. Neither workflow has been executed on GitHub from this workspace. Local verification is a separate machine and is recorded in the README only for commands that were actually run.

## What the workflow runs

On Ubuntu, for Python 3.10, 3.11, and 3.12:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests benchmarks examples
python -m ruff format --check src tests benchmarks examples
python -m pytest
python benchmarks/synthetic_cases.py --skip-scale
python -m mlpipelinecheck audit-dataset examples/data/classification.csv --target churn --task classification --html dataset-report.html --json dataset-report.json
python -m build
```

The classification audit is expected to exit 0. The workflow then runs the target-copy file and requires exit code 1:

```bash
python -m mlpipelinecheck audit-dataset examples/data/leak_target_copy.csv --target churn --task classification --min-severity high
```

HTML and JSON reports from the clean audit are uploaded as artifacts. The scale benchmark (2,000 and 10,000 rows) is local-only so the workflow stays short. Run it with:

```bash
python benchmarks/synthetic_cases.py --write
```

That command rewrites `benchmarks/results.json` and `benchmarks/RESULTS.md` from the process that just finished. Do not hand-edit the measured counts. The held-out list is a separate local command, `python benchmarks/heldout_cases.py --write`. The workflow does not run it.

## Exit codes in a job

| Code | What to do in CI |
|---|---|
| 0 | The gate did not match. The job can continue. |
| 1 | A finding met `--min-severity` (default `high`). Fail the job when that is the policy. |
| 2 | The CSV, the columns, or the config is wrong, or a check errored. Fail the job. |
| 3 | Unexpected failure. Fail the job and keep the traceback. |

`--min-severity none` disables the gate. `--rule-id` restricts it to listed ids. Unknown ids exit 2. `--fail-on` is an alias of `--min-severity`; setting both to different values exits 2.

Low and medium findings stay in the report under the default gate. Lower the threshold only when that failure is intended.
