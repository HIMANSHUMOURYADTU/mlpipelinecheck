# Contributing

Mlpipelinecheck is a small library. Changes should keep the public audit functions honest: no training, no network calls, no raw rows in reports, and no finding that claims a heuristic proves leakage.

## Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests benchmarks examples
python -m ruff format --check src tests benchmarks examples
```

Python 3.10 or newer. The local development environment used for 0.2.0 was Python 3.11.

## Adding a check

1. Add a `RuleSpec` in `src/mlpipelinecheck/rules.py`. The id is permanent once released.
2. Record the check on every path that should know about it, including the skip and error paths. A silent omission is a false pass.
3. Add a positive case and a benign case.
4. Document the id in `docs/rules.md`, including what the evidence does not prove.
5. If the synthetic benchmark labels that id, add a fixture in `benchmarks/synthetic_cases.py` and regenerate `benchmarks/RESULTS.md` by running the script. Do not type the precision table by hand.

## Tests

Prefer a hand-computable expected value over a snapshot of an entire report. Report JSON includes a timestamp, so compare fields rather than the whole document unless the test controls that field.

Do not weaken a detector so a fixture looks perfect. If a fixture and a detector disagree, fix the fixture when the detector matches its documented rule, and fix the detector when the rule is wrong.

## Privacy and safety

Do not add telemetry, optional network imports, `pickle` loading, or `fit` calls on user estimators. Do not put sample rows into a report. HTML must escape column names and other dataset-derived strings.
