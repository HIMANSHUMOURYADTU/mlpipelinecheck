# Benchmarks

`synthetic_cases.py` is the development list. The detectors were written against it. `heldout_cases.py` is a separate list. Its labels are assigned from the case construction and the documented thresholds, then the audit is run. Do not change a threshold to improve the held-out score.

Both write markdown and JSON when given `--write`. Undefined precision, recall, or false-positive rate means the denominator was zero.

# Synthetic benchmark

`synthetic_cases.py` builds labeled tables with seed 42 and compares the finding ids from a real audit to those labels. `RESULTS.md` and `results.json` are produced by:

```bash
python benchmarks/synthetic_cases.py --write
```

`--skip-scale` omits the 2,000-row and 10,000-row timing cases. The default run includes them. Wall time is `perf_counter` around the audit call. `tracemalloc_peak_bytes` is Python allocation tracing for that call, not process RSS.

## How a case is scored

A case passes when the set of finding ids equals the expected set. Extra ids are false positives. Missing ids are false negatives.

Structural rules are exact conditions: copies, overlap, schema, emptiness, missingness, non-finite values, group overlap, time order, and invalid metric inputs. Heuristic rules on this list are `DATA.HIGH_CARDINALITY_FEATURE`, `DATA.CONSTANT_FEATURE`, `SPLIT.CLASS_DISTRIBUTION_SHIFT`, and `SPLIT.NUMERIC_DISTRIBUTION_SHIFT`. `DATA.CONSTANT_FEATURE` is grouped there because the same id also covers near-constant columns. An exact constant is still a high-confidence finding.

`METRIC.WIDE_CONFIDENCE_INTERVAL` is not scored on the development list because those cases do not set `bootstrap_samples`. The interval exists only when a prediction audit requests it.

Precision and recall use only true positives, false positives, and false negatives on this list. A score of 1.000 with one true positive means that one fixture matched its label.

## What the labels assume

The positive distribution cases are built so they cross the default thresholds, and the benign cases are built so they do not. The score is a regression check on that construction. It is not an estimate of detection quality on field data, and it is not a reason to change a threshold.

The benchmark does not cover near-duplicate rows, fuzzy target copies, correlation proxies, bootstrap calibration, grouped metric uncertainty, or data outside this file. A flagged numeric shift means the means differ by the stated statistic. It does not say the shift is harmful.

No public dataset is included. A field table would need a documented license and labels assigned without reading Mlpipelinecheck's output. That combination was not established for this release.
