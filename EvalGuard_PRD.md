# EvalGuard — Product Requirements Document (PRD)

**Document version:** 1.0  
**Product status:** Proposed open-source Python library  
**Initial release target:** `0.1.0` MVP  
**Owner:** Project maintainer  
**License recommendation:** MIT  
**Primary language:** Python  
**Product category:** ML evaluation integrity, dataset/split auditing, reproducible checks, and CI quality gates

---

## 1. Executive summary

EvalGuard is an open-source Python library that helps ML practitioners find and investigate risks that can make model evaluations misleading. It audits datasets, explicit train/test splits, and supplied predictions; produces evidence-backed findings; and can fail a CI job when configured high-risk conditions are found.

EvalGuard is **not** a guarantee that a model is leakage-free, nor is every statistical warning proof of a defect. It must distinguish confirmed structural problems from heuristic risks and make limitations visible in every report.

### Product thesis

ML teams need more than a single accuracy or MAE number. They need a repeatable, reviewable way to answer:

1. Is this dataset structurally suitable for this task?
2. Could the split violate independence, time, group, or duplicate assumptions?
3. Are the metrics appropriate and sufficiently precise to interpret?
4. Which findings are supported by direct evidence, and which are only hypotheses?
5. Can the same checks run locally and in CI before results are trusted?

### One-sentence value proposition

**EvalGuard finds common evaluation-integrity risks, explains the evidence, and turns checks into reproducible reports and CI gates.**

---

## 2. Problem statement

An ML experiment may appear successful while its evaluation is fragile or invalid. Examples include duplicate records appearing in both train and test, related entities being split across partitions, temporal leakage, target columns accidentally appearing among features, a class-imbalanced classification task being summarized only by accuracy, or performance metrics being reported without uncertainty.

These issues often require a combination of data inspection, statistical judgment, source-code review, and experimentation. Existing tools already cover parts of this workflow. EvalGuard should not try to recreate every monitoring, observability, or data-quality feature. Its initial focus is an evidence-oriented evaluation audit that can be run as a small Python library or command-line tool.

### Core user pain

- The user receives a score but cannot tell whether the evaluation design is trustworthy.
- A finding may be hard to reproduce or explain to a reviewer.
- Different notebooks use different ad hoc checks.
- Data scientists want important issues to be caught before an experiment or merge is accepted.
- Teams need a simple local-first tool without a hosted service or mandatory data upload.

---

## 3. Product goals and non-goals

### Goals for the first public release

1. Provide stable, documented APIs for dataset, split, and prediction audits.
2. Detect a bounded set of meaningful, explainable risks with transparent limitations.
3. Use a common finding schema across all checks.
4. Export reports as Python objects, JSON, and standalone HTML.
5. Offer a basic CLI for auditing CSV datasets/splits and writing reports.
6. Support CI by returning a nonzero exit code when configured severity thresholds are reached.
7. Ship synthetic known-good and known-bad test cases to measure detector behavior.
8. Make execution local by default, deterministic where practical, and privacy-conscious.
9. Be installable with a standard Python packaging workflow and easy to contribute to.

### Explicit non-goals for MVP

- Guaranteeing that a model has no possible data leakage.
- Replacing Evidently, Deepchecks, NannyML, Great Expectations, or experiment trackers.
- A hosted dashboard, multi-user backend, accounts, database, telemetry, or cloud service.
- Automatically rewriting notebooks or modifying a user's training code.
- Executing arbitrary notebooks or user code as part of static analysis.
- Full support for every deep-learning framework, distributed training, streaming data, LLM/RAG evaluation, image/audio pipelines, or feature stores.
- Automatically selecting the best production model.
- Labeling suspicious correlation, unusual metrics, or a score decrease as proof of leakage.

### Follow-on ideas, not MVP commitments

- Static notebook/repository scanning with source-location evidence.
- Additional image, text, time-series, and deep-learning checks.
- Extensible user-defined checks and plugins.
- Optional integrations for MLflow, DVC, notebook environments, or CI providers.
- A benchmark suite tracking detection quality across versions.

---

## 4. Target users and use cases

### Primary personas

**A. Data science student / practitioner**  
Wants a quick audit before publishing a portfolio project, competition entry, or report.

**B. ML engineer**  
Wants repeatable checks and machine-readable results in a local pipeline or CI workflow.

**C. Researcher / reviewer**  
Wants evidence about split construction, uncertainty, and metric choices before trusting a reported result.

**D. Team lead / reviewer**  
Wants a concise report that differentiates high-confidence problems from low-confidence warnings.

### Top user stories

1. As a data scientist, I can audit a pandas DataFrame and receive findings without needing to train a model.
2. As an ML engineer, I can compare explicit train and test partitions for duplicate and group overlap.
3. As a researcher, I can calculate suitable metrics and uncertainty summaries from held-out predictions.
4. As a reviewer, I can read a finding, understand the evidence, and reproduce the check.
5. As a maintainer, I can configure the CI gate so only specified severities or finding IDs fail a build.
6. As a privacy-conscious user, I can export reports without copying raw rows or labels into the report.
7. As a contributor, I can add a check with tests and documentation without changing unrelated modules.

---

## 5. Positioning and competitive awareness

EvalGuard should differentiate through focus and transparent evidence rather than unsupported claims of uniqueness.

| Tool / category | Established focus | EvalGuard's intended relationship |
|---|---|---|
| Evidently | Broad evaluation, data quality, drift, and ML/LLM observability | Complementary; do not reproduce its entire metric/monitoring catalog |
| Deepchecks | Data/model validation, train-test checks, and ML testing workflows | Complementary; focus the first release on a lean local audit API and evidence-rich findings |
| NannyML | Post-deployment performance estimation and drift monitoring | Adjacent; not the MVP's core use case |
| scikit-learn Pipelines | Correctly chain transformations and estimators; reduce preprocessing leakage risks | Integrate with and recommend standard safe patterns rather than compete with them |

**Differentiation hypothesis to validate:** a small library that combines a clear finding model, split-integrity checks, transparent uncertainty, easy reports, and CI enforcement will be useful to developers who want a focused audit without adopting a full monitoring stack.

Do not market the product as “the first”, “the only”, or “the most accurate” unless independently demonstrated.

---

## 6. Product principles

1. **Evidence before confidence.** Every finding should include the observed evidence and the test that produced it.
2. **Warnings are not verdicts.** Heuristic findings must be labeled as risks or signals, not confirmed leakage.
3. **Reproducibility.** Stable rule IDs, deterministic defaults, explicit configuration, and versioned reports.
4. **Local-first privacy.** No network calls or upload of user data in normal execution.
5. **Useful defaults, explicit limits.** Sensible defaults, but document assumptions and allow configuration.
6. **Small composable API.** Each audit can run independently; users do not need to use every feature.
7. **Safe failure.** A failed check must not silently disappear. Report errors and unsupported cases separately from passing checks.
8. **Honest benchmarks.** Publish false positives and false negatives alongside successful detections.

---

## 7. Scope and functional requirements

Priorities: **P0 = required for MVP**, **P1 = desirable after core**, **P2 = later release**.

### 7.1 Dataset audit — P0

Input: a pandas DataFrame, target column name (optional where no target-specific check is required), task type (`classification`, `regression`, or `auto` where inference is safe), optional identifier columns, and configuration.

Checks:

- Missing values by column and overall.
- Fully duplicated rows and duplicate rates.
- Duplicate target values / label distribution summary.
- Constant and near-constant columns.
- Target accidentally included as a feature (when a target column is explicitly supplied).
- Exact or near-exact copies of the target among feature columns, when a deterministic supported comparison is possible.
- High-cardinality columns that may act as identifiers.
- Classification class imbalance and single-class targets.
- Basic numeric summaries and invalid/non-finite numeric values.
- Rows/columns count and audit warnings when sample size or data types limit a check.

Requirements:

- Return findings and compact statistics, not a mutated DataFrame.
- Never silently drop rows or columns.
- Missing values, duplicates, high cardinality, and target correlation must not automatically be marked as leakage. Classification depends on the specific check and evidence.
- Correlation-based target-proxy checks, if included, are heuristic and should default to `warning` or `info` severity.

### 7.2 Split audit — P0

Inputs: explicit train/test DataFrames or feature/target pairs; optional group column, time column, ID column, and split strategy metadata.

Checks:

- Exact row overlap between train and test using a documented row fingerprint. Handle index values separately; index equality alone is not enough.
- Duplicate rows within and across partitions.
- Group overlap when a group identifier is provided.
- Time-ordering violations when a temporal split is declared or a timestamp is provided.
- Train/test class distribution differences for classification.
- Large train/test distribution differences on numeric columns using configurable descriptive/statistical checks, with sample-size limitations explained.
- Train/test schema mismatch: missing/extra columns and incompatible dtypes.
- Target contamination or inconsistent target availability if target columns are supplied.
- Small test-partition warning where configured thresholds are exceeded.

Requirements:

- Do not assume every problem requires a time split. A temporal finding is only relevant when the user provides time metadata or declares a temporal setting.
- Group leakage requires the user to identify a meaningful group key; the library cannot infer all related entities.
- Distribution difference is not proof of leakage. Reports should say “distribution differs” and explain possible interpretations.
- Avoid loading every row into auxiliary copies if the check can be done with vectorized or bounded-memory approaches.

### 7.3 Prediction evaluation — P0

Inputs: `y_true`, `y_pred`, task type, and optional probability/score arrays.

Classification metrics:

- Accuracy, balanced accuracy, precision, recall, F1 (macro and weighted where meaningful).
- Confusion matrix and class support.
- ROC AUC only when valid labels/scores are provided and the problem formulation is supported.
- Log loss and Brier score only when suitable probabilities are provided.
- Calibration summaries may be P1 if robust binning and edge cases are tested.
- Warn when accuracy is shown as the only requested headline metric for a highly imbalanced target.

Regression metrics:

- MAE, RMSE, and R² (with documented behavior for constant targets and small samples).
- Error summaries including signed mean error.
- Avoid interpreting R² alone as an error magnitude metric.

Uncertainty:

- Add bootstrap confidence intervals for supported scalar metrics in P0/P1, with a fixed seed by default and configurable repetitions.
- Clearly state that ordinary bootstrap intervals assume an appropriate sampling unit; group/time-aware bootstrap behavior must not be implied unless implemented.
- For P0 if full bootstrap is too large to implement safely, ship the metric functions first and mark CI intervals P1 rather than faking precision.

Requirements:

- Reject mismatched lengths and unsupported score shapes with actionable errors.
- Do not silently coerce ambiguous inputs.
- Metrics must be tested against known fixtures and, where applicable, reference implementations.

### 7.4 Finding model — P0

All checks must emit a common, serializable finding object.

Required fields:

- `id`: stable rule ID, such as `SPLIT.EXACT_ROW_OVERLAP`.
- `title`: brief human-readable summary.
- `category`: e.g. `data_quality`, `split_integrity`, `metric_validity`, `statistical_uncertainty`.
- `severity`: `critical`, `high`, `medium`, `low`, or `info`.
- `confidence`: `high`, `medium`, or `low`.
- `status`: `detected`, `not_detected`, `not_applicable`, or `error` at the check-result level; finding objects represent detected issues.
- `summary`: concise explanation of what was observed.
- `evidence`: structured, privacy-safe facts (counts, ratios, column names, configuration, metric values).
- `why_it_matters`: practical consequence.
- `recommended_action`: specific next step.
- `limitations`: caveats about the check.
- `docs_url`: optional stable documentation URL or relative rule reference.

Rules:

- Finding IDs remain stable across patch versions once released.
- Do not include arbitrary raw samples by default.
- A check can return no finding; audit reports must still record which checks ran, which passed, which were skipped, and which errored.

### 7.5 Report object — P0

Required methods/properties:

- `summary()` — concise counts by severity and category.
- `findings` — structured findings.
- `checks` — all check outcomes including passed/skipped/errored checks.
- `to_dict()` — JSON-compatible representation.
- `to_json(path=None)` — return JSON string and optionally write a file.
- `to_html(path)` — create a standalone HTML report that works offline.
- `raise_for_findings(min_severity="high", rule_ids=None)` — raise a documented exception when configured findings exist.
- Include package version, report schema version, run timestamp, user-configured options, and safe dataset shape metadata.

HTML security requirements:

- Escape user-controlled strings.
- Do not execute scripts supplied by data, column names, or notebook code.
- No external CDN dependency for core report rendering.

### 7.6 Command-line interface — P0

Use standard library `argparse` initially to keep installation light.

Proposed commands:

```bash
evalguard audit-dataset data.csv --target target --task classification --html report.html --json report.json
evalguard audit-split train.csv test.csv --target target --task classification --group-column customer_id --html split-report.html
evalguard audit-predictions predictions.csv --target true --prediction-column pred --task regression --html predictions-report.html
```

CLI requirements:

- Provide `--help` with working examples.
- Support `--config` for a documented YAML/JSON configuration if added; do not make configuration parsing more complex than necessary.
- Provide `--fail-on` and/or `--min-severity` for CI gates.
- Use stable exit codes and document them. Suggested: `0` success/no threshold breach; `1` configured gate breached; `2` invalid input/configuration; `3` internal error.
- Print a short human-readable summary to stdout and keep full details in the report.
- Never print raw rows or target values by default.

### 7.7 CI integration — P0

- CLI must be suitable for a GitHub Actions step or any shell-based CI system.
- Provide a minimal example workflow in `docs/ci.md`.
- Fail only according to explicit user configuration; default behavior should not fail simply because a low-severity informational finding exists.
- Document that audit results are advisory and depend on available metadata and data size.

### 7.8 Configuration — P0

Configuration should include:

- task type;
- optional group/time/id columns;
- enabled/disabled check IDs;
- severity thresholds for CI;
- maximum rows or sampling settings for expensive checks;
- bootstrap repetitions/seed if implemented;
- privacy options controlling whether any examples are stored;
- distribution-check thresholds.

Validate unknown keys and invalid values. Provide a readable default configuration. Avoid an elaborate plugin system before the initial APIs are stable.

### 7.9 Documentation and examples — P0

- README: problem, what EvalGuard does/does not do, install, quick start, example report screenshot or textual output, limitations, contribution instructions.
- API reference for each public function/class.
- Tutorials: one classification dataset; one regression dataset; one grouped split; one temporal split; one CI run.
- Rule catalog: ID, interpretation, severity, confidence, limitations, remediation, and test fixture.
- Security/privacy statement explaining local execution and report contents.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and license file.

---

## 8. Proposed public API

Treat these signatures as the target interface. Refine only when repository inspection or implementation evidence justifies a change; document any change.

```python
from evalguard import audit_dataset, audit_split, audit_predictions

# Data quality and target/feature checks
report = audit_dataset(
    df,
    target="churn",
    task="classification",
    id_columns=["customer_id"],
)
report.summary()
report.to_html("dataset_audit.html")

# Validate two explicit partitions
split_report = audit_split(
    train_df,
    test_df,
    target="churn",
    task="classification",
    group_column="customer_id",
    time_column=None,
    split_strategy="group",
)

# Evaluate supplied held-out predictions
metric_report = audit_predictions(
    y_true=y_test,
    y_pred=y_pred,
    task="regression",
)
```

API guidelines:

- Public names and docstrings should use explicit, beginner-readable terminology.
- Functions must not mutate input data.
- Use type hints for public functions and main data classes.
- Return the same `AuditReport` abstraction from each audit entry point.
- Keep estimators out of the core API until safe evaluation protocols are well-defined. Do not unexpectedly train a user's estimator when auditing data.
- Future releases may add optional `audit_estimator` and notebook/repository static-analysis APIs as separate features.

---

## 9. Rule catalog for MVP

Rule IDs are proposed stable identifiers. Initial severity is a default that can be tuned after testing.

| Rule ID | Condition / evidence | Default severity | Confidence | Notes |
|---|---|---|---|---|
| `DATA.EMPTY_DATASET` | zero rows or zero feature columns | high | high | Cannot audit a useful dataset |
| `DATA.MISSING_VALUES` | missing values exceed configured threshold | medium | high | Not automatically a defect; context matters |
| `DATA.DUPLICATE_ROWS` | exact duplicate records found | medium | high | Duplicates can be legitimate in some domains |
| `DATA.CONSTANT_FEATURE` | a feature has one unique non-null value | low | high | May be intentional or harmless |
| `DATA.HIGH_CARDINALITY_FEATURE` | feature meets configurable unique-ratio threshold | low | medium | Potential identifier; not proof of leakage |
| `DATA.TARGET_IN_FEATURES` | target explicitly appears among feature columns | critical | high | Strong structural contamination signal |
| `DATA.TARGET_COPY_FEATURE` | feature is exactly equal to target under supported comparison | high | high | Must not infer semantic copies that it cannot verify |
| `DATA.CLASS_IMBALANCE` | class proportions cross configured thresholds | medium | high | Suggest metrics and stratification review |
| `SPLIT.SCHEMA_MISMATCH` | train/test feature schemas differ | high | high | Include exact missing/extra column names |
| `SPLIT.EXACT_ROW_OVERLAP` | matching row fingerprints across train/test | high | high | Document fingerprint normalization |
| `SPLIT.GROUP_OVERLAP` | same declared group appears in both partitions | high | high | Only applicable when a group column is supplied |
| `SPLIT.TIME_ORDER_VIOLATION` | declared temporal test data contains timestamps earlier than the training cutoff, according to the chosen policy | high | high | Only applicable with timestamp/strategy metadata |
| `SPLIT.CLASS_DISTRIBUTION_SHIFT` | class proportions differ beyond a configured measure/threshold | low | medium | Distribution difference is not proof of a bad split |
| `SPLIT.NUMERIC_DISTRIBUTION_SHIFT` | numeric feature distributions differ beyond configured criterion | low | medium | State the test and limitations |
| `METRIC.ACCURACY_ONLY_IMBALANCED` | accuracy-only summary on highly imbalanced classification | medium | high | Recommend balanced metrics; no claim the model is bad |
| `METRIC.INVALID_AUC_INPUT` | AUC requested with invalid labels/scores or only one class | high | high | Prefer explicit validation errors when computation is impossible |
| `METRIC.SMALL_EVALUATION_SAMPLE` | evaluation sample below configured threshold | low | high | Exact threshold should be configurable |
| `METRIC.WIDE_CONFIDENCE_INTERVAL` | supported CI width exceeds configured threshold | low | medium | Only if CI is implemented and interpretation is documented |

Every rule must have one or more unit tests for a positive detection and a negative/benign case.

---

## 10. Statistical and implementation correctness

### Statistical requirements

- Identify sample counts and denominators for every rate.
- Document threshold defaults and allow customization where practical.
- Account for missing values explicitly in calculations.
- Make small-sample behavior explicit.
- Use correct multiclass averaging and probability-shape conventions.
- Avoid presenting p-values as a substitute for effect size or practical significance.
- Avoid interpreting a metric improvement or decline alone as evidence of leakage.
- Explain assumptions for bootstrap intervals and distribution tests.

### Technical requirements

- Python 3.10+ support unless the maintainer chooses a narrower, documented matrix based on dependency support.
- pandas and NumPy as core runtime dependencies; scikit-learn/SciPy may be core or optional depending on the final implementation. Keep installation minimal.
- `pyproject.toml` with standard project metadata and reproducible development tooling.
- Unit tests with `pytest`; lint/format using `ruff` (or document an equivalent choice).
- Type hints on public APIs and key models.
- No network calls in normal audit execution.
- No hidden model training.
- Set random seeds for stochastic routines and expose the seed in configuration/report metadata.
- Gracefully handle empty data, all-null columns, mixed types, duplicate index labels, one-class targets, non-finite values, and unhashable object values.
- Define deterministic behavior for unsupported dtypes rather than crashing unexpectedly.
- Add bounded or configurable strategies for costly comparisons on large datasets.

---

## 11. Privacy and security requirements

1. Data stays on the user's machine by default.
2. No telemetry or external requests in the core library.
3. Reports must not include raw rows, raw target labels, or arbitrary sample values by default.
4. Include column names and aggregate counts only when needed to explain findings; document that column names themselves may be sensitive.
5. HTML output must escape untrusted content and work offline.
6. Do not execute notebook cells, scripts, serialized model artifacts, or shell commands as part of an audit.
7. Do not load pickle/joblib model files from untrusted sources in the MVP.
8. If users opt into example rows in a future version, make that opt-in explicit and warn about sensitive information.
9. Keep dependencies current and scan them through automated CI where practical.

---

## 12. User experience and report design

The default report should answer the following in the first screen:

- Overall audit status: `issues found`, `no configured issues found`, or `audit incomplete`.
- Number of findings by severity.
- Highest-priority findings, each with evidence and recommended action.
- Checks passed, skipped, or errored.
- Data shape, task type, audit version, and configuration summary.

### Finding presentation

Each finding should be scannable:

1. Title and severity.
2. Confidence label.
3. What was observed — with counts or metrics.
4. Why it could matter.
5. What to do next.
6. Limitations and conditions under which the finding may be benign.

Design requirements:

- Calm, neutral visual hierarchy; avoid exaggerated “AI risk score” language.
- Use text labels as well as colors.
- Make JSON the machine-readable source of truth and HTML a rendering of the same report data.
- Do not combine all findings into one opaque global score.

---

## 13. Quality metrics and definition of success

Do not invent performance or accuracy claims before measuring them. Establish baselines from test fixtures and publish actual results after running the benchmark.

### Engineering quality gates

- All tests pass on the supported Python matrix.
- Ruff checks pass.
- Package builds successfully and installs in a clean environment.
- Public examples run in CI.
- JSON serialization and HTML rendering round-trip successfully for representative reports.
- Invalid input has predictable, documented error behavior.
- No known unescaped HTML injection path in report-rendering tests.

### Detector benchmark

Create a labelled synthetic benchmark with known-good and known-bad cases. Report:

- Precision and recall per detector where ground truth is well-defined.
- False-positive rate on benign/ambiguous cases.
- Runtime and memory versus row count.
- Behavior across missingness, class imbalance, duplicate levels, group sizes, and time splits.

For structural checks such as explicit group overlap or exact row overlap, target near-perfect fixture performance. Do not set a numeric public target for heuristic checks until baseline data exists. A practical initial goal is no false positives on the project's curated canonical benign fixtures, but keep the fixture set diverse and acknowledge it is not proof of general correctness.

### Adoption signals after release

- Clean install success and quick-start completion.
- External issues with reproducible examples.
- External pull requests or independently written integrations.
- Downloads, stars, or forks as secondary signals only—not proof of quality.

---

## 14. Release plan

### Phase 0 — Repository foundation

- Inspect repository, keep existing work intact, settle package metadata, tests, formatting, license, and CI.
- Define report/finding schemas and stable rule IDs.

### Phase 1 — Core audit library (`0.1.0`)

- Dataset audit.
- Split audit.
- Prediction metrics validation.
- Common report/finding types.
- JSON and HTML export.
- CLI and CI gate.
- Unit tests, synthetic benchmark, README, rule catalog, CI example.

### Phase 2 — Deeper evaluation (`0.2.x`)

- Bootstrap intervals with documented assumptions.
- More robust distribution diagnostics and optional plotting.
- Improved configuration and benchmark reporting.
- Clear compatibility matrix and public API stabilization.

### Phase 3 — Code and notebook inspection (`0.3.x` or later)

- Static parsing of `.py` and `.ipynb` for a narrow set of supported anti-patterns.
- Read notebook JSON but never execute cells.
- Source cell/line references and documented confidence levels.
- Include fixtures with both anti-patterns and safe equivalent code.

### Phase 4 — Ecosystem integrations

- Optional MLflow/DVC/notebook integrations.
- Plugin checks only if there is user demand and the core rule contract is stable.
- Broader model modalities after the tabular experience is proven.

---

## 15. Suggested repository structure

```text
evalguard/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── evalguard/
│       ├── __init__.py
│       ├── _version.py
│       ├── api.py
│       ├── config.py
│       ├── exceptions.py
│       ├── findings.py
│       ├── report.py
│       ├── cli.py
│       ├── checks/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── split.py
│       │   └── metrics.py
│       └── renderers/
│           ├── __init__.py
│           ├── json.py
│           └── html.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   ├── test_cli.py
│   ├── test_report.py
│   └── test_security.py
├── benchmarks/
│   ├── README.md
│   └── synthetic_cases.py
└── docs/
    ├── quickstart.md
    ├── api.md
    ├── rules.md
    ├── ci.md
    ├── privacy.md
    └── limitations.md
```

This is a recommendation, not a requirement to reorganize an existing repository blindly. If the repo already has a sound layout, retain it and adapt the imports accordingly.

---

## 16. Acceptance criteria for MVP

The first release is acceptable only when all criteria below are met:

1. `pip install -e .` works in a clean environment.
2. All documented API examples execute from a fresh environment.
3. Dataset, split, and prediction audits return the same report abstraction.
4. Each implemented rule has a stable ID and tests for detection and a benign case.
5. Reports distinguish findings from passed, skipped, and failed checks.
6. JSON serialization is standards-compliant and includes a schema version.
7. HTML is standalone, readable offline, and safely escapes input strings.
8. CLI commands produce reports and obey documented exit codes.
9. A CI example demonstrates a severity gate.
10. The tool does not train a model or send data over the network unexpectedly.
11. The README documents limitations and makes no unsupported detection-accuracy claims.
12. The synthetic benchmark runs and reports measured results, even if results are modest.
13. Unit tests, linting, packaging checks, and example smoke tests pass.
14. Unsupported cases fail with actionable messages and do not silently pass.
15. The final release checklist verifies that the package version, changelog, public API, and documentation agree.

---

## 17. Risks and mitigations

| Risk | Mitigation |
|---|---|
| False positives undermine trust | Confidence labels, evidence, negative fixtures, tunable thresholds, and no claim that heuristics prove leakage |
| Scope expands into a full observability platform | Keep MVP to tabular evaluation integrity and explicit reports; move notebook/repo scanning to a later phase |
| Checks are mathematically inappropriate in edge cases | State assumptions, validate input shapes, use reference fixtures, and mark unsupported cases |
| Large datasets cause slow audits | Vectorize checks, document complexity, add limits/sampling where safe, and benchmark runtime/memory |
| API changes break early users | Stable schema version, semantic versioning, changelog, and tests for public interfaces |
| Reports expose sensitive data | Aggregate-only evidence by default, HTML escaping, and privacy tests |
| AI coding tool invents unfinished functionality | Require tests for every feature, prohibit placeholder implementations in MVP, and implement in phases |
| Product duplicates established tools | Keep differentiation hypothesis explicit; compare fairly and publish the exact supported scope |

---

## 18. Open decisions and recommended defaults

- **Name:** EvalGuard is a working name; verify package and trademark availability before public release.
- **License:** MIT for low-friction adoption, provided it suits the maintainer's goals.
- **Supported Python:** start at 3.10+ if compatible with chosen dependency versions; test a clear matrix and document it.
- **Input format:** pandas DataFrame in Python APIs, CSV in the first CLI.
- **Privacy:** aggregate-only report evidence by default.
- **ML framework:** framework-agnostic in v0.1; do not require PyTorch or TensorFlow.
- **Notebook/code scanning:** deferred until the report schema and static-check rules are stable.
- **Global risk score:** do not implement; severity counts and rule-level evidence are more interpretable.

---

## 19. References and technical background

The following public resources informed the scope and constraints. They are not endorsements and do not imply that EvalGuard already has these capabilities.

- Evidently — open-source evaluation, testing, and monitoring: https://github.com/evidentlyai/evidently
- Deepchecks — ML testing and validation: https://deepchecks.com/open-source/
- NannyML — post-deployment performance estimation and drift: https://github.com/NannyML/nannyml
- scikit-learn — common pitfalls and data leakage: https://scikit-learn.org/stable/common_pitfalls.html
- Python Packaging User Guide — `pyproject.toml`: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

---

## 20. Final product definition

EvalGuard v0.1 is a reliable, local-first Python package and CLI for auditing tabular datasets, explicit train/test splits, and supplied predictions. It emits structured, evidence-backed findings, produces offline HTML/JSON reports, and optionally blocks CI according to user-defined gates. It does not claim to prove the absence of leakage; it makes known risks easier to identify, reproduce, explain, and act on.
