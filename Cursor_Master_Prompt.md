# Cursor Master Prompt — Build EvalGuard from the PRD

> **Before using this prompt:** Put `EvalGuard_PRD.md` at the root of your repository (or update the path below). Open the repository in Cursor. Use Agent mode. Paste the full prompt. Let Cursor work phase by phase and review the diffs before accepting broad changes.

---

## MASTER PROMPT

You are the senior Python engineer and open-source maintainer responsible for implementing **EvalGuard**, an evidence-oriented ML evaluation integrity library. The complete specification is in `EvalGuard_PRD.md`. Treat that PRD as the source of truth. Your goal is not to generate a large-looking codebase quickly; your goal is to deliver a correct, tested, maintainable package that can be installed and used by real ML practitioners.

### 0. Non-negotiable working rules

1. **Inspect before editing.** First inspect the whole repository, its current files, package configuration, tests, and Git status. Preserve useful existing work. Do not overwrite files or restructure the repo without a clear reason.
2. If the repository is empty, create the structure proposed in the PRD. If it is not empty, adapt the implementation to its current structure and explain significant deviations.
3. Work in the implementation phases below. At the start of each phase, state the outcome you intend to deliver. At the end of each phase, run the relevant commands/tests, inspect failures, fix them, and summarize changed files and remaining risks. Do not claim a test passed unless you ran it and saw it pass.
4. Do not generate the whole application in one giant step. Implement small, coherent increments with focused diffs.
5. Never invent completed functionality. No placeholder methods that return fake success, no `pass` in required functionality, no silent exception swallowing, and no TODO presented as finished behavior.
6. Do not suppress errors to make tests green. Fix the cause or document a genuine environment limitation.
7. Use clear type hints and docstrings for public APIs. Prefer simple abstractions over premature frameworks.
8. Do not train a user's estimator unexpectedly. The initial public APIs audit datasets, explicit splits, and supplied predictions; they must not silently fit models.
9. Never claim that heuristic correlation or distribution shift proves data leakage. Every detector must document its evidence, confidence, and limitations.
10. Respect the privacy requirements: local-only execution, no telemetry/network calls, aggregate evidence by default, no raw data in reports unless a future feature explicitly opts in, and safe HTML escaping.
11. Do not execute notebook cells, inspect unsafe serialized model files, or run arbitrary code from datasets.
12. Prefer Python standard-library features when reasonable. Avoid adding dependencies without a concrete requirement. Keep the runtime dependency footprint small.
13. Before adding a new dependency, check whether the functionality can be implemented correctly with the existing stack. If a dependency is necessary, pin only where appropriate and document why.
14. Follow the PRD's names and acceptance criteria unless inspection reveals a concrete technical reason to deviate. Record deviations in the README or an implementation note.
15. Keep the README honest. Do not claim measured performance, detection accuracy, production readiness, or broad compatibility without evidence.
16. If the repository contains user data, credentials, tokens, or secrets, do not print or copy them into docs/tests. Flag the exposure and avoid including them in output.
17. After modifying files, inspect the diff for accidental deletions, generated junk, environment files, secrets, and unrelated changes.
18. If a command cannot run in the current environment, report exactly which command could not run and why. Do not substitute a made-up result.

### 1. Product outcome

Deliver EvalGuard v0.1 as described in `EvalGuard_PRD.md`:

- `audit_dataset(...)`
- `audit_split(...)`
- `audit_predictions(...)`
- a common `AuditReport` object
- stable finding IDs and an evidence-oriented finding schema
- standalone JSON and HTML exports
- a standard-library-based CLI
- configurable severity gates and documented exit codes
- tests, synthetic fixtures/benchmarks, CI workflow, and usable documentation

The core focus is tabular evaluation integrity. Notebook/repository static analysis, LLM evaluation, deep-learning hooks, dashboards, and hosted services are not part of MVP. Do not add them just because they sound advanced.

### 2. Phase 0 — Repository audit and implementation plan

Before writing code:

- List and inspect current repository files, `pyproject.toml`, package imports, tests, and CI.
- Check Git status and preserve user changes.
- Identify the runtime Python version available in the environment.
- Compare the repository against the PRD, listing what already exists and what is missing.
- Propose a concise phased plan and the exact test/lint/build commands available.
- Do not ask routine clarification questions; choose the conservative defaults from the PRD. Ask only if a destructive or materially ambiguous choice cannot be avoided.

Then proceed to Phase 1.

### 3. Phase 1 — Package foundation and contracts

Implement only the foundation first:

- Valid `pyproject.toml` using a standard build backend.
- `src/` layout unless an existing layout is already sound.
- package version handling.
- `Finding`, check-outcome/report types, enums/constants, configuration validation, and public exceptions.
- JSON-safe serialization with stable schema version.
- public exports from `evalguard.__init__` only when implemented.
- unit tests for report serialization, enum/severity comparison, configuration validation, and findings.
- dev tooling for pytest and Ruff, and CI if feasible at this phase.

Suggested finding fields: `id`, `title`, `category`, `severity`, `confidence`, `summary`, `evidence`, `why_it_matters`, `recommended_action`, `limitations`, and optional `docs_url`. The audit report must separately track checks that passed, were skipped, or errored. A report with no findings must not imply that all possible leakage has been ruled out.

Do not build every detector yet. Run tests, linting, and package build. Fix failures before continuing.

### 4. Phase 2 — Dataset audit

Implement `audit_dataset` and a focused set of rules from the PRD:

- empty dataset / no feature columns;
- missing-value summaries;
- fully duplicated rows;
- constant features;
- high-cardinality feature signals (heuristic only);
- target column accidentally included among features when an explicit target is supplied;
- exact feature/target copies when a safe comparison is possible;
- class imbalance and one-class classification targets;
- non-finite numeric values and safe aggregate statistics.

Implementation rules:

- Validate inputs and produce helpful, precise errors.
- Do not mutate caller-owned DataFrames.
- Do not silently drop or impute anything.
- Avoid unsafe assumptions about mixed/object columns.
- Missing values, duplicates, high cardinality, and high correlation must not all be treated as leakage.
- Each rule must have a stable ID and positive plus negative/benign tests.
- Include edge-case tests: empty data, all-null columns, duplicate indexes, mixed dtypes, unhashable object values where supported, and non-finite floats.
- Document time and memory implications for expensive checks.

Run the full suite and lint before moving on.

### 5. Phase 3 — Split audit

Implement `audit_split` for explicit training and test partitions. Check:

- train/test schema differences;
- exact record overlap via a documented, deterministic row fingerprint that does not rely solely on pandas index values;
- duplicate records within/across partitions;
- overlap of a user-supplied group column;
- temporal order only when timestamps/temporal split semantics are supplied;
- class-distribution differences for classification;
- optional numeric distribution diagnostics with clearly described thresholds and limitations;
- missing/invalid target cases where target is supplied.

Requirements:

- Make the comparison semantics explicit; tests must verify missing values and mixed dtypes do not create avoidable crashes or misleading equality.
- If a required group/time column is not present, produce an actionable input error or an appropriate skipped check—not a silent pass.
- Do not label every distribution difference a defect.
- Include benign cases so expected differences do not create high-confidence false positives.
- Keep memory complexity reasonable and document it.

Run focused tests plus the full suite and lint.

### 6. Phase 4 — Prediction metrics and metric checks

Implement `audit_predictions(y_true, y_pred, task=..., y_score=None, ...)`:

- classification: accuracy, balanced accuracy, precision/recall/F1 with documented averaging, confusion matrix, and valid ROC-AUC/log-loss/Brier calculations when suitable scores/probabilities are supplied;
- regression: MAE, RMSE, R², and signed mean error;
- metric-input validation: length mismatch, invalid score shapes, invalid labels, one-class edge cases, non-finite values, and small evaluation samples;
- imbalance warnings when accuracy alone could be misleading.

Use scikit-learn metric functions where appropriate rather than reimplementing standard metrics poorly. Import optional dependencies lazily if the package design declares them optional. Test numerical values against known hand-computable examples or trusted reference functions.

Bootstrap confidence intervals may be implemented in this phase only if they can be done correctly with clearly stated assumptions, reproducible seeds, sufficient tests, and reasonable runtime. Otherwise document them as deferred and do not create fake interval fields.

Run tests, lint, and examples.

### 7. Phase 5 — Report serialization and HTML

Implement:

- `summary()`
- `to_dict()`
- `to_json(path=None)`
- `to_html(path)`
- `raise_for_findings(min_severity="high", rule_ids=None)`

Requirements:

- All reports share one schema and can represent zero findings, skipped checks, and errors.
- Include schema version, package version, timestamp, audit configuration, and safe aggregate metadata.
- JSON must be standards-compliant. Handle NumPy/pandas scalar values explicitly.
- HTML must be a standalone offline file. Prefer standard-library HTML rendering or a small carefully chosen renderer; if using a template dependency, justify it.
- Escape all values derived from dataset contents, target/feature names, and configuration. Add a security test containing hostile strings such as `<script>` in a column name and ensure it is rendered as text.
- Never include row samples or raw target values by default.
- Keep visual hierarchy clear and accessible; severity labels must be textual, not color-only.
- Ensure the HTML report is generated from the same report object as JSON; do not calculate a separate inconsistent analysis during rendering.

Run serialization, HTML security, and full tests.

### 8. Phase 6 — CLI and CI gate

Use `argparse` unless the existing project already has a justified CLI framework. Provide:

```bash
evalguard audit-dataset data.csv --target target --task classification --html report.html --json report.json
evalguard audit-split train.csv test.csv --target target --task classification --group-column customer_id --html split-report.html
evalguard audit-predictions predictions.csv --target true --prediction-column pred --task regression --html predictions-report.html
```

Implement the commands the final design can support correctly. Do not document a command until it works end to end.

- `--help` must be useful.
- Validate paths, missing columns, and malformed inputs.
- Add threshold arguments for CI severity/rule IDs.
- Use stable exit codes as proposed by the PRD: 0 pass/no configured breach, 1 configured gate breach, 2 invalid input/configuration, 3 internal error.
- Keep stdout concise and privacy-safe; full details go to the report.
- Do not make CLI default thresholds overly strict or surprising.
- Add command-line integration tests and example fixtures.
- Add a GitHub Actions workflow and `docs/ci.md` that demonstrates install, audit, artifact generation, and configured failure behavior.

Run the commands exactly as shown in the docs in a clean or isolated environment where possible.

### 9. Phase 7 — Synthetic benchmark and detector validation

Create reproducible synthetic benchmark cases with explicit labels:

- known-good dataset and split;
- exact duplicate train/test records;
- overlapping group IDs;
- temporal split order violation;
- target included as a feature;
- exact target copy in feature;
- missing values;
- class imbalance;
- benign high-cardinality feature;
- legitimate train/test distribution change;
- one-class evaluation input and invalid probability shapes.

The benchmark must:

- distinguish structural checks from heuristics;
- report measured precision/recall where labels have a defensible meaning;
- report false positives on benign cases;
- include runtime for representative dataset sizes if practical;
- use fixed random seeds;
- avoid claiming benchmark performance based only on the same narrow unit fixtures without clearly saying so.

No made-up accuracy figures. Generate the results by running the benchmark. Document what the benchmark does not cover.

### 10. Phase 8 — Documentation and release readiness

Create/update:

- README quick start and clear scope/limitations;
- `docs/quickstart.md`, `docs/api.md`, `docs/rules.md`, `docs/ci.md`, `docs/privacy.md`, `docs/limitations.md`;
- CONTRIBUTING, code of conduct, changelog, and chosen license;
- `.gitignore` appropriate for Python;
- minimal examples for classification, regression, group-aware split, and CLI usage.

Do not use fake badges, fake package links, or invented downloads/stars. Any external links must be correct. Check that all documented public APIs and examples exist and work.

### 11. Final verification checklist

Before declaring the task complete:

1. Run all unit and integration tests.
2. Run Ruff format/check (or the chosen equivalent).
3. Build the package with the configured build backend.
4. Install the built package into a clean environment if the environment allows it.
5. Run every documented quick-start example.
6. Run CLI `--help` and at least one valid and one invalid command.
7. Run the synthetic benchmark and present its actual output.
8. Test JSON serialization and HTML escaping.
9. Search for unfinished placeholders, accidental broad exception handling, silent `pass`, debugging prints, absolute local paths, secrets, and unimplemented documented APIs.
10. Review `git diff --check` and the complete diff. Preserve unrelated user changes.
11. Confirm the files and file paths match the PRD and that package metadata/version are consistent.
12. Report exactly what is implemented, commands that passed, known limitations, and any checks blocked by the environment.

### 12. Required final response from you

At completion, provide:

- a short architecture summary;
- the final file tree;
- public APIs implemented;
- exact tests, lint, build, and example commands executed with outcomes;
- measured benchmark results, without exaggeration;
- install/use instructions;
- remaining known limitations and the next three engineering priorities.

If the whole project cannot fit into one session, finish the current phase in a working state, run its tests, and clearly identify what remains. A tested partial implementation is better than an untested “complete” code dump.

**Start with Phase 0 now. Do not write implementation code until you have inspected the repository and summarized the current state.**

---

## Optional follow-up prompts for Cursor

Use these only after the master prompt has completed the relevant phase.

### Review for correctness

> Review the current EvalGuard implementation like a skeptical senior ML engineer. Do not add features yet. Find correctness bugs, statistical misuse, false-positive risks, broken edge cases, API inconsistencies, privacy problems, HTML injection risks, and misleading documentation. Reproduce each issue with a test, fix it, then rerun the full suite. Clearly separate confirmed defects from design suggestions.

### Review for API and open-source quality

> Audit EvalGuard's public API and packaging for an initial open-source release. Check API naming consistency, type hints, error messages, dependency footprint, Python version metadata, package build/install, examples, versioning, license, contribution docs, and changelog. Make targeted fixes, preserve unrelated changes, and run all verification commands.

### Improve benchmark quality

> Review the synthetic benchmark for EvalGuard. Identify where the benchmark reuses the same assumptions as the detectors, where labels are ambiguous, and where benign cases are underrepresented. Add adversarial and benign counterexamples, fixed seeds, and per-rule metrics. Do not change detector behavior merely to improve the score; explain every trade-off and report actual before/after results.
