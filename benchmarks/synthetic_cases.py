"""Development fixtures for Mlpipelinecheck.

These cases were used while the detectors were written. They are not the
held-out evaluation set; that lives in ``heldout_cases.py``. A perfect score
here is a regression result on this list, not a field accuracy estimate.

Run ``python benchmarks/synthetic_cases.py``. Add ``--write`` to refresh
``benchmarks/results.json`` and ``benchmarks/RESULTS.md`` from the run.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlpipelinecheck import AuditConfig, __version__, audit_dataset, audit_predictions, audit_split
from mlpipelinecheck.exceptions import MlpipelinecheckInputError
from mlpipelinecheck.rules import DATASET_RULE_IDS, PREDICTION_RULE_IDS, SPLIT_RULE_IDS

SEED = 42
HEURISTIC_RULES = frozenset(
    {
        "DATA.HIGH_CARDINALITY_FEATURE",
        "DATA.CONSTANT_FEATURE",
        "SPLIT.CLASS_DISTRIBUTION_SHIFT",
        "SPLIT.NUMERIC_DISTRIBUTION_SHIFT",
    }
)
# Near-constant features share DATA.CONSTANT_FEATURE with exact constants.
# This fixture list only labels exact constants, but the rule can also fire
# on the frequency heuristic, so it is scored with the heuristics.
UNSCORED_RULES = frozenset({"METRIC.WIDE_CONFIDENCE_INTERVAL"})


@dataclass
class CaseResult:
    name: str
    audit: str
    expected: list[str]
    actual: list[str]
    false_positives: list[str] = field(default_factory=list)
    false_negatives: list[str] = field(default_factory=list)
    passed: bool = False
    detail: str = ""


def _frame(n: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.linspace(18.0, 70.0, n),
            "tenure": np.resize(np.arange(10), n),
            "plan": np.resize(np.array(["a", "b", "c", "d"]), n),
            "churn": np.resize(np.array([0, 1]), n),
        }
    )


def _partitions(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = {
        "age": np.linspace(20.0, 60.0, n),
        "plan": np.resize(np.array(["a", "b"]), n),
        "churn": np.resize(np.array([0, 1]), n),
    }
    train = pd.DataFrame({**columns, "customer_id": np.arange(n)})
    test = pd.DataFrame({**columns, "customer_id": np.arange(n, 2 * n)})
    return train, test


def _ids(report: Any) -> list[str]:
    return sorted(finding.id for finding in report.findings)


def _score_case(
    name: str, audit: str, universe: tuple[str, ...], expected: set[str], actual: set[str]
) -> CaseResult:
    scoped_expected = expected - UNSCORED_RULES
    scoped_actual = actual - UNSCORED_RULES
    # Rules outside this audit are not labeled by the case.
    labeled = set(universe) - UNSCORED_RULES
    false_positives = sorted((scoped_actual & labeled) - scoped_expected)
    false_negatives = sorted(scoped_expected - scoped_actual)
    unexpected = sorted(scoped_actual - labeled)
    return CaseResult(
        name=name,
        audit=audit,
        expected=sorted(scoped_expected),
        actual=sorted(scoped_actual),
        false_positives=false_positives + unexpected,
        false_negatives=false_negatives,
        passed=not false_positives and not false_negatives and not unexpected,
    )


def _dataset_cases() -> list[CaseResult]:
    results: list[CaseResult] = []

    def add(name: str, frame: pd.DataFrame, expected: set[str], **kwargs: Any) -> None:
        report = audit_dataset(frame, **kwargs)
        results.append(_score_case(name, "dataset", DATASET_RULE_IDS, expected, set(_ids(report))))

    good = _frame()
    add("known_good_dataset", good, set(), target="churn", task="classification")

    duplicated = pd.concat([good, good.iloc[:8]], ignore_index=True)
    add(
        "exact_duplicate_rows",
        duplicated,
        {"DATA.DUPLICATE_ROWS"},
        target="churn",
        task="classification",
    )

    copied = good.copy()
    copied["churn_copy"] = copied["churn"]
    add(
        "exact_target_copy",
        copied,
        {"DATA.TARGET_COPY_FEATURE"},
        target="churn",
        task="classification",
    )

    add(
        "target_included_as_feature",
        good,
        {"DATA.TARGET_IN_FEATURES"},
        target="churn",
        task="classification",
        feature_columns=["age", "tenure", "plan", "churn"],
    )

    missing = good.copy()
    missing.loc[::2, "tenure"] = np.nan
    add("missing_values", missing, {"DATA.MISSING_VALUES"}, target="churn", task="classification")

    imbalanced = _frame(100)
    imbalanced["churn"] = [0] * 95 + [1] * 5
    add(
        "class_imbalance",
        imbalanced,
        {"DATA.CLASS_IMBALANCE"},
        target="churn",
        task="classification",
    )

    one_class = _frame(40)
    one_class["churn"] = 1
    add(
        "one_class_target",
        one_class,
        {"DATA.CLASS_IMBALANCE"},
        target="churn",
        task="classification",
    )

    non_finite = good.copy()
    non_finite.loc[0, "age"] = np.inf
    add(
        "non_finite_value",
        non_finite,
        {"DATA.NON_FINITE_VALUES"},
        target="churn",
        task="classification",
    )

    empty = pd.DataFrame({"age": pd.Series(dtype="float"), "churn": pd.Series(dtype="int")})
    add("empty_dataset", empty, {"DATA.EMPTY_DATASET"}, target="churn", task="classification")

    constant = good.copy()
    constant["fixed"] = "x"
    add(
        "constant_feature",
        constant,
        {"DATA.CONSTANT_FEATURE"},
        target="churn",
        task="classification",
    )

    benign_card = _frame(40)
    benign_card["measurement"] = np.linspace(0.0, 1.0, 40)
    benign_card["user_id"] = [f"user-{index}" for index in range(40)]
    add(
        "benign_high_cardinality_float_and_declared_id",
        benign_card,
        set(),
        target="churn",
        task="classification",
        id_columns=["user_id"],
    )

    identifier = _frame(40)
    identifier["code"] = [f"code-{index}" for index in range(40)]
    add(
        "undeclared_high_cardinality_code",
        identifier,
        {"DATA.HIGH_CARDINALITY_FEATURE"},
        target="churn",
        task="classification",
    )
    return results


def _split_cases() -> list[CaseResult]:
    results: list[CaseResult] = []

    def add(
        name: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        expected: set[str],
        **kwargs: Any,
    ) -> None:
        report = audit_split(train, test, target="churn", task="classification", **kwargs)
        results.append(_score_case(name, "split", SPLIT_RULE_IDS, expected, set(_ids(report))))

    train, test = _partitions()
    add(
        "known_good_split",
        train,
        test,
        set(),
        group_column="customer_id",
        id_columns=["customer_id"],
    )

    copied_train, copied_test = _partitions()
    for column in copied_train.columns:
        copied_test.loc[copied_test.index[:5], column] = copied_train.loc[
            copied_train.index[:5], column
        ].to_numpy()
    add(
        "exact_duplicate_train_test_records",
        copied_train,
        copied_test,
        {"SPLIT.EXACT_ROW_OVERLAP", "SPLIT.GROUP_OVERLAP"},
        group_column="customer_id",
        id_columns=["customer_id"],
    )

    group_train, group_test = _partitions()
    group_test.loc[0, "customer_id"] = group_train.loc[0, "customer_id"]
    group_test.loc[0, "plan"] = "zzz"
    add(
        "overlapping_group_ids",
        group_train,
        group_test,
        {"SPLIT.GROUP_OVERLAP"},
        group_column="customer_id",
        id_columns=["customer_id"],
    )

    time_train, time_test = _partitions()
    time_train["event_time"] = pd.date_range("2024-01-01", periods=len(time_train), freq="D")
    time_test["event_time"] = pd.date_range("2024-01-20", periods=len(time_test), freq="D")
    add(
        "temporal_split_order_violation",
        time_train,
        time_test,
        {"SPLIT.TIME_ORDER_VIOLATION"},
        group_column="customer_id",
        id_columns=["customer_id"],
        time_column="event_time",
        split_strategy="temporal",
    )

    ordered_train, ordered_test = _partitions()
    ordered_train["event_time"] = pd.date_range("2024-01-01", periods=len(ordered_train), freq="D")
    ordered_test["event_time"] = pd.date_range("2024-03-01", periods=len(ordered_test), freq="D")
    add(
        "benign_temporal_order",
        ordered_train,
        ordered_test,
        set(),
        group_column="customer_id",
        id_columns=["customer_id"],
        time_column="event_time",
        split_strategy="temporal",
    )

    text_train = pd.DataFrame(
        {
            "value": pd.Series(np.arange(40), dtype=object),
            "churn": np.resize(np.array([0, 1]), 40),
        }
    )
    text_test = pd.DataFrame(
        {
            "value": pd.Series([str(value) for value in range(40)], dtype=object),
            "churn": np.resize(np.array([0, 1]), 40),
        }
    )
    add("text_does_not_equal_number", text_train, text_test, set())

    mismatch_train, mismatch_test = _partitions()
    add(
        "schema_mismatch",
        mismatch_train,
        mismatch_test.drop(columns=["plan"]),
        {"SPLIT.SCHEMA_MISMATCH"},
        group_column="customer_id",
        id_columns=["customer_id"],
    )

    small_train, small_test = _partitions()
    add(
        "small_test_partition",
        small_train,
        small_test.head(10),
        {"SPLIT.SMALL_TEST_PARTITION"},
        group_column="customer_id",
        id_columns=["customer_id"],
    )

    shift_train = pd.DataFrame(
        {"age": np.linspace(0.0, 1.0, 100), "churn": np.resize(np.array([0, 1]), 100)}
    )
    legitimate = pd.DataFrame(
        {"age": np.linspace(5.0, 6.0, 100), "churn": np.resize(np.array([0, 1]), 100)}
    )
    add(
        "legitimate_train_test_distribution_change",
        shift_train,
        legitimate,
        {"SPLIT.NUMERIC_DISTRIBUTION_SHIFT"},
    )
    mild = pd.DataFrame(
        {"age": np.linspace(0.05, 1.05, 100), "churn": np.resize(np.array([0, 1]), 100)}
    )
    add("benign_mild_numeric_shift", shift_train, mild, set())

    class_train = pd.DataFrame(
        {
            "age": np.linspace(0.0, 1.0, 80),
            "row_id": np.arange(80),
            "churn": [0, 1] * 40,
        }
    )
    class_test = pd.DataFrame(
        {
            "age": np.linspace(0.0, 1.0, 80),
            "row_id": np.arange(80, 160),
            "churn": [0] * 20 + [1] * 60,
        }
    )
    add(
        "class_proportion_difference",
        class_train,
        class_test,
        {"SPLIT.CLASS_DISTRIBUTION_SHIFT"},
        id_columns=["row_id"],
    )
    return results


def _prediction_cases() -> tuple[list[CaseResult], list[dict[str, Any]]]:
    results: list[CaseResult] = []
    validation: list[dict[str, Any]] = []
    balanced = audit_predictions(
        [0, 1] * 20,
        [0, 1] * 20,
        task="classification",
        y_score=[0.2, 0.8] * 20,
        config=AuditConfig(small_evaluation_rows=30),
    )
    results.append(
        _score_case(
            "known_good_predictions",
            "predictions",
            PREDICTION_RULE_IDS,
            set(),
            set(_ids(balanced)),
        )
    )
    one_class = audit_predictions(
        [0] * 40,
        [0] * 40,
        task="classification",
        y_score=np.linspace(0.05, 0.95, 40),
    )
    results.append(
        _score_case(
            "one_class_evaluation",
            "predictions",
            PREDICTION_RULE_IDS,
            {"METRIC.INVALID_AUC_INPUT", "METRIC.ACCURACY_ONLY_IMBALANCED"},
            set(_ids(one_class)),
        )
    )
    small = audit_predictions([0, 1, 0, 1], [0, 1, 1, 1], task="classification")
    results.append(
        _score_case(
            "small_evaluation_sample",
            "predictions",
            PREDICTION_RULE_IDS,
            {"METRIC.SMALL_EVALUATION_SAMPLE"},
            set(_ids(small)),
        )
    )
    try:
        audit_predictions([0, 1] * 5, [0, 1] * 5, task="classification", y_score=np.ones((10, 3)))
    except MlpipelinecheckInputError as exc:
        validation.append(
            {
                "name": "invalid_probability_shape",
                "expected": "MlpipelinecheckInputError",
                "actual": type(exc).__name__,
                "passed": True,
            }
        )
    else:
        validation.append(
            {
                "name": "invalid_probability_shape",
                "expected": "MlpipelinecheckInputError",
                "actual": "no_error",
                "passed": False,
            }
        )
    return results, validation


def _rule_metrics(cases: list[CaseResult]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for case in cases:
        expected = set(case.expected)
        actual = set(case.actual)
        for rule_id in expected & actual:
            counts[rule_id]["tp"] += 1
        for rule_id in set(case.false_positives):
            counts[rule_id]["fp"] += 1
        for rule_id in expected - actual:
            counts[rule_id]["fn"] += 1
    reported: dict[str, Any] = {}
    for rule_id, stats in sorted(counts.items()):
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        reported[rule_id] = {
            "kind": "heuristic" if rule_id in HEURISTIC_RULES else "structural",
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if tp + fn == 0 else tp / (tp + fn),
        }
    return reported


def _micro(rules: dict[str, Any], kind: str) -> dict[str, Any]:
    tp = fp = fn = 0
    for stats in rules.values():
        if stats["kind"] != kind:
            continue
        tp += stats["tp"]
        fp += stats["fp"]
        fn += stats["fn"]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
    }


def _time_call(fn: Any) -> dict[str, Any]:
    tracemalloc.start()
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"seconds": elapsed, "tracemalloc_peak_bytes": peak}


def _scale() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_rows in (2000, 10000):
        frame = _frame(n_rows)
        train = frame.iloc[: n_rows // 2].reset_index(drop=True)
        test = frame.iloc[n_rows // 2 :].reset_index(drop=True)
        test["customer_id"] = np.arange(n_rows // 2, n_rows)
        train["customer_id"] = np.arange(n_rows // 2)
        dataset_time = _time_call(
            lambda frame=frame: audit_dataset(frame, target="churn", task="classification")
        )
        split_time = _time_call(
            lambda train=train, test=test: audit_split(
                train,
                test,
                target="churn",
                task="classification",
                id_columns=["customer_id"],
            )
        )
        rows.append({"n_rows": n_rows, "dataset_audit": dataset_time, "split_audit": split_time})
    return rows


def run_benchmark(include_scale: bool = True) -> dict[str, Any]:
    """Run the labeled fixtures and, optionally, the runtime cases."""
    np.random.seed(SEED)
    cases = _dataset_cases() + _split_cases()
    prediction_cases, validation = _prediction_cases()
    cases.extend(prediction_cases)
    rules = _rule_metrics(cases)
    benign = [
        case.name
        for case in cases
        if case.name.startswith("known_good")
        or case.name.startswith("benign_")
        or case.name == "text_does_not_equal_number"
    ]
    benign_false_positives = [
        {"case": case.name, "rules": case.false_positives}
        for case in cases
        if case.name in benign and case.false_positives
    ]
    return {
        "package_version": __version__,
        "seed": SEED,
        "scope": (
            "Labeled synthetic fixtures constructed for this repository. "
            "Heuristic thresholds match the detector defaults, so the score is a "
            "regression check on this list, not a field accuracy estimate. "
            "A precision or recall of 1.000 often comes from a single labeled fixture. "
            "DATA.CONSTANT_FEATURE is grouped with the heuristics because that id also "
            "covers near-constant columns; an exact one-value column remains a high-confidence finding."
        ),
        "cases": [
            {
                "name": case.name,
                "audit": case.audit,
                "passed": case.passed,
                "expected": case.expected,
                "actual": case.actual,
                "false_positives": case.false_positives,
                "false_negatives": case.false_negatives,
            }
            for case in cases
        ],
        "rules": rules,
        "structural": _micro(rules, "structural"),
        "heuristic": _micro(rules, "heuristic"),
        "benign_false_positives": benign_false_positives,
        "validation_cases": validation,
        "scale": _scale() if include_scale else [],
        "not_covered": [
            "Near-duplicate rows, fuzzy target copies, and correlation proxies.",
            "Bootstrap calibration. The development list does not request intervals. "
            "IID intervals are opt-in and are not a grouped or temporal estimate.",
            "Grouped or temporal dependence in metric uncertainty.",
            "Real-world datasets, multiple industries, or adversarial users outside this list.",
            "Whether a flagged distribution shift is harmful. The fixture only says the means differ.",
            "Notebook, image, text, and deep-learning evaluation.",
        ],
    }


def _format(results: dict[str, Any]) -> str:
    lines = [
        "# Mlpipelinecheck synthetic benchmark",
        "",
        f"Package {results['package_version']}. Seed {results['seed']}.",
        "",
        results["scope"],
        "",
        "## Case results",
        "",
        "| Case | Passed | Expected | Actual |",
        "|---|---|---|---|",
    ]
    for case in results["cases"]:
        lines.append(
            f"| {case['name']} | {case['passed']} | {', '.join(case['expected']) or 'none'} | "
            f"{', '.join(case['actual']) or 'none'} |"
        )
    lines.extend(["", "## Per-rule counts on this fixture list", ""])
    lines.append("| Rule | Kind | TP | FP | FN | Precision | Recall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for rule_id, stats in results["rules"].items():
        precision = "n/a" if stats["precision"] is None else f"{stats['precision']:.3f}"
        recall = "n/a" if stats["recall"] is None else f"{stats['recall']:.3f}"
        lines.append(
            f"| {rule_id} | {stats['kind']} | {stats['tp']} | {stats['fp']} | {stats['fn']} | "
            f"{precision} | {recall} |"
        )
    for label in ("structural", "heuristic"):
        stats = results[label]
        lines.append("")
        lines.append(
            f"{label.title()} micro precision={stats['precision']} recall={stats['recall']} "
            f"(tp={stats['tp']}, fp={stats['fp']}, fn={stats['fn']}). "
            "These counts are fixture hits and misses, not a detection rate on external data."
        )
    lines.extend(["", "## Validation cases", ""])
    for case in results["validation_cases"]:
        lines.append(f"- {case['name']}: passed={case['passed']} actual={case['actual']}")
    if results["scale"]:
        lines.extend(["", "## Runtime", ""])
        lines.append(
            "Seconds are wall time around the audit call. "
            "`tracemalloc_peak_bytes` is Python allocation tracing for that call, not process RSS."
        )
        lines.append("")
        for row in results["scale"]:
            lines.append(
                f"- {row['n_rows']} rows: dataset {row['dataset_audit']['seconds']:.3f}s "
                f"(peak {row['dataset_audit']['tracemalloc_peak_bytes']} bytes), "
                f"split {row['split_audit']['seconds']:.3f}s "
                f"(peak {row['split_audit']['tracemalloc_peak_bytes']} bytes)"
            )
    lines.extend(["", "## Not covered", ""])
    for item in results["not_covered"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Mlpipelinecheck synthetic benchmark.")
    parser.add_argument("--write", action="store_true", help="Write results.json and RESULTS.md.")
    parser.add_argument("--skip-scale", action="store_true", help="Skip the larger runtime cases.")
    args = parser.parse_args(argv)
    results = run_benchmark(include_scale=not args.skip_scale)
    text = _format(results)
    print(text)
    if args.write:
        directory = Path(__file__).resolve().parent
        (directory / "results.json").write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / "RESULTS.md").write_text(text, encoding="utf-8")
    failed = [case["name"] for case in results["cases"] if not case["passed"]]
    failed.extend(case["name"] for case in results["validation_cases"] if not case["passed"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
