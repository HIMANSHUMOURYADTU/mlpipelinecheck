"""Held-out fixtures for Mlpipelinecheck.

Labels are assigned from the case construction and from the documented
thresholds. This module does not read detector output in order to decide what
a case should find. ``benchmarks/synthetic_cases.py`` is the development list
and is not reused here.

Run ``python benchmarks/heldout_cases.py``. ``--write`` refreshes
``benchmarks/heldout_RESULTS.md`` and ``benchmarks/heldout_results.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlpipelinecheck import AuditConfig, __version__, audit_dataset, audit_predictions, audit_split
from mlpipelinecheck.exceptions import MlpipelinecheckInputError

# Same rule classification as the development fixture list. A test checks equality.
HEURISTIC_RULES = frozenset(
    {
        "DATA.HIGH_CARDINALITY_FEATURE",
        "DATA.CONSTANT_FEATURE",
        "SPLIT.CLASS_DISTRIBUTION_SHIFT",
        "SPLIT.NUMERIC_DISTRIBUTION_SHIFT",
    }
)
SEED = 7
SMALL_SPLIT = AuditConfig(small_test_rows=1)


@dataclass
class Case:
    """One held-out audit and the label assigned before the audit runs."""

    name: str
    family: str
    expected: frozenset[str]
    rationale: str
    structural: bool
    audit: Any
    coverage_gap: str = ""
    forbid_leakage_claim: bool = False


def _pooled_smd(left: np.ndarray, right: np.ndarray) -> float | None:
    """Pooled standardized mean difference used as the labeling rule.

    This is the definition in ``docs/rules.md``: sample variances with
    ``ddof=1``, then the absolute mean difference divided by the pooled
    standard deviation. It is not a call into Mlpipelinecheck.
    """
    if left.size < 2 or right.size < 2:
        return None
    var_left = float(np.var(left, ddof=1))
    var_right = float(np.var(right, ddof=1))
    pooled = ((left.size - 1) * var_left + (right.size - 1) * var_right) / (
        left.size + right.size - 2
    )
    if pooled <= 0 or not math.isfinite(pooled):
        return None
    return abs(float(np.mean(left) - np.mean(right)) / math.sqrt(pooled))


def _balanced_frame(n: int, *, copy: bool = False, doubled: bool = False) -> pd.DataFrame:
    churn = [index % 2 for index in range(n)]
    frame = pd.DataFrame(
        {
            "age": [20 + index for index in range(n)],
            "plan": ["a" if index % 2 == 0 else "b" for index in range(n)],
            "churn": churn,
        }
    )
    if copy:
        frame["mirror"] = [float(value) for value in churn]
    if doubled:
        frame["twice"] = [2 * value for value in churn]
    return frame


def _dataset_cases() -> list[Case]:
    exact = _balanced_frame(8, copy=True)
    doubled = _balanced_frame(8, doubled=True)
    correlated = _balanced_frame(8)
    correlated["score"] = [0.2, 0.7, 0.3, 0.8, 0.1, 0.6, 0.25, 0.9]
    missing = _balanced_frame(8).astype({"age": float})
    missing.loc[0, "age"] = np.nan
    non_finite = _balanced_frame(8).astype({"age": float})
    non_finite.loc[1, "age"] = np.inf
    imbalanced = pd.DataFrame(
        {
            "age": [20 + (index % 6) for index in range(20)],
            "visits": [(index % 5) + 1 for index in range(20)],
            "churn": [0] * 18 + [1, 1],
        }
    )
    one_class = _balanced_frame(8)
    one_class["churn"] = 0
    codes = pd.DataFrame(
        {
            "code": [f"k{index:03d}" for index in range(25)],
            "plan": ["a" if index % 2 == 0 else "b" for index in range(25)],
            "churn": [index % 2 for index in range(25)],
        }
    )
    declared = codes.copy()
    declared["measurement"] = np.linspace(0.0, 1.0, 25)
    repeated = _balanced_frame(8)
    repeated.loc[1] = repeated.loc[0]
    constant = _balanced_frame(8)
    constant["fixed"] = "x"
    empty = pd.DataFrame({"age": pd.Series(dtype="float"), "churn": pd.Series(dtype="int")})
    return [
        Case(
            "exact_target_copy",
            "dataset",
            frozenset({"DATA.TARGET_COPY_FEATURE"}),
            "mirror is float(churn) on every row, and it is part of the default feature set.",
            True,
            lambda: audit_dataset(exact, target="churn", task="classification"),
        ),
        Case(
            "target_listed_as_feature",
            "dataset",
            frozenset({"DATA.TARGET_IN_FEATURES"}),
            "feature_columns explicitly lists churn, the target. The label is that caller choice, not a detector result.",
            True,
            lambda: audit_dataset(
                _balanced_frame(8),
                target="churn",
                task="classification",
                feature_columns=["age", "churn"],
            ),
        ),
        Case(
            "doubled_target_is_not_an_exact_copy",
            "dataset",
            frozenset(),
            "twice is 2*churn. The zero rows match and the one rows do not, so the column is not an exact copy.",
            True,
            lambda: audit_dataset(doubled, target="churn", task="classification"),
            coverage_gap="No rule claims affine or noisy target derivatives. This is a coverage gap, not a miss of the exact-copy rule.",
        ),
        Case(
            "legitimate_correlation",
            "dataset",
            frozenset(),
            "score moves with churn and is not equal to it. Correlation is not an exact copy.",
            True,
            lambda: audit_dataset(correlated, target="churn", task="classification"),
            coverage_gap="A strong legitimate association is not labeled as leakage.",
        ),
        Case(
            "missing_value",
            "dataset",
            frozenset({"DATA.MISSING_VALUES"}),
            "One age cell is missing. The default threshold flags any positive missing fraction.",
            True,
            lambda: audit_dataset(missing, target="churn", task="classification"),
        ),
        Case(
            "non_finite_value",
            "dataset",
            frozenset({"DATA.NON_FINITE_VALUES"}),
            "One age cell is positive infinity.",
            True,
            lambda: audit_dataset(non_finite, target="churn", task="classification"),
        ),
        Case(
            "class_imbalance",
            "dataset",
            frozenset({"DATA.CLASS_IMBALANCE"}),
            "18 of 20 labels are 0, so the minority fraction is 0.1, which meets the default <= 0.1 rule.",
            True,
            lambda: audit_dataset(imbalanced, target="churn", task="classification"),
        ),
        Case(
            "one_class_target",
            "dataset",
            frozenset({"DATA.CLASS_IMBALANCE"}),
            "Every label is 0.",
            True,
            lambda: audit_dataset(one_class, target="churn", task="classification"),
        ),
        Case(
            "undeclared_identifier",
            "dataset",
            frozenset({"DATA.HIGH_CARDINALITY_FEATURE"}),
            "code has 25 distinct values in 25 rows. The documented heuristic flags a non-float column at or above unique ratio 0.95 with at least 20 uniques and 20 rows.",
            False,
            lambda: audit_dataset(codes, target="churn", task="classification"),
        ),
        Case(
            "declared_id_and_unique_float",
            "dataset",
            frozenset(),
            "code is declared as an identifier. measurement is a float, which the default heuristic skips.",
            False,
            lambda: audit_dataset(
                declared, target="churn", task="classification", id_columns=["code"]
            ),
        ),
        Case(
            "benign_repeated_measurement",
            "dataset",
            frozenset({"DATA.DUPLICATE_ROWS"}),
            "Two rows are exact copies. That is a duplicate-row fact. It is not, by the rule text, proof of leakage.",
            True,
            lambda: audit_dataset(repeated, target="churn", task="classification"),
            forbid_leakage_claim=True,
        ),
        Case(
            "constant_feature",
            "dataset",
            frozenset({"DATA.CONSTANT_FEATURE"}),
            "fixed has one non-null value. That is the exact-constant branch of DATA.CONSTANT_FEATURE.",
            False,
            lambda: audit_dataset(constant, target="churn", task="classification"),
        ),
        Case(
            "empty_dataset",
            "dataset",
            frozenset({"DATA.EMPTY_DATASET"}),
            "The table has column names and no rows.",
            True,
            lambda: audit_dataset(empty, target="churn", task="classification"),
        ),
    ]


def _split_frame(start: int, n: int, *, day: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"g{index:03d}" for index in range(start, start + n)],
            "age": [30 + (index % 5) for index in range(n)],
            "churn": [index % 2 for index in range(n)],
            "event_date": pd.date_range(day, periods=n, freq="D"),
        }
    )


def _split_cases() -> list[Case]:
    train = _split_frame(0, 8, day="2024-01-01")
    copied_test = _split_frame(100, 8, day="2024-06-01")
    for column in train.columns:
        copied_test.loc[copied_test.index[:4], column] = train.loc[
            train.index[:4], column
        ].to_numpy()
    group_train = _split_frame(0, 8, day="2024-01-01")
    group_test = _split_frame(0, 8, day="2024-06-01")
    group_test["age"] = group_test["age"] + 1
    group_test["churn"] = 1 - group_test["churn"]
    later_train = _split_frame(0, 8, day="2024-01-01")
    later_test = _split_frame(50, 8, day="2024-03-01")
    early_test = _split_frame(50, 8, day="2023-01-01")
    wide_train = _split_frame(0, 8, day="2024-01-01")
    wide_test = _split_frame(50, 8, day="2024-03-01").drop(columns=["age"])
    duplicate_train = _split_frame(0, 8, day="2024-01-01")
    duplicate_train.iloc[1] = duplicate_train.iloc[0]
    duplicate_test = _split_frame(50, 8, day="2024-03-01")
    rng = np.random.default_rng(SEED)
    shift_train_age = rng.normal(0.0, 1.0, 80)
    shift_test_age = rng.normal(4.0, 1.0, 80)
    labeled_smd = _pooled_smd(shift_train_age, shift_test_age)
    shift_expected = (
        frozenset({"SPLIT.NUMERIC_DISTRIBUTION_SHIFT"})
        if labeled_smd is not None and labeled_smd >= 0.8
        else frozenset()
    )
    mild_train_age = np.linspace(0.0, 1.0, 40)
    mild_test_age = np.linspace(0.05, 1.05, 40)
    mild_smd = _pooled_smd(mild_train_age, mild_test_age)
    mild_expected = (
        frozenset({"SPLIT.NUMERIC_DISTRIBUTION_SHIFT"})
        if mild_smd is not None and mild_smd >= 0.8
        else frozenset()
    )
    return [
        Case(
            "exact_row_and_group_copy",
            "split",
            frozenset({"SPLIT.EXACT_ROW_OVERLAP", "SPLIT.GROUP_OVERLAP"}),
            "The first four test rows are copied from train, including customer_id, so both the row fingerprint and the group overlap.",
            True,
            lambda: audit_split(
                train,
                copied_test,
                target="churn",
                task="classification",
                group_column="customer_id",
                config=SMALL_SPLIT,
            ),
        ),
        Case(
            "group_overlap_without_row_copy",
            "split",
            frozenset({"SPLIT.GROUP_OVERLAP"}),
            "customer_id values match, while age is shifted by 1 and churn is flipped, so the full rows differ.",
            True,
            lambda: audit_split(
                group_train,
                group_test,
                target="churn",
                task="classification",
                group_column="customer_id",
                config=SMALL_SPLIT,
            ),
        ),
        Case(
            "reversed_time_order",
            "split",
            frozenset({"SPLIT.TIME_ORDER_VIOLATION"}),
            "Every test date is in 2023 and every training date is in 2024, so the test is before the training maximum.",
            True,
            lambda: audit_split(
                later_train,
                early_test,
                target="churn",
                task="classification",
                time_column="event_date",
                split_strategy="temporal",
                id_columns=["customer_id"],
                config=SMALL_SPLIT,
            ),
        ),
        Case(
            "ordered_time_split",
            "split",
            frozenset(),
            "Test dates start on 2024-03-01, after the training window that starts on 2024-01-01 and lasts 8 days.",
            True,
            lambda: audit_split(
                later_train,
                later_test,
                target="churn",
                task="classification",
                time_column="event_date",
                split_strategy="temporal",
                id_columns=["customer_id"],
                config=SMALL_SPLIT,
            ),
        ),
        Case(
            "schema_mismatch",
            "split",
            frozenset({"SPLIT.SCHEMA_MISMATCH"}),
            "The test frame drops age. Exact overlap is not labeled because the schemas differ.",
            True,
            lambda: audit_split(
                wide_train,
                wide_test,
                target="churn",
                task="classification",
                config=SMALL_SPLIT,
            ),
        ),
        Case(
            "duplicate_inside_train_only",
            "split",
            frozenset({"DATA.DUPLICATE_ROWS"}),
            "Train row 1 copies train row 0. Test customer ids and dates are different, so this is not a cross-partition copy.",
            True,
            lambda: audit_split(
                duplicate_train,
                duplicate_test,
                target="churn",
                task="classification",
                group_column="customer_id",
                config=SMALL_SPLIT,
            ),
            forbid_leakage_claim=True,
        ),
        Case(
            "large_numeric_location_shift",
            "split",
            shift_expected,
            (
                "Labeling rule: flag SPLIT.NUMERIC_DISTRIBUTION_SHIFT when the pooled "
                f"absolute SMD is at least 0.8. Independent calculation gave {labeled_smd}."
            ),
            False,
            lambda: audit_split(
                _shift_frame(shift_train_age, 0),
                _shift_frame(shift_test_age, 1000),
                target="churn",
                task="classification",
                id_columns=["row_id"],
                group_column="customer_id",
            ),
        ),
        Case(
            "mild_numeric_location_shift",
            "split",
            mild_expected,
            (
                "Labeling rule: the same 0.8 pooled absolute SMD cutoff. "
                f"Independent calculation gave {mild_smd}."
            ),
            False,
            lambda: audit_split(
                _shift_frame(mild_train_age, 0),
                _shift_frame(mild_test_age, 1000),
                target="churn",
                task="classification",
                id_columns=["row_id"],
                group_column="customer_id",
            ),
        ),
    ]


def _shift_frame(age: np.ndarray, start: int) -> pd.DataFrame:
    n = int(age.size)
    return pd.DataFrame(
        {
            "row_id": np.arange(start, start + n),
            "customer_id": [f"c{index:04d}" for index in range(start, start + n)],
            "age": age,
            "churn": [0 if index < (n // 2) else 1 for index in range(n)],
        }
    )


def _class_shift_case() -> Case:
    n = 80
    train_churn = [0] * 40 + [1] * 40
    test_churn = [0] * 64 + [1] * 16
    train_fraction = train_churn.count(1) / n
    test_fraction = test_churn.count(1) / n
    difference = abs(train_fraction - test_fraction)
    expected = frozenset({"SPLIT.CLASS_DISTRIBUTION_SHIFT"}) if difference > 0.1 else frozenset()
    train = _shift_frame(np.linspace(0.0, 1.0, n), 0)
    test = _shift_frame(np.linspace(0.0, 1.0, n), 1000)
    train["churn"] = train_churn
    test["churn"] = test_churn
    return Case(
        "class_proportion_difference",
        "split",
        expected,
        (
            "Labeling rule: flag when the maximum absolute class-proportion difference "
            f"is greater than 0.1. The positive-class fractions are {train_fraction} and "
            f"{test_fraction}, so the difference is {difference}."
        ),
        False,
        lambda: audit_split(
            train,
            test,
            target="churn",
            task="classification",
            id_columns=["row_id"],
            group_column="customer_id",
        ),
    )


def _prediction_cases() -> list[Case]:
    good_true = [index % 2 for index in range(40)]
    good_pred = list(good_true)
    one_class = [0] * 40
    return [
        Case(
            "known_good_predictions",
            "predictions",
            frozenset(),
            "Forty balanced rows, matching labels, and no scores. The sample meets the default size of 30.",
            True,
            lambda: audit_predictions(good_true, good_pred, task="classification"),
        ),
        Case(
            "one_class_evaluation",
            "predictions",
            frozenset({"METRIC.ACCURACY_ONLY_IMBALANCED", "METRIC.INVALID_AUC_INPUT"}),
            "y_true has one class and a one-dimensional score was supplied, so AUC is undefined and accuracy is not a class-wise summary.",
            True,
            lambda: audit_predictions(
                one_class, one_class, task="classification", y_score=[0.2] * 40
            ),
        ),
        Case(
            "small_evaluation_sample",
            "predictions",
            frozenset({"METRIC.SMALL_EVALUATION_SAMPLE"}),
            "Five rows are below the default minimum of 30. Both classes are present and no scores are supplied.",
            True,
            lambda: audit_predictions([0, 1, 0, 1, 0], [0, 1, 1, 1, 0], task="classification"),
        ),
    ]


def _invalid_shape() -> dict[str, Any]:
    try:
        audit_predictions(
            [0, 1] * 5,
            [0, 1] * 5,
            task="classification",
            y_score=np.zeros((10, 3)),
        )
    except MlpipelinecheckInputError as exc:
        return {"name": "invalid_probability_shape", "passed": True, "actual": type(exc).__name__}
    return {"name": "invalid_probability_shape", "passed": False, "actual": "no exception"}


def _score(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0}
    )
    for case in cases:
        universe = case["rules_checked"]
        for rule_id in universe:
            bucket = counts[rule_id]
            bucket["n"] += 1
            expected = rule_id in case["expected"]
            actual = rule_id in case["actual"]
            if expected and actual:
                bucket["tp"] += 1
            elif actual and not expected:
                bucket["fp"] += 1
            elif expected and not actual:
                bucket["fn"] += 1
            else:
                bucket["tn"] += 1
    rules = {}
    for rule_id, bucket in sorted(counts.items()):
        tp, fp, fn, tn = bucket["tp"], bucket["fp"], bucket["fn"], bucket["tn"]
        rules[rule_id] = {
            **bucket,
            "kind": "heuristic" if rule_id in HEURISTIC_RULES else "structural",
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if tp + fn == 0 else tp / (tp + fn),
            "false_positive_rate": None if fp + tn == 0 else fp / (fp + tn),
        }
    return rules


def run_heldout() -> dict[str, Any]:
    """Run the held-out cases and score them against the predeclared labels."""
    cases = _dataset_cases() + _split_cases() + [_class_shift_case()] + _prediction_cases()
    rows: list[dict[str, Any]] = []
    structural_failures: list[str] = []
    for case in cases:
        report = case.audit()
        actual = frozenset(finding.id for finding in report.findings)
        passed = actual == case.expected
        leakage_ok = True
        if case.forbid_leakage_claim:
            for finding in report.findings:
                if finding.id not in case.expected:
                    continue
                text = finding.limitations.lower()
                if "proof of leakage" not in text or "not" not in text:
                    leakage_ok = False
        if case.structural and (not passed or not leakage_ok):
            structural_failures.append(case.name)
        rows.append(
            {
                "name": case.name,
                "family": case.family,
                "structural_case": case.structural,
                "expected": sorted(case.expected),
                "actual": sorted(actual),
                "passed": passed and leakage_ok,
                "rationale": case.rationale,
                "coverage_gap": case.coverage_gap,
                "rules_checked": [check.id for check in report.checks],
            }
        )
    rules = _score(rows)
    structural_rules = {key: value for key, value in rules.items() if value["kind"] == "structural"}
    heuristic_rules = {key: value for key, value in rules.items() if value["kind"] == "heuristic"}
    return {
        "package_version": __version__,
        "seed": SEED,
        "scope": (
            "Held-out synthetic fixtures. Labels were assigned from the construction "
            "and from the documented thresholds before the audit result was read. "
            "This is not a measure of performance on field data."
        ),
        "cases": rows,
        "rules": rules,
        "structural_micro": _micro(structural_rules),
        "heuristic_micro": _micro(heuristic_rules),
        "structural_failures": structural_failures,
        "validation": [_invalid_shape()],
        "not_covered": [
            "Near-duplicate rows and fuzzy or affine target copies are coverage gaps when the exact-copy rule is silent.",
            "Grouped and temporal uncertainty. Those intervals are not implemented.",
            "Real-world tables, adversarial users, and industries outside this list.",
            "Whether a distribution shift is harmful. The numeric labels only say whether the documented SMD cutoff was crossed.",
        ],
    }


def _micro(rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tp = sum(item["tp"] for item in rules.values())
    fp = sum(item["fp"] for item in rules.values())
    fn = sum(item["fn"] for item in rules.values())
    tn = sum(item["tn"] for item in rules.values())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": tp + fp + fn + tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "false_positive_rate": None if fp + tn == 0 else fp / (fp + tn),
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.3f}"


def format_markdown(result: dict[str, Any]) -> str:
    """Render the held-out score without filling undefined metrics."""
    lines = [
        "# Mlpipelinecheck held-out fixtures",
        "",
        f"Package {result['package_version']}. Seed {result['seed']}.",
        "",
        result["scope"],
        "",
        "## Cases",
        "",
        "| Case | Structural label | Passed | Expected | Actual |",
        "|---|---|---|---|---|",
    ]
    for case in result["cases"]:
        expected = ", ".join(case["expected"]) or "none"
        actual = ", ".join(case["actual"]) or "none"
        lines.append(
            f"| {case['name']} | {case['structural_case']} | {case['passed']} | {expected} | {actual} |"
        )
    lines.extend(
        [
            "",
            "## Per-rule counts",
            "",
            "| Rule | Kind | n | TP | FP | FN | TN | Precision | Recall | FPR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rule_id, stats in result["rules"].items():
        lines.append(
            f"| {rule_id} | {stats['kind']} | {stats['n']} | {stats['tp']} | {stats['fp']} | "
            f"{stats['fn']} | {stats['tn']} | {_fmt(stats['precision'])} | {_fmt(stats['recall'])} | "
            f"{_fmt(stats['false_positive_rate'])} |"
        )
    for label, key in (("Structural", "structural_micro"), ("Heuristic", "heuristic_micro")):
        stats = result[key]
        lines.extend(
            [
                "",
                (
                    f"{label} micro: n={stats['n']}, tp={stats['tp']}, fp={stats['fp']}, "
                    f"fn={stats['fn']}, tn={stats['tn']}, precision={_fmt(stats['precision'])}, "
                    f"recall={_fmt(stats['recall'])}, false-positive rate={_fmt(stats['false_positive_rate'])}."
                ),
            ]
        )
    lines.extend(["", "## Coverage notes", ""])
    for case in result["cases"]:
        if case["coverage_gap"]:
            lines.append(f"- {case['name']}: {case['coverage_gap']}")
    lines.extend(["", "## Not covered", ""])
    for item in result["not_covered"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the held-out fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run_heldout()
    text = format_markdown(result)
    print(text)
    if args.write:
        directory = Path(__file__).resolve().parent
        (directory / "heldout_RESULTS.md").write_text(text, encoding="utf-8")
        (directory / "heldout_results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
