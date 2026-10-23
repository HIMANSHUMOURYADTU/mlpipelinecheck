"""Dataset audit tests, including benign cases and edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlpipelinecheck import AuditConfig, audit_dataset
from mlpipelinecheck.exceptions import MlpipelinecheckInputError
from mlpipelinecheck.findings import CheckStatus


def _ids(report) -> set[str]:
    return {finding.id for finding in report.findings}


def _status(report, rule_id: str) -> str:
    return next(check.status.value for check in report.checks if check.id == rule_id)


def test_balanced_frame_has_no_findings_and_does_not_mutate() -> None:
    frame = pd.DataFrame(
        {
            "age": [25, 40, 33, 52],
            "plan": ["a", "b", "a", "b"],
            "churn": [0, 1, 0, 1],
        }
    )
    original = frame.copy(deep=True)
    report = audit_dataset(frame, target="churn", task="classification")
    pd.testing.assert_frame_equal(original, frame)
    assert report.summary()["status"] == "no configured issues found"
    assert _ids(report) == set()
    assert "does not mean" in report.summary()["disclaimer"]


def test_empty_and_no_feature_columns() -> None:
    empty = pd.DataFrame({"age": pd.Series(dtype="float"), "churn": pd.Series(dtype="int")})
    report = audit_dataset(empty, target="churn", task="classification")
    assert "DATA.EMPTY_DATASET" in _ids(report)
    assert _status(report, "DATA.MISSING_VALUES") == CheckStatus.NOT_APPLICABLE.value

    target_only = pd.DataFrame({"churn": [0, 1, 0, 1]})
    only = audit_dataset(target_only, target="churn", task="classification")
    assert "DATA.EMPTY_DATASET" in _ids(only)
    assert "no_feature_columns" in next(
        finding.evidence["reasons"]
        for finding in only.findings
        if finding.id == "DATA.EMPTY_DATASET"
    )


def test_missing_threshold_is_strictly_greater_than() -> None:
    frame = pd.DataFrame({"age": [1.0, None], "churn": [0, 1]})
    at_half = audit_dataset(
        frame,
        target="churn",
        task="classification",
        config=AuditConfig(missing_fraction_threshold=0.5),
    )
    assert "DATA.MISSING_VALUES" not in _ids(at_half)
    above = audit_dataset(
        frame,
        target="churn",
        task="classification",
        config=AuditConfig(missing_fraction_threshold=0.49),
    )
    assert "DATA.MISSING_VALUES" in _ids(above)
    all_null = pd.DataFrame({"age": [None, None, None], "churn": [0, 1, 0]})
    report = audit_dataset(all_null, target="churn", task="classification")
    finding = next(item for item in report.findings if item.id == "DATA.MISSING_VALUES")
    assert finding.evidence["flagged"]["columns"][0]["all_null"] is True
    assert "DATA.CONSTANT_FEATURE" not in _ids(report)


def test_duplicate_rows_ignore_index_and_match_missing_tokens() -> None:
    frame = pd.DataFrame(
        {"age": [1.0, 1.0, np.nan, 2.0], "city": ["a", "a", None, "b"], "churn": [0, 0, 1, 1]},
        index=[5, 5, 9, 9],
    )
    report = audit_dataset(frame, target="churn", task="classification")
    finding = next(item for item in report.findings if item.id == "DATA.DUPLICATE_ROWS")
    assert finding.evidence["n_extra_copies"] == 1
    assert finding.evidence["index_used"] is False
    distinct = pd.DataFrame({"age": [1, 2, 3], "churn": [0, 1, 0]}, index=[1, 1, 1])
    assert "DATA.DUPLICATE_ROWS" not in _ids(
        audit_dataset(distinct, target="churn", task="classification")
    )


def test_unhashable_duplicate_rows_are_detected_and_unsupported_values_error() -> None:
    lists = pd.DataFrame({"items": [[1], [1], [2]], "churn": [0, 0, 1]})
    report = audit_dataset(lists, target="churn", task="classification")
    assert (
        next(
            item.evidence["n_extra_copies"]
            for item in report.findings
            if item.id == "DATA.DUPLICATE_ROWS"
        )
        == 1
    )

    class Widget:
        pass

    broken = pd.DataFrame({"item": [Widget(), Widget()], "churn": [0, 1]})
    failed = audit_dataset(broken, target="churn", task="classification")
    assert _status(failed, "DATA.DUPLICATE_ROWS") == CheckStatus.ERROR.value
    assert failed.summary()["status"] == "audit incomplete"
    assert "DATA.DUPLICATE_ROWS" not in _ids(failed)


def test_constant_near_constant_and_benign_variation() -> None:
    frame = pd.DataFrame(
        {
            "fixed": ["x"] * 200,
            "almost": ["a"] * 199 + ["b"],
            "varied": ["a", "b"] * 100,
            "churn": [0, 1] * 100,
        }
    )
    report = audit_dataset(frame, target="churn", task="classification")
    finding = next(item for item in report.findings if item.id == "DATA.CONSTANT_FEATURE")
    assert finding.evidence["constant_columns"]["columns"][0]["column"] == "fixed"
    assert finding.severity.value == "low"
    near_only = frame.drop(columns=["fixed"])
    near = audit_dataset(near_only, target="churn", task="classification")
    near_finding = next(item for item in near.findings if item.id == "DATA.CONSTANT_FEATURE")
    assert near_finding.severity.value == "info"
    assert near_finding.evidence["constant_columns"]["n_columns"] == 0
    balanced = pd.DataFrame({"varied": ["a", "b"] * 10, "churn": [0, 1] * 10})
    assert "DATA.CONSTANT_FEATURE" not in _ids(
        audit_dataset(
            balanced,
            target="churn",
            task="classification",
            config=AuditConfig(near_constant_fraction=None),
        )
    )


def test_high_cardinality_ignores_floats_and_declared_ids() -> None:
    rows = 30
    frame = pd.DataFrame(
        {
            "measurement": np.linspace(0.0, 1.0, rows),
            "user_id": [f"u{i}" for i in range(rows)],
            "plan": ["a", "b", "c"] * 10,
            "churn": [0, 1] * 15,
        }
    )
    benign = audit_dataset(frame, target="churn", task="classification", id_columns=["user_id"])
    assert "DATA.HIGH_CARDINALITY_FEATURE" not in _ids(benign)
    flagged = audit_dataset(
        frame.drop(columns=["user_id"]).assign(code=[f"c{i}" for i in range(rows)]),
        target="churn",
        task="classification",
    )
    assert "DATA.HIGH_CARDINALITY_FEATURE" in _ids(flagged)
    finding = next(item for item in flagged.findings if item.id == "DATA.HIGH_CARDINALITY_FEATURE")
    assert finding.confidence.value == "medium"
    assert "not proof" in finding.limitations.lower() or "not proof" in finding.limitations


def test_target_in_features_and_exact_copy_rules() -> None:
    frame = pd.DataFrame(
        {
            "age": [1, 2, 3, 4],
            "churn": [0, 1, 0, 1],
            "churn_copy": [0.0, 1.0, 0.0, 1.0],
            "label_text": ["0", "1", "0", "1"],
        }
    )
    copied = audit_dataset(frame, target="churn", task="classification")
    assert "DATA.TARGET_COPY_FEATURE" in _ids(copied)
    assert "DATA.TARGET_IN_FEATURES" not in _ids(copied)
    assert "label_text" not in str(
        next(
            item.evidence["copied_columns"]
            for item in copied.findings
            if item.id == "DATA.TARGET_COPY_FEATURE"
        )
    )
    listed = audit_dataset(
        frame.drop(columns=["churn_copy"]),
        target="churn",
        task="classification",
        feature_columns=["age", "churn", "label_text"],
    )
    assert "DATA.TARGET_IN_FEATURES" in _ids(listed)
    missing = pd.DataFrame({"churn": [1, None, 0], "mirror": [1, None, 0]})
    assert "DATA.TARGET_COPY_FEATURE" in _ids(
        audit_dataset(missing, target="churn", task="classification")
    )


def test_class_imbalance_one_class_and_balanced_negative() -> None:
    imbalanced = pd.DataFrame({"age": range(20), "churn": [0] * 18 + [1, 1]})
    report = audit_dataset(imbalanced, target="churn", task="classification")
    finding = next(item for item in report.findings if item.id == "DATA.CLASS_IMBALANCE")
    assert finding.severity.value == "medium"
    assert finding.evidence["single_class"] is False
    one = pd.DataFrame({"age": [1, 2, 3], "churn": [1, 1, 1]})
    one_report = audit_dataset(one, target="churn", task="classification")
    one_finding = next(item for item in one_report.findings if item.id == "DATA.CLASS_IMBALANCE")
    assert one_finding.severity.value == "high"
    balanced = pd.DataFrame({"age": [1, 2, 3, 4], "churn": [0, 0, 1, 1]})
    assert "DATA.CLASS_IMBALANCE" not in _ids(
        audit_dataset(balanced, target="churn", task="classification")
    )
    regression = pd.DataFrame({"age": [1, 2, 3, 4], "y": [0.1, 0.2, 0.3, 0.4]})
    assert _status(
        audit_dataset(regression, target="y", task="regression"), "DATA.CLASS_IMBALANCE"
    ) == (CheckStatus.NOT_APPLICABLE.value)


def test_non_finite_values_and_safe_means() -> None:
    frame = pd.DataFrame({"age": [1.0, np.inf, -np.inf, 3.0], "churn": [0, 1, 0, 1]})
    report = audit_dataset(frame, target="churn", task="classification")
    finding = next(item for item in report.findings if item.id == "DATA.NON_FINITE_VALUES")
    column = finding.evidence["flagged"]["columns"][0]
    assert column["n_positive_inf"] == 1
    assert column["n_negative_inf"] == 1
    assert report.metadata["column_profiles"]["age"]["mean"] == pytest.approx(2.0)
    finite = pd.DataFrame({"age": [1.0, 2.0, None], "churn": [0, 1, 0]})
    assert "DATA.NON_FINITE_VALUES" not in _ids(
        audit_dataset(finite, target="churn", task="classification")
    )


def test_ambiguous_task_is_not_guessed_and_inputs_are_validated() -> None:
    frame = pd.DataFrame({"age": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 1.0, 2.0, 3.0]})
    report = audit_dataset(frame, target="y", task="auto")
    assert report.metadata["task"] is None
    assert report.metadata["task_source"] == "not_inferred"
    assert _status(report, "DATA.CLASS_IMBALANCE") == CheckStatus.NOT_APPLICABLE.value
    with pytest.raises(MlpipelinecheckInputError, match="Missing target column"):
        audit_dataset(frame, target="missing")
    with pytest.raises(MlpipelinecheckInputError, match="DataFrame"):
        audit_dataset([1, 2, 3], target="y")  # type: ignore[arg-type]
    duplicated = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(MlpipelinecheckInputError, match="duplicate column"):
        audit_dataset(duplicated)
    with pytest.raises(MlpipelinecheckInputError, match="regression"):
        audit_dataset(pd.DataFrame({"y": ["a", "b"]}), target="y", task="regression")


def test_disabled_check_is_skipped_not_passed() -> None:
    frame = pd.DataFrame({"age": [1.0, None], "churn": [0, 1]})
    report = audit_dataset(
        frame,
        target="churn",
        task="classification",
        config=AuditConfig(disabled_checks=["DATA.MISSING_VALUES"]),
    )
    assert "DATA.MISSING_VALUES" not in _ids(report)
    assert _status(report, "DATA.MISSING_VALUES") == CheckStatus.NOT_APPLICABLE.value


def test_max_rows_skips_expensive_checks_without_passing_them() -> None:
    frame = pd.DataFrame({"age": [1, 1, 1], "churn": [0, 0, 1]})
    report = audit_dataset(
        frame, target="churn", task="classification", config=AuditConfig(max_rows=2)
    )
    assert _status(report, "DATA.DUPLICATE_ROWS") == CheckStatus.NOT_APPLICABLE.value
    assert (
        "not a pass"
        in next(
            check.summary for check in report.checks if check.id == "DATA.DUPLICATE_ROWS"
        ).lower()
    )


def test_mixed_object_column_does_not_crash() -> None:
    frame = pd.DataFrame({"mixed": [1, "a", None], "churn": [0, 1, 0]})
    report = audit_dataset(frame, target="churn", task="classification")
    assert report.audit_type == "dataset"
    assert _status(report, "DATA.MISSING_VALUES") == CheckStatus.DETECTED.value


def test_target_copy_ignores_identifiers_and_unselected_columns() -> None:
    identified = pd.DataFrame({"row_id": [0, 1, 0, 1], "age": [3, 4, 5, 6], "churn": [0, 1, 0, 1]})
    report = audit_dataset(identified, target="churn", task="classification", id_columns=["row_id"])
    assert "DATA.TARGET_COPY_FEATURE" not in _ids(report)
    assert report.metadata["feature_columns"] == ["age"]
    excluded = pd.DataFrame(
        {"age": [1, 2, 3, 4], "churn": [0, 1, 0, 1], "churn_copy": [0, 1, 0, 1]}
    )
    narrowed = audit_dataset(
        excluded, target="churn", task="classification", feature_columns=["age"]
    )
    assert "DATA.TARGET_COPY_FEATURE" not in _ids(narrowed)
