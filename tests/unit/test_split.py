"""Split audit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlpipelinecheck import AuditConfig, audit_split
from mlpipelinecheck.exceptions import MlpipelinecheckInputError
from mlpipelinecheck.findings import CheckStatus


def _cfg(**kwargs: object) -> AuditConfig:
    base = dict(small_test_rows=1, class_shift_min_rows=4, numeric_shift_min_rows=4)
    base.update(kwargs)
    return AuditConfig(**base)  # type: ignore[arg-type]


def _ids(report) -> set[str]:
    return {finding.id for finding in report.findings}


def _status(report, rule_id: str) -> str:
    return next(check.status.value for check in report.checks if check.id == rule_id)


def _pair(n_train: int = 40, n_test: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "age": np.linspace(20, 60, n_train),
            "plan": ["a", "b"] * (n_train // 2),
            "customer_id": np.arange(n_train),
            "churn": [0, 1] * (n_train // 2),
        }
    )
    test = pd.DataFrame(
        {
            "age": np.linspace(20, 60, n_test),
            "plan": ["a", "b"] * (n_test // 2),
            "customer_id": np.arange(n_train, n_train + n_test),
            "churn": [0, 1] * (n_test // 2),
        }
    )
    return train, test


def test_clean_split_has_no_high_confidence_overlap() -> None:
    train, test = _pair()
    original_train = train.copy(deep=True)
    original_test = test.copy(deep=True)
    report = audit_split(
        train,
        test,
        target="churn",
        task="classification",
        group_column="customer_id",
        id_columns=["customer_id"],
        config=_cfg(),
    )
    pd.testing.assert_frame_equal(original_train, train)
    pd.testing.assert_frame_equal(original_test, test)
    assert "SPLIT.EXACT_ROW_OVERLAP" not in _ids(report)
    assert "SPLIT.GROUP_OVERLAP" not in _ids(report)
    assert "SPLIT.SCHEMA_MISMATCH" not in _ids(report)
    assert _status(report, "SPLIT.TIME_ORDER_VIOLATION") == CheckStatus.NOT_APPLICABLE.value


def test_exact_overlap_ignores_index_and_treats_missing_values_as_equal() -> None:
    train = pd.DataFrame({"age": [1.0, np.nan], "churn": [0, 1]}, index=[7, 7])
    test = pd.DataFrame({"age": [np.nan, 3.0], "churn": [1, 0]}, index=[7, 1])
    report = audit_split(train, test, target="churn", task="classification", config=_cfg())
    finding = next(item for item in report.findings if item.id == "SPLIT.EXACT_ROW_OVERLAP")
    assert finding.evidence["n_shared_fingerprints"] == 1
    assert finding.evidence["index_used"] is False
    text_mismatch = audit_split(
        pd.DataFrame({"age": [1], "churn": [0]}),
        pd.DataFrame({"age": ["1"], "churn": [0]}),
        target="churn",
        task="classification",
        config=_cfg(),
    )
    assert "SPLIT.EXACT_ROW_OVERLAP" not in _ids(text_mismatch)


def test_integer_and_integer_valued_float_rows_overlap() -> None:
    train = pd.DataFrame({"age": [1, 2], "churn": [0, 1]})
    test = pd.DataFrame({"age": [1.0, 9.0], "churn": [0, 1]})
    report = audit_split(train, test, target="churn", task="classification", config=_cfg())
    assert "SPLIT.EXACT_ROW_OVERLAP" in _ids(report)
    assert "SPLIT.SCHEMA_MISMATCH" not in _ids(report)


def test_schema_mismatch_blocks_overlap_claim() -> None:
    train = pd.DataFrame({"age": [1, 2], "churn": [0, 1]})
    test = pd.DataFrame({"age": [1, 2], "extra": [3, 4], "churn": [0, 1]})
    report = audit_split(train, test, target="churn", task="classification", config=_cfg())
    assert "SPLIT.SCHEMA_MISMATCH" in _ids(report)
    assert _status(report, "SPLIT.EXACT_ROW_OVERLAP") == CheckStatus.NOT_APPLICABLE.value
    incompatible = audit_split(
        pd.DataFrame({"age": [1, 2], "churn": [0, 1]}),
        pd.DataFrame({"age": ["1", "2"], "churn": [0, 1]}),
        target="churn",
        task="classification",
        config=_cfg(),
    )
    assert "SPLIT.SCHEMA_MISMATCH" in _ids(incompatible)


def test_group_overlap_counts_ids_without_copying_them() -> None:
    train = pd.DataFrame(
        {
            "age": [1, 2, 3, 4],
            "customer_id": ["CUSTOMER_SECRET_123", "b", "c", "d"],
            "churn": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "age": [9, 8, 7, 6],
            "customer_id": ["CUSTOMER_SECRET_123", "e", "f", "g"],
            "churn": [1, 0, 1, 0],
        }
    )
    report = audit_split(
        train,
        test,
        target="churn",
        task="classification",
        group_column="customer_id",
        config=_cfg(),
    )
    finding = next(item for item in report.findings if item.id == "SPLIT.GROUP_OVERLAP")
    assert finding.evidence["n_overlapping_groups"] == 1
    assert "CUSTOMER_SECRET_123" not in report.to_json()
    assert "SPLIT.EXACT_ROW_OVERLAP" not in _ids(report)
    with pytest.raises(MlpipelinecheckInputError, match="group column"):
        audit_split(train, test, group_column="missing", config=_cfg())
    skipped = audit_split(train, test, target="churn", task="classification", config=_cfg())
    assert _status(skipped, "SPLIT.GROUP_OVERLAP") == CheckStatus.NOT_APPLICABLE.value


def test_temporal_policy_and_missing_time_column() -> None:
    train = pd.DataFrame(
        {
            "age": np.linspace(0, 1, 8),
            "event_time": pd.date_range("2024-01-01", periods=8, freq="D"),
            "churn": [0, 1] * 4,
        }
    )
    test = train.copy()
    test["event_time"] = pd.date_range("2024-01-05", periods=8, freq="D")
    test["age"] = np.linspace(0, 1, 8)
    violation = audit_split(
        train,
        test,
        target="churn",
        task="classification",
        time_column="event_time",
        split_strategy="temporal",
        config=_cfg(class_shift_min_rows=4),
    )
    assert "SPLIT.TIME_ORDER_VIOLATION" in _ids(violation)
    later = test.copy()
    later["event_time"] = pd.date_range("2024-02-01", periods=8, freq="D")
    clean = audit_split(
        train,
        later,
        target="churn",
        task="classification",
        time_column="event_time",
        config=_cfg(),
    )
    assert "SPLIT.TIME_ORDER_VIOLATION" not in _ids(clean)
    with pytest.raises(MlpipelinecheckInputError, match="time_column"):
        audit_split(
            train.drop(columns=["event_time"]),
            later.drop(columns=["event_time"]),
            split_strategy="temporal",
        )
    with pytest.raises(MlpipelinecheckInputError, match="epochs"):
        audit_split(
            pd.DataFrame({"when": [1, 2, 3, 4]}),
            pd.DataFrame({"when": [5, 6, 7, 8]}),
            time_column="when",
            config=_cfg(),
        )


def test_class_and_numeric_shift_are_low_confidence_signals() -> None:
    train = pd.DataFrame({"age": np.linspace(0, 1, 40), "churn": [0, 1] * 20})
    mild = pd.DataFrame({"age": np.linspace(0.05, 1.05, 40), "churn": [0] * 20 + [1] * 20})
    benign = audit_split(
        train, mild, target="churn", task="classification", config=_cfg(numeric_shift_min_rows=30)
    )
    assert "SPLIT.NUMERIC_DISTRIBUTION_SHIFT" not in _ids(benign)
    assert "SPLIT.CLASS_DISTRIBUTION_SHIFT" not in _ids(benign)
    shifted = pd.DataFrame({"age": np.linspace(5, 6, 40), "churn": [0] * 36 + [1] * 4})
    report = audit_split(
        train,
        shifted,
        target="churn",
        task="classification",
        config=_cfg(numeric_shift_min_rows=30, class_shift_min_rows=30),
    )
    numeric = next(
        item for item in report.findings if item.id == "SPLIT.NUMERIC_DISTRIBUTION_SHIFT"
    )
    classes = next(item for item in report.findings if item.id == "SPLIT.CLASS_DISTRIBUTION_SHIFT")
    assert numeric.severity.value == "low"
    assert numeric.confidence.value == "medium"
    assert (
        "not proof of leakage" in numeric.summary.lower()
        or "not proof of leakage" in numeric.limitations.lower()
    )
    assert classes.severity.value == "low"
    tiny = audit_split(
        train.head(5),
        mild.head(5),
        target="churn",
        task="classification",
        config=AuditConfig(small_test_rows=1, class_shift_min_rows=30, numeric_shift_min_rows=30),
    )
    assert _status(tiny, "SPLIT.CLASS_DISTRIBUTION_SHIFT") == CheckStatus.NOT_APPLICABLE.value
    assert _status(tiny, "SPLIT.NUMERIC_DISTRIBUTION_SHIFT") == CheckStatus.NOT_APPLICABLE.value


def test_small_test_duplicate_rows_and_target_problems() -> None:
    train = pd.DataFrame({"age": [1, 1, 2, 3], "copied": [0, 0, 1, 0], "churn": [0, 0, 1, 0]})
    test = pd.DataFrame({"age": [4], "copied": [1], "churn": [1]})
    report = audit_split(train, test, target="churn", task="classification")
    assert "SPLIT.SMALL_TEST_PARTITION" in _ids(report)
    assert "DATA.DUPLICATE_ROWS" in _ids(report)
    copied = audit_split(
        train.assign(mirror=train["churn"]),
        test.assign(mirror=test["churn"]),
        target="churn",
        task="classification",
        config=_cfg(),
    )
    assert "DATA.TARGET_COPY_FEATURE" in _ids(copied)
    missing_target = audit_split(
        pd.DataFrame({"age": [1, 2], "churn": [0, None]}),
        pd.DataFrame({"age": [3, 4], "churn": [1, 1]}),
        target="churn",
        task="classification",
        config=_cfg(missing_fraction_threshold=0.0),
    )
    assert "DATA.MISSING_VALUES" in _ids(missing_target)


def test_within_partition_duplicates_and_same_values_with_different_index() -> None:
    train = pd.DataFrame({"age": [1, 2], "churn": [0, 1]}, index=[0, 0])
    test = pd.DataFrame({"age": [3, 4], "churn": [1, 0]}, index=[0, 0])
    report = audit_split(train, test, target="churn", task="classification", config=_cfg())
    assert "SPLIT.EXACT_ROW_OVERLAP" not in _ids(report)
    assert "DATA.DUPLICATE_ROWS" not in _ids(report)


def test_object_numeric_time_values_are_not_parsed_as_epochs() -> None:
    train = pd.DataFrame(
        {"age": [1, 2], "when": pd.Series([2020, 2021], dtype=object), "churn": [0, 1]}
    )
    test = pd.DataFrame(
        {"age": [3, 4], "when": pd.Series([2019, 2018], dtype=object), "churn": [0, 1]}
    )
    with pytest.raises(MlpipelinecheckInputError, match="numeric"):
        audit_split(
            train, test, target="churn", task="classification", time_column="when", config=_cfg()
        )


def test_explicit_target_in_features_and_empty_frames() -> None:
    train = pd.DataFrame({"age": [1, 2, 3, 4], "churn": [0, 1, 0, 1]})
    test = pd.DataFrame({"age": [2, 3, 4, 5], "churn": [0, 1, 0, 1]})
    listed = audit_split(
        train,
        test,
        target="churn",
        task="classification",
        feature_columns=["age", "churn"],
        config=_cfg(),
    )
    assert "DATA.TARGET_IN_FEATURES" in _ids(listed)
    empty = audit_split(
        pd.DataFrame({"age": pd.Series(dtype="float"), "churn": pd.Series(dtype="int")}),
        pd.DataFrame({"age": pd.Series(dtype="float"), "churn": pd.Series(dtype="int")}),
        target="churn",
        task="classification",
        config=_cfg(),
    )
    assert empty.summary()["status"] in {
        "issues found",
        "no configured issues found",
        "audit incomplete",
    }
    assert _status(empty, "SPLIT.EXACT_ROW_OVERLAP") == CheckStatus.NOT_APPLICABLE.value
    with pytest.raises(MlpipelinecheckInputError, match="duplicate column"):
        audit_split(
            pd.DataFrame([[1, 0], [2, 1]], columns=["age", "age"]),
            test,
            target="churn",
            config=_cfg(),
        )


def test_partial_numeric_comparison_is_not_described_as_a_full_match() -> None:
    train = pd.DataFrame(
        {
            "age": [1.0, 2.0, 3.0, 4.0],
            "sparse": [1.0, np.nan, np.nan, np.nan],
            "churn": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "age": [1.0, 2.0, 3.0, 4.0],
            "sparse": [1.0, np.nan, np.nan, np.nan],
            "churn": [0, 1, 0, 1],
        }
    )
    report = audit_split(train, test, target="churn", task="classification", config=_cfg())
    check = next(item for item in report.checks if item.id == "SPLIT.NUMERIC_DISTRIBUTION_SHIFT")
    assert check.status is CheckStatus.NOT_DETECTED
    assert "not compared" in check.summary.lower()
