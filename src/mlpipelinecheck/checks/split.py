"""Split checks for explicit train and test partitions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from mlpipelinecheck._encoding import (
    UnsupportedCell,
    UnsupportedValue,
    cell_token,
    fingerprint_frame,
    is_missing,
)
from mlpipelinecheck._frame import dtype_group, dtypes_compatible, is_numeric_feature
from mlpipelinecheck.checks._recorder import CheckRecorder, make_finding
from mlpipelinecheck.checks.dataset import (
    _class_counts,
    _column_name,
    _float_array,
    _plain,
    _values_equal,
)
from mlpipelinecheck.config import AuditConfig
from mlpipelinecheck.exceptions import MlpipelinecheckInputError
from mlpipelinecheck.findings import Severity
from mlpipelinecheck.rules import SPLIT_RULE_IDS

_LIST_LIMIT = 50


def _over_limit(frame: pd.DataFrame, config: AuditConfig) -> bool:
    return config.max_rows is not None and len(frame) > int(config.max_rows)


def _limit_details(train: pd.DataFrame, test: pd.DataFrame, config: AuditConfig) -> dict[str, Any]:
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "max_rows": config.max_rows,
    }


def _schema(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, Any], bool]:
    missing_in_test = [
        _column_name(column) for column in train.columns if column not in test.columns
    ]
    missing_in_train = [
        _column_name(column) for column in test.columns if column not in train.columns
    ]
    incompatible: list[dict[str, str]] = []
    for column in train.columns:
        if column not in test.columns:
            continue
        if not dtypes_compatible(train[column], test[column]):
            incompatible.append(
                {
                    "column": _column_name(column),
                    "train_dtype": str(train[column].dtype),
                    "test_dtype": str(test[column].dtype),
                    "train_group": dtype_group(train[column]),
                    "test_group": dtype_group(test[column]),
                }
            )
    mismatch = bool(missing_in_test or missing_in_train or incompatible)
    evidence = {
        "missing_in_test": missing_in_test[:_LIST_LIMIT],
        "missing_in_train": missing_in_train[:_LIST_LIMIT],
        "n_missing_in_test": len(missing_in_test),
        "n_missing_in_train": len(missing_in_train),
        "incompatible_dtypes": incompatible[:_LIST_LIMIT],
        "n_incompatible_dtypes": len(incompatible),
        "names_truncated": len(missing_in_test) > _LIST_LIMIT
        or len(missing_in_train) > _LIST_LIMIT,
    }
    return evidence, mismatch


def _duplicate_stats(frame: pd.DataFrame) -> dict[str, Any]:
    fingerprints = fingerprint_frame(frame)
    n_rows = len(fingerprints)
    n_unique = len(set(fingerprints))
    n_extra = n_rows - n_unique
    return {
        "n_rows": n_rows,
        "n_unique_rows": n_unique,
        "n_extra_copies": n_extra,
        "duplicate_rate": (n_extra / n_rows) if n_rows else None,
    }


def _group_tokens(series: pd.Series) -> tuple[set[str] | None, int, str | None]:
    tokens: set[str] = set()
    n_missing = 0
    for value in series.tolist():
        if is_missing(value):
            n_missing += 1
            continue
        try:
            tokens.add(cell_token(value))
        except UnsupportedValue as exc:
            return None, n_missing, exc.type_name
    return tokens, n_missing, None


def _parse_time(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_bool_dtype(series):
        raise MlpipelinecheckInputError(
            f"Time column {label!r} is boolean. Convert it to datetimes before auditing."
        )
    if pd.api.types.is_numeric_dtype(series):
        raise MlpipelinecheckInputError(
            f"Time column {label!r} is numeric. Mlpipelinecheck does not interpret numbers as epochs. "
            "Convert the column to datetimes before auditing."
        )
    numeric_objects = 0
    for value in series.tolist():
        if is_missing(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            raise MlpipelinecheckInputError(
                f"Time column {label!r} contains boolean values. "
                "Convert it to datetimes before auditing."
            )
        if isinstance(value, (int, np.integer, float, np.floating)):
            numeric_objects += 1
    if numeric_objects:
        raise MlpipelinecheckInputError(
            f"Time column {label!r} contains {numeric_objects} numeric value(s) stored outside "
            "a numeric dtype. Mlpipelinecheck does not interpret numbers as epochs. "
            "Convert the column to datetimes before auditing."
        )
    converted = pd.to_datetime(series, errors="coerce")
    failed = series.notna() & converted.isna()
    if bool(failed.any()):
        raise MlpipelinecheckInputError(
            f"Time column {label!r} has {int(failed.sum())} non-missing values that "
            "could not be parsed as timestamps."
        )
    return converted


def _align_clocks(train_time: pd.Series, test_time: pd.Series) -> tuple[pd.Series, pd.Series]:
    train_tz = getattr(train_time.dtype, "tz", None)
    test_tz = getattr(test_time.dtype, "tz", None)
    if (train_tz is None) != (test_tz is None):
        raise MlpipelinecheckInputError(
            "Train and test time columns must both be timezone-aware or both be timezone-naive."
        )
    if train_tz is not None and test_tz is not None:
        return train_time.dt.tz_convert("UTC"), test_time.dt.tz_convert("UTC")
    return train_time, test_time


def _smd(left: np.ndarray, right: np.ndarray) -> float | None:
    n_left = int(left.size)
    n_right = int(right.size)
    if n_left < 2 or n_right < 2:
        return None
    var_left = float(np.var(left, ddof=1))
    var_right = float(np.var(right, ddof=1))
    pooled = ((n_left - 1) * var_left + (n_right - 1) * var_right) / (n_left + n_right - 2)
    if pooled <= 0 or not math.isfinite(pooled):
        return None
    return (float(np.mean(left)) - float(np.mean(right))) / math.sqrt(pooled)


def _finite_values(series: pd.Series) -> np.ndarray | None:
    array = _float_array(series)
    if array is None:
        return None
    return array[np.isfinite(array)]


def run_split_checks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str | None,
    task: str | None,
    task_source: str,
    explicit_features: list[str] | None,
    features: list[Any],
    config: AuditConfig,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Run split rules. Callers have already validated required columns."""
    recorder = CheckRecorder(config.disabled_checks)
    schema_evidence, schema_mismatch = _schema(train, test)
    metadata: dict[str, Any] = {
        "train": {"n_rows": int(len(train)), "n_columns": int(train.shape[1])},
        "test": {"n_rows": int(len(test)), "n_columns": int(test.shape[1])},
        "target": target,
        "task": task,
        "task_source": task_source,
        "group_column": config.group_column,
        "time_column": config.time_column,
        "split_strategy": config.split_strategy,
        "train_and_test_are_same_object": train is test,
        "row_fingerprint": {
            "algorithm": "sha256 of length-prefixed canonical cell tokens",
            "index_included": False,
            "column_order_ignored": True,
            "missing_values_equal": True,
            "integer_valued_floats_match_integers": True,
            "text_coerced_to_number": False,
            "evaluated_only_when_schemas_match": True,
        },
        "complexity_note": (
            "Exact overlap and within-partition duplicates store one digest per row. "
            "Memory grows with the number of rows. Group overlap stores one token per "
            "distinct non-missing group. Distribution checks use column aggregates and "
            "do not copy the full tables. Set max_rows to skip fingerprint checks."
        ),
    }
    _schema_check(recorder, schema_evidence, schema_mismatch)
    _overlap_check(recorder, train, test, config, schema_mismatch=schema_mismatch)
    _within_duplicates(recorder, train, test, config)
    _group_check(recorder, train, test, config)
    _time_check(recorder, train, test, config)
    _class_shift(recorder, train, test, target=target, task=task, config=config)
    _numeric_shift(
        recorder,
        train,
        test,
        target=target,
        explicit_features=explicit_features,
        config=config,
    )
    _small_test(recorder, train, test, config)
    _target_in_features(recorder, train, test, target=target, explicit_features=explicit_features)
    _target_copies(recorder, train, test, target=target, features=features, config=config)
    _target_missing(recorder, train, test, target=target, config=config)
    _one_class(recorder, train, test, target=target, task=task)
    seen = [check.id for check in recorder.checks]
    if seen != list(SPLIT_RULE_IDS):
        raise RuntimeError(f"Split checks were {seen}, expected {list(SPLIT_RULE_IDS)}.")
    return recorder.findings, recorder.checks, metadata


def _schema_check(recorder: CheckRecorder, evidence: dict[str, Any], mismatch: bool) -> None:
    rule_id = "SPLIT.SCHEMA_MISMATCH"
    if not recorder.enabled(rule_id):
        return
    if mismatch:
        recorder.detect(make_finding(rule_id, "Train and test schemas differ.", evidence))
        return
    recorder.pass_(rule_id, "Train and test column names and dtype groups match.", evidence)


def _overlap_check(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AuditConfig,
    *,
    schema_mismatch: bool,
) -> None:
    rule_id = "SPLIT.EXACT_ROW_OVERLAP"
    if not recorder.enabled(rule_id):
        return
    if len(train) == 0 or len(test) == 0:
        recorder.skip(
            rule_id,
            "Exact row overlap was not evaluated because one partition has no rows. This is not a pass.",
        )
        return
    if schema_mismatch:
        recorder.skip(
            rule_id,
            "Exact row overlap was not evaluated because the schemas differ. "
            "See SPLIT.SCHEMA_MISMATCH. This is not a claim that the rows are distinct.",
        )
        return
    if _over_limit(train, config) or _over_limit(test, config):
        recorder.skip(
            rule_id,
            "Exact row overlap was not run because a partition exceeds max_rows. This is not a pass.",
            _limit_details(train, test, config),
        )
        return
    try:
        train_fps = fingerprint_frame(train)
        test_fps = fingerprint_frame(test)
    except UnsupportedCell as exc:
        recorder.fail_check(
            rule_id,
            f"{exc} Overlap was not evaluated and is not a pass.",
            {"column": exc.column, "value_type": exc.type_name},
        )
        return
    train_set = set(train_fps)
    test_set = set(test_fps)
    shared = train_set & test_set
    n_test_rows = sum(1 for digest in test_fps if digest in train_set)
    n_train_rows = sum(1 for digest in train_fps if digest in test_set)
    evidence = {
        "n_shared_fingerprints": len(shared),
        "n_train_rows_with_a_test_copy": n_train_rows,
        "n_test_rows_with_a_train_copy": n_test_rows,
        "n_train_rows": len(train_fps),
        "n_test_rows": len(test_fps),
        "fraction_test_rows_overlapping": n_test_rows / len(test_fps),
        "index_used": False,
    }
    if shared:
        recorder.detect(
            make_finding(
                rule_id,
                f"{n_test_rows} test row(s) have the same fingerprint as a training row.",
                evidence,
            )
        )
        return
    recorder.pass_(rule_id, "No exact row fingerprint is shared by train and test.", evidence)


def _within_duplicates(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AuditConfig,
) -> None:
    rule_id = "DATA.DUPLICATE_ROWS"
    if not recorder.enabled(rule_id):
        return
    if len(train) == 0 or len(test) == 0 or train.shape[1] == 0 or test.shape[1] == 0:
        recorder.skip(
            rule_id, "Within-partition duplicates were not evaluated on an empty partition."
        )
        return
    if _over_limit(train, config) or _over_limit(test, config):
        recorder.skip(
            rule_id,
            "Within-partition duplicates were not run because a partition exceeds max_rows. "
            "This is not a pass.",
            _limit_details(train, test, config),
        )
        return
    partitions: list[dict[str, Any]] = []
    for name, frame in (("train", train), ("test", test)):
        try:
            stats = _duplicate_stats(frame)
        except UnsupportedCell as exc:
            recorder.fail_check(
                rule_id,
                f"{name} partition: {exc} Duplicate detection did not complete and is not a pass.",
                {"partition": name, "column": exc.column, "value_type": exc.type_name},
            )
            return
        partitions.append({"partition": name, **stats})
    evidence = {"partitions": partitions, "index_used": False}
    if any(item["n_extra_copies"] for item in partitions):
        recorder.detect(
            make_finding(
                rule_id,
                "At least one partition contains exact duplicate rows.",
                evidence,
            )
        )
        return
    recorder.pass_(rule_id, "Neither partition contains exact duplicate rows.", evidence)


def _group_check(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AuditConfig,
) -> None:
    rule_id = "SPLIT.GROUP_OVERLAP"
    if not recorder.enabled(rule_id):
        return
    if config.group_column is None:
        recorder.skip(
            rule_id,
            "No group column was supplied. Group independence was not checked and is not a pass.",
        )
        return
    column = config.group_column
    train_tokens, train_missing, train_error = _group_tokens(train[column])
    test_tokens, test_missing, test_error = _group_tokens(test[column])
    if train_error or test_error or train_tokens is None or test_tokens is None:
        recorder.fail_check(
            rule_id,
            "Group values could not be compared. Group overlap was not established and is not a pass.",
            {"column": column, "train_error": train_error, "test_error": test_error},
        )
        return
    if not train_tokens and not test_tokens:
        recorder.fail_check(
            rule_id,
            "The group column has no non-null values, so overlap cannot be assessed. This is not a pass.",
            {"column": column, "n_missing_train": train_missing, "n_missing_test": test_missing},
        )
        return
    n_overlap = len(train_tokens & test_tokens)
    evidence = {
        "column": column,
        "n_train_groups": len(train_tokens),
        "n_test_groups": len(test_tokens),
        "n_overlapping_groups": n_overlap,
        "fraction_test_groups_in_train": (n_overlap / len(test_tokens)) if test_tokens else None,
        "n_missing_train": train_missing,
        "n_missing_test": test_missing,
        "identifiers_included": False,
    }
    if n_overlap:
        recorder.detect(
            make_finding(
                rule_id,
                f"{n_overlap} group value(s) appear in both partitions.",
                evidence,
            )
        )
        return
    recorder.pass_(rule_id, "No non-missing group value appears in both partitions.", evidence)


def _time_check(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AuditConfig,
) -> None:
    rule_id = "SPLIT.TIME_ORDER_VIOLATION"
    if not recorder.enabled(rule_id):
        return
    if config.time_column is None:
        recorder.skip(
            rule_id,
            "No time column or temporal strategy was supplied. "
            "Temporal order was not checked and is not a pass.",
        )
        return
    column = config.time_column
    train_time, test_time = _align_clocks(
        _parse_time(train[column], f"train[{column}]"),
        _parse_time(test[column], f"test[{column}]"),
    )
    train_valid = train_time.dropna()
    test_valid = test_time.dropna()
    details = {
        "column": column,
        "policy": "test_not_before_train_max",
        "n_missing_train": int(train_time.isna().sum()),
        "n_missing_test": int(test_time.isna().sum()),
        "n_train_timestamps": int(len(train_valid)),
        "n_test_timestamps": int(len(test_valid)),
    }
    if train_valid.empty or test_valid.empty:
        recorder.fail_check(
            rule_id,
            "Temporal order cannot be assessed because train or test has no usable timestamps. "
            "This is not a pass.",
            details,
        )
        return
    cutoff = train_valid.max()
    n_before = int((test_valid < cutoff).sum())
    evidence = {
        **details,
        "train_cutoff": cutoff.isoformat(),
        "n_test_rows_before_cutoff": n_before,
        "ties_at_cutoff_flagged": False,
    }
    if n_before:
        recorder.detect(
            make_finding(
                rule_id,
                f"{n_before} test timestamp(s) are earlier than the maximum training timestamp.",
                evidence,
            )
        )
        return
    recorder.pass_(
        rule_id,
        "No test timestamp is earlier than the maximum training timestamp.",
        evidence,
    )


def _class_shift(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str | None,
    task: str | None,
    config: AuditConfig,
) -> None:
    rule_id = "SPLIT.CLASS_DISTRIBUTION_SHIFT"
    if not recorder.enabled(rule_id):
        return
    if target is None or task != "classification":
        recorder.skip(
            rule_id,
            "Class distributions were not compared because this is not a classification audit "
            "with a target. This is not a claim that the distributions match.",
        )
        return
    train_counts, train_error = _class_counts(train[target])
    test_counts, test_error = _class_counts(test[target])
    if train_error == "unhashable" or test_error == "unhashable":
        recorder.fail_check(
            rule_id,
            "Class distributions could not be compared because target values are not hashable.",
            {"target": target},
        )
        return
    train_map = (
        {}
        if train_counts is None
        else {_plain(label): int(value) for label, value in train_counts.items()}
    )
    test_map = (
        {}
        if test_counts is None
        else {_plain(label): int(value) for label, value in test_counts.items()}
    )
    n_train = sum(train_map.values())
    n_test = sum(test_map.values())
    if (
        train_counts is None
        or test_counts is None
        or n_train < config.class_shift_min_rows
        or n_test < config.class_shift_min_rows
    ):
        recorder.skip(
            rule_id,
            "Class distributions were not compared because a partition has fewer non-null "
            "targets than class_shift_min_rows. This is not a pass.",
            {
                "target": target,
                "n_train_non_null": n_train,
                "n_test_non_null": n_test,
                "min_rows": config.class_shift_min_rows,
            },
        )
        return
    labels = set(train_map) | set(test_map)
    rows: list[dict[str, Any]] = []
    max_abs = 0.0
    variation = 0.0
    for label in sorted(labels, key=str):
        p_train = train_map.get(label, 0) / n_train
        p_test = test_map.get(label, 0) / n_test
        diff = abs(p_train - p_test)
        max_abs = max(max_abs, diff)
        variation += diff
        rows.append(
            {
                "label": _plain(label),
                "train_fraction": p_train,
                "test_fraction": p_test,
                "abs_diff": diff,
            }
        )
    total_variation = 0.5 * variation
    evidence = {
        "target": target,
        "max_abs_proportion_diff": max_abs,
        "total_variation": total_variation,
        "threshold": config.class_shift_max_abs_diff,
        "comparison": "flag when max_abs_proportion_diff > threshold",
        "n_train_non_null": n_train,
        "n_test_non_null": n_test,
        "classes": rows[:20],
        "classes_truncated": len(rows) > 20,
    }
    if max_abs > config.class_shift_max_abs_diff:
        recorder.detect(
            make_finding(
                rule_id,
                "Class proportions differ between train and test by more than the configured threshold.",
                evidence,
            )
        )
        return
    recorder.pass_(
        rule_id,
        "Class proportions do not exceed the configured difference threshold.",
        evidence,
    )


def _numeric_shift(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str | None,
    explicit_features: list[str] | None,
    config: AuditConfig,
) -> None:
    rule_id = "SPLIT.NUMERIC_DISTRIBUTION_SHIFT"
    if not recorder.enabled(rule_id):
        return
    excluded = set(config.id_columns)
    if target is not None:
        excluded.add(target)
    if config.group_column is not None:
        excluded.add(config.group_column)
    if config.time_column is not None:
        excluded.add(config.time_column)
    shared = [
        column for column in train.columns if column in test.columns and column not in excluded
    ]
    if explicit_features is not None:
        allowed = set(explicit_features)
        shared = [column for column in shared if column in allowed]
    candidates = [
        column
        for column in shared
        if is_numeric_feature(train[column]) and is_numeric_feature(test[column])
    ]
    if not candidates:
        recorder.skip(
            rule_id,
            "No shared numeric feature columns were available. "
            "Identifiers, the target, the group column, and the time column are excluded.",
        )
        return
    flagged: list[dict[str, Any]] = []
    not_compared: list[dict[str, Any]] = []
    for column in candidates:
        left = _finite_values(train[column])
        right = _finite_values(test[column])
        if left is None or right is None:
            not_compared.append({"column": _column_name(column), "reason": "conversion_failed"})
            continue
        if left.size < config.numeric_shift_min_rows or right.size < config.numeric_shift_min_rows:
            not_compared.append(
                {
                    "column": _column_name(column),
                    "reason": "below_min_rows",
                    "n_train_finite": int(left.size),
                    "n_test_finite": int(right.size),
                }
            )
            continue
        value = _smd(left, right)
        if value is None:
            not_compared.append({"column": _column_name(column), "reason": "undefined_scale"})
            continue
        if abs(value) >= config.numeric_shift_min_abs_smd:
            flagged.append(
                {
                    "column": _column_name(column),
                    "standardized_mean_difference": value,
                    "mean_train": float(np.mean(left)),
                    "mean_test": float(np.mean(right)),
                    "n_train_finite": int(left.size),
                    "n_test_finite": int(right.size),
                }
            )
    evidence = {
        "threshold": config.numeric_shift_min_abs_smd,
        "min_rows": config.numeric_shift_min_rows,
        "statistic": "absolute pooled standardized mean difference",
        "flagged": flagged[:_LIST_LIMIT],
        "n_flagged": len(flagged),
        "not_compared": not_compared[:_LIST_LIMIT],
        "n_not_compared": len(not_compared),
    }
    compared = len(candidates) - len(not_compared)
    if flagged:
        recorder.detect(
            make_finding(
                rule_id,
                f"{len(flagged)} numeric feature(s) differ by at least the configured "
                "standardized mean difference. This is not proof of leakage.",
                evidence,
            )
        )
        return
    if compared == 0:
        recorder.skip(
            rule_id,
            "Numeric distributions were not compared because no column met the sample-size "
            "and scale requirements. This is not a pass.",
            evidence,
        )
        return
    summary = "No compared numeric feature reached the standardized mean difference threshold."
    if not_compared:
        summary += (
            f" {len(not_compared)} column(s) were not compared. "
            "That is not evidence that those columns match."
        )
    recorder.pass_(rule_id, summary, evidence)


def _small_test(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AuditConfig,
) -> None:
    rule_id = "SPLIT.SMALL_TEST_PARTITION"
    if not recorder.enabled(rule_id):
        return
    n_test = int(len(test))
    n_total = int(len(train) + len(test))
    fraction = (n_test / n_total) if n_total else 0.0
    reasons: list[str] = []
    if n_test < config.small_test_rows:
        reasons.append("row_count")
    if config.small_test_fraction is not None and fraction < config.small_test_fraction:
        reasons.append("fraction")
    evidence = {
        "n_test_rows": n_test,
        "n_train_rows": int(len(train)),
        "test_fraction": fraction,
        "small_test_rows": config.small_test_rows,
        "small_test_fraction": config.small_test_fraction,
        "reasons": reasons,
    }
    if reasons:
        recorder.detect(
            make_finding(
                rule_id,
                "The test partition is below a configured size threshold.",
                evidence,
            )
        )
        return
    recorder.pass_(rule_id, "The test partition meets the configured size thresholds.", evidence)


def _target_in_features(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str | None,
    explicit_features: list[str] | None,
) -> None:
    rule_id = "DATA.TARGET_IN_FEATURES"
    if not recorder.enabled(rule_id):
        return
    if target is None:
        recorder.skip(rule_id, "No target was supplied, so target contamination was not checked.")
        return
    reasons: list[str] = []
    if explicit_features is not None and target in explicit_features:
        reasons.append("target_listed_as_feature")
    if list(train.columns).count(target) > 1 or list(test.columns).count(target) > 1:
        reasons.append("duplicate_target_label")
    evidence = {"target": target, "reasons": reasons}
    if reasons:
        recorder.detect(
            make_finding(rule_id, "The target column is included among the features.", evidence)
        )
        return
    recorder.pass_(rule_id, "The target column is not included in the feature set.", evidence)


def _target_copies(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str | None,
    features: list[Any],
    config: AuditConfig,
) -> None:
    rule_id = "DATA.TARGET_COPY_FEATURE"
    if not recorder.enabled(rule_id):
        return
    if target is None:
        recorder.skip(rule_id, "No target was supplied, so exact target copies were not checked.")
        return
    if _over_limit(train, config) or _over_limit(test, config):
        recorder.skip(
            rule_id,
            "Target-copy detection was not run because a partition exceeds max_rows. This is not a pass.",
            _limit_details(train, test, config),
        )
        return
    copies: list[dict[str, str]] = []
    unassessed: list[dict[str, str]] = []
    for name, frame in (("train", train), ("test", test)):
        if target not in frame.columns or len(frame) == 0:
            continue
        for column in features:
            if column == target or column not in frame.columns:
                continue
            outcome = _values_equal(frame[column], frame[target])
            if outcome is None:
                unassessed.append(
                    {
                        "partition": name,
                        "column": _column_name(column),
                        "reason": "values could not be compared safely",
                    }
                )
            elif outcome:
                copies.append(
                    {"partition": name, "column": _column_name(column), "comparison": "exact"}
                )
    evidence = {"target": target, "copied_columns": copies, "unassessed_columns": unassessed}
    if copies:
        recorder.detect(
            make_finding(
                rule_id,
                f"{len(copies)} column(s) are exact copies of the target.",
                evidence,
            )
        )
        return
    if unassessed:
        recorder.fail_check(
            rule_id,
            "Some columns could not be compared with the target. This is not a pass.",
            evidence,
        )
        return
    compared = [
        column
        for column in features
        if column != target and (column in train.columns or column in test.columns)
    ]
    if not compared:
        recorder.pass_(
            rule_id,
            "No feature column was available to compare with the target. "
            "Identifier columns and columns outside the feature list are not treated as copies.",
            evidence,
        )
        return
    recorder.pass_(rule_id, "No feature column is an exact copy of the target.", evidence)


def _target_missing(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str | None,
    config: AuditConfig,
) -> None:
    rule_id = "DATA.MISSING_VALUES"
    if not recorder.enabled(rule_id):
        return
    if target is None:
        recorder.skip(rule_id, "No target was supplied, so target missingness was not checked.")
        return
    partitions = []
    flagged = False
    for name, frame in (("train", train), ("test", test)):
        n_rows = int(len(frame))
        n_missing = int(frame[target].isna().sum()) if n_rows else 0
        fraction = (n_missing / n_rows) if n_rows else None
        partitions.append(
            {
                "partition": name,
                "n_missing": n_missing,
                "n_rows": n_rows,
                "fraction": fraction,
            }
        )
        if fraction is not None and fraction > config.missing_fraction_threshold:
            flagged = True
    evidence = {
        "scope": "target",
        "target": target,
        "threshold": config.missing_fraction_threshold,
        "comparison": "flag when missing_fraction > threshold",
        "partitions": partitions,
    }
    if flagged:
        recorder.detect(
            make_finding(
                rule_id,
                "The target exceeds the missing-value threshold in at least one partition.",
                evidence,
            )
        )
        return
    recorder.pass_(
        rule_id, "Target missingness does not exceed the configured threshold.", evidence
    )


def _one_class(
    recorder: CheckRecorder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str | None,
    task: str | None,
) -> None:
    rule_id = "DATA.CLASS_IMBALANCE"
    if not recorder.enabled(rule_id):
        return
    if target is None or task != "classification":
        recorder.skip(
            rule_id,
            "Single-class targets were not checked because this is not a classification audit "
            "with a target.",
        )
        return
    single: list[dict[str, Any]] = []
    for name, frame in (("train", train), ("test", test)):
        counts, error = _class_counts(frame[target])
        if error == "unhashable":
            recorder.fail_check(
                rule_id,
                "Single-class detection could not hash the target. This is not a pass.",
                {"partition": name, "target": target},
            )
            return
        n_classes = 0 if counts is None else int(len(counts))
        if n_classes == 1:
            single.append(
                {"partition": name, "n_non_null": int(counts.sum()) if counts is not None else 0}
            )
    evidence = {
        "target": target,
        "scope": "single_class_only",
        "single_class_partitions": single,
        "note": (
            "During a split audit this rule reports a single observed class. "
            "Milder imbalance is covered by SPLIT.CLASS_DISTRIBUTION_SHIFT and by audit_dataset."
        ),
    }
    if single:
        recorder.detect(
            make_finding(
                rule_id,
                "At least one partition has a single-class target.",
                evidence,
                severity=Severity.HIGH,
                title="A partition has only one class",
            )
        )
        return
    recorder.pass_(rule_id, "Neither partition has a single-class target.", evidence)
