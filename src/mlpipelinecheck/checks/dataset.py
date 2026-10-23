"""Dataset checks for Mlpipelinecheck 0.1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlpipelinecheck._encoding import (
    UnsupportedCell,
    UnsupportedValue,
    cell_token,
    fingerprint_frame,
)
from mlpipelinecheck._frame import is_boolean, is_float_feature, is_numeric_feature
from mlpipelinecheck.checks._recorder import CheckRecorder, make_finding
from mlpipelinecheck.config import AuditConfig
from mlpipelinecheck.findings import Severity
from mlpipelinecheck.rules import DATASET_RULE_IDS

_PROFILE_LIMIT = 200
_LIST_LIMIT = 50


def _variation(series: pd.Series) -> tuple[dict[str, Any] | None, str | None]:
    """Return unique-count stats, or an error reason when values cannot be compared."""
    try:
        missing = series.isna()
    except (TypeError, ValueError):
        return None, "values could not be scanned for missingness"
    non_null = series.loc[~missing]
    n_non_null = int(len(non_null))
    if n_non_null == 0:
        return {"n_non_null": 0, "n_unique_non_null": 0, "dominant_fraction": None}, None
    try:
        counts = non_null.value_counts(dropna=True)
    except TypeError:
        tokens: list[str] = []
        for value in non_null.tolist():
            try:
                tokens.append(cell_token(value))
            except UnsupportedValue as exc:
                return None, f"unsupported value type {exc.type_name}"
        counts = pd.Series(tokens, dtype="object").value_counts()
    n_unique = int(len(counts))
    dominant = float(counts.iloc[0] / n_non_null) if n_unique else None
    return {
        "n_non_null": n_non_null,
        "n_unique_non_null": n_unique,
        "dominant_fraction": dominant,
    }, None


def _float_array(series: pd.Series) -> np.ndarray | None:
    if not is_numeric_feature(series):
        return None
    try:
        array = series.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        return None
    if getattr(array, "ndim", None) != 1:
        return None
    return np.asarray(array, dtype=float)


def _column_name(column: object) -> str:
    return str(column)


def _capped(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "columns": records[:_LIST_LIMIT],
        "n_columns": len(records),
        "truncated": len(records) > _LIST_LIMIT,
    }


def _profiles(frame: pd.DataFrame) -> tuple[dict[str, Any], bool]:
    items: dict[str, Any] = {}
    truncated = len(frame.columns) > _PROFILE_LIMIT
    for column in list(frame.columns)[:_PROFILE_LIMIT]:
        series = frame[column]
        profile: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()) if _isna_safe(series) else None,
        }
        array = _float_array(series)
        if array is None:
            stats, _error = _variation(series)
            profile["n_unique_non_null"] = None if stats is None else stats["n_unique_non_null"]
        else:
            finite = array[np.isfinite(array)]
            n_positive = int(np.isposinf(array).sum())
            n_negative = int(np.isneginf(array).sum())
            profile.update(
                {
                    "n_finite": int(finite.size),
                    "n_positive_inf": n_positive,
                    "n_negative_inf": n_negative,
                    "mean": float(np.mean(finite)) if finite.size else None,
                    "std": float(np.std(finite, ddof=1)) if finite.size >= 2 else None,
                    "min": float(np.min(finite)) if finite.size else None,
                    "max": float(np.max(finite)) if finite.size else None,
                }
            )
        items[_column_name(column)] = profile
    return items, truncated


def _isna_safe(series: pd.Series) -> bool:
    try:
        series.isna()
    except (TypeError, ValueError):
        return False
    return True


def _unassessed_status(
    recorder: CheckRecorder,
    rule_id: str,
    unassessed: list[dict[str, str]],
    *,
    pass_summary: str,
) -> None:
    if unassessed:
        recorder.fail_check(
            rule_id,
            "The check did not complete because some columns could not be assessed. "
            "This is not a pass.",
            {"unassessed_columns": _capped(unassessed)},
        )
        return
    recorder.pass_(rule_id, pass_summary)


def _empty(
    recorder: CheckRecorder,
    *,
    n_rows: int,
    n_columns: int,
    n_features: int,
) -> None:
    rule_id = "DATA.EMPTY_DATASET"
    if not recorder.enabled(rule_id):
        return
    reasons: list[str] = []
    if n_rows == 0:
        reasons.append("no_rows")
    if n_features == 0:
        reasons.append("no_feature_columns")
    evidence = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_feature_columns": n_features,
        "reasons": reasons,
    }
    if reasons:
        recorder.detect(
            make_finding(
                rule_id,
                "The dataset has no rows or no feature columns to audit.",
                evidence,
            )
        )
        return
    recorder.pass_(rule_id, "The dataset has rows and at least one feature column.", evidence)


def _missing(recorder: CheckRecorder, frame: pd.DataFrame, config: AuditConfig) -> None:
    rule_id = "DATA.MISSING_VALUES"
    if not recorder.enabled(rule_id):
        return
    if len(frame) == 0 or frame.shape[1] == 0:
        recorder.skip(rule_id, "Missingness was not evaluated because the dataset has no cells.")
        return
    flagged: list[dict[str, Any]] = []
    unassessed: list[dict[str, str]] = []
    n_rows = len(frame)
    total_missing = 0
    for column in frame.columns:
        series = frame[column]
        if not _isna_safe(series):
            unassessed.append({"column": _column_name(column), "reason": "missingness scan failed"})
            continue
        n_missing = int(series.isna().sum())
        total_missing += n_missing
        fraction = n_missing / n_rows
        if fraction > config.missing_fraction_threshold:
            flagged.append(
                {
                    "column": _column_name(column),
                    "n_missing": n_missing,
                    "n_rows": n_rows,
                    "fraction": fraction,
                    "all_null": n_missing == n_rows,
                }
            )
    evidence = {
        "threshold": config.missing_fraction_threshold,
        "comparison": "flag when missing_fraction > threshold",
        "n_rows": n_rows,
        "n_columns": int(frame.shape[1]),
        "n_missing_cells": total_missing,
        "flagged": _capped(flagged),
        "unassessed_columns": unassessed,
    }
    if flagged:
        recorder.detect(
            make_finding(
                rule_id,
                f"{len(flagged)} column(s) exceed the missing-value threshold "
                f"of {config.missing_fraction_threshold}.",
                evidence,
            )
        )
        return
    if unassessed:
        recorder.fail_check(
            rule_id,
            "Missingness could not be evaluated for every column. This is not a pass.",
            evidence,
        )
        return
    recorder.pass_(rule_id, "No column exceeds the missing-value threshold.", evidence)


def _duplicates(recorder: CheckRecorder, frame: pd.DataFrame, config: AuditConfig) -> None:
    rule_id = "DATA.DUPLICATE_ROWS"
    if not recorder.enabled(rule_id):
        return
    n_rows = len(frame)
    if n_rows == 0 or frame.shape[1] == 0:
        recorder.skip(rule_id, "Duplicate rows were not evaluated because the table has no cells.")
        return
    if config.max_rows is not None and n_rows > config.max_rows:
        recorder.skip(
            rule_id,
            "Duplicate detection was not run because the table exceeds max_rows. "
            "This is not a pass.",
            {"n_rows": n_rows, "max_rows": config.max_rows},
        )
        return
    try:
        fingerprints = fingerprint_frame(frame)
    except UnsupportedCell as exc:
        recorder.fail_check(
            rule_id,
            f"{exc} Duplicate detection did not complete and is not a pass.",
            {"column": exc.column, "value_type": exc.type_name},
        )
        return
    n_unique = len(set(fingerprints))
    n_extra = n_rows - n_unique
    evidence = {
        "n_rows": n_rows,
        "n_unique_rows": n_unique,
        "n_extra_copies": n_extra,
        "duplicate_rate": n_extra / n_rows,
        "index_used": False,
    }
    if n_extra:
        recorder.detect(
            make_finding(
                rule_id,
                f"{n_extra} extra row(s) are exact copies of another row.",
                evidence,
            )
        )
        return
    recorder.pass_(rule_id, "No exact duplicate rows were found.", evidence)


def _constants(
    recorder: CheckRecorder,
    frame: pd.DataFrame,
    features: list[Any],
    config: AuditConfig,
) -> None:
    rule_id = "DATA.CONSTANT_FEATURE"
    if not recorder.enabled(rule_id):
        return
    if len(frame) == 0:
        recorder.skip(
            rule_id, "Constant features were not evaluated because the dataset has no rows."
        )
        return
    if not features:
        recorder.skip(
            rule_id, "Constant features were not evaluated because there are no feature columns."
        )
        return
    constants: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    unassessed: list[dict[str, str]] = []
    for column in features:
        stats, error = _variation(frame[column])
        if error is not None or stats is None:
            unassessed.append({"column": _column_name(column), "reason": error or "unknown"})
            continue
        n_unique = int(stats["n_unique_non_null"])
        if n_unique == 1:
            constants.append({"column": _column_name(column), **stats})
        elif (
            config.near_constant_fraction is not None
            and n_unique > 1
            and stats["dominant_fraction"] is not None
            and float(stats["dominant_fraction"]) >= config.near_constant_fraction
        ):
            near.append(
                {
                    "column": _column_name(column),
                    **stats,
                    "threshold": config.near_constant_fraction,
                }
            )
    evidence = {
        "constant_columns": _capped(constants),
        "near_constant_columns": _capped(near),
        "near_constant_fraction": config.near_constant_fraction,
        "unassessed_columns": unassessed,
    }
    if constants or near:
        severity = Severity.LOW if constants else Severity.INFO
        title = "A feature has no variation" if constants else "A feature is near-constant"
        summary = (
            f"{len(constants)} feature(s) have one non-null value"
            if constants
            else f"{len(near)} feature(s) meet the near-constant fraction."
        )
        recorder.detect(make_finding(rule_id, summary, evidence, severity=severity, title=title))
        return
    _unassessed_status(
        recorder,
        rule_id,
        unassessed,
        pass_summary="No constant or near-constant feature columns were found.",
    )


def _high_cardinality(
    recorder: CheckRecorder,
    frame: pd.DataFrame,
    features: list[Any],
    id_columns: list[str],
    config: AuditConfig,
) -> None:
    rule_id = "DATA.HIGH_CARDINALITY_FEATURE"
    if not recorder.enabled(rule_id):
        return
    if len(frame) == 0:
        recorder.skip(
            rule_id,
            "Cardinality was not evaluated because the dataset has no rows.",
        )
        return
    candidates = [
        column for column in features if column not in id_columns and not is_boolean(frame[column])
    ]
    if not candidates:
        recorder.skip(
            rule_id,
            "No non-identifier feature columns were available for the cardinality heuristic.",
        )
        return
    flagged: list[dict[str, Any]] = []
    unassessed: list[dict[str, str]] = []
    for column in candidates:
        series = frame[column]
        if is_float_feature(series) and not config.high_cardinality_include_floats:
            continue
        stats, error = _variation(series)
        if error is not None or stats is None:
            unassessed.append({"column": _column_name(column), "reason": error or "unknown"})
            continue
        n_non_null = int(stats["n_non_null"])
        n_unique = int(stats["n_unique_non_null"])
        if n_non_null < config.high_cardinality_min_rows or n_non_null == 0:
            continue
        ratio = n_unique / n_non_null
        if (
            n_unique >= config.high_cardinality_min_unique
            and ratio >= config.high_cardinality_ratio
        ):
            flagged.append(
                {
                    "column": _column_name(column),
                    "n_non_null": n_non_null,
                    "n_unique_non_null": n_unique,
                    "unique_ratio": ratio,
                }
            )
    evidence = {
        "unique_ratio_threshold": config.high_cardinality_ratio,
        "min_unique": config.high_cardinality_min_unique,
        "min_rows": config.high_cardinality_min_rows,
        "floats_included": config.high_cardinality_include_floats,
        "flagged": _capped(flagged),
        "unassessed_columns": unassessed,
    }
    if flagged:
        recorder.detect(
            make_finding(
                rule_id,
                f"{len(flagged)} feature(s) meet the high-cardinality heuristic.",
                evidence,
            )
        )
        return
    _unassessed_status(
        recorder,
        rule_id,
        unassessed,
        pass_summary="No feature met the high-cardinality heuristic.",
    )


def _target_in_features(
    recorder: CheckRecorder,
    frame: pd.DataFrame,
    *,
    target: str | None,
    explicit_features: list[str] | None,
) -> None:
    rule_id = "DATA.TARGET_IN_FEATURES"
    if not recorder.enabled(rule_id):
        return
    if target is None:
        recorder.skip(rule_id, "No target was supplied, so feature contamination was not checked.")
        return
    reasons: list[str] = []
    if explicit_features is not None and target in explicit_features:
        reasons.append("target_listed_as_feature")
    if list(frame.columns).count(target) > 1:
        reasons.append("duplicate_target_label")
    evidence = {"target": target, "reasons": reasons}
    if reasons:
        recorder.detect(
            make_finding(
                rule_id,
                "The target column is included among the features.",
                evidence,
            )
        )
        return
    recorder.pass_(
        rule_id,
        "The target column is not included in the feature set.",
        evidence,
    )


def _values_equal(left: pd.Series, right: pd.Series) -> bool | None:
    """Return whether two columns match, or ``None`` if a cell cannot be compared."""
    if is_numeric_feature(left) and is_numeric_feature(right):
        left_array = _float_array(left)
        right_array = _float_array(right)
        if left_array is None or right_array is None:
            return None
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    if (is_numeric_feature(left) and is_boolean(right)) or (
        is_boolean(left) and is_numeric_feature(right)
    ):
        left_array = _numeric_or_bool_array(left)
        right_array = _numeric_or_bool_array(right)
        if left_array is None or right_array is None:
            return None
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    if is_boolean(left) and is_boolean(right):
        left_missing = left.isna().to_numpy()
        right_missing = right.isna().to_numpy()
        if not np.array_equal(left_missing, right_missing):
            return False
        comparable = ~left_missing
        return bool(
            np.array_equal(
                left.to_numpy(dtype=object)[comparable].astype(bool),
                right.to_numpy(dtype=object)[comparable].astype(bool),
            )
        )
    left_values = left.to_numpy(copy=True)
    right_values = right.to_numpy(copy=True)
    for left_value, right_value in zip(left_values, right_values, strict=True):
        try:
            left_missing = _missing_scalar(left_value)
            right_missing = _missing_scalar(right_value)
        except (TypeError, ValueError):
            return None
        if left_missing and right_missing:
            continue
        if left_missing or right_missing:
            return False
        try:
            same = left_value == right_value
        except (TypeError, ValueError):
            return None
        if not isinstance(same, (bool, np.bool_)):
            return None
        if not bool(same):
            return False
    return True


def _missing_scalar(value: object) -> bool:
    from mlpipelinecheck._encoding import is_missing

    return is_missing(value)


def _numeric_or_bool_array(series: pd.Series) -> np.ndarray | None:
    try:
        if is_boolean(series):
            array = series.astype("float").to_numpy(dtype=float, na_value=np.nan)
        else:
            array = series.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        return None
    return np.asarray(array, dtype=float)


def _target_copy(
    recorder: CheckRecorder,
    frame: pd.DataFrame,
    *,
    target: str | None,
    features: list[Any],
    config: AuditConfig,
) -> None:
    rule_id = "DATA.TARGET_COPY_FEATURE"
    if not recorder.enabled(rule_id):
        return
    if target is None:
        recorder.skip(rule_id, "No target was supplied, so exact copies were not checked.")
        return
    if len(frame) == 0:
        recorder.skip(rule_id, "Target copies were not evaluated because the dataset has no rows.")
        return
    if config.max_rows is not None and len(frame) > config.max_rows:
        recorder.skip(
            rule_id,
            "Target-copy detection was not run because the table exceeds max_rows. This is not a pass.",
            {"n_rows": len(frame), "max_rows": config.max_rows},
        )
        return
    copies: list[dict[str, str]] = []
    unassessed: list[dict[str, str]] = []
    target_series = frame[target]
    comparable = [column for column in features if column != target and column in frame.columns]
    if not comparable:
        recorder.pass_(
            rule_id,
            "No feature column was available to compare with the target. "
            "Identifier columns and columns outside the feature list are not treated as copies.",
        )
        return
    for column in comparable:
        outcome = _values_equal(frame[column], target_series)
        if outcome is None:
            unassessed.append(
                {"column": _column_name(column), "reason": "values could not be compared safely"}
            )
        elif outcome:
            copies.append({"column": _column_name(column), "comparison": "exact"})
    evidence = {
        "target": target,
        "copied_columns": copies,
        "unassessed_columns": unassessed,
        "text_coerced_to_number": False,
    }
    if copies:
        recorder.detect(
            make_finding(
                rule_id,
                f"{len(copies)} column(s) are exact copies of the target.",
                evidence,
            )
        )
        return
    _unassessed_status(
        recorder,
        rule_id,
        unassessed,
        pass_summary="No other column is an exact copy of the target.",
    )


def _class_counts(series: pd.Series) -> tuple[pd.Series | None, str | None]:
    non_null = series.dropna()
    if non_null.empty:
        return None, "empty"
    try:
        return non_null.value_counts(dropna=True), None
    except TypeError:
        return None, "unhashable"


def _imbalance(
    recorder: CheckRecorder,
    frame: pd.DataFrame,
    *,
    target: str | None,
    task: str | None,
    config: AuditConfig,
) -> None:
    rule_id = "DATA.CLASS_IMBALANCE"
    if not recorder.enabled(rule_id):
        return
    if target is None:
        recorder.skip(rule_id, "No target was supplied.")
        return
    if task != "classification":
        recorder.skip(
            rule_id,
            "Class imbalance applies to classification. "
            f"Resolved task: {task if task is not None else 'not inferred'}.",
        )
        return
    if len(frame) == 0:
        recorder.skip(rule_id, "Class balance was not evaluated because the dataset has no rows.")
        return
    counts, error = _class_counts(frame[target])
    n_missing = int(frame[target].isna().sum())
    if error == "unhashable":
        recorder.fail_check(
            rule_id,
            "Class balance could not be evaluated because target values are not hashable. "
            "This is not a pass.",
            {"target": target},
        )
        return
    if counts is None:
        recorder.skip(
            rule_id,
            "Class balance was not evaluated because the target has no non-null values.",
            {"target": target, "n_missing": n_missing},
        )
        return
    n_non_null = int(counts.sum())
    n_classes = int(len(counts))
    majority_count = int(counts.iloc[0])
    minority_count = int(counts.iloc[-1])
    majority_fraction = majority_count / n_non_null
    minority_fraction = minority_count / n_non_null
    classes = [
        {"label": _plain(label), "count": int(count), "fraction": float(count / n_non_null)}
        for label, count in counts.items()
    ]
    evidence = {
        "target": target,
        "n_non_null": n_non_null,
        "n_missing": n_missing,
        "n_classes": n_classes,
        "majority_fraction": majority_fraction,
        "minority_fraction": minority_fraction,
        "majority_count": majority_count,
        "minority_count": minority_count,
        "minority_fraction_threshold": config.imbalance_minority_fraction,
        "majority_fraction_threshold": config.imbalance_majority_fraction,
        "single_class": n_classes == 1,
        "classes": classes[:20],
        "classes_truncated": len(classes) > 20,
    }
    single_class = n_classes == 1
    imbalanced = (
        minority_fraction <= config.imbalance_minority_fraction
        or majority_fraction >= config.imbalance_majority_fraction
    )
    if single_class or imbalanced:
        recorder.detect(
            make_finding(
                rule_id,
                "The classification target has only one observed class."
                if single_class
                else "The classification target crosses the configured imbalance thresholds.",
                evidence,
                severity=Severity.HIGH if single_class else Severity.MEDIUM,
                title=(
                    "Classification target has only one class"
                    if single_class
                    else "The classification target is imbalanced"
                ),
            )
        )
        return
    recorder.pass_(
        rule_id, "The classification target does not cross the imbalance thresholds.", evidence
    )


def _plain(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _non_finite(recorder: CheckRecorder, frame: pd.DataFrame) -> None:
    rule_id = "DATA.NON_FINITE_VALUES"
    if not recorder.enabled(rule_id):
        return
    if len(frame) == 0:
        recorder.skip(
            rule_id, "Non-finite values were not evaluated because the dataset has no rows."
        )
        return
    flagged: list[dict[str, Any]] = []
    unassessed: list[dict[str, str]] = []
    numeric_seen = False
    for column in frame.columns:
        series = frame[column]
        if not is_numeric_feature(series):
            continue
        numeric_seen = True
        array = _float_array(series)
        if array is None:
            unassessed.append(
                {"column": _column_name(column), "reason": "numeric conversion failed"}
            )
            continue
        n_positive = int(np.isposinf(array).sum())
        n_negative = int(np.isneginf(array).sum())
        if n_positive or n_negative:
            flagged.append(
                {
                    "column": _column_name(column),
                    "n_positive_inf": n_positive,
                    "n_negative_inf": n_negative,
                    "n_non_finite": n_positive + n_negative,
                }
            )
    if not numeric_seen and not unassessed:
        recorder.skip(rule_id, "No numeric columns were available to check for non-finite values.")
        return
    evidence = {"flagged": _capped(flagged), "unassessed_columns": unassessed}
    if flagged:
        recorder.detect(
            make_finding(
                rule_id,
                f"{len(flagged)} numeric column(s) contain non-finite values.",
                evidence,
            )
        )
        return
    _unassessed_status(
        recorder,
        rule_id,
        unassessed,
        pass_summary="No non-finite values were found in numeric columns.",
    )


def run_dataset_checks(
    frame: pd.DataFrame,
    *,
    target: str | None,
    task: str | None,
    task_source: str,
    explicit_features: list[str] | None,
    features: list[Any],
    config: AuditConfig,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Run the dataset rule list and return findings, checks, and metadata."""
    recorder = CheckRecorder(config.disabled_checks)
    profiles, truncated = _profiles(frame)
    metadata: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "n_feature_columns": len(features),
        "feature_columns": [_column_name(column) for column in features],
        "id_columns": list(config.id_columns),
        "target": target,
        "task": task,
        "task_source": task_source,
        "column_profiles": profiles,
        "column_profiles_truncated": truncated,
        "statistic_note": (
            "Numeric mean, sample standard deviation (ddof=1), min, and max exclude "
            "missing and non-finite values. No rows are imputed or dropped."
        ),
        "complexity_note": (
            "Missing, constant, cardinality, and non-finite checks are vectorized passes "
            "over the table. Duplicate and target-copy checks fingerprint cells at a cost "
            "proportional to rows times columns, plus memory for one digest per row. "
            "Set max_rows to skip those two checks instead of sampling."
        ),
    }
    _empty(
        recorder,
        n_rows=len(frame),
        n_columns=int(frame.shape[1]),
        n_features=len(features),
    )
    _missing(recorder, frame, config)
    _duplicates(recorder, frame, config)
    _constants(recorder, frame, features, config)
    _high_cardinality(recorder, frame, features, list(config.id_columns), config)
    _target_in_features(
        recorder,
        frame,
        target=target,
        explicit_features=explicit_features,
    )
    _target_copy(recorder, frame, target=target, features=features, config=config)
    _imbalance(recorder, frame, target=target, task=task, config=config)
    _non_finite(recorder, frame)
    _assert_rule_coverage(recorder)
    return recorder.findings, recorder.checks, metadata


def _assert_rule_coverage(recorder: CheckRecorder) -> None:
    seen = [check.id for check in recorder.checks]
    expected = list(DATASET_RULE_IDS)
    if seen != expected:
        raise RuntimeError(f"Dataset checks were {seen}, expected {expected}.")
