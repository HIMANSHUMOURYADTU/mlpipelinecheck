"""DataFrame helpers shared by the audits."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from mlpipelinecheck.exceptions import MlpipelinecheckInputError


def require_dataframe(value: object, name: str) -> pd.DataFrame:
    """Return ``value`` if it is a DataFrame, otherwise raise an input error."""
    if not isinstance(value, pd.DataFrame):
        raise MlpipelinecheckInputError(
            f"{name} must be a pandas DataFrame, got {type(value).__name__}."
        )
    return value


def reject_duplicate_columns(frame: pd.DataFrame, name: str) -> None:
    """Reject duplicate labels, which would make column checks ambiguous."""
    duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicates:
        rendered = ", ".join(repr(column) for column in duplicates)
        raise MlpipelinecheckInputError(
            f"{name} has duplicate column names: {rendered}. "
            "Rename them so each column can be addressed unambiguously."
        )


def require_columns(frame: pd.DataFrame, columns: Sequence[str], *, role: str) -> None:
    """Raise when any required column is absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        rendered = ", ".join(repr(column) for column in missing)
        available = ", ".join(repr(column) for column in list(frame.columns)[:40])
        suffix = "" if len(frame.columns) <= 40 else f" ({len(frame.columns)} columns total)"
        raise MlpipelinecheckInputError(
            f"Missing {role}: {rendered}. Available columns: {available}{suffix}."
        )


def is_boolean(series: pd.Series) -> bool:
    return bool(pd.api.types.is_bool_dtype(series))


def is_numeric_feature(series: pd.Series) -> bool:
    """True for real numeric columns, excluding booleans."""
    return bool(pd.api.types.is_numeric_dtype(series) and not is_boolean(series))


def is_float_feature(series: pd.Series) -> bool:
    return bool(pd.api.types.is_float_dtype(series) and not is_boolean(series))


def dtype_group(series: pd.Series) -> str:
    """Return a coarse dtype group used for schema comparison.

    Integers and floats share ``numeric``. Booleans stay separate.
    Object, string, and categorical columns share ``text``.
    """
    if is_boolean(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_timedelta64_dtype(series):
        return "timedelta"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(series):
        return "text"
    if pd.api.types.is_object_dtype(series):
        return "text"
    return f"other:{series.dtype}"


def dtypes_compatible(left: pd.Series, right: pd.Series) -> bool:
    return dtype_group(left) == dtype_group(right)


def resolve_feature_columns(
    frame: pd.DataFrame,
    *,
    target: str | None,
    id_columns: Sequence[str],
    feature_columns: Sequence[str] | None,
) -> list[str]:
    """Return the feature columns without modifying ``frame``.

    When ``feature_columns`` is omitted, the target and declared identifiers
    are excluded. An explicit list is validated and kept, including the target
    if the caller put it there; that case is a finding, not a silent drop.
    """
    if feature_columns is None:
        excluded = set(id_columns)
        if target is not None:
            excluded.add(target)
        return [column for column in frame.columns if column not in excluded]
    require_columns(frame, feature_columns, role="feature columns")
    return list(feature_columns)


def regression_target_ok(series: pd.Series) -> bool:
    return is_numeric_feature(series)


def infer_task(series: pd.Series) -> str | None:
    """Infer classification or regression only when the signal is unambiguous.

    Returns ``None`` when guessing would be unsafe. Callers then skip
    task-specific checks instead of labeling the column incorrectly.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None
    if is_boolean(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return "classification"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        try:
            n_unique = int(non_null.nunique(dropna=True))
        except TypeError:
            return None
        if 1 <= n_unique <= 20:
            return "classification"
        return None
    if not is_numeric_feature(series):
        return None
    n_unique = int(non_null.nunique(dropna=True))
    if is_float_feature(series):
        if n_unique <= 2:
            return "classification"
        if n_unique > 10:
            return "regression"
        return None
    if n_unique <= 10:
        return "classification"
    if n_unique > 20:
        return "regression"
    return None


def resolve_task(series: pd.Series | None, requested: str) -> tuple[str | None, str]:
    """Return ``(task, source)`` where task may be ``None`` when inference is unsafe."""
    if requested == "regression":
        if series is not None and not regression_target_ok(series):
            raise MlpipelinecheckInputError(
                "task='regression' requires a numeric, non-boolean target. "
                "Boolean and text targets should use task='classification', "
                "or convert the target before auditing."
            )
        return "regression", "explicit"
    if requested == "classification":
        return "classification", "explicit"
    if series is None:
        return None, "no_target"
    inferred = infer_task(series)
    if inferred is None:
        return None, "not_inferred"
    return inferred, "inferred"
