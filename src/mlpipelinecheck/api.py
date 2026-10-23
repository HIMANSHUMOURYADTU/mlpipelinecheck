"""Public audit entry points.

These functions audit tables, explicit partitions, and supplied predictions.
They do not fit estimators and they do not modify caller-owned data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from mlpipelinecheck._frame import (
    reject_duplicate_columns,
    require_columns,
    require_dataframe,
    resolve_feature_columns,
    resolve_task,
)
from mlpipelinecheck.checks.dataset import run_dataset_checks
from mlpipelinecheck.checks.metrics import run_prediction_checks
from mlpipelinecheck.checks.split import run_split_checks
from mlpipelinecheck.config import PREDICTION_TASKS, AuditConfig
from mlpipelinecheck.exceptions import MlpipelinecheckConfigError, MlpipelinecheckInputError
from mlpipelinecheck.report import AuditReport


def _config(
    config: AuditConfig | Mapping[str, Any] | None,
    **overrides: Any,
) -> AuditConfig:
    if isinstance(config, AuditConfig):
        base = config
    else:
        base = AuditConfig.from_mapping(config)
    return base.with_overrides(**overrides)


def _reject_prediction_only_options(config: AuditConfig) -> None:
    """Dataset and split audits do not resample rows."""
    if (
        config.bootstrap_samples is not None
        or config.observation_dependence != "independent"
        or config.confidence_interval_max_width is not None
    ):
        raise MlpipelinecheckConfigError(
            "bootstrap_samples, observation_dependence, and confidence_interval_max_width "
            "apply to audit_predictions. A dataset or split audit does not use them."
        )


def audit_dataset(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    id_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    config: AuditConfig | Mapping[str, Any] | None = None,
) -> AuditReport:
    """Audit a tabular dataset without training a model.

    ``target`` is optional. Task-specific checks run only for classification,
    either because ``task`` is explicit or because it can be inferred safely.
    When ``feature_columns`` is omitted, the target and ``id_columns`` are
    excluded from the feature set. An explicit list that still contains the
    target is reported as ``DATA.TARGET_IN_FEATURES``.

    The input frame is not modified, dropped, or imputed.
    """
    frame = require_dataframe(df, "df")
    cfg = _config(
        config,
        task=task,
        id_columns=id_columns,
        feature_columns=feature_columns,
    )
    _reject_prediction_only_options(cfg)
    reject_duplicate_columns(frame, "df")
    if target is not None:
        require_columns(frame, [target], role="target column")
    if cfg.id_columns:
        require_columns(frame, cfg.id_columns, role="identifier columns")
    explicit = cfg.feature_columns
    features = resolve_feature_columns(
        frame,
        target=target,
        id_columns=cfg.id_columns,
        feature_columns=explicit,
    )
    series = None if target is None else frame[target]
    resolved_task, task_source = resolve_task(series, cfg.task)
    findings, checks, metadata = run_dataset_checks(
        frame,
        target=target,
        task=resolved_task,
        task_source=task_source,
        explicit_features=explicit,
        features=features,
        config=cfg,
    )
    return AuditReport(
        "dataset",
        findings,
        checks,
        config=cfg.to_dict(),
        metadata=metadata,
    )


def audit_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    split_strategy: str | None = None,
    id_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    config: AuditConfig | Mapping[str, Any] | None = None,
) -> AuditReport:
    """Audit two explicit partitions.

    Row overlap uses a fingerprint of cell values and ignores the index.
    Group and time checks run only when those columns are supplied. A temporal
    strategy without ``time_column``, or a group strategy without
    ``group_column``, is an input error rather than a passed check.

    Distribution differences are descriptive. They are not labeled as leakage.
    Neither frame is modified.
    """
    train = require_dataframe(train_df, "train_df")
    test = require_dataframe(test_df, "test_df")
    cfg = _config(
        config,
        task=task,
        group_column=group_column,
        time_column=time_column,
        split_strategy=split_strategy,
        id_columns=id_columns,
        feature_columns=feature_columns,
    )
    if cfg.split_strategy == "group" and cfg.group_column is None:
        raise MlpipelinecheckInputError(
            "split_strategy='group' requires group_column. "
            "The group check is not treated as passed."
        )
    if cfg.split_strategy == "temporal" and cfg.time_column is None:
        raise MlpipelinecheckInputError(
            "split_strategy='temporal' requires time_column. "
            "Temporal order is not treated as passed."
        )
    reject_duplicate_columns(train, "train_df")
    reject_duplicate_columns(test, "test_df")
    if target is not None:
        require_columns(train, [target], role="target column")
        require_columns(test, [target], role="target column")
    if cfg.group_column is not None:
        require_columns(train, [cfg.group_column], role="group column")
        require_columns(test, [cfg.group_column], role="group column")
    if cfg.time_column is not None:
        require_columns(train, [cfg.time_column], role="time column")
        require_columns(test, [cfg.time_column], role="time column")
    if cfg.id_columns:
        require_columns(train, cfg.id_columns, role="identifier columns")
        require_columns(test, cfg.id_columns, role="identifier columns")
    if cfg.feature_columns is not None:
        require_columns(train, cfg.feature_columns, role="feature columns")
        require_columns(test, cfg.feature_columns, role="feature columns")
    train_target = None if target is None else train[target]
    if cfg.task == "regression" and target is not None:
        resolve_task(test[target], "regression")
    resolved_task, task_source = resolve_task(train_target, cfg.task)
    _reject_prediction_only_options(cfg)
    features = resolve_feature_columns(
        train,
        target=target,
        id_columns=cfg.id_columns,
        feature_columns=cfg.feature_columns,
    )
    findings, checks, metadata = run_split_checks(
        train,
        test,
        target=target,
        task=resolved_task,
        task_source=task_source,
        explicit_features=cfg.feature_columns,
        features=features,
        config=cfg,
    )
    return AuditReport(
        "split",
        findings,
        checks,
        config=cfg.to_dict(),
        metadata=metadata,
    )


def audit_predictions(
    y_true: Any,
    y_pred: Any,
    task: str,
    y_score: Any | None = None,
    groups: Any | None = None,
    timestamps: Any | None = None,
    config: AuditConfig | Mapping[str, Any] | None = None,
) -> AuditReport:
    """Audit supplied predictions.

    The function does not fit a model. ``task`` must be ``classification`` or
    ``regression``. Classification metrics use scikit-learn. Optional scores
    follow the shape conventions in :mod:`mlpipelinecheck.checks.metrics`.

    Comparison is positional. A Series index is not used to align ``y_true``
    and ``y_pred``.

    ``groups`` and ``timestamps`` declare dependence. They are counted, not
    copied into the report. A bootstrap interval is computed only for
    ``observation_dependence='independent'`` when ``bootstrap_samples`` is set.
    Grouped and temporal intervals are refused rather than replaced with an
    iid interval.
    """
    if task not in PREDICTION_TASKS:
        raise MlpipelinecheckInputError(
            "audit_predictions requires task='classification' or task='regression'."
        )
    cfg = _config(config, task=task)
    findings, checks, metadata = run_prediction_checks(
        y_true,
        y_pred,
        y_score,
        groups=groups,
        timestamps=timestamps,
        config=cfg,
    )
    return AuditReport(
        "predictions",
        findings,
        checks,
        config=cfg.to_dict(),
        metadata=metadata,
    )
