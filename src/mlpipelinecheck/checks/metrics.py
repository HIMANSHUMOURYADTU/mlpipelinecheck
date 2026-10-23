"""Prediction metrics and metric-validity checks.

Rows are compared by position. Indexes are ignored and are not aligned.
``y_pred`` must contain class labels for classification, not probabilities.
Probabilities and ranking scores belong in ``y_score``.

Binary ``y_score`` of shape ``(n,)`` is the score for the greater sorted label,
matching scikit-learn. Shape ``(n, 2)`` uses sorted label order, so the second
column is that positive class. Multiclass scores must have shape
``(n, n_classes)`` in that same order. Log loss and Brier score are computed
only when the supplied values are probabilities; other scores can still be used
for ROC AUC.

An optional iid percentile interval is computed only when ``bootstrap_samples`` is set
and ``observation_dependence`` is ``independent``. Grouped and temporal intervals
are not computed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from mlpipelinecheck._encoding import UnsupportedValue, cell_token, is_missing
from mlpipelinecheck.checks._recorder import CheckRecorder, make_finding
from mlpipelinecheck.config import AuditConfig
from mlpipelinecheck.exceptions import MlpipelinecheckError, MlpipelinecheckInputError
from mlpipelinecheck.findings import Severity
from mlpipelinecheck.rules import PREDICTION_RULE_IDS

_CLASS_LIST_LIMIT = 20


def _kind(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def _plain(value: object) -> object:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_1d(name: str, values: object) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        raise MlpipelinecheckInputError(
            f"{name} must be a one-dimensional series or array, not a DataFrame."
        )
    if isinstance(values, pd.Series):
        array = values.to_numpy()
    else:
        array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = np.reshape(array, -1)
    if array.ndim != 1:
        raise MlpipelinecheckInputError(
            f"{name} must be one-dimensional; got shape {tuple(int(dim) for dim in array.shape)}."
        )
    if array.size == 0:
        raise MlpipelinecheckInputError(f"{name} is empty.")
    return array


def _reject_missing(name: str, array: np.ndarray) -> None:
    for value in array.tolist():
        if is_missing(value):
            raise MlpipelinecheckInputError(f"{name} contains missing values.")


def _classification_labels(name: str, array: np.ndarray) -> list[object]:
    _reject_missing(name, array)
    labels = [_plain(value) for value in array.tolist()]
    kinds = {_kind(value) for value in labels}
    if len(kinds) != 1:
        raise MlpipelinecheckInputError(
            f"{name} mixes label types {sorted(kinds)}. Pass one label type; "
            "Mlpipelinecheck does not coerce labels."
        )
    return labels


def _regression_values(name: str, array: np.ndarray) -> np.ndarray:
    numbers: list[float] = []
    for value in array.tolist():
        if is_missing(value):
            raise MlpipelinecheckInputError(f"{name} contains missing values.")
        if isinstance(value, (bool, np.bool_)):
            raise MlpipelinecheckInputError(
                f"{name} contains boolean values. Regression metrics expect real numbers."
            )
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            number = float(value)
        elif isinstance(value, (float, np.floating)):
            number = float(value)
        else:
            raise MlpipelinecheckInputError(
                f"{name} contains non-numeric values of type {type(value).__name__}."
            )
        if not math.isfinite(number):
            raise MlpipelinecheckInputError(f"{name} contains non-finite values.")
        numbers.append(number)
    return np.asarray(numbers, dtype=float)


def _score_array(y_score: object, n_rows: int) -> np.ndarray:
    if isinstance(y_score, pd.DataFrame):
        raw = y_score.to_numpy()
    elif isinstance(y_score, pd.Series):
        raw = y_score.to_numpy()
    else:
        raw = np.asarray(y_score)
    if raw.size == 0:
        raise MlpipelinecheckInputError("y_score is empty.")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MlpipelinecheckInputError("y_score must contain only numeric values.") from exc
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim not in {1, 2}:
        raise MlpipelinecheckInputError(
            f"y_score must have 1 or 2 dimensions; got shape {tuple(int(dim) for dim in array.shape)}."
        )
    if not np.isfinite(array).all():
        raise MlpipelinecheckInputError("y_score contains missing or non-finite values.")
    if array.shape[0] != n_rows:
        raise MlpipelinecheckInputError(
            f"y_score has {array.shape[0]} rows but y_true has {n_rows} values."
        )
    return array


def _require_score_shape(array: np.ndarray, n_classes: int) -> None:
    if array.ndim == 1:
        if n_classes > 2:
            raise MlpipelinecheckInputError(
                "Multiclass y_score must have shape (n_samples, n_classes) in sorted label order; "
                f"got a one-dimensional score for {n_classes} classes."
            )
        return
    n_columns = int(array.shape[1])
    if n_classes > 2 and n_columns != n_classes:
        raise MlpipelinecheckInputError(
            f"y_score has {n_columns} columns but y_true has {n_classes} classes. "
            "Provide one probability column per class in sorted label order."
        )
    if n_classes <= 2 and n_columns != 2:
        raise MlpipelinecheckInputError(
            "For binary labels, y_score must have shape (n_samples,) or (n_samples, 2) "
            f"in sorted label order; got {n_columns} columns."
        )


def _sorted_labels(labels: list[object]) -> list[object]:
    if not labels:
        return []
    if all(isinstance(label, (bool, np.bool_)) for label in labels):
        return sorted({bool(label) for label in labels})
    if all(isinstance(label, (int, float)) and not isinstance(label, bool) for label in labels):
        return sorted(set(labels))
    if all(isinstance(label, str) for label in labels):
        return sorted(set(labels))
    return sorted(set(labels), key=str)


def _probability_matrix(
    scores: np.ndarray,
    n_classes: int,
) -> np.ndarray | None:
    """Return class probabilities in sorted-label order, or None when scores are not probabilities."""
    if n_classes < 2:
        return None
    if scores.ndim == 1:
        if n_classes != 2:
            return None
        if np.any(scores < 0) or np.any(scores > 1):
            return None
        return np.column_stack([1.0 - scores, scores])
    if scores.shape[1] != n_classes:
        return None
    if np.any(scores < -1e-8) or np.any(scores > 1.0 + 1e-8):
        return None
    if np.any(np.abs(scores.sum(axis=1) - 1.0) > 1e-3):
        return None
    return np.clip(scores, 0.0, 1.0)


def _import_sklearn() -> dict[str, Any]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            brier_score_loss,
            confusion_matrix,
            log_loss,
            mean_absolute_error,
            mean_squared_error,
            precision_recall_fscore_support,
            r2_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise MlpipelinecheckError(
            "scikit-learn is required to audit predictions. "
            "Install the mlpipelinecheck dependencies and retry."
        ) from exc
    return {
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "brier_score_loss": brier_score_loss,
        "confusion_matrix": confusion_matrix,
        "log_loss": log_loss,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "r2_score": r2_score,
        "roc_auc_score": roc_auc_score,
    }


def _structure_summary(name: str, values: object, n_rows: int) -> dict[str, int]:
    """Count group values without copying them into the report."""
    array = _as_1d(name, values)
    if len(array) != n_rows:
        raise MlpipelinecheckInputError(
            f"{name} has {len(array)} values but y_true has {n_rows} values."
        )
    n_missing = 0
    unique: set[str] = set()
    for value in array.tolist():
        if is_missing(value):
            n_missing += 1
            continue
        try:
            unique.add(cell_token(value))
        except UnsupportedValue as exc:
            raise MlpipelinecheckInputError(
                f"{name} contains unsupported value type {exc.type_name}."
            ) from exc
    if n_missing == n_rows:
        raise MlpipelinecheckInputError(f"{name} has no non-missing values.")
    return {"n_rows": n_rows, "n_missing": n_missing, "n_unique": len(unique)}


def _timestamp_summary(name: str, values: object, n_rows: int) -> dict[str, int]:
    """Validate timestamps and return counts, not the timestamp values."""
    from mlpipelinecheck.checks.split import _parse_time

    array = _as_1d(name, values)
    if len(array) != n_rows:
        raise MlpipelinecheckInputError(
            f"{name} has {len(array)} values but y_true has {n_rows} values."
        )
    parsed = _parse_time(pd.Series(array.tolist()), name)
    n_missing = int(parsed.isna().sum())
    n_usable = int(parsed.notna().sum())
    if n_usable == 0:
        raise MlpipelinecheckInputError(f"{name} has no usable timestamps.")
    return {"n_rows": n_rows, "n_missing": n_missing, "n_usable": n_usable}


def _prepare_dependence(
    config: AuditConfig,
    n_rows: int,
    groups: object | None,
    timestamps: object | None,
) -> dict[str, Any]:
    """Record the sampling assumption without inventing an interval."""
    dependence = config.observation_dependence
    if dependence == "independent":
        if groups is not None or timestamps is not None:
            raise MlpipelinecheckInputError(
                "groups or timestamps were supplied while observation_dependence is "
                "independent. Set observation_dependence to 'grouped' or 'temporal'. "
                "Those values are not dropped and then treated as iid."
            )
        return {"dependence": dependence}
    if dependence == "grouped":
        if groups is None:
            raise MlpipelinecheckInputError(
                "observation_dependence='grouped' requires groups. "
                "The rows were not treated as independent."
            )
        summary: dict[str, Any] = {
            "dependence": dependence,
            "groups": _structure_summary("groups", groups, n_rows),
        }
    else:
        if timestamps is None:
            raise MlpipelinecheckInputError(
                "observation_dependence='temporal' requires timestamps. "
                "The rows were not treated as independent."
            )
        summary = {
            "dependence": dependence,
            "timestamps": _timestamp_summary("timestamps", timestamps, n_rows),
        }
    if config.bootstrap_samples is not None:
        raise MlpipelinecheckInputError(
            f"A {dependence} bootstrap interval is not implemented. "
            "An iid interval was not computed in its place. Omit bootstrap_samples, "
            "or set observation_dependence='independent' only when that assumption is acceptable."
        )
    return summary


def run_prediction_checks(
    y_true: object,
    y_pred: object,
    y_score: object | None,
    *,
    groups: object | None = None,
    timestamps: object | None = None,
    config: AuditConfig,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Validate inputs, compute documented metrics, and record metric checks."""
    true_array = _as_1d("y_true", y_true)
    pred_array = _as_1d("y_pred", y_pred)
    if len(true_array) != len(pred_array):
        raise MlpipelinecheckInputError(
            f"y_true has {len(true_array)} values but y_pred has {len(pred_array)} values. "
            "Values are compared by position, not by index."
        )
    dependence = _prepare_dependence(config, len(true_array), groups, timestamps)
    if config.task == "classification":
        return _classification(true_array, pred_array, y_score, config, dependence)
    if config.task == "regression":
        return _regression(true_array, pred_array, y_score, config, dependence)
    raise MlpipelinecheckInputError(
        "audit_predictions requires task='classification' or task='regression'."
    )


def _classification(
    true_array: np.ndarray,
    pred_array: np.ndarray,
    y_score: object | None,
    config: AuditConfig,
    dependence: dict[str, Any],
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    y_true = _classification_labels("y_true", true_array)
    y_pred = _classification_labels("y_pred", pred_array)
    if _kind(y_true[0]) != _kind(y_pred[0]):
        raise MlpipelinecheckInputError(
            f"y_pred labels are {_kind(y_pred[0])} but y_true labels are {_kind(y_true[0])}. "
            "Mlpipelinecheck does not coerce label types."
        )
    observed = _sorted_labels(y_true)
    n_classes = len(observed)
    scores = None
    if y_score is not None:
        scores = _score_array(y_score, len(y_true))
        _require_score_shape(scores, n_classes)
    sklearn = _import_sklearn()
    label_list = _sorted_labels(y_true + y_pred)
    metrics, notes = _classification_metrics(
        y_true,
        y_pred,
        observed,
        label_list,
        scores,
        sklearn,
    )
    recorder = CheckRecorder(config.disabled_checks)
    _imbalance_check(recorder, y_true, observed, config)
    _auc_check(recorder, scores, n_classes, metrics.get("roc_auc"), metrics.get("roc_auc_reason"))
    _sample_check(recorder, len(y_true), config)
    interval = _resample_interval(
        y_true,
        y_pred,
        metric_name="accuracy",
        point=float(metrics["accuracy"]),
        score=lambda left, right: float(sklearn["accuracy_score"](left, right)),
        config=config,
    )
    _bootstrap_check(recorder, interval, config)
    _assert_coverage(recorder)
    metadata = _metadata(
        "classification", len(y_true), metrics, notes, config, interval, dependence
    )
    return recorder.findings, recorder.checks, metadata


def _classification_metrics(
    y_true: list[object],
    y_pred: list[object],
    observed: list[object],
    label_list: list[object],
    scores: np.ndarray | None,
    sklearn: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    import warnings

    notes: list[str] = []
    n_classes = len(observed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        accuracy = float(sklearn["accuracy_score"](y_true, y_pred))
        per_precision, per_recall, per_f1, per_support = sklearn["precision_recall_fscore_support"](
            y_true,
            y_pred,
            labels=label_list,
            average=None,
            zero_division=0,
        )
        precision_macro, recall_macro, f1_macro, _ = sklearn["precision_recall_fscore_support"](
            y_true,
            y_pred,
            labels=label_list,
            average="macro",
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = sklearn[
            "precision_recall_fscore_support"
        ](
            y_true,
            y_pred,
            labels=label_list,
            average="weighted",
            zero_division=0,
        )
        balanced: float | None
        if n_classes >= 2:
            balanced = float(sklearn["balanced_accuracy_score"](y_true, y_pred))
        else:
            balanced = None
            notes.append(
                "Balanced accuracy is undefined when y_true has one class and was omitted."
            )
        matrix: list[list[int]] | None
        if len(label_list) <= _CLASS_LIST_LIMIT:
            matrix = sklearn["confusion_matrix"](y_true, y_pred, labels=label_list).tolist()
            per_class = [
                {
                    "label": label,
                    "precision": float(per_precision[index]),
                    "recall": float(per_recall[index]),
                    "f1": float(per_f1[index]),
                    "support": int(per_support[index]),
                }
                for index, label in enumerate(label_list)
            ]
        else:
            matrix = None
            per_class = []
            notes.append(
                "Confusion matrix and per-class rates were omitted because more than 20 labels are present."
            )
    metrics: dict[str, Any] = {
        "n_rows": len(y_true),
        "n_classes_in_y_true": n_classes,
        "label_order": observed,
        "positive_label": observed[-1] if n_classes == 2 else None,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": label_list,
            "matrix": matrix,
            "omitted": matrix is None,
        },
        "averaging": (
            "macro is the unweighted mean across the labels present in y_true or y_pred; "
            "weighted is the support-weighted mean. zero_division is 0."
        ),
    }
    _add_score_metrics(metrics, notes, y_true, observed, scores, sklearn)
    return metrics, notes


def _add_score_metrics(
    metrics: dict[str, Any],
    notes: list[str],
    y_true: list[object],
    observed: list[object],
    scores: np.ndarray | None,
    sklearn: dict[str, Any],
) -> None:
    metrics["roc_auc"] = None
    metrics["roc_auc_reason"] = None
    metrics["log_loss"] = None
    metrics["brier"] = None
    if scores is None:
        metrics["roc_auc_reason"] = "not_requested"
        notes.append(
            "ROC AUC, log loss, and Brier score were not computed because y_score was omitted."
        )
        return
    n_classes = len(observed)
    if n_classes < 2:
        metrics["roc_auc_reason"] = "single_class"
        notes.append("ROC AUC was not computed because y_true has one class.")
        return
    try:
        if scores.ndim == 1:
            auc = float(sklearn["roc_auc_score"](y_true, scores, labels=observed))
        elif n_classes == 2:
            auc = float(sklearn["roc_auc_score"](y_true, scores[:, 1], labels=observed))
        else:
            auc = float(
                sklearn["roc_auc_score"](
                    y_true,
                    scores,
                    labels=observed,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except ValueError:
        metrics["roc_auc_reason"] = "roc_auc_rejected"
        notes.append("ROC AUC was rejected for these labels and scores and was omitted.")
        return
    metrics["roc_auc"] = auc
    metrics["roc_auc_average"] = (
        "binary score for the greater sorted label" if n_classes == 2 else "weighted one-vs-rest"
    )
    probabilities = _probability_matrix(scores, n_classes)
    if probabilities is None:
        notes.append(
            "Log loss and Brier score were omitted because y_score is not a probability matrix "
            "in sorted label order. ROC AUC can still use ranking scores."
        )
        return
    true_positions = [observed.index(label) for label in y_true]
    true_probability = probabilities[np.arange(len(y_true)), true_positions]
    if np.any(true_probability <= 0.0):
        metrics["log_loss"] = None
        notes.append(
            "Log loss is undefined because at least one true-class probability is 0. "
            "A clipped finite substitute was not reported."
        )
    else:
        metrics["log_loss"] = float(sklearn["log_loss"](y_true, probabilities, labels=observed))
        notes.append(
            "Log loss is the scikit-learn value. Probabilities of exactly 0 or 1 can be "
            "clipped inside that implementation, so a perfect probability can be reported "
            "as a tiny positive number rather than 0."
        )
    if n_classes == 2:
        metrics["brier"] = float(
            sklearn["brier_score_loss"](y_true, probabilities[:, 1], pos_label=observed[-1])
        )
    else:
        notes.append("Brier score is reported only for binary probabilities.")


def _imbalance_check(
    recorder: CheckRecorder,
    y_true: list[object],
    observed: list[object],
    config: AuditConfig,
) -> None:
    rule_id = "METRIC.ACCURACY_ONLY_IMBALANCED"
    if not recorder.enabled(rule_id):
        return
    counts: dict[object, int] = {}
    for label in y_true:
        counts[label] = counts.get(label, 0) + 1
    n_rows = len(y_true)
    majority = max(counts.values()) / n_rows
    minority = min(counts.values()) / n_rows
    single_class = len(observed) == 1
    evidence = {
        "n_rows": n_rows,
        "n_classes": len(observed),
        "majority_fraction": majority,
        "minority_fraction": minority,
        "minority_fraction_threshold": config.imbalance_minority_fraction,
        "majority_fraction_threshold": config.imbalance_majority_fraction,
        "single_class": single_class,
    }
    imbalanced = (
        minority <= config.imbalance_minority_fraction
        or majority >= config.imbalance_majority_fraction
    )
    if single_class or imbalanced:
        recorder.detect(
            make_finding(
                rule_id,
                "The evaluation target has only one class, so accuracy is not a class-wise summary."
                if single_class
                else "Accuracy can mislead because the evaluation target crosses the imbalance thresholds.",
                evidence,
                severity=Severity.HIGH if single_class else Severity.MEDIUM,
                title=(
                    "Evaluation target contains a single class"
                    if single_class
                    else "Accuracy is a weak summary of this classification target"
                ),
            )
        )
        return
    recorder.pass_(
        rule_id,
        "The evaluation target does not cross the configured imbalance thresholds.",
        evidence,
    )


def _auc_check(
    recorder: CheckRecorder,
    scores: np.ndarray | None,
    n_classes: int,
    auc: object,
    reason: object,
) -> None:
    rule_id = "METRIC.INVALID_AUC_INPUT"
    if not recorder.enabled(rule_id):
        return
    if scores is None:
        recorder.skip(rule_id, "ROC AUC was not requested because y_score was omitted.")
        return
    if auc is None:
        recorder.detect(
            make_finding(
                rule_id,
                "ROC AUC could not be computed from the supplied labels and scores.",
                {"reason": reason, "n_classes_in_y_true": n_classes},
            )
        )
        return
    recorder.pass_(
        rule_id,
        "ROC AUC was computed from the supplied labels and scores.",
        {"n_classes_in_y_true": n_classes},
    )


def _sample_check(recorder: CheckRecorder, n_rows: int, config: AuditConfig) -> None:
    rule_id = "METRIC.SMALL_EVALUATION_SAMPLE"
    if not recorder.enabled(rule_id):
        return
    evidence = {"n_rows": n_rows, "minimum": config.small_evaluation_rows}
    if n_rows < config.small_evaluation_rows:
        recorder.detect(
            make_finding(
                rule_id,
                f"The evaluation sample has {n_rows} rows, below the configured minimum "
                f"of {config.small_evaluation_rows}.",
                evidence,
            )
        )
        return
    recorder.pass_(rule_id, "The evaluation sample meets the configured size minimum.", evidence)


def _resample_interval(
    y_true: list[object] | np.ndarray,
    y_pred: list[object] | np.ndarray,
    *,
    metric_name: str,
    point: float,
    score: Any,
    config: AuditConfig,
) -> dict[str, Any]:
    """Percentile interval for one metric under an iid row resample.

    Grouped and temporal requests are rejected before this function runs.
    """
    if config.bootstrap_samples is None:
        if config.observation_dependence == "independent":
            reason = "Bootstrap samples were not requested. No interval was computed."
        else:
            reason = (
                f"observation_dependence is {config.observation_dependence}. "
                "No interval was computed. Grouped and temporal bootstrap are not "
                "implemented, and an iid interval was not substituted."
            )
        return {"implemented": False, "random_seed_used": False, "reason": reason}
    n_rows = len(y_true)
    if n_rows < 2:
        return {
            "implemented": False,
            "random_seed_used": False,
            "metric": metric_name,
            "reason": "Fewer than 2 rows. A resampling interval was not computed.",
        }
    rng = np.random.default_rng(config.random_seed)
    stats = np.empty(int(config.bootstrap_samples), dtype=float)
    for draw in range(int(config.bootstrap_samples)):
        chosen = rng.integers(0, n_rows, size=n_rows)
        if isinstance(y_true, np.ndarray):
            stats[draw] = float(score(y_true[chosen], y_pred[chosen]))
        else:
            stats[draw] = float(
                score(
                    [y_true[int(index)] for index in chosen],
                    [y_pred[int(index)] for index in chosen],
                )
            )
    low, high = (float(value) for value in np.quantile(stats, [0.025, 0.975], method="linear"))
    return {
        "implemented": True,
        "method": "percentile",
        "level": 0.95,
        "samples": int(config.bootstrap_samples),
        "random_seed": int(config.random_seed),
        "random_seed_used": True,
        "dependence": "independent",
        "assumption": (
            "Rows are resampled independently with replacement. Predictions stay fixed; "
            "the model is not retrained. This is not a grouped or temporal interval, and "
            "it does not include training variability."
        ),
        "metric": metric_name,
        "point": float(point),
        "low": low,
        "high": high,
        "width": high - low,
    }


def _bootstrap_check(
    recorder: CheckRecorder,
    interval: dict[str, Any],
    config: AuditConfig,
) -> None:
    rule_id = "METRIC.WIDE_CONFIDENCE_INTERVAL"
    if not recorder.enabled(rule_id):
        return
    if not interval["implemented"]:
        recorder.skip(rule_id, str(interval["reason"]), {"implemented": False})
        return
    evidence = {
        "metric": interval["metric"],
        "level": interval["level"],
        "samples": interval["samples"],
        "low": interval["low"],
        "high": interval["high"],
        "width": interval["width"],
        "threshold": config.confidence_interval_max_width,
        "dependence": "independent",
        "comparison": "flag when width > threshold",
    }
    if config.confidence_interval_max_width is None:
        recorder.skip(
            rule_id,
            "An iid percentile interval was computed, but no maximum width was configured, "
            "so the width was not judged.",
            evidence,
        )
        return
    if float(interval["width"]) > float(config.confidence_interval_max_width):
        recorder.detect(
            make_finding(
                rule_id,
                "The iid percentile interval is wider than the configured maximum.",
                evidence,
            )
        )
        return
    recorder.pass_(
        rule_id,
        "The iid percentile interval is within the configured maximum width.",
        evidence,
    )


def _regression(
    true_array: np.ndarray,
    pred_array: np.ndarray,
    y_score: object | None,
    config: AuditConfig,
    dependence: dict[str, Any],
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    y_true = _regression_values("y_true", true_array)
    y_pred = _regression_values("y_pred", pred_array)
    if y_score is not None:
        raise MlpipelinecheckInputError(
            "y_score is not used for regression. Omit it; do not pass class probabilities."
        )
    sklearn = _import_sklearn()
    mae = float(sklearn["mean_absolute_error"](y_true, y_pred))
    rmse = float(math.sqrt(float(sklearn["mean_squared_error"](y_true, y_pred))))
    signed = float(np.mean(y_pred - y_true))
    notes: list[str] = []
    if float(np.min(y_true)) == float(np.max(y_true)):
        r2 = None
        notes.append("R^2 is undefined when y_true is constant and was omitted.")
    else:
        r2 = float(sklearn["r2_score"](y_true, y_pred))
    metrics = {
        "n_rows": int(y_true.size),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "signed_mean_error": signed,
        "signed_mean_error_definition": (
            "mean(y_pred - y_true); positive means predictions are larger than actuals on average"
        ),
        "r2_note": "R^2 is not an error magnitude. A constant y_true makes R^2 undefined.",
    }
    recorder = CheckRecorder(config.disabled_checks)
    recorder.skip(
        "METRIC.ACCURACY_ONLY_IMBALANCED",
        "This check applies to classification, not regression.",
    )
    recorder.skip(
        "METRIC.INVALID_AUC_INPUT",
        "ROC AUC applies to classification, not regression.",
    )
    _sample_check(recorder, int(y_true.size), config)
    interval = _resample_interval(
        y_true,
        y_pred,
        metric_name="mae",
        point=mae,
        score=lambda left, right: float(sklearn["mean_absolute_error"](left, right)),
        config=config,
    )
    _bootstrap_check(recorder, interval, config)
    _assert_coverage(recorder)
    return (
        recorder.findings,
        recorder.checks,
        _metadata("regression", int(y_true.size), metrics, notes, config, interval, dependence),
    )


def _metadata(
    task: str,
    n_rows: int,
    metrics: dict[str, Any],
    notes: list[str],
    config: AuditConfig,
    interval: dict[str, Any],
    dependence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": task,
        "n_rows": n_rows,
        "comparison": "positional; indexes are not aligned",
        "metrics": metrics,
        "metric_notes": notes,
        "observation_dependence": dependence,
        "bootstrap": interval,
    }


def _assert_coverage(recorder: CheckRecorder) -> None:
    seen = [check.id for check in recorder.checks]
    if seen != list(PREDICTION_RULE_IDS):
        raise RuntimeError(f"Prediction checks were {seen}, expected {list(PREDICTION_RULE_IDS)}.")
