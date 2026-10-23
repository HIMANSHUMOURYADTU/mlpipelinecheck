"""Prediction metric tests against hand-computed values."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from mlpipelinecheck import AuditConfig, audit_predictions
from mlpipelinecheck.exceptions import MlpipelinecheckInputError
from mlpipelinecheck.findings import CheckStatus


def _ids(report) -> set[str]:
    return {finding.id for finding in report.findings}


def test_binary_metrics_match_hand_calculation() -> None:
    report = audit_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        task="classification",
        y_score=[0.1, 0.4, 0.35, 0.8],
        config=AuditConfig(small_evaluation_rows=1),
    )
    metrics = report.metadata["metrics"]
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["precision_macro"] == pytest.approx((1.0 + (2.0 / 3.0)) / 2.0)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["positive_label"] == 1
    assert "METRIC.INVALID_AUC_INPUT" not in _ids(report)
    assert "METRIC.ACCURACY_ONLY_IMBALANCED" not in _ids(report)


def test_probability_matrix_log_loss_and_brier() -> None:
    y_true = [0, 1]
    y_score = np.array([[0.8, 0.2], [0.4, 0.6]])
    report = audit_predictions(
        y_true,
        [0, 1],
        task="classification",
        y_score=y_score,
        config=AuditConfig(small_evaluation_rows=1),
    )
    metrics = report.metadata["metrics"]
    expected_brier = ((0.2 - 0.0) ** 2 + (0.6 - 1.0) ** 2) / 2.0
    assert metrics["brier"] == pytest.approx(expected_brier)
    expected_log_loss = -(math.log(0.8) + math.log(0.6)) / 2.0
    assert metrics["log_loss"] == pytest.approx(expected_log_loss)


def test_regression_metrics_and_undefined_r2() -> None:
    report = audit_predictions(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 6.0],
        task="regression",
        config=AuditConfig(small_evaluation_rows=1),
    )
    metrics = report.metadata["metrics"]
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(3.0))
    assert metrics["signed_mean_error"] == pytest.approx(1.0)
    assert metrics["r2"] == pytest.approx(-3.5)
    constant = audit_predictions(
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 3.0],
        task="regression",
        config=AuditConfig(small_evaluation_rows=1),
    )
    assert constant.metadata["metrics"]["r2"] is None
    assert constant.metadata["bootstrap"]["implemented"] is False
    assert (
        "interval" not in constant.to_json()
        or "No interval" in constant.to_json()
        or "not implemented" in constant.to_json().lower()
        or "deferred" in constant.to_json().lower()
    )


def test_imbalance_warning_and_misleading_accuracy() -> None:
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10
    report = audit_predictions(
        y_true, y_pred, task="classification", config=AuditConfig(small_evaluation_rows=1)
    )
    assert report.metadata["metrics"]["accuracy"] == pytest.approx(0.9)
    assert report.metadata["metrics"]["balanced_accuracy"] == pytest.approx(0.5)
    finding = next(item for item in report.findings if item.id == "METRIC.ACCURACY_ONLY_IMBALANCED")
    assert finding.severity.value == "medium"
    assert "poor" not in finding.summary.lower()


def test_one_class_and_invalid_shapes() -> None:
    one = audit_predictions(
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        task="classification",
        y_score=[0.2, 0.4, 0.1, 0.3],
        config=AuditConfig(small_evaluation_rows=1),
    )
    assert "METRIC.INVALID_AUC_INPUT" in _ids(one)
    imbalance = next(item for item in one.findings if item.id == "METRIC.ACCURACY_ONLY_IMBALANCED")
    assert imbalance.severity.value == "high"
    assert one.metadata["metrics"]["roc_auc"] is None
    with pytest.raises(MlpipelinecheckInputError, match="3 columns|columns"):
        audit_predictions(
            [0, 1, 0, 1], [0, 1, 0, 1], task="classification", y_score=np.ones((4, 3))
        )
    with pytest.raises(MlpipelinecheckInputError, match="one-dimensional|n_classes|columns"):
        audit_predictions(
            [0, 1, 2, 0], [0, 1, 2, 1], task="classification", y_score=[0.1, 0.2, 0.3, 0.4]
        )


def test_input_validation() -> None:
    with pytest.raises(MlpipelinecheckInputError, match="3 values"):
        audit_predictions([0, 1, 0], [0, 1], task="classification")
    with pytest.raises(MlpipelinecheckInputError, match="missing"):
        audit_predictions([0, np.nan], [0, 1], task="classification")
    with pytest.raises(MlpipelinecheckInputError, match="non-finite"):
        audit_predictions([0.0, 1.0], [0.0, np.inf], task="regression")
    with pytest.raises(MlpipelinecheckInputError, match="does not coerce"):
        audit_predictions([0, 1], ["0", "1"], task="classification")
    with pytest.raises(MlpipelinecheckInputError, match="task"):
        audit_predictions([0, 1], [0, 1], task="auto")
    y_true = pd.Series([0, 1, 0], index=[2, 1, 0])
    y_pred = pd.Series([0, 1, 0], index=[0, 1, 2])
    report = audit_predictions(
        y_true, y_pred, task="classification", config=AuditConfig(small_evaluation_rows=1)
    )
    assert report.metadata["metrics"]["accuracy"] == pytest.approx(1.0)
    assert report.metadata["comparison"].startswith("positional")


def test_small_sample_and_deferred_interval() -> None:
    report = audit_predictions([0, 1], [0, 1], task="classification")
    assert "METRIC.SMALL_EVALUATION_SAMPLE" in _ids(report)
    assert next(
        check.status for check in report.checks if check.id == "METRIC.WIDE_CONFIDENCE_INTERVAL"
    ) is (CheckStatus.NOT_APPLICABLE)
    text = report.to_json()
    assert "bootstrap_low" not in text
    assert report.metadata["bootstrap"]["implemented"] is False
    assert "not requested" in report.metadata["bootstrap"]["reason"].lower()


def test_ranking_scores_do_not_invent_log_loss() -> None:
    report = audit_predictions(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        task="classification",
        y_score=[-1.0, -0.2, 0.3, 2.0],
        config=AuditConfig(small_evaluation_rows=1),
    )
    assert report.metadata["metrics"]["roc_auc"] is not None
    assert report.metadata["metrics"]["log_loss"] is None
    assert report.metadata["metrics"]["brier"] is None


def test_regression_sentinel_value_is_not_copied_into_the_report() -> None:
    sentinel = 42424242.0
    report = audit_predictions([0.0, 0.0, 0.0, sentinel], [0.0, 0.0, 0.0, 0.0], task="regression")
    payload = json.loads(report.to_json())
    # created_at is an ISO timestamp. A short digit sentinel can appear in the clock.
    payload.pop("created_at")
    rendered = json.dumps(payload, sort_keys=True)
    assert "42424242" not in rendered
    assert payload["metadata"]["metrics"]["mae"] == pytest.approx(sentinel / 4)


def test_zero_true_class_probability_does_not_report_clipped_log_loss() -> None:
    report = audit_predictions(
        [0, 1],
        [0, 1],
        task="classification",
        y_score=[[0.0, 1.0], [1.0, 0.0]],
        config=AuditConfig(small_evaluation_rows=1),
    )
    assert report.metadata["metrics"]["log_loss"] is None
    assert report.metadata["metrics"]["brier"] == pytest.approx(1.0)
    assert "undefined" in " ".join(report.metadata["metric_notes"]).lower()


def test_iid_bootstrap_matches_an_independent_resample() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    samples = 40
    report = audit_predictions(
        y_true,
        y_pred,
        task="classification",
        config=AuditConfig(bootstrap_samples=samples, random_seed=0, small_evaluation_rows=1),
    )
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(samples):
        chosen = rng.integers(0, 4, size=4)
        stats.append(float(np.mean([y_true[int(index)] == y_pred[int(index)] for index in chosen])))
    low, high = np.quantile(stats, [0.025, 0.975], method="linear")
    interval = report.metadata["bootstrap"]
    assert interval["implemented"] is True
    assert interval["metric"] == "accuracy"
    assert interval["point"] == pytest.approx(0.75)
    assert interval["low"] == pytest.approx(float(low))
    assert interval["high"] == pytest.approx(float(high))
    other = audit_predictions(
        y_true,
        y_pred,
        task="classification",
        config=AuditConfig(bootstrap_samples=samples, random_seed=1, small_evaluation_rows=1),
    )
    assert (other.metadata["bootstrap"]["low"], other.metadata["bootstrap"]["high"]) != (
        interval["low"],
        interval["high"],
    )


def test_grouped_bootstrap_is_refused_and_group_ids_are_not_copied() -> None:
    from mlpipelinecheck.exceptions import MlpipelinecheckConfigError

    with pytest.raises(MlpipelinecheckConfigError, match="random_seed"):
        AuditConfig(random_seed=4)
    with pytest.raises(MlpipelinecheckInputError, match="not implemented"):
        audit_predictions(
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            task="classification",
            groups=["SECRET_GROUP_ZZZ", "SECRET_GROUP_ZZZ", "OTHER_GROUP_YYY", "OTHER_GROUP_YYY"],
            config=AuditConfig(
                bootstrap_samples=20,
                observation_dependence="grouped",
                small_evaluation_rows=1,
            ),
        )
    report = audit_predictions(
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        task="classification",
        groups=["SECRET_GROUP_ZZZ", "SECRET_GROUP_ZZZ", "OTHER_GROUP_YYY", "OTHER_GROUP_YYY"],
        config=AuditConfig(observation_dependence="grouped", small_evaluation_rows=1),
    )
    assert "SECRET_GROUP_ZZZ" not in report.to_json()
    assert report.metadata["observation_dependence"]["groups"]["n_unique"] == 2
    assert report.metadata["bootstrap"]["implemented"] is False
