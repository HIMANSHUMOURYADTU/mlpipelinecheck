"""Foundation tests: config, findings, severity, and report serialization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlpipelinecheck import (
    SCHEMA_VERSION,
    AuditConfig,
    AuditReport,
    CheckStatus,
    Finding,
    Severity,
    __version__,
)
from mlpipelinecheck.exceptions import MlpipelinecheckConfigError, MlpipelinecheckFindingsError
from mlpipelinecheck.findings import Category, CheckOutcome, Confidence, severity_at_least
from mlpipelinecheck.rules import RULES


def _finding(rule_id: str = "DATA.EMPTY_DATASET", severity: Severity = Severity.HIGH) -> Finding:
    spec = RULES[rule_id]
    return Finding(
        id=spec.id,
        title=spec.title,
        category=spec.category,
        severity=severity,
        confidence=spec.default_confidence,
        summary="Observed during a test.",
        evidence={"n_rows": np.int64(0)},
        why_it_matters=spec.why_it_matters,
        recommended_action=spec.recommended_action,
        limitations=spec.limitations,
        docs_url=spec.docs_url,
    )


def _check(rule_id: str, status: CheckStatus) -> CheckOutcome:
    return CheckOutcome(id=rule_id, status=status, summary="Checked.")


def test_version_and_schema_are_set() -> None:
    assert __version__ == "0.2.0"
    assert SCHEMA_VERSION == "1.1.0"


def test_severity_order() -> None:
    assert severity_at_least(Severity.CRITICAL, Severity.HIGH)
    assert severity_at_least(Severity.HIGH, Severity.HIGH)
    assert not severity_at_least(Severity.MEDIUM, Severity.HIGH)
    assert not severity_at_least(Severity.INFO, Severity.LOW)
    assert list(Severity)[0] is Severity.CRITICAL


def test_config_rejects_unknown_keys_and_invalid_values() -> None:
    with pytest.raises(MlpipelinecheckConfigError, match="Unknown configuration keys"):
        AuditConfig.from_mapping({"not_a_key": 1})
    with pytest.raises(MlpipelinecheckConfigError):
        AuditConfig(task="forecast")
    with pytest.raises(MlpipelinecheckConfigError):
        AuditConfig(missing_fraction_threshold=1.5)
    with pytest.raises(MlpipelinecheckConfigError):
        AuditConfig(random_seed=True)  # type: ignore[arg-type]
    with pytest.raises(MlpipelinecheckConfigError, match="privacy_include_examples"):
        AuditConfig(privacy_include_examples=True)
    with pytest.raises(MlpipelinecheckConfigError, match="Unknown check"):
        AuditConfig(disabled_checks=["NOT.A.RULE"])


def test_config_round_trip_and_override_does_not_mutate() -> None:
    original = AuditConfig(missing_fraction_threshold=0.2, id_columns=["user_id"])
    updated = original.with_overrides(task="classification")
    assert original.task == "auto"
    assert updated.task == "classification"
    assert updated.missing_fraction_threshold == 0.2
    assert original.id_columns == ["user_id"]
    restored = AuditConfig.from_mapping(original.to_dict())
    assert restored.to_dict() == original.to_dict()


def test_report_with_no_findings_is_not_a_clearance() -> None:
    report = AuditReport(
        "dataset",
        [],
        [_check("DATA.EMPTY_DATASET", CheckStatus.NOT_DETECTED)],
        config=AuditConfig().to_dict(),
        metadata={"n_rows": 3},
    )
    summary = report.summary()
    assert summary["status"] == "no configured issues found"
    assert summary["by_severity"] == {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    assert "does not mean" in summary["disclaimer"]
    assert report.raise_for_findings(min_severity="high") is None


def test_report_sorts_findings_and_serializes_numpy() -> None:
    low = _finding("DATA.CONSTANT_FEATURE", Severity.LOW)
    high = _finding("DATA.TARGET_COPY_FEATURE", Severity.HIGH)
    report = AuditReport(
        "dataset",
        [low, high],
        [
            _check("DATA.CONSTANT_FEATURE", CheckStatus.DETECTED),
            _check("DATA.TARGET_COPY_FEATURE", CheckStatus.DETECTED),
        ],
        config={"task": "classification"},
        metadata={
            "mean": np.float64("nan"),
            "count": np.int64(2),
            "stamp": pd.Timestamp("2024-01-02"),
        },
    )
    assert [item.id for item in report.findings] == [
        "DATA.TARGET_COPY_FEATURE",
        "DATA.CONSTANT_FEATURE",
    ]
    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "1.1.0"
    assert payload["package_version"] == "0.2.0"
    assert payload["metadata"]["mean"] is None
    assert payload["metadata"]["count"] == 2
    assert payload["findings"][1]["evidence"]["n_rows"] == 0
    assert payload["summary"]["status"] == "issues found"
    json.dumps(payload, allow_nan=False)
    text = report.to_json()
    assert text == json.dumps(
        json.loads(text), indent=2, allow_nan=False, ensure_ascii=False, sort_keys=True
    )


def test_errored_check_marks_the_audit_incomplete() -> None:
    report = AuditReport(
        "dataset",
        [],
        [_check("DATA.DUPLICATE_ROWS", CheckStatus.ERROR)],
        config={},
        metadata={},
    )
    assert report.summary()["status"] == "audit incomplete"
    assert report.summary()["checks_by_status"]["error"] == 1


def test_raise_for_findings_respects_severity_and_rule_ids() -> None:
    report = AuditReport(
        "dataset",
        [_finding("DATA.MISSING_VALUES", Severity.MEDIUM)],
        [_check("DATA.MISSING_VALUES", CheckStatus.DETECTED)],
        config={},
        metadata={},
    )
    report.raise_for_findings(min_severity="high")
    with pytest.raises(MlpipelinecheckFindingsError) as caught:
        report.raise_for_findings(min_severity="medium")
    assert caught.value.findings[0].id == "DATA.MISSING_VALUES"
    report.raise_for_findings(min_severity="low", rule_ids=["DATA.EMPTY_DATASET"])
    with pytest.raises(MlpipelinecheckConfigError):
        report.raise_for_findings(rule_ids=["NO.SUCH.RULE"])


def test_finding_requires_text_and_consistent_checks() -> None:
    with pytest.raises(ValueError):
        Finding(
            id="BAD",
            title="t",
            category=Category.DATA_QUALITY,
            severity=Severity.LOW,
            confidence=Confidence.HIGH,
            summary="s",
            evidence={},
            why_it_matters="w",
            recommended_action="a",
            limitations="l",
        )
    with pytest.raises(ValueError, match="detected"):
        AuditReport(
            "dataset",
            [],
            [_check("DATA.EMPTY_DATASET", CheckStatus.DETECTED)],
            config={},
            metadata={},
        )
