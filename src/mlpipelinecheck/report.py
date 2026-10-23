"""Audit report object shared by every public audit."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlpipelinecheck._version import __version__
from mlpipelinecheck.exceptions import (
    MlpipelinecheckConfigError,
    MlpipelinecheckFindingsError,
    MlpipelinecheckInputError,
)
from mlpipelinecheck.findings import (
    SEVERITY_RANK,
    Category,
    CheckOutcome,
    CheckStatus,
    Finding,
    Severity,
    parse_severity,
)
from mlpipelinecheck.rules import RULE_IDS
from mlpipelinecheck.serialization import to_jsonable

SCHEMA_VERSION = "1.1.0"

DISCLAIMER = (
    "No findings does not mean leakage or other evaluation problems have been ruled out. "
    "Mlpipelinecheck reports only the configured checks that ran. "
    "Heuristic signals are not proof of leakage."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditReport:
    """Evidence-oriented result of one dataset, split, or prediction audit.

    A report with zero findings means the checks that ran did not meet their
    configured conditions. It is not a certificate that the evaluation is valid.
    """

    def __init__(
        self,
        audit_type: str,
        findings: Sequence[Finding],
        checks: Sequence[CheckOutcome],
        *,
        config: dict[str, Any],
        metadata: dict[str, Any],
        created_at: str | None = None,
        package_version: str | None = None,
    ) -> None:
        if audit_type not in {"dataset", "split", "predictions"}:
            raise ValueError(f"Unknown audit_type {audit_type!r}.")
        self.audit_type = audit_type
        self.schema_version = SCHEMA_VERSION
        self.package_version = package_version or __version__
        self.created_at = created_at or _utc_now()
        self._config = dict(config)
        self._metadata = dict(metadata)
        self._findings = tuple(
            sorted(findings, key=lambda finding: (-SEVERITY_RANK[finding.severity], finding.id))
        )
        self._checks = tuple(checks)
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        finding_ids = [finding.id for finding in self._findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("An audit report cannot contain two findings with the same id.")
        check_ids = [check.id for check in self._checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("An audit report cannot contain two checks with the same id.")
        detected = {check.id for check in self._checks if check.status is CheckStatus.DETECTED}
        if detected != set(finding_ids):
            raise ValueError(
                "Every detected check needs one finding with the same id, "
                "and every finding needs a detected check."
            )

    @property
    def findings(self) -> list[Finding]:
        """Detected findings, highest severity first."""
        return list(self._findings)

    @property
    def checks(self) -> list[CheckOutcome]:
        """Every check that was considered, including passes, skips, and errors."""
        return list(self._checks)

    @property
    def config(self) -> dict[str, Any]:
        """Configuration recorded for this run."""
        return dict(self._config)

    @property
    def metadata(self) -> dict[str, Any]:
        """Aggregate context such as shape, task, and computed metrics."""
        return dict(self._metadata)

    def summary(self) -> dict[str, Any]:
        """Return counts by severity, category, and check status.

        ``status`` is ``audit incomplete`` when any check errored, even if
        findings were also detected. ``no configured issues found`` does not
        mean the evaluation was proven safe.
        """
        by_severity = {severity.value: 0 for severity in Severity}
        by_category = {category.value: 0 for category in Category}
        for finding in self._findings:
            by_severity[finding.severity.value] += 1
            by_category[finding.category.value] += 1
        by_status = {status.value: 0 for status in CheckStatus}
        for check in self._checks:
            by_status[check.status.value] += 1
        if by_status[CheckStatus.ERROR.value]:
            status = "audit incomplete"
        elif self._findings:
            status = "issues found"
        else:
            status = "no configured issues found"
        return {
            "status": status,
            "n_findings": len(self._findings),
            "by_severity": by_severity,
            "by_category": by_category,
            "n_checks": len(self._checks),
            "checks_by_status": by_status,
            "disclaimer": DISCLAIMER,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-ready report.

        HTML rendering uses this dictionary. It does not recompute the audit.
        """
        payload = {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "audit_type": self.audit_type,
            "created_at": self.created_at,
            "summary": self.summary(),
            "disclaimer": DISCLAIMER,
            "config": self._config,
            "metadata": self._metadata,
            "findings": [finding.to_dict() for finding in self._findings],
            "checks": [check.to_dict() for check in self._checks],
        }
        return to_jsonable(payload)

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize the report to a JSON string and optionally write it to ``path``.

        The file is UTF-8. Parent directories are not created. Non-finite numbers
        are rejected rather than written as non-standard JSON.
        """
        text = json.dumps(
            self.to_dict(),
            indent=indent,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )
        if path is not None:
            self._write_text(path, text + "\n")
        return text

    def to_html(self, path: str | Path) -> Path:
        """Write a standalone offline HTML rendering of :meth:`to_dict`."""
        from mlpipelinecheck.renderers.html import render_html

        destination = self._write_text(path, render_html(self.to_dict()))
        return destination

    @staticmethod
    def _write_text(path: str | Path, text: str) -> Path:
        destination = Path(path)
        if destination.exists() and destination.is_dir():
            raise MlpipelinecheckInputError(f"Report path is a directory: {destination}")
        try:
            destination.write_text(text, encoding="utf-8")
        except OSError as exc:
            reason = exc.strerror or exc.__class__.__name__
            raise MlpipelinecheckInputError(
                f"Could not write report to {destination}: {reason}"
            ) from exc
        return destination

    def matching_findings(
        self,
        min_severity: str = "high",
        rule_ids: Sequence[str] | None = None,
    ) -> list[Finding]:
        """Return findings that meet a gate.

        ``min_severity="none"`` matches nothing. When ``rule_ids`` is provided,
        a finding must also have one of those ids. Unknown rule ids raise
        :class:`MlpipelinecheckConfigError`.
        """
        if min_severity == "none":
            return []
        try:
            minimum = parse_severity(min_severity)
        except ValueError as exc:
            raise MlpipelinecheckConfigError(str(exc)) from exc
        allowed: set[str] | None = None
        if rule_ids is not None:
            unknown = sorted(set(rule_ids) - RULE_IDS)
            if unknown:
                raise MlpipelinecheckConfigError(f"Unknown rule ids in the gate: {unknown}.")
            allowed = set(rule_ids)
        matched: list[Finding] = []
        for finding in self._findings:
            if SEVERITY_RANK[finding.severity] >= SEVERITY_RANK[minimum] and (
                allowed is None or finding.id in allowed
            ):
                matched.append(finding)
        return matched

    def raise_for_findings(
        self,
        min_severity: str = "high",
        rule_ids: Sequence[str] | None = None,
    ) -> None:
        """Raise :class:`MlpipelinecheckFindingsError` when the gate matches any finding."""
        matched = self.matching_findings(min_severity=min_severity, rule_ids=rule_ids)
        if matched:
            raise MlpipelinecheckFindingsError(matched, min_severity=min_severity)

    def __repr__(self) -> str:
        status = self.summary()["status"]
        return (
            f"AuditReport(audit_type={self.audit_type!r}, status={status!r}, "
            f"n_findings={len(self._findings)})"
        )
