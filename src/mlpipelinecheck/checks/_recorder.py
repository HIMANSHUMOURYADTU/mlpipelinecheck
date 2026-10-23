"""Small helper that records every check exactly once."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mlpipelinecheck.findings import CheckOutcome, CheckStatus, Finding
from mlpipelinecheck.rules import RULES


class CheckRecorder:
    """Collect findings and check outcomes for one audit."""

    def __init__(self, disabled_checks: Sequence[str]) -> None:
        self.disabled = set(disabled_checks)
        self.findings: list[Finding] = []
        self.checks: list[CheckOutcome] = []

    def enabled(self, rule_id: str) -> bool:
        """Return whether the check should run, recording a skip when disabled."""
        if rule_id not in self.disabled:
            return True
        self.checks.append(
            CheckOutcome(
                id=rule_id,
                status=CheckStatus.NOT_APPLICABLE,
                summary="Check disabled by configuration.",
            )
        )
        return False

    def pass_(self, rule_id: str, summary: str, details: dict[str, Any] | None = None) -> None:
        self._outcome(rule_id, CheckStatus.NOT_DETECTED, summary, details)

    def skip(self, rule_id: str, summary: str, details: dict[str, Any] | None = None) -> None:
        self._outcome(rule_id, CheckStatus.NOT_APPLICABLE, summary, details)

    def fail_check(self, rule_id: str, summary: str, details: dict[str, Any] | None = None) -> None:
        """Record a check that could not be completed.

        This is not a detected finding and it is not a pass.
        """
        self._outcome(rule_id, CheckStatus.ERROR, summary, details)

    def detect(self, finding: Finding) -> None:
        self.findings.append(finding)
        self.checks.append(
            CheckOutcome(
                id=finding.id,
                status=CheckStatus.DETECTED,
                summary=finding.summary,
            )
        )

    def _outcome(
        self,
        rule_id: str,
        status: CheckStatus,
        summary: str,
        details: dict[str, Any] | None,
    ) -> None:
        self.checks.append(
            CheckOutcome(
                id=rule_id,
                status=status,
                summary=summary,
                details={} if details is None else details,
            )
        )


def make_finding(
    rule_id: str,
    summary: str,
    evidence: dict[str, Any],
    *,
    severity: Any = None,
    confidence: Any = None,
    title: str | None = None,
) -> Finding:
    """Build a finding from the catalog, optionally overriding severity or title."""
    spec = RULES[rule_id]
    return Finding(
        id=spec.id,
        title=title or spec.title,
        category=spec.category,
        severity=severity or spec.default_severity,
        confidence=confidence or spec.default_confidence,
        summary=summary,
        evidence=evidence,
        why_it_matters=spec.why_it_matters,
        recommended_action=spec.recommended_action,
        limitations=spec.limitations,
        docs_url=spec.docs_url,
    )
