"""Finding schema, check outcomes, and severity ordering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Impact label for a detected finding.

    Comparison uses :data:`SEVERITY_RANK`. ``critical`` is the highest rank.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Confidence(str, Enum):
    """How directly the evidence supports the finding.

    ``high`` means a structural comparison succeeded. ``medium`` and ``low``
    are heuristic signals and are not claims that leakage occurred.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CheckStatus(str, Enum):
    """Outcome of one configured check.

    Findings are recorded only for ``detected``. ``not_detected`` means the
    check ran and its condition was not met. It does not mean every possible
    evaluation problem was ruled out.
    """

    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class Category(str, Enum):
    """Grouping used to summarize findings."""

    DATA_QUALITY = "data_quality"
    SPLIT_INTEGRITY = "split_integrity"
    METRIC_VALIDITY = "metric_validity"
    STATISTICAL_UNCERTAINTY = "statistical_uncertainty"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

SEVERITY_NAMES: frozenset[str] = frozenset(item.value for item in Severity)
GATE_SEVERITY_NAMES: frozenset[str] = SEVERITY_NAMES | {"none"}


def parse_severity(value: str) -> Severity:
    """Return a :class:`Severity` from its public string value."""
    try:
        return Severity(value)
    except ValueError as exc:
        known = ", ".join(item.value for item in Severity)
        raise ValueError(f"Unknown severity {value!r}. Expected one of: {known}.") from exc


def severity_at_least(value: Severity, minimum: Severity) -> bool:
    """Return whether ``value`` reaches ``minimum``."""
    return SEVERITY_RANK[value] >= SEVERITY_RANK[minimum]


@dataclass(frozen=True)
class Finding:
    """One detected, evidence-backed issue.

    ``evidence`` contains aggregate facts such as counts, rates, column names,
    and metric values. It must not contain raw rows.
    """

    id: str
    title: str
    category: Category
    severity: Severity
    confidence: Confidence
    summary: str
    evidence: dict[str, Any]
    why_it_matters: str
    recommended_action: str
    limitations: str
    docs_url: str | None = None

    def __post_init__(self) -> None:
        if not self.id or "." not in self.id:
            raise ValueError(f"Finding id must look like 'AREA.RULE', got {self.id!r}.")
        for field_name in (
            "title",
            "summary",
            "why_it_matters",
            "recommended_action",
            "limitations",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Finding {self.id} is missing {field_name}.")
        object.__setattr__(self, "evidence", dict(self.evidence))
        if not isinstance(self.category, Category):
            raise ValueError("Finding category must be a Category value.")
        if not isinstance(self.severity, Severity):
            raise ValueError("Finding severity must be a Severity value.")
        if not isinstance(self.confidence, Confidence):
            raise ValueError("Finding confidence must be a Confidence value.")

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-ready dictionary with a stable key order."""
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category.value,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "summary": self.summary,
            "evidence": self.evidence,
            "why_it_matters": self.why_it_matters,
            "recommended_action": self.recommended_action,
            "limitations": self.limitations,
            "docs_url": self.docs_url,
        }


@dataclass(frozen=True)
class CheckOutcome:
    """Result of running one check, whether or not it produced a finding."""

    id: str
    status: CheckStatus
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Check outcome id is required.")
        if not isinstance(self.status, CheckStatus):
            raise ValueError("Check status must be a CheckStatus value.")
        if not str(self.summary).strip():
            raise ValueError(f"Check {self.id} is missing a summary.")
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        """Return the check outcome as a JSON-ready dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "summary": self.summary,
            "details": self.details,
        }
