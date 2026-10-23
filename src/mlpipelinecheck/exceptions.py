"""Public exceptions raised by Mlpipelinecheck."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlpipelinecheck.findings import Finding


class MlpipelinecheckError(Exception):
    """Base class for errors raised by Mlpipelinecheck."""


class MlpipelinecheckInputError(MlpipelinecheckError):
    """The audit input is missing, malformed, or cannot be interpreted safely."""


class MlpipelinecheckConfigError(MlpipelinecheckError):
    """Configuration or gate settings contain unknown or invalid values."""


class MlpipelinecheckFindingsError(MlpipelinecheckError):
    """Raised when findings meet a configured severity gate."""

    def __init__(self, findings: list[Finding], min_severity: str) -> None:
        self.findings = list(findings)
        self.min_severity = min_severity
        parts = [f"{finding.id} ({finding.severity.value})" for finding in self.findings]
        listed = ", ".join(parts)
        super().__init__(
            f"{len(self.findings)} finding(s) at or above severity {min_severity!r}: {listed}"
        )
