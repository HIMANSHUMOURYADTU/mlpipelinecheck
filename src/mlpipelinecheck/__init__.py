"""Mlpipelinecheck public API.

Import audit functions and report types from this package. Detector modules are
internal and may change. IID bootstrap intervals are opt-in on prediction audits.
Grouped and temporal intervals are not computed.
"""

from mlpipelinecheck._version import __version__
from mlpipelinecheck.api import audit_dataset, audit_predictions, audit_split
from mlpipelinecheck.config import AuditConfig
from mlpipelinecheck.exceptions import (
    MlpipelinecheckConfigError,
    MlpipelinecheckError,
    MlpipelinecheckFindingsError,
    MlpipelinecheckInputError,
)
from mlpipelinecheck.findings import (
    Category,
    CheckOutcome,
    CheckStatus,
    Confidence,
    Finding,
    Severity,
)
from mlpipelinecheck.report import SCHEMA_VERSION, AuditReport

__all__ = [
    "SCHEMA_VERSION",
    "AuditConfig",
    "AuditReport",
    "Category",
    "CheckOutcome",
    "CheckStatus",
    "Confidence",
    "MlpipelinecheckConfigError",
    "MlpipelinecheckError",
    "MlpipelinecheckFindingsError",
    "MlpipelinecheckInputError",
    "Finding",
    "Severity",
    "__version__",
    "audit_dataset",
    "audit_predictions",
    "audit_split",
]
