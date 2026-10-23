"""Validated audit configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mlpipelinecheck.exceptions import MlpipelinecheckConfigError
from mlpipelinecheck.findings import GATE_SEVERITY_NAMES
from mlpipelinecheck.rules import RULE_IDS

TASKS: frozenset[str] = frozenset({"auto", "classification", "regression"})
PREDICTION_TASKS: frozenset[str] = frozenset({"classification", "regression"})
SPLIT_STRATEGIES: frozenset[str] = frozenset({"random", "stratified", "group", "temporal"})

_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "task",
        "id_columns",
        "feature_columns",
        "group_column",
        "time_column",
        "split_strategy",
        "missing_fraction_threshold",
        "near_constant_fraction",
        "high_cardinality_ratio",
        "high_cardinality_min_unique",
        "high_cardinality_min_rows",
        "high_cardinality_include_floats",
        "imbalance_minority_fraction",
        "imbalance_majority_fraction",
        "class_shift_max_abs_diff",
        "class_shift_min_rows",
        "numeric_shift_min_abs_smd",
        "numeric_shift_min_rows",
        "small_test_rows",
        "small_test_fraction",
        "small_evaluation_rows",
        "max_rows",
        "disabled_checks",
        "min_severity",
        "random_seed",
        "bootstrap_samples",
        "observation_dependence",
        "confidence_interval_max_width",
        "privacy_include_examples",
    }
)

DEPENDENCE_VALUES: frozenset[str] = frozenset({"independent", "grouped", "temporal"})


def _reject_bool_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MlpipelinecheckConfigError(f"{name} must be a number, not {type(value).__name__}.")


def _as_float(
    name: str,
    value: object,
    *,
    low: float,
    high: float,
    inclusive_low: bool = True,
    inclusive_high: bool = True,
) -> float:
    _reject_bool_number(name, value)
    number = float(value)
    if not math.isfinite(number):
        raise MlpipelinecheckConfigError(f"{name} must be a finite number.")
    below = number < low if inclusive_low else number <= low
    above = number > high if inclusive_high else number >= high
    if below or above:
        left = "[" if inclusive_low else "("
        right = "]" if inclusive_high else ")"
        raise MlpipelinecheckConfigError(f"{name} must be in {left}{low}, {high}{right}.")
    return number


def _as_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MlpipelinecheckConfigError(f"{name} must be an integer >= {minimum}.")
    if value < minimum:
        raise MlpipelinecheckConfigError(f"{name} must be an integer >= {minimum}.")
    return value


def _as_optional_float(
    name: str,
    value: object,
    *,
    low: float,
    high: float,
    inclusive_low: bool = True,
    inclusive_high: bool = True,
) -> float | None:
    if value is None:
        return None
    return _as_float(
        name,
        value,
        low=low,
        high=high,
        inclusive_low=inclusive_low,
        inclusive_high=inclusive_high,
    )


def _as_string_list(name: str, value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise MlpipelinecheckConfigError(f"{name} must be a list of strings.")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise MlpipelinecheckConfigError(f"{name} must contain non-empty strings.")
        items.append(item)
    return items


def _as_optional_string_list(name: str, value: object) -> list[str] | None:
    if value is None:
        return None
    return _as_string_list(name, value)


def _as_optional_str(name: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MlpipelinecheckConfigError(f"{name} must be a non-empty string or null.")
    return value


@dataclass
class AuditConfig:
    """Thresholds and column roles for an audit.

    Unknown keys are rejected. The defaults are intentionally not a leakage
    verdict: several checks are heuristic and default severities stay below the
    CLI gate of ``high``.

    ``random_seed`` is used only when ``bootstrap_samples`` is set. A non-zero seed
    without a bootstrap request is rejected, because the seed would not change
    the audit. The default seed of 0 with no bootstrap request is unused and is
    recorded as such on prediction reports.

    ``observation_dependence`` selects the sampling assumption for a prediction
    interval. Grouped and temporal intervals are not computed. Requesting a
    bootstrap under those assumptions is an error rather than an iid interval.

    ``privacy_include_examples`` must stay false. Example rows are not implemented.
    """

    task: str = "auto"
    id_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] | None = None
    group_column: str | None = None
    time_column: str | None = None
    split_strategy: str | None = None
    missing_fraction_threshold: float = 0.0
    near_constant_fraction: float | None = 0.99
    high_cardinality_ratio: float = 0.95
    high_cardinality_min_unique: int = 20
    high_cardinality_min_rows: int = 20
    high_cardinality_include_floats: bool = False
    imbalance_minority_fraction: float = 0.1
    imbalance_majority_fraction: float = 0.9
    class_shift_max_abs_diff: float = 0.1
    class_shift_min_rows: int = 30
    numeric_shift_min_abs_smd: float = 0.8
    numeric_shift_min_rows: int = 30
    small_test_rows: int = 30
    small_test_fraction: float | None = None
    small_evaluation_rows: int = 30
    max_rows: int | None = None
    disabled_checks: list[str] = field(default_factory=list)
    min_severity: str = "high"
    random_seed: int = 0
    bootstrap_samples: int | None = None
    observation_dependence: str = "independent"
    confidence_interval_max_width: float | None = None
    privacy_include_examples: bool = False

    def __post_init__(self) -> None:
        self.id_columns = _as_string_list("id_columns", self.id_columns)
        self.feature_columns = _as_optional_string_list("feature_columns", self.feature_columns)
        self.group_column = _as_optional_str("group_column", self.group_column)
        self.time_column = _as_optional_str("time_column", self.time_column)
        self.split_strategy = _as_optional_str("split_strategy", self.split_strategy)
        if not isinstance(self.task, str) or self.task not in TASKS:
            known = ", ".join(sorted(TASKS))
            raise MlpipelinecheckConfigError(f"task must be one of: {known}.")
        if self.split_strategy is not None and self.split_strategy not in SPLIT_STRATEGIES:
            known = ", ".join(sorted(SPLIT_STRATEGIES))
            raise MlpipelinecheckConfigError(f"split_strategy must be one of: {known}.")
        self.missing_fraction_threshold = _as_float(
            "missing_fraction_threshold", self.missing_fraction_threshold, low=0.0, high=1.0
        )
        self.near_constant_fraction = _as_optional_float(
            "near_constant_fraction",
            self.near_constant_fraction,
            low=0.0,
            high=1.0,
            inclusive_low=False,
        )
        self.high_cardinality_ratio = _as_float(
            "high_cardinality_ratio",
            self.high_cardinality_ratio,
            low=0.0,
            high=1.0,
            inclusive_low=False,
        )
        self.high_cardinality_min_unique = _as_int(
            "high_cardinality_min_unique", self.high_cardinality_min_unique, minimum=2
        )
        self.high_cardinality_min_rows = _as_int(
            "high_cardinality_min_rows", self.high_cardinality_min_rows, minimum=1
        )
        if not isinstance(self.high_cardinality_include_floats, bool):
            raise MlpipelinecheckConfigError("high_cardinality_include_floats must be a boolean.")
        self.imbalance_minority_fraction = _as_float(
            "imbalance_minority_fraction", self.imbalance_minority_fraction, low=0.0, high=1.0
        )
        self.imbalance_majority_fraction = _as_float(
            "imbalance_majority_fraction", self.imbalance_majority_fraction, low=0.0, high=1.0
        )
        self.class_shift_max_abs_diff = _as_float(
            "class_shift_max_abs_diff",
            self.class_shift_max_abs_diff,
            low=0.0,
            high=1.0,
            inclusive_low=False,
        )
        self.class_shift_min_rows = _as_int(
            "class_shift_min_rows", self.class_shift_min_rows, minimum=1
        )
        self.numeric_shift_min_abs_smd = _as_float(
            "numeric_shift_min_abs_smd",
            self.numeric_shift_min_abs_smd,
            low=0.0,
            high=1e9,
            inclusive_low=False,
        )
        self.numeric_shift_min_rows = _as_int(
            "numeric_shift_min_rows", self.numeric_shift_min_rows, minimum=2
        )
        self.small_test_rows = _as_int("small_test_rows", self.small_test_rows, minimum=1)
        self.small_test_fraction = _as_optional_float(
            "small_test_fraction",
            self.small_test_fraction,
            low=0.0,
            high=1.0,
            inclusive_low=False,
        )
        self.small_evaluation_rows = _as_int(
            "small_evaluation_rows", self.small_evaluation_rows, minimum=1
        )
        if self.max_rows is not None:
            self.max_rows = _as_int("max_rows", self.max_rows, minimum=1)
        self.disabled_checks = _as_string_list("disabled_checks", self.disabled_checks)
        unknown_checks = sorted(set(self.disabled_checks) - RULE_IDS)
        if unknown_checks:
            raise MlpipelinecheckConfigError(
                "Unknown check IDs in disabled_checks: "
                f"{unknown_checks}. Known IDs include {sorted(RULE_IDS)}."
            )
        if not isinstance(self.min_severity, str) or self.min_severity not in GATE_SEVERITY_NAMES:
            known = ", ".join(sorted(GATE_SEVERITY_NAMES))
            raise MlpipelinecheckConfigError(f"min_severity must be one of: {known}.")
        self.random_seed = _as_int("random_seed", self.random_seed, minimum=0)
        if self.bootstrap_samples is not None:
            self.bootstrap_samples = _as_int(
                "bootstrap_samples", self.bootstrap_samples, minimum=20
            )
        if self.observation_dependence not in DEPENDENCE_VALUES:
            known = ", ".join(sorted(DEPENDENCE_VALUES))
            raise MlpipelinecheckConfigError(f"observation_dependence must be one of: {known}.")
        self.confidence_interval_max_width = _as_optional_float(
            "confidence_interval_max_width",
            self.confidence_interval_max_width,
            low=0.0,
            high=1e9,
        )
        if self.confidence_interval_max_width is not None and self.bootstrap_samples is None:
            raise MlpipelinecheckConfigError(
                "confidence_interval_max_width requires bootstrap_samples. "
                "A width gate without an interval would not change the audit."
            )
        if self.bootstrap_samples is None and self.random_seed != 0:
            raise MlpipelinecheckConfigError(
                "random_seed does not change an audit unless bootstrap_samples is set. "
                "Set bootstrap_samples to compute an iid percentile interval, or leave "
                "random_seed at 0."
            )
        if not isinstance(self.privacy_include_examples, bool):
            raise MlpipelinecheckConfigError("privacy_include_examples must be a boolean.")
        if self.privacy_include_examples:
            raise MlpipelinecheckConfigError(
                "privacy_include_examples=true is not supported in Mlpipelinecheck 0.2.0. "
                "Reports stay aggregate-only; leave this value false."
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> AuditConfig:
        """Build a config from a mapping, rejecting unknown keys."""
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise MlpipelinecheckConfigError("Configuration must be a mapping of named options.")
        unknown = sorted(set(data) - _ALLOWED_KEYS)
        if unknown:
            raise MlpipelinecheckConfigError(f"Unknown configuration keys: {unknown}.")
        try:
            return cls(**dict(data))
        except TypeError as exc:
            raise MlpipelinecheckConfigError(f"Invalid configuration: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy of the configuration."""
        return {
            "task": self.task,
            "id_columns": list(self.id_columns),
            "feature_columns": None if self.feature_columns is None else list(self.feature_columns),
            "group_column": self.group_column,
            "time_column": self.time_column,
            "split_strategy": self.split_strategy,
            "missing_fraction_threshold": self.missing_fraction_threshold,
            "near_constant_fraction": self.near_constant_fraction,
            "high_cardinality_ratio": self.high_cardinality_ratio,
            "high_cardinality_min_unique": self.high_cardinality_min_unique,
            "high_cardinality_min_rows": self.high_cardinality_min_rows,
            "high_cardinality_include_floats": self.high_cardinality_include_floats,
            "imbalance_minority_fraction": self.imbalance_minority_fraction,
            "imbalance_majority_fraction": self.imbalance_majority_fraction,
            "class_shift_max_abs_diff": self.class_shift_max_abs_diff,
            "class_shift_min_rows": self.class_shift_min_rows,
            "numeric_shift_min_abs_smd": self.numeric_shift_min_abs_smd,
            "numeric_shift_min_rows": self.numeric_shift_min_rows,
            "small_test_rows": self.small_test_rows,
            "small_test_fraction": self.small_test_fraction,
            "small_evaluation_rows": self.small_evaluation_rows,
            "max_rows": self.max_rows,
            "disabled_checks": list(self.disabled_checks),
            "min_severity": self.min_severity,
            "random_seed": self.random_seed,
            "bootstrap_samples": self.bootstrap_samples,
            "observation_dependence": self.observation_dependence,
            "confidence_interval_max_width": self.confidence_interval_max_width,
            "privacy_include_examples": self.privacy_include_examples,
        }

    def with_overrides(self, **overrides: Any) -> AuditConfig:
        """Return a copy with non-null overrides applied.

        The caller's config is not modified. ``None`` keeps the current value,
        which lets API defaults mean "use the config".
        """
        data = self.to_dict()
        for key, value in overrides.items():
            if key not in _ALLOWED_KEYS:
                raise MlpipelinecheckConfigError(f"Unknown configuration override: {key}.")
            if value is not None:
                data[key] = value
        return AuditConfig.from_mapping(data)
