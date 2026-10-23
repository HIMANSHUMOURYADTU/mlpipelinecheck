"""Command-line interface for Mlpipelinecheck.

Exit codes:
0. The audit finished and the configured gate did not match.
1. The configured severity gate matched one or more findings.
2. The input or configuration is invalid, or a check could not be completed.
3. An unexpected internal error occurred.

A check recorded with status ``error`` returns 2 even when no finding meets the
gate, so an incomplete audit is not reported as a clean pass. If the gate also
matches, the process still returns 2 and the summary says the gate would have
breached. Reports are written before the exit code is chosen.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mlpipelinecheck.api import audit_dataset, audit_predictions, audit_split
from mlpipelinecheck.config import DEPENDENCE_VALUES, SPLIT_STRATEGIES, AuditConfig
from mlpipelinecheck.exceptions import (
    MlpipelinecheckConfigError,
    MlpipelinecheckError,
    MlpipelinecheckInputError,
)
from mlpipelinecheck.findings import GATE_SEVERITY_NAMES, CheckStatus
from mlpipelinecheck.report import AuditReport


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser used by the ``mlpipelinecheck`` command."""
    parser = argparse.ArgumentParser(
        prog="mlpipelinecheck",
        description=(
            "Audit tabular datasets, explicit train/test splits, and supplied predictions. "
            "Mlpipelinecheck reports evidence. It does not prove that an evaluation is free of leakage."
        ),
        epilog=(
            "Examples:\n"
            "  mlpipelinecheck audit-dataset data.csv --target target --task classification "
            "--html report.html --json report.json\n"
            "  mlpipelinecheck audit-split train.csv test.csv --target target --task classification "
            "--group-column customer_id --html split-report.html\n"
            "  mlpipelinecheck audit-predictions predictions.csv --target true --prediction-column pred "
            "--task regression --html predictions-report.html\n\n"
            "Exit codes: 0 no configured gate breach; 1 gate breached; "
            "2 invalid input or an incomplete check; 3 internal error.\n"
            "The default gate is severity high. Informational and low findings do not fail it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("audit-dataset", help="Audit one CSV dataset.")
    dataset.add_argument("data", help="CSV file to audit.")
    dataset.add_argument(
        "--target", help="Target column name. Omit to skip target-specific checks."
    )
    dataset.add_argument(
        "--task",
        choices=["auto", "classification", "regression"],
        help="Task type. Defaults to auto, which infers a task only when that is unambiguous.",
    )
    dataset.add_argument(
        "--id-column",
        action="append",
        dest="id_columns",
        help="Identifier column to exclude from features. Repeatable.",
    )
    dataset.add_argument(
        "--feature-column",
        action="append",
        dest="feature_columns",
        help="Feature column. Repeatable. If omitted, every column except the target and identifiers is a feature.",
    )
    _add_common(dataset)

    split = subparsers.add_parser("audit-split", help="Audit explicit train and test CSV files.")
    split.add_argument("train", help="Training CSV.")
    split.add_argument("test", help="Test CSV.")
    split.add_argument("--target", help="Target column present in both files.")
    split.add_argument("--task", choices=["auto", "classification", "regression"])
    split.add_argument(
        "--group-column", help="Entity identifier that should not appear in both partitions."
    )
    split.add_argument("--time-column", help="Timestamp column used only when supplied.")
    split.add_argument("--split-strategy", choices=sorted(SPLIT_STRATEGIES))
    split.add_argument("--id-column", action="append", dest="id_columns")
    split.add_argument("--feature-column", action="append", dest="feature_columns")
    _add_common(split)

    predictions = subparsers.add_parser(
        "audit-predictions",
        help="Audit a CSV of labels and predictions.",
    )
    predictions.add_argument("data", help="CSV with label and prediction columns.")
    predictions.add_argument(
        "--target", required=True, help="Column of true labels, for example 'true'."
    )
    predictions.add_argument(
        "--prediction-column",
        required=True,
        help="Column of predicted labels or regression predictions.",
    )
    predictions.add_argument(
        "--score-column",
        action="append",
        dest="score_columns",
        help=(
            "Optional score or probability column. Repeat once per class, in sorted label order, "
            "for multiclass probabilities."
        ),
    )
    predictions.add_argument("--task", required=True, choices=["classification", "regression"])
    predictions.add_argument(
        "--bootstrap-samples",
        type=int,
        help="Number of iid bootstrap resamples. Omit to skip intervals. Minimum 20.",
    )
    predictions.add_argument(
        "--random-seed",
        type=int,
        help="Seed for the bootstrap resample. Requires --bootstrap-samples unless it is 0.",
    )
    predictions.add_argument(
        "--observation-dependence",
        choices=sorted(DEPENDENCE_VALUES),
        help="independent, grouped, or temporal. Grouped and temporal intervals are refused.",
    )
    predictions.add_argument(
        "--max-interval-width",
        type=float,
        help="Fail the width check when the interval is wider than this. Requires a bootstrap.",
    )
    _add_common(predictions)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--html", help="Write a standalone HTML report to this path.")
    parser.add_argument("--json", dest="json_path", help="Write a JSON report to this path.")
    parser.add_argument(
        "--config", help="JSON object of AuditConfig fields. CLI flags override it."
    )
    parser.add_argument(
        "--min-severity",
        choices=sorted(GATE_SEVERITY_NAMES),
        help="Fail when a finding reaches this severity. Default: high. Use 'none' to disable the gate.",
    )
    parser.add_argument(
        "--fail-on",
        choices=sorted(GATE_SEVERITY_NAMES),
        help="Alias of --min-severity. If both are set, they must match.",
    )
    parser.add_argument(
        "--rule-id",
        action="append",
        dest="rule_ids",
        help="Also require this rule id for the gate. Repeatable. Unknown ids are an error.",
    )
    parser.add_argument(
        "--disable-check",
        action="append",
        dest="disabled_checks",
        help="Skip this rule id. Repeatable.",
    )
    parser.add_argument(
        "--max-rows", type=int, help="Skip fingerprint-based checks above this row count."
    )
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding. Default: utf-8.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 2
    try:
        return _run(args)
    except (MlpipelinecheckInputError, MlpipelinecheckConfigError) as exc:
        print(f"Mlpipelinecheck error: {exc}", file=sys.stderr)
        return 2
    except MlpipelinecheckError as exc:
        print(f"Mlpipelinecheck error: {exc}", file=sys.stderr)
        return 3
    except Exception:
        print("Mlpipelinecheck internal error.", file=sys.stderr)
        traceback.print_exc()
        return 3


def _run(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if args.command == "audit-dataset":
        frame = _read_csv(args.data, args.encoding)
        report = audit_dataset(
            frame,
            target=args.target,
            task=args.task,
            id_columns=args.id_columns,
            feature_columns=args.feature_columns,
            config=config,
        )
    elif args.command == "audit-split":
        report = audit_split(
            _read_csv(args.train, args.encoding),
            _read_csv(args.test, args.encoding),
            target=args.target,
            task=args.task,
            group_column=args.group_column,
            time_column=args.time_column,
            split_strategy=args.split_strategy,
            id_columns=args.id_columns,
            feature_columns=args.feature_columns,
            config=config,
        )
    elif args.command == "audit-predictions":
        frame = _read_csv(args.data, args.encoding)
        y_true, y_pred, y_score = _prediction_columns(frame, args)
        report = audit_predictions(y_true, y_pred, task=args.task, y_score=y_score, config=config)
    else:
        raise MlpipelinecheckInputError(f"Unknown command {args.command!r}.")
    _write_reports(report, args)
    gate = args.min_severity or args.fail_on or config.min_severity
    matched = report.matching_findings(min_severity=gate, rule_ids=args.rule_ids)
    errored = any(check.status is CheckStatus.ERROR for check in report.checks)
    _print_summary(report, gate=gate, matched=len(matched), errored=errored)
    if errored:
        return 2
    if matched:
        return 1
    return 0


def _load_config(args: argparse.Namespace) -> AuditConfig:
    if (
        args.min_severity is not None
        and args.fail_on is not None
        and args.min_severity != args.fail_on
    ):
        raise MlpipelinecheckConfigError("--min-severity and --fail-on do not match.")
    if args.config is None:
        base = AuditConfig()
    else:
        path = Path(args.config)
        if not path.is_file():
            raise MlpipelinecheckInputError(f"Configuration file not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MlpipelinecheckConfigError("Configuration file is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise MlpipelinecheckConfigError("Configuration JSON must be an object.")
        base = AuditConfig.from_mapping(payload)
    disabled = list(base.disabled_checks)
    if args.disabled_checks:
        disabled.extend(args.disabled_checks)
    overrides: dict[str, object] = {
        "min_severity": args.min_severity or args.fail_on,
        "disabled_checks": disabled if args.disabled_checks else None,
        "max_rows": args.max_rows,
    }
    for attribute, key in (
        ("bootstrap_samples", "bootstrap_samples"),
        ("random_seed", "random_seed"),
        ("observation_dependence", "observation_dependence"),
        ("max_interval_width", "confidence_interval_max_width"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[key] = value
    return base.with_overrides(**overrides)


def _read_csv(path: str, encoding: str) -> pd.DataFrame:
    file = Path(path)
    if not file.is_file():
        raise MlpipelinecheckInputError(f"File not found: {file}")
    try:
        return pd.read_csv(file, encoding=encoding)
    except pd.errors.EmptyDataError as exc:
        raise MlpipelinecheckInputError(
            f"Could not read {file.name}: the file has no columns."
        ) from exc
    except pd.errors.ParserError as exc:
        raise MlpipelinecheckInputError(
            f"Could not read {file.name}: the file is not valid CSV."
        ) from exc
    except UnicodeError as exc:
        raise MlpipelinecheckInputError(
            f"Could not read {file.name} with encoding {encoding!r}."
        ) from exc


def _prediction_columns(
    frame: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.Series, pd.Series, pd.Series | pd.DataFrame | None]:
    missing = [name for name in (args.target, args.prediction_column) if name not in frame.columns]
    if args.score_columns:
        missing.extend(name for name in args.score_columns if name not in frame.columns)
    if missing:
        raise MlpipelinecheckInputError(
            "Missing column(s): " + ", ".join(repr(name) for name in missing) + "."
        )
    y_score: pd.Series | pd.DataFrame | None
    if not args.score_columns:
        y_score = None
    elif len(args.score_columns) == 1:
        y_score = frame[args.score_columns[0]]
    else:
        y_score = frame[list(args.score_columns)]
    return frame[args.target], frame[args.prediction_column], y_score


def _write_reports(report: AuditReport, args: argparse.Namespace) -> None:
    if args.json_path:
        report.to_json(args.json_path)
    if args.html:
        report.to_html(args.html)


def _print_summary(report: AuditReport, *, gate: str, matched: int, errored: bool) -> None:
    summary = report.summary()
    counts = summary["by_severity"]
    checks = summary["checks_by_status"]
    print(f"Mlpipelinecheck {report.audit_type} audit")
    print(f"Status: {summary['status']}")
    print(
        "Findings: "
        + " ".join(
            f"{name}={counts[name]}" for name in ("critical", "high", "medium", "low", "info")
        )
    )
    print(
        "Checks: "
        + " ".join(
            f"{name}={checks[name]}"
            for name in ("detected", "not_detected", "not_applicable", "error")
        )
    )
    for finding in report.findings:
        print(f"- [{finding.severity.value}] {finding.id}: {finding.title}")
    if gate == "none":
        print("Gate: disabled")
    elif matched:
        print(f"Gate: breached at severity >= {gate} ({matched} finding(s))")
    else:
        print(f"Gate: no breach at severity >= {gate}")
    if errored and matched:
        print("One or more checks could not be completed. The gate would also have breached.")
    elif errored:
        print("One or more checks could not be completed. See the report.")
    print("Note: no findings does not prove an evaluation is leakage-free.")


if __name__ == "__main__":
    sys.exit(main())
