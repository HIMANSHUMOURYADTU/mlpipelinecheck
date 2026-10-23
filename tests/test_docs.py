"""Documentation stays aligned with the rule catalog and the public API."""

from __future__ import annotations

from pathlib import Path

from mlpipelinecheck.rules import RULES

ROOT = Path(__file__).resolve().parents[1]


def test_rules_doc_lists_every_id() -> None:
    text = (ROOT / "docs" / "rules.md").read_text(encoding="utf-8")
    for rule_id in RULES:
        assert f"## {rule_id}" in text


def test_api_doc_lists_public_entry_points() -> None:
    text = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    for name in (
        "audit_dataset",
        "audit_split",
        "audit_predictions",
        "AuditReport",
        "AuditConfig",
        "to_json",
        "to_html",
        "raise_for_findings",
        "summary",
    ):
        assert name in text
