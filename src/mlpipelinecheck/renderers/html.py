"""Standalone offline HTML rendering for an AuditReport dictionary."""

from __future__ import annotations

import html
from typing import Any


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _render_value(value: object, depth: int = 0) -> str:
    if depth > 8:
        return "<span>truncated</span>"
    if isinstance(value, dict):
        if not value:
            return "<span>none</span>"
        items = "".join(
            f"<li><strong>{_esc(key)}</strong>: {_render_value(item, depth + 1)}</li>"
            for key, item in value.items()
        )
        return f"<ul>{items}</ul>"
    if isinstance(value, list):
        if not value:
            return "<span>none</span>"
        items = "".join(f"<li>{_render_value(item, depth + 1)}</li>" for item in value)
        return f"<ul>{items}</ul>"
    if value is None:
        return "<span>null</span>"
    if isinstance(value, bool):
        return _esc("true" if value else "false")
    return _esc(value)


_CSS = """
body { margin: 0; background: #f4f1ea; color: #1d1a16; font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif; }
main { max-width: 920px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }
h1, h2, h3 { font-family: "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.25; }
h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.25rem; margin-top: 2rem; }
.banner, article, table, .note { background: #fffdf8; border: 1px solid #d9d3c5; }
.banner, .note, article { padding: 0.9rem 1rem; margin: 1rem 0; }
.meta { color: #3d3832; }
.sev { font-weight: 700; text-transform: lowercase; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
th, td { border-bottom: 1px solid #d9d3c5; text-align: left; vertical-align: top; padding: 0.45rem 0.5rem; font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 0.95rem; }
th { background: #efeae0; }
ul { margin: 0.3rem 0 0.3rem 1.1rem; padding: 0; }
"""


def render_html(report: dict[str, Any]) -> str:
    """Render a report dictionary produced by :meth:`AuditReport.to_dict`.

    Every inserted value is escaped. The page has no scripts and no remote assets.
    """
    summary = report["summary"]
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Mlpipelinecheck report</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<main>",
        "<h1>Mlpipelinecheck report</h1>",
        f'<p class="meta">Audit: {_esc(report.get("audit_type"))}. '
        f"Schema {_esc(report.get('schema_version'))}. "
        f"Package {_esc(report.get('package_version'))}. "
        f"Created {_esc(report.get('created_at'))}.</p>",
        '<section class="banner">',
        f"<h2>Status: {_esc(summary.get('status'))}</h2>",
        f"<p>{_esc(summary.get('disclaimer'))}</p>",
        "<p><strong>Findings by severity.</strong> "
        + " ".join(
            f"{_esc(name)}={_esc(summary['by_severity'][name])}"
            for name in ("critical", "high", "medium", "low", "info")
        )
        + "</p>",
        "<p><strong>Checks.</strong> "
        + " ".join(
            f"{_esc(name)}={_esc(summary['checks_by_status'][name])}"
            for name in ("detected", "not_detected", "not_applicable", "error")
        )
        + "</p>",
        "</section>",
        "<h2>Findings</h2>",
    ]
    findings = report.get("findings") or []
    if not findings:
        parts.append("<p>No configured issues were detected by the checks that ran.</p>")
    for finding in findings:
        parts.append("<article>")
        parts.append(
            f'<h3><span class="sev">{_esc(finding.get("severity"))}</span> '
            f"{_esc(finding.get('title'))}</h3>"
        )
        parts.append(
            f"<p><strong>{_esc(finding.get('id'))}</strong>. "
            f"Confidence: {_esc(finding.get('confidence'))}. "
            f"Category: {_esc(finding.get('category'))}.</p>"
        )
        parts.append(f"<p>{_esc(finding.get('summary'))}</p>")
        parts.append("<h4>Evidence</h4>")
        parts.append(_render_value(finding.get("evidence")))
        parts.append(
            f"<p><strong>Why it matters.</strong> {_esc(finding.get('why_it_matters'))}</p>"
        )
        parts.append(
            f"<p><strong>Recommended action.</strong> {_esc(finding.get('recommended_action'))}</p>"
        )
        parts.append(f"<p><strong>Limitations.</strong> {_esc(finding.get('limitations'))}</p>")
        if finding.get("docs_url"):
            parts.append(f"<p>Rule reference: {_esc(finding.get('docs_url'))}</p>")
        parts.append("</article>")
    parts.append("<h2>Checks</h2>")
    parts.append(
        "<table><thead><tr><th>Check</th><th>Status</th><th>Summary</th></tr></thead><tbody>"
    )
    for check in report.get("checks") or []:
        parts.append(
            "<tr>"
            f"<td>{_esc(check.get('id'))}</td>"
            f"<td>{_esc(check.get('status'))}</td>"
            f"<td>{_esc(check.get('summary'))}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    parts.append("<h2>Context</h2>")
    parts.append('<div class="note">')
    parts.append("<h3>Configuration</h3>")
    parts.append(_render_value(report.get("config")))
    parts.append("<h3>Metadata</h3>")
    parts.append(_render_value(report.get("metadata")))
    parts.append("</div>")
    parts.append(
        '<p class="meta">This file is standalone. It does not load remote assets or execute scripts. '
        "It renders the same report object as the JSON export.</p>"
    )
    parts.append("</main></body></html>")
    return "\n".join(parts) + "\n"
