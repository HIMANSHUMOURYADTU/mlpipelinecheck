"""HTML escaping and aggregate-only reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlpipelinecheck import audit_dataset, audit_predictions


def test_html_escapes_hostile_column_names_and_labels(tmp_path: Path) -> None:
    hostile = "<script>alert(1)</script>"
    frame = pd.DataFrame({hostile: [1.0, None], "churn": [0, 1]})
    report = audit_dataset(frame, target="churn", task="classification")
    html_path = tmp_path / "report.html"
    report.to_html(html_path)
    text = html_path.read_text(encoding="utf-8")
    assert "<script>" not in text.lower()
    assert "&lt;script&gt;" in text
    assert "alert(1)" in text

    metric = audit_predictions(
        [hostile, "ok", hostile, "ok"],
        [hostile, "ok", "ok", "ok"],
        task="classification",
    )
    metric_html = tmp_path / "metrics.html"
    metric.to_html(metric_html)
    rendered = metric_html.read_text(encoding="utf-8")
    assert "<script>" not in rendered.lower()
    assert "&lt;script&gt;" in rendered


def test_reports_do_not_include_raw_feature_values(tmp_path: Path) -> None:
    sentinel = "RAW_ROW_SENTINEL_9f3a"
    frame = pd.DataFrame(
        {
            "note": [sentinel, "ordinary", "ordinary", "ordinary"],
            "churn": [0, 1, 0, 1],
        }
    )
    report = audit_dataset(frame, target="churn", task="classification")
    html_path = tmp_path / "report.html"
    report.to_html(html_path)
    blob = report.to_json() + html_path.read_text(encoding="utf-8")
    assert sentinel not in blob
