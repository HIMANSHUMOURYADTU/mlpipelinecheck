"""Classification dataset audit from docs/quickstart.md."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from mlpipelinecheck import audit_dataset


def main() -> None:
    """Audit a small balanced classification table and write both report formats."""
    frame = pd.DataFrame(
        {
            "age": [25, 40, 33, 52],
            "plan": ["a", "b", "a", "b"],
            "churn": [0, 1, 0, 1],
        }
    )
    report = audit_dataset(frame, target="churn", task="classification")
    status = report.summary()["status"]
    print(status)
    if status != "no configured issues found":
        raise SystemExit(1)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        json_path = folder / "classification.json"
        html_path = folder / "classification.html"
        report.to_json(json_path)
        report.to_html(html_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        html = html_path.read_text(encoding="utf-8")
        if payload["summary"]["status"] != status or status not in html:
            raise SystemExit("JSON and HTML reports did not match the audit")
    print("wrote JSON and HTML reports")


if __name__ == "__main__":
    main()
