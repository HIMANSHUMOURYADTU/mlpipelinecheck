"""CLI exit codes and documented commands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mlpipelinecheck.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def test_help_and_invalid_command(capsys) -> None:
    assert main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "audit-dataset" in help_text
    assert "Exit codes" in help_text
    assert main(["audit-dataset", "missing.csv"]) == 2


def test_dataset_gate_and_reports(tmp_path: Path, capsys) -> None:
    clean = tmp_path / "clean.csv"
    _write_csv(
        clean,
        pd.DataFrame(
            {
                "age": [25, 40, 33, 52],
                "plan": ["a", "b", "a", "b"],
                "churn": [0, 1, 0, 1],
            }
        ),
    )
    html = tmp_path / "report.html"
    json_path = tmp_path / "report.json"
    assert (
        main(
            [
                "audit-dataset",
                str(clean),
                "--target",
                "churn",
                "--task",
                "classification",
                "--html",
                str(html),
                "--json",
                str(json_path),
            ]
        )
        == 0
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.1.0"
    assert html.is_file()
    assert "Gate: no breach" in capsys.readouterr().out

    leaked = tmp_path / "leak.csv"
    _write_csv(
        leaked,
        pd.DataFrame(
            {
                "age": [25, 40, 33, 52],
                "churn": [0, 1, 0, 1],
                "churn_copy": [0, 1, 0, 1],
            }
        ),
    )
    assert (
        main(["audit-dataset", str(leaked), "--target", "churn", "--task", "classification"]) == 1
    )
    assert (
        main(
            [
                "audit-dataset",
                str(leaked),
                "--target",
                "churn",
                "--task",
                "classification",
                "--min-severity",
                "critical",
            ]
        )
        == 0
    )


def test_split_and_prediction_commands(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    _write_csv(
        train,
        pd.DataFrame(
            {"age": [20, 30, 40, 50], "customer_id": [1, 2, 3, 4], "target": [0, 1, 0, 1]}
        ),
    )
    _write_csv(
        test,
        pd.DataFrame(
            {"age": [21, 31, 41, 51], "customer_id": [5, 6, 7, 8], "target": [1, 0, 1, 0]}
        ),
    )
    assert (
        main(
            [
                "audit-split",
                str(train),
                str(test),
                "--target",
                "target",
                "--task",
                "classification",
                "--group-column",
                "customer_id",
                "--html",
                str(tmp_path / "split.html"),
            ]
        )
        == 0
    )
    predictions = tmp_path / "predictions.csv"
    _write_csv(predictions, pd.DataFrame({"true": [1.0, 2.0, 3.0], "pred": [1.1, 1.9, 3.2]}))
    assert (
        main(
            [
                "audit-predictions",
                str(predictions),
                "--target",
                "true",
                "--prediction-column",
                "pred",
                "--task",
                "regression",
                "--html",
                str(tmp_path / "predictions.html"),
                "--min-severity",
                "high",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "audit-predictions",
                str(predictions),
                "--target",
                "missing",
                "--prediction-column",
                "pred",
                "--task",
                "regression",
            ]
        )
        == 2
    )


def test_documented_examples_if_present() -> None:
    data = ROOT / "examples" / "data"
    if not (data / "classification.csv").exists():
        return
    commands = [
        [
            sys.executable,
            "-m",
            "mlpipelinecheck",
            "audit-dataset",
            str(data / "classification.csv"),
            "--target",
            "churn",
            "--task",
            "classification",
        ],
        [
            sys.executable,
            "-m",
            "mlpipelinecheck",
            "audit-split",
            str(data / "train.csv"),
            str(data / "test.csv"),
            "--target",
            "churn",
            "--task",
            "classification",
            "--group-column",
            "customer_id",
        ],
        [
            sys.executable,
            "-m",
            "mlpipelinecheck",
            "audit-predictions",
            str(data / "predictions.csv"),
            "--target",
            "true",
            "--prediction-column",
            "pred",
            "--task",
            "regression",
        ],
    ]
    for command in commands:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
