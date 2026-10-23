"""The synthetic benchmark must keep its labeled fixtures consistent."""

from __future__ import annotations

from benchmarks.synthetic_cases import run_benchmark


def test_labeled_fixtures_match_detector_output() -> None:
    results = run_benchmark(include_scale=False)
    failures = [case["name"] for case in results["cases"] if not case["passed"]]
    assert failures == []
    assert results["structural"]["precision"] == 1
    assert results["structural"]["recall"] == 1
    assert results["heuristic"]["fp"] == 0
    assert results["benign_false_positives"] == []
    assert results["validation_cases"][0]["passed"] is True
    assert "not a field accuracy" in results["scope"] or "not an estimate" in results["scope"]
