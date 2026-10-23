"""Held-out fixture scoring. Structural mismatches fail. Heuristic misses are reported."""

from __future__ import annotations

from benchmarks.heldout_cases import HEURISTIC_RULES as HELD_OUT_HEURISTICS
from benchmarks.heldout_cases import run_heldout
from benchmarks.synthetic_cases import HEURISTIC_RULES as DEVELOPMENT_HEURISTICS


def test_heldout_denominators_and_structural_labels() -> None:
    assert HELD_OUT_HEURISTICS == DEVELOPMENT_HEURISTICS
    result = run_heldout()
    assert result["structural_failures"] == []
    assert result["validation"][0]["passed"] is True
    for stats in result["rules"].values():
        assert stats["tp"] + stats["fp"] + stats["fn"] + stats["tn"] == stats["n"]
        if stats["tp"] + stats["fp"] == 0:
            assert stats["precision"] is None
        if stats["tp"] + stats["fn"] == 0:
            assert stats["recall"] is None
        if stats["fp"] + stats["tn"] == 0:
            assert stats["false_positive_rate"] is None
    assert result["heuristic_micro"]["n"] > 0
    assert result["structural_micro"]["n"] > 0
