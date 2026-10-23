"""Cell token and fingerprint semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from mlpipelinecheck._encoding import UnsupportedValue, cell_token, fingerprint_frame


def test_missing_tokens_match_and_text_does_not() -> None:
    assert cell_token(None) == cell_token(float("nan")) == cell_token(pd.NA)
    assert cell_token("nan") != cell_token(float("nan"))
    assert cell_token("<NA>") != cell_token(pd.NA)


def test_numbers_bools_and_text_are_distinct() -> None:
    assert cell_token(1) == cell_token(1.0)
    assert cell_token(True) != cell_token(1)
    assert cell_token("1") != cell_token(1)
    assert cell_token(0.5) != cell_token(1)


def test_containers_and_unsupported_values() -> None:
    assert cell_token([1, {"b": 2, "a": 1}]) == cell_token([1, {"a": 1, "b": 2}])
    with pytest.raises(UnsupportedValue):
        cell_token({1, 2})


def test_fingerprint_ignores_index_and_column_order() -> None:
    left = pd.DataFrame({"b": [1, 2], "a": ["x", "y"]}, index=[9, 9])
    right = pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]}, index=[0, 1])
    assert fingerprint_frame(left) == fingerprint_frame(right)
