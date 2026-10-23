"""Canonical cell encoding and row fingerprints.

The fingerprint ignores the pandas index. Two cells match when their tokens
match. Rules:

* ``None``, ``NaN``, ``pd.NA``, and ``NaT`` share one missing token.
* The text ``"nan"`` or ``"<NA>"`` is not missing.
* Booleans are not treated as integers, so ``True`` does not match ``1``.
* Finite integer-valued floats inside the exact integer range match that integer,
  so ``1`` matches ``1.0``. Floats with a fractional part use a stable ``repr``.
* Text is not parsed as a number, so ``"1"`` does not match ``1``.
* Lists and dicts use canonical JSON. Sets and arbitrary objects are unsupported.

Row hashes are SHA-256 of a length-prefixed UTF-8 encoding, so values cannot
collide just by containing the separator. Building the hash set takes time and
memory proportional to ``rows * columns`` plus the stored digests.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

_MAX_EXACT_INT = 2**53


class UnsupportedValue(Exception):
    """A cell cannot be given a deterministic token."""

    def __init__(self, type_name: str) -> None:
        self.type_name = type_name
        super().__init__(type_name)


class UnsupportedCell(Exception):
    """A named column contains a value that cannot be fingerprinted."""

    def __init__(self, column: str, type_name: str) -> None:
        self.column = column
        self.type_name = type_name
        super().__init__(
            f"Column {column!r} contains unsupported value type {type_name} "
            "and could not be fingerprinted."
        )


def is_missing(value: object) -> bool:
    """Return whether a scalar should be treated as missing."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (str, bytes, bytearray, list, tuple, dict, set)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(float(value)))
    if isinstance(value, pd.Timestamp):
        return bool(pd.isna(value))
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _json_fragment(value: object) -> object:
    if is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise UnsupportedValue(type(value).__name__)
        return number
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_fragment(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise UnsupportedValue("non-string-dict-key")
        return {str(key): _json_fragment(item) for key, item in value.items()}
    raise UnsupportedValue(type(value).__name__)


def cell_token(value: object) -> str:
    """Return the canonical token for one cell."""
    if is_missing(value):
        return "m"
    if isinstance(value, (bool, np.bool_)):
        return "b1" if bool(value) else "b0"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"i{int(value)}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isinf(number):
            return "inf+" if number > 0 else "inf-"
        if number.is_integer() and abs(number) <= _MAX_EXACT_INT:
            return f"i{int(number)}"
        return f"f{format(number, '.17g')}"
    if isinstance(value, str):
        return f"s{value}"
    if isinstance(value, (bytes, bytearray)):
        return f"x{bytes(value).hex()}"
    if isinstance(value, pd.Timestamp):
        stamp = value
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC")
        return f"t{stamp.isoformat()}"
    if isinstance(value, np.datetime64):
        return cell_token(pd.Timestamp(value))
    if hasattr(value, "isoformat") and value.__class__.__module__ == "datetime":
        return f"t{value.isoformat()}"
    if isinstance(value, (list, tuple, dict)):
        fragment = _json_fragment(value)
        encoded = json.dumps(
            fragment,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return f"j{encoded}"
    raise UnsupportedValue(type(value).__name__)


def hash_tokens(tokens: list[str]) -> str:
    """Hash field tokens with an unambiguous length prefix."""
    chunks: list[bytes] = []
    for token in tokens:
        raw = token.encode("utf-8")
        chunks.append(str(len(raw)).encode("ascii") + b":" + raw)
    payload = b"\x1e".join(chunks)
    return hashlib.sha256(payload).hexdigest()


def fingerprint_frame(frame: pd.DataFrame) -> list[str]:
    """Return one SHA-256 fingerprint per row, in row order.

    Column order does not matter: tokens follow sorted column names.
    The index is not an input. Duplicate column names are rejected by callers
    before this function runs.
    """
    columns = [str(column) for column in frame.columns]
    order = sorted(range(len(columns)), key=lambda position: columns[position])
    values = frame.to_numpy(copy=True)
    fingerprints: list[str] = []
    for row in values:
        tokens: list[str] = []
        for position in order:
            cell = row[position]
            try:
                tokens.append(cell_token(cell))
            except UnsupportedValue as exc:
                raise UnsupportedCell(columns[position], exc.type_name) from exc
        fingerprints.append(hash_tokens(tokens))
    return fingerprints
