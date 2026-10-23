"""Source constraints: no network, training, pickle loading, or placeholders."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "mlpipelinecheck"
FORBIDDEN_MODULES = {"requests", "socket", "pickle", "joblib", "urllib"}


def _modules(node: ast.AST) -> list[str]:
    found: list[str] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Import):
            found.extend(alias.name for alias in item.names)
        elif isinstance(item, ast.ImportFrom) and item.module:
            found.append(item.module)
    return found


def test_library_does_not_import_network_pickle_or_fit_models() -> None:
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _modules(tree):
            assert module.split(".")[0] not in FORBIDDEN_MODULES, path
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"fit", "fit_transform", "fit_predict"}, path
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    raise AssertionError(f"bare except in {path}")
                if isinstance(node.type, ast.Name) and node.type.id in {
                    "Exception",
                    "BaseException",
                }:
                    assert path.name == "cli.py"
            if isinstance(node, ast.Pass):
                raise AssertionError(f"pass statement in {path}")


def test_no_placeholders_or_local_paths() -> None:
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "todo" not in lowered
        assert "fixme" not in lowered
        assert "notimplementederror" not in lowered
        assert "placeholder" not in lowered
        assert "c:\\users" not in lowered
        assert "d:\\" not in lowered
