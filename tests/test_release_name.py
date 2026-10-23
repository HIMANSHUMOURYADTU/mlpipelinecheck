"""The release gate refuses the PyPI name that already belongs to another project."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from scripts.refuse_occupied_pypi_name import filename_distribution, release_blockers

from mlpipelinecheck import __version__


def _wheel(path: Path, import_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{import_name}/__init__.py", "")


def test_filename_normalization_treats_separators_as_the_same_name() -> None:
    assert filename_distribution("evalguard-0.2.0-py3-none-any.whl") == "evalguard"
    assert filename_distribution("evalguard_tabular-0.2.0.tar.gz") == "evalguard-tabular"


def test_current_distribution_name_is_refused(tmp_path: Path) -> None:
    _wheel(tmp_path / "evalguard-0.2.0-py3-none-any.whl", "evalguard")
    blockers = release_blockers(tmp_path)
    assert any("distribution name" in item for item in blockers)
    assert any("top-level import" in item for item in blockers)


def test_distinct_distribution_and_import_are_allowed(tmp_path: Path) -> None:
    _wheel(tmp_path / "mlpipelinecheck-0.2.0-py3-none-any.whl", "mlpipelinecheck")
    assert release_blockers(tmp_path) == []


def test_import_collision_is_refused_even_with_a_new_distribution_name(tmp_path: Path) -> None:
    _wheel(tmp_path / "evalguard_tabular-0.2.0-py3-none-any.whl", "evalguard")
    blockers = release_blockers(tmp_path)
    assert len(blockers) == 1
    assert "top-level import" in blockers[0]


def test_release_workflow_cannot_publish_from_a_pull_request() -> None:
    root = Path(__file__).resolve().parents[1]
    release = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    testpypi = (root / ".github" / "workflows" / "testpypi.yml").read_text(encoding="utf-8")
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pull_request" not in release
    assert "pull_request" not in testpypi
    assert "https://test.pypi.org/legacy/" in testpypi
    assert "workflow_dispatch" in testpypi
    assert "pull_request_target" not in release
    assert "pypa/gh-action-pypi-publish@release/v1" in release
    assert "id-token: write" in release
    assert "password:" not in release
    assert "PYPI_API_TOKEN" not in release
    assert "pull_request:" in ci
    assert "id-token" not in ci
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    version = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in pyproject.splitlines()
        if line.startswith("version = ")
    )
    assert version == __version__


def test_missing_dist_directory_is_a_blocker(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        release_blockers(tmp_path / "missing")
