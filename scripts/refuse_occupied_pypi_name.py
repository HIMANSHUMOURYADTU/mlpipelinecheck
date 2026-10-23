"""Refuse a release whose name collides with the existing PyPI project.

https://pypi.org/project/evalguard/ is a different package. This script is the
gate in ``.github/workflows/release.yml``. It does not upload anything.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

OCCUPIED_DISTRIBUTION = "evalguard"
OCCUPIED_IMPORT = "evalguard"


def normalized_name(name: str) -> str:
    """Apply the PyPI name normalization used for comparison."""
    return re.sub(r"[-_.]+", "-", name).strip("-").lower()


def filename_distribution(filename: str) -> str:
    """Return the normalized distribution name encoded in a wheel or sdist filename."""
    match = re.match(r"^(?P<name>.+)-(?P<version>\d+\.\d+.*)$", filename)
    if match is None:
        raise ValueError(f"Unrecognized distribution filename: {filename}")
    return normalized_name(match.group("name"))


def wheel_import_names(path: Path) -> set[str]:
    """Return top-level import packages stored in a wheel."""
    imports: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for item in archive.namelist():
            parts = item.split("/")
            if (
                len(parts) >= 2
                and parts[1] == "__init__.py"
                and not parts[0].endswith(".dist-info")
            ):
                imports.add(parts[0])
    return imports


def release_blockers(dist: Path) -> list[str]:
    """Return human-readable reasons this directory must not be uploaded."""
    files = sorted(
        path
        for path in dist.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if not files:
        return [f"{dist} contains no wheel or sdist."]
    blockers: list[str] = []
    for path in files:
        try:
            dist_name = filename_distribution(path.name)
        except ValueError as exc:
            blockers.append(str(exc))
            continue
        if dist_name == OCCUPIED_DISTRIBUTION:
            blockers.append(
                f"{path.name} uses the distribution name {OCCUPIED_DISTRIBUTION!r}, "
                "which is already registered to a different project at "
                "https://pypi.org/project/evalguard/."
            )
        if path.suffix == ".whl" and OCCUPIED_IMPORT in wheel_import_names(path):
            blockers.append(
                f"{path.name} installs the top-level import {OCCUPIED_IMPORT!r}. "
                "The existing PyPI project uses that import. Rename the package "
                "before the first upload."
            )
    return blockers


def main(argv: list[str] | None = None) -> int:
    """Check ``dist/`` or a directory passed as the only argument."""
    args = list(sys.argv[1:] if argv is None else argv)
    dist = Path(args[0] if args else "dist")
    if not dist.is_dir():
        print(f"{dist} is not a directory.", file=sys.stderr)
        return 1
    blockers = release_blockers(dist)
    if blockers:
        print("Refusing to publish:", file=sys.stderr)
        for blocker in blockers:
            print(f"- {blocker}", file=sys.stderr)
        return 1
    print(f"{dist} does not use the occupied name {OCCUPIED_DISTRIBUTION!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
