"""Where the released surface baseline comes from, and the facts every tier
is measured against: the repository root, the index, and the markers the
version-check workflow reads back.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tomllib

# The extractor is a sibling module. Avoiding bytecode keeps a read-only checkout
# unchanged and does not require an ignore rule for tests/__pycache__.
sys.dont_write_bytecode = True

# facts.py sits in tests/baseline/, so the repository root is three levels up.
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
BASELINE = ROOT / "released-surface.json"
INDEX = "https://pypi.org/pypi"

# Markers. The workflow branches on these, so they are constants here and referenced by
# name, never retyped as prose. `REGISTRY_MARKERS` is the set that asserts "a registry
# serves this exact version"; anything else asserts the opposite.
SDIST = "pypi-sdist"
WHEEL = "pypi-wheel"
ARCHIVE = "git-archive"
HEAD = "head"
REGISTRY_MARKERS = (SDIST, WHEEL)

# The tag question is asked of ORIGIN, never of the working copy. `git ls-remote` spells
# a tag ref this way and lists an ANNOTATED tag a second time under the peel suffix,
# that second line naming the commit the tag points at.
REMOTE = "origin"
TAG_REF_PREFIX = "refs/tags/"
PEELED_SUFFIX = "^{}"


def run(*command: str) -> str:
    """A git command's output, or a loud failure."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), *command], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise SystemExit(f"git {' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def declared(manifest: pathlib.Path) -> tuple:
    """The distribution name and version a `pyproject.toml` declares."""
    metadata = tomllib.loads(manifest.read_text())
    project = metadata.get("project", {})
    name, version = project.get("name"), project.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise SystemExit(
            f"{manifest} declares no static [project] name and version, so there is no "
            "declared version to build a baseline around"
        )
    return name, version


def version_key(version: str) -> tuple:
    """Order versions without pulling in a parser: numeric parts numerically."""
    pieces = []
    for piece in version.split("."):
        pieces.append((int(False), int(piece)) if piece.isdigit() else (int(True), piece))
    return tuple(pieces)

