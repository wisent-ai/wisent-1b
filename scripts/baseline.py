"""Write `released-surface.json`: the surface of the version actually published.

The baseline the shared versioning rule compares against must describe an artifact
somebody can install, not whichever working tree someone happened to have checked out.
So this asks PyPI what exists:

  * If the distribution is published, the sdist for that exact version is downloaded,
    unpacked, and `scripts/surface.py` is run against the unpacked tree. The recorded
    surface is then the one that shipped, recovered rather than assumed -- which is the
    whole reason the extractor reads source with `ast` instead of importing.
  * If nothing is published, HEAD is the only thing anyone can consume (a git install of
    the version `pyproject.toml` declares), and `source` says so plainly along with the
    commit, so a later reader can tell the baseline was never a release.

It never invents a version: the one written is the one `pyproject.toml` declares. And it
never guesses -- if PyPI is unreachable it fails rather than recording a baseline that
might be measured against the wrong artifact.

Usage:
    python3 scripts/baseline.py            # rewrite released-surface.json
    python3 scripts/baseline.py --stdout   # print it instead
"""

from __future__ import annotations

import http
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request

# The extractor is a sibling script, not an installed module. Importing it would leave
# a scripts/__pycache__ behind, and this repository has no .gitignore to absorb it.
sys.dont_write_bytecode = True

import surface as extractor

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "released-surface.json"
INDEX = "https://pypi.org/pypi"

# The workflow reads `source` back and refuses a baseline that claims a release PyPI
# does not have, so the phrase that marks "recovered from a published sdist" is a
# contract between this script and .github/workflows/version-check.yml, not prose.
PUBLISHED = "sdist published on PyPI"


def declared() -> tuple:
    """The distribution name and the version this repository declares."""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = metadata.get("project", {})
    name, version = project.get("name"), project.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise SystemExit(
            "pyproject.toml declares no static [project] name and version, so there is "
            "no declared version to build a baseline around"
        )
    return name, version


def commit() -> str:
    """The commit HEAD points at, so a HEAD-based baseline stays identifiable."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SystemExit(f"git rev-parse HEAD failed: {result.stderr.strip()}")
    return result.stdout.strip()


def published(name: str, version: str) -> dict | None:
    """PyPI's metadata for this exact version, or None when it is not published."""
    try:
        with urllib.request.urlopen(f"{INDEX}/{name}/{version}/json") as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == http.HTTPStatus.NOT_FOUND:
            return None
        raise SystemExit(
            f"PyPI answered {error.code} for {name} {version}, so whether it is published "
            "is unknown and a baseline must not be guessed"
        ) from error
    except urllib.error.URLError as error:
        raise SystemExit(
            f"cannot reach PyPI to establish whether {name} {version} is published, and "
            f"a baseline must not be guessed: {error}"
        ) from error


def sdist_surface(metadata: dict, name: str, version: str) -> list:
    """Download the published sdist and read its surface with the same extractor."""
    sdists = [f for f in metadata.get("urls", ()) if f.get("packagetype") == "sdist"]
    if not sdists:
        raise SystemExit(
            f"{name} {version} is published but ships no sdist, so the source surface it "
            "released cannot be recovered; publish an sdist to make the baseline provable"
        )
    with tempfile.TemporaryDirectory() as work:
        area = pathlib.Path(work)
        archive = area / "sdist.tar.gz"
        with urllib.request.urlopen(next(iter(sdists))["url"]) as response:
            archive.write_bytes(response.read())
        unpacked = area / "src"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        try:
            (tree,) = [entry for entry in unpacked.iterdir() if entry.is_dir()]
        except ValueError as error:
            raise SystemExit(
                f"the sdist for {name} {version} does not unpack to a single directory, "
                "so the tree to read the surface from is ambiguous"
            ) from error
        return extractor.surface(tree)


def build() -> dict:
    """The baseline document."""
    name, version = declared()
    metadata = published(name, version)
    if metadata is None:
        return {
            "version": version,
            "source": (
                f"HEAD at {commit()}. Nothing is published for {name}: PyPI has no such "
                "version, so the only thing anyone can consume is a git install of the "
                "version pyproject.toml declares, and this surface is read from the tree "
                "that declares it. Regenerate from the sdist once a release exists."
            ),
            "surface": extractor.surface(ROOT),
        }
    return {
        "version": version,
        "source": f"{PUBLISHED} as {name} {version}",
        "surface": sdist_surface(metadata, name, version),
    }


def main(argv: list) -> int:
    document = json.dumps(build(), indent=int(True) + int(True)) + "\n"
    if "--stdout" in argv:
        sys.stdout.write(document)
    else:
        BASELINE.write_text(document)
        print(f"wrote {BASELINE.relative_to(ROOT)}")
    return int(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[int(True) :]))
