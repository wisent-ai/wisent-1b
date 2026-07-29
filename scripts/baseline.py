"""Write `released-surface.json`: the surface of the best artifact anyone can actually get.

The baseline the shared versioning rule compares against must describe a real artifact,
not whichever working tree someone happened to have checked out. So this reaches for the
best available source, in the fleet's order of preference:

    pypi-sdist  >  pypi-wheel  >  stado  >  git-archive  >  head

and records which one it used as the first token of `source`. That token is a marker the
version-check workflow reads back, so the two files are coupled by a constant rather
than by prose:

    pypi-sdist:<filename>   recovered from a published sdist
    pypi-wheel:<filename>   recovered from a published pure-Python wheel
    stado:<object>          recovered from a published Stado channel artifact
    git-archive:<tag>       reproduced from a git tag
    head:<sha>              last resort: nothing published, no usable tag

Two rules keep the baseline honest.

The version recorded is the LATEST PUBLISHED one, never the version `pyproject.toml`
declares. Looking up only the declared version means that the moment someone bumps ahead
of a release the lookup 404s, the generator quietly falls back to HEAD, and every later
comparison is measured against something nobody released.

A tag is trusted only if the tree it points at declares the version its name claims. A
tag that says one version while its own `pyproject.toml` says another is reported and
skipped, because reproducing it would put a surface under a version that never had it.

It never guesses. If a registry is unreachable, or the only published artifact is of a
kind this script cannot read, it fails and says what to do instead of silently dropping
to a weaker tier.

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

# Markers. The workflow branches on these, so they are constants here and referenced by
# name, never retyped as prose. `REGISTRY_MARKERS` is the set that asserts "a registry
# serves this exact version"; anything else asserts the opposite.
SDIST = "pypi-sdist"
WHEEL = "pypi-wheel"
ARCHIVE = "git-archive"
HEAD = "head"
REGISTRY_MARKERS = (SDIST, WHEEL)


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


def ask_pypi(path: str) -> dict | None:
    """PyPI's JSON at `path`, or None when it has no such thing."""
    try:
        with urllib.request.urlopen(f"{INDEX}/{path}/json") as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == http.HTTPStatus.NOT_FOUND:
            return None
        raise SystemExit(
            f"PyPI answered {error.code} for {path}, so whether it is published is "
            "unknown and a baseline must not be guessed"
        ) from error
    except urllib.error.URLError as error:
        raise SystemExit(
            f"cannot reach PyPI to establish what is published, and a baseline must not "
            f"be guessed: {error}"
        ) from error


def from_registry(name: str) -> dict | None:
    """The latest published version's surface, or None when nothing is published."""
    project = ask_pypi(name)
    if project is None:
        return None
    # PyPI reports the newest STABLE release here, ignoring prereleases, so this does
    # not baseline onto an rc the day someone uploads one. Visuals measured that across
    # django, numpy and urllib3. The one state where it misbehaves is a project whose
    # ONLY releases are prereleases: info.version is then a prerelease and the baseline
    # would be pinned to it. Documented rather than coded around -- this repository
    # publishes nothing, so writing speculative handling would be untestable here.
    version = project.get("info", {}).get("version")
    if not isinstance(version, str):
        raise SystemExit(f"PyPI serves {name} but names no latest version")
    release = ask_pypi(f"{name}/{version}")
    if release is None:
        raise SystemExit(f"PyPI names {name} {version} as latest but does not serve it")

    files = release.get("urls", ())
    sdists = [f for f in files if f.get("packagetype") == "sdist"]
    if not sdists:
        wheels = [f for f in files if f.get("packagetype") == "bdist_wheel"]
        raise SystemExit(
            f"{name} {version} is published but ships no sdist"
            + (
                f" -- only wheels ({', '.join(w['filename'] for w in wheels)}). A wheel "
                "carries no pyproject.toml, so the console-script half of the contract "
                "would have to be read from dist-info/entry_points.txt instead. This "
                "script does not implement that tier, and will not quietly record a "
                "weaker baseline in its place: publish an sdist, or extend it."
                if wheels
                else ", and no wheel either, so there is nothing to recover."
            )
        )

    chosen = next(iter(sdists))
    with tempfile.TemporaryDirectory() as work:
        area = pathlib.Path(work)
        archive = area / chosen["filename"]
        with urllib.request.urlopen(chosen["url"]) as response:
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
        return {
            "version": version,
            "source": f"{SDIST}:{chosen['filename']} unpacked and read with scripts/surface.py",
            "surface": extractor.surface(tree),
        }


def from_tag() -> dict | None:
    """The highest tag whose tree declares the version its name claims."""
    tags = [tag for tag in run("tag", "--list").splitlines() if tag.strip()]
    if not tags:
        return None

    usable, mismatched = [], []
    for tag in tags:
        claimed = tag.lstrip("v")
        with tempfile.TemporaryDirectory() as work:
            tree = pathlib.Path(work)
            archive = tree / "tag.tar"
            archive.write_bytes(
                subprocess.run(
                    ["git", "-C", str(ROOT), "archive", "--format=tar", tag],
                    capture_output=True,
                    check=True,
                ).stdout
            )
            content = tree / "content"
            with tarfile.open(archive) as tar:
                tar.extractall(content, filter="data")
            manifest = content / "pyproject.toml"
            if not manifest.is_file():
                mismatched.append(f"{tag} (no pyproject.toml)")
                continue
            _, actual = declared(manifest)
            if actual != claimed:
                # Main's warning: a tag can point at a commit that still declares an
                # older version. Reproducing it would file this surface under a version
                # that tree never had.
                mismatched.append(f"{tag} declares {actual}")
                continue
            usable.append((version_key(actual), actual, tag, extractor.surface(content)))

    for entry in mismatched:
        print(f"skipping tag: {entry}", file=sys.stderr)
    if not usable:
        return None

    _, version, tag, names = max(usable)
    return {
        "version": version,
        "source": (
            f"{ARCHIVE}:{tag} reproduced with `git archive` and read with "
            "scripts/surface.py; nothing is published for this distribution, so a tag "
            "whose tree declares this exact version is the strongest artifact available."
        ),
        "surface": names,
    }


def from_head() -> dict:
    """Last resort: the tree itself, saying so."""
    name, version = declared(ROOT / "pyproject.toml")
    return {
        "version": version,
        "source": (
            f"{HEAD}:{run('rev-parse', 'HEAD')} -- nothing is published for {name} on "
            "PyPI and no tag declares a version, so the only thing anyone can consume is "
            "a git install of the version pyproject.toml declares, and this surface is "
            "read from the tree that declares it. Regenerate once a release or tag exists."
        ),
        "surface": extractor.surface(ROOT),
    }


def build() -> dict:
    """The baseline, from the best tier that actually exists."""
    name, _ = declared(ROOT / "pyproject.toml")
    return from_registry(name) or from_tag() or from_head()


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
