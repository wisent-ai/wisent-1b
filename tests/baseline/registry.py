"""The published artifact tiers: what PyPI serves for this distribution, and
the surface recovered from the best one it has."""
from __future__ import annotations

import http
import json
import pathlib
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

sys.dont_write_bytecode = True

from .. import surface as extractor
from .facts import INDEX, ROOT, SDIST, WHEEL, declared, version_key


def ask_pypi(path: str) -> dict | None:
    """PyPI's JSON at `path`, or None when the index STATES it has no such thing.

    Only a not-found answer returns None. Every other outcome -- a refusal, a throttle,
    a server error, an unreachable host, or a success carrying something that is not the
    JSON document asked for -- means the question was not answered, and is refused rather
    than folded into "not published". Those are the cases that would otherwise let a
    published project read as absent, which is the one mistake this whole file exists to
    avoid. Each failure says which kind it is, because an unannotated traceback in a red
    build reads as an unrelated fault and invites a rerun until it passes.
    """
    try:
        with urllib.request.urlopen(f"{INDEX}/{path}/json") as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == http.HTTPStatus.NOT_FOUND:
            return None
        raise SystemExit(
            f"PyPI answered {error.code} for {path}. That is a refusal, a throttle or a "
            "server fault -- not a statement about whether it is published -- so a "
            "baseline must not be guessed from it"
        ) from error
    except urllib.error.URLError as error:
        raise SystemExit(
            f"cannot reach PyPI to establish what is published. This is a transport "
            f"failure, not evidence about {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"PyPI answered for {path} with something that is not JSON, which is what a "
            f"throttle or an error page looks like on a successful request: {error}"
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
            "source": f"{SDIST}:{chosen['filename']} unpacked and read with tests/surface.py",
            "surface": extractor.surface(tree),
        }

