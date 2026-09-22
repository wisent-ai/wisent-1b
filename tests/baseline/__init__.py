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
    python3 -m tests.baseline            # rewrite released-surface.json
    python3 -m tests.baseline --stdout   # print it instead
"""

from .build import build, main

__all__ = ["build", "main"]
