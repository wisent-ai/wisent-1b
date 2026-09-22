"""The tag tier: which tags ORIGIN serves, that this clone holds the same
object for each of them, and the surface of the highest tag whose own tree
declares the version its name claims. Below it, the working tree itself."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

from .. import surface as extractor
from .facts import (
    ARCHIVE,
    HEAD,
    PEELED_SUFFIX,
    REMOTE,
    ROOT,
    TAG_REF_PREFIX,
    declared,
    run,
    version_key,
)


def remote_tags() -> dict:
    """Every tag ORIGIN serves, mapped to the object ids it serves for that tag.

    Never `git tag --list`, which is wrong in two opposite directions. On a runner
    `actions/checkout@v4` fetches no tags at all, so a local listing comes back empty
    however many the remote holds -- and this generator would then call `head:` the best
    reachable tier at exactly the moment a tag made that false, blind to the one thing
    the workflow re-runs it with `--stdout` to notice. In a fork the error inverts: a
    fork shares the upstream's object store, so a locally visible tag can be the
    upstream's, and filing its surface here would claim a release nobody made in this
    repository. `git ls-remote` asks the only party that knows which tags are ours.

    An annotated tag arrives as two lines -- the tag object, and the commit it peels to
    under `<ref>^{}` -- so both ids are collected under the one name, and the local
    commit has to match one of them.
    """
    served: dict = {}
    for line in run("ls-remote", "--tags", REMOTE).splitlines():
        fields = line.split()
        if len(fields) != len(("object", "ref")):
            continue
        object_id, ref = fields
        if not ref.startswith(TAG_REF_PREFIX):
            continue
        name = ref[len(TAG_REF_PREFIX) :].removesuffix(PEELED_SUFFIX)
        if name:
            served.setdefault(name, set()).add(object_id)
    return served


def assert_tag_is_readable_here(tag: str, objects: set) -> None:
    """A tag ORIGIN serves must exist in this clone, and be the same object.

    Neither half may degrade to a skip. A remote tag this clone never fetched cannot be
    reproduced with `git archive`, and passing over it would restore exactly the
    blindness `remote_tags` exists to remove: a `head:` baseline reported while the
    remote holds a tag. A shallow clone fails the same way for a different reason -- the
    tag's tree is simply absent. And a local tag naming a different commit than the
    remote's is not the artifact anybody resolved, so its surface is not the published
    one.
    """
    if run("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit(
            f"{REMOTE} serves tag {tag}, but this clone is shallow, so that tag's tree "
            f"is absent and `git archive {tag}` cannot reproduce it. Reporting a lower "
            f"tier from here would be blindness rather than a finding; run: "
            f"git fetch --force --tags --unshallow"
        )
    try:
        local = run("rev-parse", f"{tag}^{{commit}}")
    except SystemExit as error:
        raise SystemExit(
            f"{REMOTE} serves tag {tag}, but this clone cannot resolve it ({error}). "
            f"Skipping it would report a weaker baseline than the remote can prove; "
            f"run: git fetch --force --tags"
        ) from error
    if local not in objects:
        raise SystemExit(
            f"tag {tag} is {local} here, but {REMOTE} serves {sorted(objects)} for it. "
            f"Refusing to build a baseline from a tag whose identity is unsettled: a tag "
            f"names an artifact only if the remote and this clone agree what it is."
        )


def from_tag() -> dict | None:
    """The highest tag ORIGIN serves whose tree declares the version its name claims.

    The candidates come from the remote, never from `git tag --list`: see `remote_tags`.
    """
    served = remote_tags()
    if not served:
        return None

    usable, mismatched = [], []
    for tag, objects in served.items():
        assert_tag_is_readable_here(tag, objects)
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
            "tests/surface.py; nothing is published for this distribution, so a tag "
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

