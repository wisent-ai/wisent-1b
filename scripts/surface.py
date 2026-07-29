"""Print this package's public surface: what `rej-1b` promises the people who install it.

Two things break a caller of this distribution, and nothing else does.

The first is the importable API. `rej_1b/__init__.py` names it explicitly in `__all__`
-- the config types, the two model classes, the tokenizer, `generate` and the
checkpoint helpers. That list is the whole reason anyone writes `from rej_1b import
...`, so an entry disappearing from it is a break and an entry appearing is a new
capability. Everything reachable only as `rej_1b.model.<something>` is deliberately
excluded: the package went out of its way to say which names it stands behind, and
honouring that is what makes the list worth maintaining.

The second is the console scripts declared in `[project.scripts]`. `rej-1b-train` and
`rej-1b-generate` land on a user's PATH at install time; dropping or renaming one
breaks a script that ran yesterday just as surely as removing a class does.

Deliberately *not* part of the contract: the model weights and the architecture they
imply. This repository is a model definition as well as a library, and a retrained
checkpoint is not something the versioning rule can see or should try to. Only the
code surface is compared here.

Read with `ast` and `tomllib`, never by importing. Importing `rej_1b` pulls in `torch`,
`transformers` and `datasets`, and a release decision must not depend on a machine
having them -- nor should it be able to fail because an unrelated dependency moved. It
also means this runs unchanged against an unpacked sdist, so the surface of an already
published version can be recovered exactly rather than assumed.

Anything unreadable is refused rather than skipped. A smaller surface is how the rule
spells "capability removed", so guessing here would fabricate a breaking change.

Usage:
    python3 scripts/surface.py [root]     # root defaults to the repository
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import tomllib

PACKAGE = "rej_1b"


def parse(source: pathlib.Path) -> ast.Module:
    """The module tree, or a loud failure."""
    try:
        return ast.parse(source.read_text(), filename=str(source))
    except OSError as error:
        raise SystemExit(f"{source}: {error}") from error
    except SyntaxError as error:
        # Refuse rather than skip. A module that does not parse cannot be imported
        # either, so its names are unreachable at runtime; skipping it would report a
        # smaller surface, and the rule would read that as a removed capability. The
        # surface is unknown here, not shrunk.
        raise SystemExit(
            f"{source}: does not parse, so the surface is unknown: {error}"
        ) from error


def exported_names(source: pathlib.Path) -> list:
    """The strings assigned to `__all__` at module level."""
    tree = parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target], node.value
        else:
            continue
        if not any(target.id == "__all__" for target in targets):
            continue
        if not isinstance(value, (ast.List, ast.Tuple)):
            raise SystemExit(
                f"{source}: __all__ is not a literal list, so the exported API cannot "
                "be read without importing the package -- refusing rather than "
                "reporting a partial surface"
            )
        names = []
        for element in value.elts:
            if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
                raise SystemExit(
                    f"{source}: __all__ holds a computed entry, so the exported API is "
                    "not knowable statically -- refusing rather than guessing"
                )
            names.append(element.value)
        return names
    raise SystemExit(
        f"{source}: no module-level __all__. The importable API is the contract of "
        "this package, so its absence means the surface is unknown, not empty"
    )


def console_scripts(manifest: pathlib.Path) -> list:
    """The command names `[project.scripts]` puts on a user's PATH."""
    try:
        metadata = tomllib.loads(manifest.read_text())
    except OSError as error:
        raise SystemExit(f"{manifest}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(f"{manifest}: does not parse: {error}") from error
    scripts = metadata.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        raise SystemExit(f"{manifest}: [project.scripts] is not a table")
    return list(scripts)


def surface(root: pathlib.Path) -> list:
    """Everything a user of the installed distribution would notice disappearing."""
    init = root / PACKAGE / "__init__.py"
    if not init.is_file():
        raise SystemExit(f"{init} is missing; is {root} the repository root?")
    manifest = root / "pyproject.toml"
    if not manifest.is_file():
        raise SystemExit(f"{manifest} is missing; is {root} the repository root?")

    names = {f"api:{name}" for name in exported_names(init)}
    names.update(f"cli:{command}" for command in console_scripts(manifest))
    if not names:
        raise SystemExit(
            f"no public names found for {PACKAGE}. Either __all__ emptied or the "
            "package moved -- both change what this distribution promises, so "
            "refusing rather than reporting an empty surface"
        )
    return sorted(names)


def main(argv: list) -> int:
    positional = [arg for arg in argv if not arg.startswith("-")]
    root = (
        pathlib.Path(positional[int(False)])
        if positional
        else pathlib.Path(__file__).resolve().parent.parent
    )
    print(json.dumps({"surface": surface(root)}, indent=int(True) + int(True)))
    return int(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[int(True) :]))
