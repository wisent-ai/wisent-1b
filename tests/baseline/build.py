"""Writing the baseline: the best tier that actually exists, as JSON."""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True

from .facts import BASELINE, ROOT, declared
from .registry import from_registry
from .tags import from_head, from_tag


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
