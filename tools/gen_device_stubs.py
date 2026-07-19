"""Generate ``if TYPE_CHECKING:`` ``__init__`` stubs for synthesized device models.

Declarative device classes (subclasses of
:class:`quchip.declarative.models.DeviceModel`) get their real ``__init__``
synthesized at runtime by
:func:`quchip.declarative.models._synthesize_device_init` from their declared
``parameter()`` fields, so static type checkers see no constructor signature
by default. This script inspects the synthesized signature of every such
class and writes a machine-generated ``if TYPE_CHECKING:`` stub into the
class body, delimited by marker comments, so IDEs and mypy see the real
keyword-argument surface.

Usage
-----
``python tools/gen_device_stubs.py`` rewrites stale stub files in place and
prints which files changed. ``python tools/gen_device_stubs.py --check``
exits nonzero and prints a diff summary instead of writing, for CI.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import inspect
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MARKER_START = "    # --- generated __init__ stub (tools/gen_device_stubs.py); do not edit ---"
MARKER_END = "    # --- end generated stub ---"

# Fixed annotations for the trailing structural parameters every synthesized
# __init__ carries after the declared fields: `levels` and `label`
# (quchip.declarative.models._synthesize_device_init), then the noise kwargs
# BaseDevice.__init__ itself declares. Mirrors those annotations by hand;
# keep in sync with quchip/devices/base.py::BaseDevice.__init__.
_TRAILING_ANNOTATIONS = {
    "levels": "int",
    "label": "str | None",
    "T1": "float | None",
    "T2": "float | None",
    "thermal_population": "float | None",
}

_FIELD_RE = re.compile(r"^ {4}\w+\s*:\s*.*=\s*parameter\(")


def _synthesized_device_classes() -> list[type]:
    """Return every concrete DeviceModel subclass under quchip with a synthesized __init__."""
    import quchip  # noqa: F401  (side effect: imports every built-in device module)
    from quchip.declarative.models import DeviceModel

    seen: set[type] = set()
    stack = [DeviceModel]
    classes: list[type] = []
    while stack:
        current = stack.pop()
        for sub in current.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            stack.append(sub)
            if sub.__module__.startswith("quchip") and _is_synthesized(sub):
                classes.append(sub)
    classes.sort(key=lambda cls: (cls.__module__, cls.__qualname__))
    return classes


def _is_synthesized(cls: type) -> bool:
    """Return whether *cls* carries the synthesized ``__init__`` rather than a hand-written one."""
    init = cls.__dict__.get("__init__")
    return init is not None and getattr(init, "__doc__", None) == (
        f"Initialize {cls.__name__} from its declared parameters."
    )


def _field_annotation(cls: type, name: str) -> str:
    """Return the declared string annotation for *name*, walking the MRO most-derived first."""
    for klass in cls.__mro__:
        annotations = klass.__dict__.get("__annotations__", {})
        if name in annotations:
            return annotations[name]
    raise RuntimeError(f"{cls.__name__} declares no annotation for parameter {name!r}")


def _stub_lines(cls: type) -> list[str]:
    """Render the marker-delimited stub block for *cls* as a list of lines (no trailing newlines)."""
    signature = inspect.signature(cls.__init__)
    params = list(signature.parameters.values())[1:]  # drop `self`

    lines = [MARKER_START, "    if TYPE_CHECKING:", "        def __init__(", "            self,"]
    seen_keyword_only = False
    for param in params:
        if param.kind is inspect.Parameter.KEYWORD_ONLY and not seen_keyword_only:
            lines.append("            *,")
            seen_keyword_only = True
        annotation = _TRAILING_ANNOTATIONS.get(param.name)
        if annotation is None:
            annotation = _field_annotation(cls, param.name)
        piece = f"            {param.name}: {annotation}"
        if param.default is not inspect.Parameter.empty:
            piece += f" = {param.default!r}"
        lines.append(piece + ",")
    lines.append("        ) -> None: ...")
    lines.append(MARKER_END)
    return lines


def _find_insertion_point(lines: list[str]) -> int:
    """Return the line index right after the last ``parameter()`` field declaration."""
    last_end = None
    i = 0
    while i < len(lines):
        if _FIELD_RE.match(lines[i]):
            depth = lines[i].count("(") - lines[i].count(")")
            j = i
            while depth > 0:
                j += 1
                depth += lines[j].count("(") - lines[j].count(")")
            last_end = j
            i = j + 1
            continue
        i += 1
    if last_end is None:
        raise RuntimeError("no parameter() field declaration found to anchor the generated stub")
    return last_end + 1


def _splice_stub(class_src: str, stub_lines: list[str]) -> str:
    """Insert or replace the marker-delimited stub block within *class_src*."""
    lines = class_src.splitlines(keepends=True)
    start_idx = end_idx = None
    for i, line in enumerate(lines):
        if line.rstrip("\n") == MARKER_START:
            start_idx = i
        elif line.rstrip("\n") == MARKER_END:
            end_idx = i
            break

    block = [line + "\n" for line in stub_lines]
    if start_idx is not None and end_idx is not None:
        new_lines = lines[:start_idx] + block + lines[end_idx + 1 :]
        return "".join(new_lines)

    after_fields = _find_insertion_point(lines)
    consume_from = after_fields
    if consume_from < len(lines) and lines[consume_from].strip() == "":
        consume_from += 1
    new_lines = lines[:after_fields] + ["\n"] + block + ["\n"] + lines[consume_from:]
    return "".join(new_lines)


def _apply_stub(source: str, cls: type) -> str:
    """Return *source* with *cls*'s generated stub inserted or refreshed."""
    class_src = inspect.getsource(cls)
    start = source.index(class_src)
    end = start + len(class_src)
    new_class_src = _splice_stub(class_src, _stub_lines(cls))
    return source[:start] + new_class_src + source[end:]


def _ensure_type_checking_import(source: str) -> str:
    """Ensure ``TYPE_CHECKING`` is importable from ``typing`` in *source*."""
    if re.search(r"^from typing import\b.*\bTYPE_CHECKING\b", source, re.MULTILINE):
        return source

    match = re.search(r"^from typing import (.+)$", source, re.MULTILINE)
    if match is not None:
        names = [n.strip() for n in match.group(1).split(",")]
        names.append("TYPE_CHECKING")
        # All-caps constants sort before CapWords/lowercase names (house style).
        names.sort(key=lambda n: (not n.isupper(), n))
        new_line = f"from typing import {', '.join(names)}"
        return source[: match.start()] + new_line + source[match.end() :]

    future_match = re.search(r"^from __future__ import annotations\n", source, re.MULTILINE)
    if future_match is not None:
        idx = future_match.end()
        return source[:idx] + "\nfrom typing import TYPE_CHECKING\n" + source[idx:]
    return "from typing import TYPE_CHECKING\n\n" + source


def _classes_by_file(classes: list[type]) -> dict[Path, list[type]]:
    groups: dict[Path, list[type]] = {}
    for cls in classes:
        path = Path(importlib.import_module(cls.__module__).__file__).resolve()
        groups.setdefault(path, []).append(cls)
    return groups


def compute_changes() -> dict[Path, tuple[str, str]]:
    """Return ``{path: (original, updated)}`` for every stub file that would change."""
    classes = _synthesized_device_classes()
    changes: dict[Path, tuple[str, str]] = {}
    for path, classes_in_file in _classes_by_file(classes).items():
        original = path.read_text()
        updated = original
        for cls in classes_in_file:
            updated = _apply_stub(updated, cls)
        updated = _ensure_type_checking_import(updated)
        if updated != original:
            changes[path] = (original, updated)
    return changes


def check_stubs_current() -> list[Path]:
    """Return the paths whose committed stub would change under regeneration (empty if current)."""
    return sorted(compute_changes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero and print a diff summary instead of writing",
    )
    args = parser.parse_args(argv)

    changes = compute_changes()
    if not changes:
        print("gen_device_stubs: all stubs up to date")
        return 0

    for path in sorted(changes):
        rel = path.relative_to(REPO_ROOT)
        original, updated = changes[path]
        if args.check:
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
            print(f"gen_device_stubs: {rel} is stale")
            print("".join(diff))
        else:
            path.write_text(updated)
            print(f"gen_device_stubs: wrote {rel}")

    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
