"""Command-line inventory and candidate report for the prose audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import Sequence

from . import Candidate, Passage
from .extract import extract_html, extract_markdown, extract_python
from .rules import find_candidates


def _tracked_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    return [Path(item.decode()) for item in result.stdout.split(b"\0") if item]


def _targets(
    root: Path,
    rendered_html: Path | None,
    extra_paths: Sequence[Path],
) -> list[tuple[Path, str, bool]]:
    targets: dict[Path, tuple[str, bool]] = {}
    for path in _tracked_paths(root):
        suffix = path.suffix.casefold()
        if suffix in {".py", ".pyi"}:
            targets[path] = ("python", False)
        elif suffix == ".md":
            targets[path] = ("markdown", False)
        elif suffix in {".html", ".htm"}:
            targets[path] = ("html", False)

    if rendered_html is not None and rendered_html.exists():
        resolved_root = root.resolve()
        for absolute in sorted(rendered_html.rglob("*")):
            if not absolute.is_file() or absolute.suffix.casefold() not in {".html", ".htm"}:
                continue
            try:
                path = absolute.resolve().relative_to(resolved_root)
            except ValueError:
                path = absolute.resolve()
            previous = targets.get(path)
            targets[path] = ("html", previous[1] if previous else True)

    resolved_root = root.resolve()
    for extra in extra_paths:
        absolute = extra if extra.is_absolute() else root / extra
        try:
            path = absolute.resolve().relative_to(resolved_root)
        except ValueError:
            path = absolute.resolve()
        suffix = path.suffix.casefold()
        if suffix in {".py", ".pyi"}:
            targets[path] = ("python", False)
        elif suffix == ".md":
            targets[path] = ("markdown", False)
        elif suffix in {".html", ".htm"}:
            targets[path] = ("html", False)

    return [(path, *targets[path]) for path in sorted(targets, key=lambda item: str(item))]


def _extract(root: Path, path: Path, target_format: str) -> list[Passage]:
    absolute = path if path.is_absolute() else root / path
    source = absolute.read_text(encoding="utf-8")
    display = path if not path.is_absolute() else absolute
    if target_format == "python":
        return extract_python(display, source)
    if target_format == "markdown":
        return extract_markdown(display, source)
    return extract_html(display, source)


def _candidate_dict(candidate: Candidate) -> dict[str, object]:
    data = asdict(candidate)
    passage = data.pop("passage")
    return {**passage, **data}


def build_report(
    root: Path,
    rendered_html: Path | None = None,
    extra_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Build a stable JSON-compatible repository coverage and candidate report."""
    root = root.resolve()
    passages: list[Passage] = []
    candidates: list[Candidate] = []
    failures: list[dict[str, str]] = []
    target_rows: list[dict[str, object]] = []

    for path, target_format, generated in _targets(root, rendered_html, extra_paths):
        display = str(path)
        try:
            extracted = _extract(root, path, target_format)
        except (OSError, SyntaxError, UnicodeError, ValueError) as error:
            failures.append({"path": display, "error": f"{type(error).__name__}: {error}"})
            target_rows.append(
                {
                    "path": display,
                    "format": target_format,
                    "generated": generated,
                    "passages": 0,
                    "candidates": 0,
                    "status": "failed",
                }
            )
            continue

        target_candidates = [candidate for passage in extracted for candidate in find_candidates(passage)]
        passages.extend(extracted)
        candidates.extend(target_candidates)
        target_rows.append(
            {
                "path": display,
                "format": target_format,
                "generated": generated,
                "passages": len(extracted),
                "candidates": len(target_candidates),
                "status": "candidate" if target_candidates else "clean",
            }
        )

    target_formats = [row["format"] for row in target_rows]
    inventory = {
        "python_files": target_formats.count("python"),
        "markdown_files": target_formats.count("markdown"),
        "html_files": target_formats.count("html"),
        "docstrings": sum(item.kind == "docstring" for item in passages),
        "markdown_passages": sum(item.kind.startswith("markdown-") for item in passages),
        "html_passages": sum(item.kind.startswith("html-") for item in passages),
        "total_passages": len(passages),
        "high_confidence_candidates": sum(item.confidence == "high" for item in candidates),
        "contextual_candidates": sum(item.confidence == "contextual" for item in candidates),
        "failures": len(failures),
        "unaccounted_targets": sum(row["status"] not in {"candidate", "clean", "failed"} for row in target_rows),
    }
    return {
        "inventory": inventory,
        "targets": target_rows,
        "passages": [asdict(item) for item in sorted(passages, key=lambda item: (item.path, item.line, item.kind))],
        "candidates": [
            _candidate_dict(item)
            for item in sorted(candidates, key=lambda item: (item.passage.path, item.passage.line, item.rule_id))
        ],
        "failures": failures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--rendered-html", type=Path)
    parser.add_argument("--extra-path", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the report and return nonzero only for incomplete extraction."""
    args = _parser().parse_args(argv)
    rendered = args.rendered_html
    if rendered is not None and not rendered.is_absolute():
        rendered = args.root / rendered
    report = build_report(args.root, rendered, args.extra_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    inventory = report["inventory"]
    assert isinstance(inventory, dict)
    return int(bool(inventory["failures"] or inventory["unaccounted_targets"]))


if __name__ == "__main__":  # pragma: no cover - exercised through the command
    raise SystemExit(main())
