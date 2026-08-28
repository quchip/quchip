"""Copy textual outputs from executed example notebooks into their Markdown pairs.

Jupytext's plain Markdown format preserves source cells but not notebook
outputs. The documentation includes the Markdown files, so this small bridge
keeps the rendered guides honest: code comes from the paired notebook and each
textual result is committed directly below the cell that produced it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = (
    "00_hello_chip",
    "01_resolve_and_sweep",
    "02_reduce_and_replay",
    "03_differentiate_a_driven_chip",
)
OUTPUT_START = "<!-- executed-output:start -->"
OUTPUT_END = "<!-- executed-output:end -->"
OUTPUT_RE = re.compile(
    rf"\n*{re.escape(OUTPUT_START)}.*?{re.escape(OUTPUT_END)}\n*",
    re.DOTALL,
)
CODE_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _text(value: str | list[str]) -> str:
    return "".join(value) if isinstance(value, list) else value


def _fenced_text(value: str) -> str:
    value = ANSI_RE.sub("", value).rstrip()
    return f"```text\n{value}\n```"


def _render_output(output: dict[str, Any]) -> str | None:
    output_type = output.get("output_type")
    if output_type == "stream":
        value = _text(output.get("text", ""))
        return _fenced_text(value) if value.strip() else None

    if output_type == "error":
        traceback = "\n".join(output.get("traceback", ()))
        return _fenced_text(traceback)

    if output_type not in {"display_data", "execute_result"}:
        return None

    data = output.get("data", {})
    if "text/markdown" in data:
        return _text(data["text/markdown"]).strip()
    if "text/latex" in data:
        latex = _text(data["text/latex"]).strip()
        if latex.startswith("$") and latex.endswith("$"):
            latex = latex[1:-1]
        return f"```{{math}}\n{latex}\n```"

    plain = _text(data.get("text/plain", ""))
    if "image/png" in data and plain.startswith("<Figure size"):
        return None
    return _fenced_text(plain) if plain.strip() else None


def _render_cell_outputs(cell: dict[str, Any]) -> str:
    rendered = [part for output in cell.get("outputs", ()) if (part := _render_output(output))]
    if not rendered:
        return ""
    body = "\n\n".join(rendered)
    return f"\n\n{OUTPUT_START}\n\nOutput:\n\n{body}\n\n{OUTPUT_END}"


def synchronized_markdown(markdown_path: Path, notebook_path: Path) -> str:
    source = OUTPUT_RE.sub("\n\n", markdown_path.read_text(encoding="utf-8"))
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    matches = list(CODE_RE.finditer(source))

    if len(matches) != len(code_cells):
        raise ValueError(
            f"{markdown_path}: found {len(matches)} Python blocks but "
            f"{notebook_path} has {len(code_cells)} code cells"
        )

    additions: list[tuple[int, str]] = []
    for index, (match, cell) in enumerate(zip(matches, code_cells, strict=True), start=1):
        markdown_code = match.group(1).rstrip()
        notebook_code = _text(cell.source).rstrip()
        if markdown_code != notebook_code:
            raise ValueError(
                f"{markdown_path}: code block {index} differs from the executed notebook"
            )
        if cell.execution_count is None:
            raise ValueError(f"{notebook_path}: code cell {index} has not been executed")
        if any(output.get("output_type") == "error" for output in cell.outputs):
            raise ValueError(f"{notebook_path}: code cell {index} contains an error output")
        has_figure = any("image/png" in output.get("data", {}) for output in cell.outputs)
        if has_figure:
            next_code_start = matches[index].start() if index < len(matches) else len(source)
            following_markdown = source[match.end():next_code_start]
            if "savefig(" not in notebook_code or "```{figure}" not in following_markdown:
                raise ValueError(
                    f"{markdown_path}: plotted output after code block {index} must be "
                    "saved by that cell and shown with a figure directive"
                )
        additions.append((match.end(), _render_cell_outputs(cell)))

    for offset, addition in reversed(additions):
        source = source[:offset] + addition + source[offset:]
    return source.rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="*", default=DEFAULT_PAIRS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a Markdown file does not contain the current notebook outputs",
    )
    args = parser.parse_args()

    stale: list[Path] = []
    for stem in args.stems:
        markdown_path = ROOT / "examples" / f"{stem}.md"
        notebook_path = ROOT / "examples" / f"{stem}.ipynb"
        synchronized = synchronized_markdown(markdown_path, notebook_path)
        current = markdown_path.read_text(encoding="utf-8")
        if current == synchronized:
            continue
        if args.check:
            stale.append(markdown_path)
        else:
            markdown_path.write_text(synchronized, encoding="utf-8")
            print(f"updated {markdown_path.relative_to(ROOT)}")

    if stale:
        for path in stale:
            print(f"stale executed output: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
