"""Execute the Python blocks of documentation pages and refresh their shown output.

The guides show each printed result in an ``Output:`` block below the Python
block that prints it. After a change alters what a page prints, run

    python3 tools/execute_markdown.py docs/guides/model-reduction.md

Each page runs in one fresh namespace, block by block, from ``docs/images``,
where its figures are saved. The tool then writes the printed text below each
block and removes ``Output:`` blocks that no longer print. The tests replay the
same pages with ``execute_page()`` and compare the output with the committed
text.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCK_RE = re.compile(
    r"```python\n(?P<code>.*?)\n```(?:\n\nOutput:\n\n```text\n(?P<text>.*?)\n```)?",
    re.DOTALL,
)


def execute_page(path: Path, workdir: Path) -> list[tuple[re.Match[str], str]]:
    """Run each Python block of a page in order and return it with the text that it prints."""
    import matplotlib

    matplotlib.use("Agg")
    path = Path(path).resolve()
    namespace: dict[str, object] = {"__name__": "__docs_page__"}
    executed = []
    with contextlib.chdir(workdir):
        for index, block in enumerate(BLOCK_RE.finditer(path.read_text(encoding="utf-8")), start=1):
            captured = io.StringIO()
            with warnings.catch_warnings(), contextlib.redirect_stdout(captured):
                warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
                exec(compile(block["code"], f"{path}#block-{index}", "exec"), namespace)
            executed.append((block, captured.getvalue()))
    return executed


def refresh_page(path: Path) -> bool:
    """Execute a page from docs/images and write its printed output. Return whether the page changed."""
    path = Path(path).resolve()
    source = path.read_text(encoding="utf-8")
    pieces, end = [], 0
    for block, printed in execute_page(path, ROOT / "docs" / "images"):
        pieces.append(source[end:block.start()])
        pieces.append(f"```python\n{block['code']}\n```")
        if printed.strip():
            pieces.append(f"\n\nOutput:\n\n```text\n{printed.rstrip()}\n```")
        end = block.end()
    pieces.append(source[end:])
    updated = "".join(pieces)
    if updated != source:
        path.write_text(updated, encoding="utf-8")
    return updated != source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="+", type=Path, help="Markdown pages under docs/")
    args = parser.parse_args(argv)
    for page in args.pages:
        print(f"{page}: {'updated' if refresh_page(page) else 'unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
