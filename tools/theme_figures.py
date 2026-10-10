"""Derive the published variants of each light SVG under docs/images.

Every figure is drawn once with the light quchip tokens as ``<name>.svg``.
This script writes ``<name>-dark.svg`` by swapping tokens, ``<name>.pdf``
from the light SVG (glyphs are already outlines, so the PDF stays small), and
``<name>-dark.png`` beside any light ``<name>.png`` (the README pair). Colors
outside the token map are reported so figures stay on the palette.

``rsvg-convert`` output differs between runs, so the script renders a PDF or
dark PNG only when its light SVG differs from ``HEAD``, is untracked, or has
no rendered file yet. ``--force`` renders every one.

    python3 tools/theme_figures.py            # all figures
    python3 tools/theme_figures.py clear_iq   # one stem
    python3 tools/theme_figures.py --force    # render every PDF and PNG
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"

# light token -> dark token (quchip identity, tokens/quchip.tokens.json)
TOKENS = {
    "#16181c": "#f2f4f6",  # ink
    "#c92f33": "#ef6661",  # accent
    "#246fa8": "#4e8fc4",  # blue
    "#50565a": "#9fa6aa",  # muted
    "#6d7277": "#85919a",  # soft
    "#9aa0a8": "#6d7277",  # faded
    "#dbdee1": "#22272b",  # line
    "#e7eaee": "#2e3437",  # grid
    "#f2f4f6": "#1b1e24",  # panel -> card
    "#ffffff": "#14161a",  # white surfaces -> paper
    "#fafbfc": "#14161a",  # paper
}
HEX_RE = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")


def _expand(color: str) -> str:
    color = color.lower()
    if len(color) == 4:
        color = "#" + "".join(2 * c for c in color[1:])
    return color


def theme_svg(source: Path) -> tuple[Path, set[str]]:
    text = source.read_text(encoding="utf-8")
    unknown: set[str] = set()

    def swap(match: re.Match[str]) -> str:
        color = _expand(match.group(0))
        if color in TOKENS:
            return TOKENS[color]
        unknown.add(color)
        return color

    themed = HEX_RE.sub(swap, text)
    target = source.with_name(f"{source.stem}-dark.svg")
    target.write_text(themed, encoding="utf-8")
    return target, unknown


def _matches_head(path: Path) -> bool:
    """Return whether ``path`` is tracked and identical to its ``HEAD`` version."""
    try:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True)
        if tracked.returncode != 0:
            return False
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=ROOT)
    except FileNotFoundError:
        return False
    return diff.returncode == 0


def render_pdf(svg: Path) -> Path | None:
    if shutil.which("rsvg-convert") is None:
        return None
    target = svg.with_suffix(".pdf")
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(target), str(svg)], check=True)
    return target


def render_png(svg: Path, reference_png: Path) -> Path | None:
    if shutil.which("rsvg-convert") is None:
        return None
    width = subprocess.run(
        ["sips", "-g", "pixelWidth", str(reference_png)], capture_output=True, text=True, check=True,
    ).stdout.split()[-1]
    target = svg.with_suffix(".png")
    subprocess.run(["rsvg-convert", "-w", width, "-o", str(target), str(svg)], check=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="*", help="figure stems (default: every light SVG)")
    parser.add_argument("--force", action="store_true", help="render PDFs and PNGs even when the SVG matches HEAD")
    args = parser.parse_args()
    sources = (
        [IMAGES / f"{stem}.svg" for stem in args.stems]
        if args.stems
        else sorted(path for path in IMAGES.glob("*.svg") if not path.stem.endswith("-dark"))
    )
    status = 0
    for source in sources:
        if not source.exists():
            print(f"missing: {source}", file=sys.stderr)
            status = 1
            continue
        target, unknown = theme_svg(source)
        line = f"{source.name} -> {target.name}"
        unchanged = not args.force and _matches_head(source)
        if unchanged and source.with_suffix(".pdf").exists():
            line += " (SVG matches HEAD: PDF kept)"
        else:
            pdf = render_pdf(source)
            line += f" + {pdf.name}" if pdf else " (rsvg-convert missing: no PDF)"
        light_png = source.with_suffix(".png")
        if light_png.exists():
            if unchanged and target.with_suffix(".png").exists():
                line += " (dark PNG kept)"
            else:
                png = render_png(target, light_png)
                line += f" + {png.name}" if png else " (rsvg-convert missing: no dark PNG)"
        print(line)
        if unknown:
            print(f"  off-palette colors: {', '.join(sorted(unknown))}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
