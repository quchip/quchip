"""Extract prose from repository formats without rewriting source files."""

from __future__ import annotations

import ast
from html.parser import HTMLParser
from pathlib import Path
import re

from . import Passage

_DocstringNode = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


def _docstring_line(node: _DocstringNode) -> int:
    first = node.body[0]
    assert isinstance(first, ast.Expr)
    return first.lineno


def extract_python(path: Path, source: str) -> list[Passage]:
    """Return module, class, and callable docstrings in source order."""
    tree = ast.parse(source, filename=str(path))
    passages: list[Passage] = []

    def visit(node: _DocstringNode, parents: tuple[str, ...]) -> None:
        text = ast.get_docstring(node, clean=True)
        if text is not None:
            owner = ".".join(parents) if parents else "<module>"
            passages.append(
                Passage(
                    path=str(path),
                    line=_docstring_line(node),
                    kind="docstring",
                    owner=owner,
                    text=text,
                )
            )
        for child in node.body:
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, (*parents, child.name))

    visit(tree, ())
    return sorted(passages, key=lambda item: item.line)


_IMAGE_RE = re.compile(r"!\[([^]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"(?<!!)\[([^]]+)\]\([^)]*\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LIST_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+(.*)$")


def _clean_markdown_inline(text: str) -> str:
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"[*_~]", "", text)
    return " ".join(text.split())


def extract_markdown(path: Path, source: str) -> list[Passage]:
    """Return authored Markdown blocks while excluding literal code and math."""
    passages: list[Passage] = []
    paragraph: list[str] = []
    paragraph_line = 0
    fenced = False
    math = False
    front_matter = False
    directive: str | None = None

    def flush() -> None:
        nonlocal paragraph, paragraph_line
        if paragraph:
            text = _clean_markdown_inline(" ".join(paragraph))
            if text:
                passages.append(Passage(str(path), paragraph_line, "markdown-prose", "<document>", text))
        paragraph = []
        paragraph_line = 0

    for line_number, raw in enumerate(source.splitlines(), start=1):
        stripped = raw.strip()
        if line_number == 1 and stripped == "---":
            front_matter = True
            continue
        if front_matter:
            if stripped == "---":
                front_matter = False
            continue
        myst = re.match(r"^```\{([^}]+)\}", stripped)
        if myst and not fenced:
            flush()
            directive = myst.group(1).casefold() if myst.group(1).casefold() in {"figure", "image"} else None
            fenced = directive is None
            continue
        if directive is not None and stripped == "```":
            flush()
            directive = None
            continue
        if stripped.startswith(("```", "~~~")):
            flush()
            fenced = not fenced
            continue
        if fenced:
            continue
        if directive is not None and stripped.startswith(":"):
            option = re.match(r"^:([^:]+):\s*(.*)$", stripped)
            if option and option.group(1).casefold() == "alt" and option.group(2):
                passages.append(
                    Passage(str(path), line_number, "markdown-alt", "<document>", option.group(2).strip())
                )
            continue
        if stripped == "$$":
            flush()
            math = not math
            continue
        if math:
            continue
        if not stripped:
            flush()
            continue

        for alt in _IMAGE_RE.findall(raw):
            cleaned_alt = _clean_markdown_inline(alt)
            if cleaned_alt:
                passages.append(Passage(str(path), line_number, "markdown-alt", "<document>", cleaned_alt))
        without_images = _IMAGE_RE.sub("", raw).strip()
        if not without_images:
            flush()
            continue

        heading = re.match(r"^#{1,6}\s+(.*)$", without_images)
        if heading:
            flush()
            text = _clean_markdown_inline(heading.group(1))
            passages.append(Passage(str(path), line_number, "markdown-heading", "<document>", text))
            continue

        listed = _LIST_RE.match(without_images)
        if listed:
            flush()
            text = _clean_markdown_inline(listed.group(1))
            if text:
                passages.append(Passage(str(path), line_number, "markdown-list", "<document>", text))
            continue

        if without_images.startswith("|") and without_images.endswith("|"):
            flush()
            cells = [cell.strip() for cell in without_images.strip("|").split("|")]
            if cells and not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                text = "; ".join(_clean_markdown_inline(cell) for cell in cells if cell)
                if text:
                    passages.append(Passage(str(path), line_number, "markdown-table", "<document>", text))
            continue

        if without_images.startswith(">"):
            flush()
            text = _clean_markdown_inline(without_images.lstrip("> "))
            if text:
                passages.append(Passage(str(path), line_number, "markdown-quote", "<document>", text))
            continue

        if not paragraph:
            paragraph_line = line_number
        paragraph.append(without_images)

    flush()
    return sorted(passages, key=lambda item: (item.line, item.kind))


class _VisibleHTMLParser(HTMLParser):
    _SUPPRESSED_TAGS = {"script", "style", "nav", "pre", "code", "svg"}
    _SUPPRESSED_CLASSES = {
        "admonition-title",
        "headerlink",
        "highlight",
        "related-pages",
        "sidebar",
        "sphinxsidebar",
        "toc",
    }

    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.path = path
        self.passages: list[Passage] = []
        self._suppressed: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        suppress = tag in self._SUPPRESSED_TAGS or bool(classes & self._SUPPRESSED_CLASSES)
        if self._suppressed or suppress:
            self._suppressed.append(tag)
            return
        line, _ = self.getpos()
        for attr, kind in (("alt", "html-alt"), ("aria-label", "html-aria")):
            text = " ".join((attributes.get(attr) or "").split())
            if text:
                self.passages.append(Passage(str(self.path), line, kind, "<rendered>", text))

    def handle_endtag(self, tag: str) -> None:
        if self._suppressed:
            self._suppressed.pop()

    def handle_data(self, data: str) -> None:
        if self._suppressed:
            return
        text = " ".join(data.split())
        if text:
            line, _ = self.getpos()
            self.passages.append(Passage(str(self.path), line, "html-visible", "<rendered>", text))


def extract_html(path: Path, source: str) -> list[Passage]:
    """Return visible and accessibility prose from an HTML document."""
    parser = _VisibleHTMLParser(path)
    parser.feed(source)
    parser.close()
    return sorted(parser.passages, key=lambda item: (item.line, item.kind, item.text))
