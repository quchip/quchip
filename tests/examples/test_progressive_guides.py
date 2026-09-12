"""The guides and SQA page begin with small public-API examples."""

from __future__ import annotations

import contextlib
import io
import re
import subprocess
import sys
import warnings
from pathlib import Path

import jupytext
import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
EXECUTED_BLOCK_RE = re.compile(
    r"```python\n(.*?)\n```(?:\n\nOutput:\n\n```text\n(.*?)\n```)?",
    re.DOTALL,
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))"
    r"(?:[eE][-+]?\d+)?(?![A-Za-z_])"
)
GUIDE_OUTPUT_RTOL = 1e-10
# Two hertz for GHz-valued receipts, below the guide's displayed precision.
GUIDE_OUTPUT_ATOL = 2e-9


def _assert_output_matches(actual: str, expected: str | None) -> None:
    """Compare guide output exactly except for numerical solver roundoff."""
    actual = actual.strip()
    expected = (expected or "").strip()
    assert NUMBER_RE.sub("<number>", actual) == NUMBER_RE.sub("<number>", expected)

    actual_numbers = np.array([float(value) for value in NUMBER_RE.findall(actual)])
    expected_numbers = np.array([float(value) for value in NUMBER_RE.findall(expected)])
    np.testing.assert_allclose(
        actual_numbers,
        expected_numbers,
        rtol=GUIDE_OUTPUT_RTOL,
        atol=GUIDE_OUTPUT_ATOL,
    )


def _run_opening_example(path: str) -> dict[str, object]:
    notebook = jupytext.read(ROOT / path)
    namespace: dict[str, object] = {"__name__": "__guide_example__"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        with contextlib.chdir(ROOT / "examples"):
            started = False
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    started = True
                    exec(compile(cell.source, str(ROOT / path), "exec"), namespace)
                elif started and ("\n## " in "\n" + cell.source or "```{figure}" in cell.source):
                    break
    return namespace


@pytest.mark.examples
def test_statics_guide_starts_with_a_small_declaration() -> None:
    """The opening statics example reads one declared chip before sweeping it."""
    example = _run_opening_example("examples/01_resolve_and_sweep.md")
    assert tuple(device.label for device in example["chip"].devices) == ("q1", "q2", "bus")
    assert float(example["chip"].freq("q1")) > 5.0


@pytest.mark.examples
@pytest.mark.optional_backend
def test_differentiability_guide_starts_with_static_shapes() -> None:
    """The opening differentiability cell returns one gradient and one Jacobian."""
    pytest.importorskip("dynamiqs")
    example = _run_opening_example("examples/03_differentiate_a_driven_chip.md")
    assert example["static_gradient"].shape == (3,)
    assert example["static_jacobian"].shape == (2, 3)


def test_committed_markdown_contains_current_notebook_outputs() -> None:
    """Every canonical Markdown guide contains outputs from its executed pair."""
    check = subprocess.run(
        [sys.executable, "tools/sync_example_outputs.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    for path in sorted((ROOT / "examples").glob("0[0-5]_*.md")):
        assert "<!-- executed-output:start -->" in path.read_text(encoding="utf-8")


@pytest.mark.examples
@pytest.mark.parametrize("guide", ["defining-and-inspecting-a-chip", "steady-state-and-vna", "slh-networks"])
def test_guide_outputs_match_a_fresh_execution(guide: str) -> None:
    """Standalone guides execute their physical checks and reproduce shown output."""
    path = ROOT / "docs" / "guides" / f"{guide}.md"
    source = path.read_text(encoding="utf-8")
    blocks = EXECUTED_BLOCK_RE.findall(source)
    assert blocks, "the guide must contain executable examples with displayed output"

    namespace: dict[str, object] = {"__name__": "__defining_guide__"}
    with contextlib.chdir(ROOT / "docs" / "images"):
        for index, (code, expected) in enumerate(blocks, start=1):
            captured = io.StringIO()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
                with contextlib.redirect_stdout(captured):
                    exec(compile(code, f"{path}#cell-{index}", "exec"), namespace)
            _assert_output_matches(captured.getvalue(), expected)


@pytest.mark.examples
@pytest.mark.optional_backend
def test_sqa_snippets_execute_independently() -> None:
    """Every SQA snippet runs alone and produces its displayed output."""
    pytest.importorskip("dynamiqs")
    path = ROOT / "docs" / "guides" / "from-sqa-2026.md"
    snippets = EXECUTED_BLOCK_RE.findall(path.read_text(encoding="utf-8"))
    assert len(snippets) == 5
    for index, (snippet, expected) in enumerate(snippets):
        namespace: dict[str, object] = {"__name__": f"__sqa_snippet_{index}__"}
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(snippet, f"{path}#snippet-{index + 1}", "exec"), namespace)
        _assert_output_matches(captured.getvalue(), expected)
