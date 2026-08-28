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
CODE_BLOCK_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
EXECUTED_BLOCK_RE = re.compile(
    r"```python\n(.*?)\n```\n\nOutput:\n\n```text\n(.*?)\n```",
    re.DOTALL,
)


def _run_first_cell(path: str) -> dict[str, object]:
    notebook = jupytext.read(ROOT / path)
    first = next(cell for cell in notebook.cells if cell.cell_type == "code")
    namespace: dict[str, object] = {"__name__": "__guide_example__"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        with contextlib.chdir(ROOT / "examples"):
            exec(compile(first.source, str(ROOT / path), "exec"), namespace)
    return namespace


@pytest.mark.examples
def test_statics_guide_starts_with_a_small_declaration() -> None:
    """The opening statics cell reads one declared chip before sweeping it."""
    example = _run_first_cell("examples/01_resolve_and_sweep.md")
    assert tuple(device.label for device in example["chip"].devices) == ("q1", "q2", "bus")
    assert float(example["chip"].freq("q1")) > 5.0


@pytest.mark.examples
def test_dynamics_guide_starts_with_one_pulse_and_one_solve() -> None:
    """The opening dynamics cell returns one finite population trajectory."""
    example = _run_first_cell("examples/00_hello_chip.md")
    population = np.asarray(example["result"].population("q", level=1))
    assert population.shape == (161,)
    assert np.isfinite(population).all()


@pytest.mark.examples
def test_transformations_guide_starts_with_one_immutable_parameter_change() -> None:
    """The opening transformation cell changes one number on an isolated copy."""
    example = _run_first_cell("examples/02_reduce_and_replay.md")
    assert example["chip"].parameters["q.freq"] == 5.0
    assert example["rebound"].parameters["q.freq"] == 5.1
    assert example["cloned"] is not example["chip"]


@pytest.mark.examples
@pytest.mark.optional_backend
def test_differentiability_guide_starts_with_static_shapes() -> None:
    """The opening differentiability cell returns one gradient and one Jacobian."""
    pytest.importorskip("dynamiqs")
    example = _run_first_cell("examples/03_differentiate_a_driven_chip.md")
    assert example["static_gradient"].shape == (3,)
    assert example["static_jacobian"].shape == (2, 3)


def test_sqa_snippets_are_small_and_link_once_per_topic() -> None:
    """The SQA page contains five runnable snippets and canonical links."""
    source = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    snippets = CODE_BLOCK_RE.findall(source)
    assert len(snippets) == 5
    assert all(len(snippet.splitlines()) <= 36 for snippet in snippets)
    for route in (
        "defining-and-inspecting-a-chip",
        "statics-and-parameter-studies",
        "dynamics-pulses-and-readout",
        "chip-transformations",
        "differentiability",
    ):
        assert source.count(f"https://docs.quchip.org/guides/{route}") == 1
    assert "github.com" not in source


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
    for path in sorted((ROOT / "examples").glob("0[0-3]_*.md")):
        assert "<!-- executed-output:start -->" in path.read_text(encoding="utf-8")


@pytest.mark.examples
def test_defining_guide_outputs_match_a_fresh_execution() -> None:
    """The standalone introductory guide shows the output its cells produce."""
    pytest.importorskip("scqubits")
    path = ROOT / "docs" / "guides" / "defining-and-inspecting-a-chip.md"
    source = path.read_text(encoding="utf-8")
    blocks = EXECUTED_BLOCK_RE.findall(source)
    assert len(blocks) == 10
    for required in (
        "CrossKerr",
        "cross_kerr",
        '"exchange_rate"',
        "constraints=constraints",
        "fit.history",
        "np.minimum.accumulate",
        "fit_a_dress_convergence.png",
    ):
        assert required in source

    namespace: dict[str, object] = {"__name__": "__defining_guide__"}
    with contextlib.chdir(path.parent):
        for index, (code, expected) in enumerate(blocks, start=1):
            captured = io.StringIO()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
                with contextlib.redirect_stdout(captured):
                    exec(compile(code, f"{path}#cell-{index}", "exec"), namespace)
            assert captured.getvalue().strip() == expected.strip()


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
        assert captured.getvalue().strip() == expected.strip()
