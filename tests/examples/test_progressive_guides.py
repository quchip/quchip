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
NUMBER_RE = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))"
    r"(?:[eE][-+]?\d+)?(?![A-Za-z_])"
)
GUIDE_OUTPUT_RTOL = 1e-10
# Two hertz for GHz-valued receipts, below the guide's displayed precision.
GUIDE_OUTPUT_ATOL = 2e-9


def _assert_output_matches(actual: str, expected: str) -> None:
    """Compare guide output exactly except for numerical solver roundoff."""
    actual = actual.strip()
    expected = expected.strip()
    assert NUMBER_RE.sub("<number>", actual) == NUMBER_RE.sub("<number>", expected)

    actual_numbers = np.array([float(value) for value in NUMBER_RE.findall(actual)])
    expected_numbers = np.array([float(value) for value in NUMBER_RE.findall(expected)])
    np.testing.assert_allclose(
        actual_numbers,
        expected_numbers,
        rtol=GUIDE_OUTPUT_RTOL,
        atol=GUIDE_OUTPUT_ATOL,
    )


def test_guide_output_comparison_accepts_solver_roundoff() -> None:
    """Platform-level numerical roundoff does not stale an executed guide."""
    _assert_output_matches(
        "dressed f01: 4.998533473435458",
        "dressed f01: 4.99853347343543",
    )
    _assert_output_matches(
        "fit residual: -1.4e-08",
        "fit residual: -1.3e-08",
    )


def test_guide_output_comparison_rejects_meaningful_changes() -> None:
    """Guide checks still reject changed prose and changed results."""
    with pytest.raises(AssertionError):
        _assert_output_matches("dressed f01: 4.9", "dressed f01: 5.0")
    with pytest.raises(AssertionError):
        _assert_output_matches("bare f01: 5.0", "dressed f01: 5.0")
    with pytest.raises(AssertionError):
        _assert_output_matches("fit residual: 1e-6", "fit residual: 0.0")


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


def test_sqa_page_has_one_snippet_and_link_per_topic() -> None:
    """The SQA page gives each topic one runnable snippet and canonical link."""
    source = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    snippets = CODE_BLOCK_RE.findall(source)
    assert len(snippets) == 5
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
    assert blocks, "the guide must contain executable examples with displayed output"

    namespace: dict[str, object] = {"__name__": "__defining_guide__"}
    with contextlib.chdir(path.parent):
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
