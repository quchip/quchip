"""The guides and SQA page begin with small public-API examples."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import jupytext
import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
CODE_BLOCK_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def _run_first_cell(path: str) -> dict[str, object]:
    notebook = jupytext.read(ROOT / path)
    first = next(cell for cell in notebook.cells if cell.cell_type == "code")
    namespace: dict[str, object] = {"__name__": "__guide_example__"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        exec(compile(first.source, str(ROOT / path), "exec"), namespace)
    return namespace


@pytest.mark.examples
def test_statics_guide_starts_with_a_small_sweep() -> None:
    """The opening statics cell resolves the intended avoided crossing."""
    example = _run_first_cell("examples/01_resolve_and_sweep.md")
    assert 4.3e-3 < float(np.min(example["splitting"])) < 4.5e-3


@pytest.mark.examples
def test_dynamics_guide_starts_with_one_pulse_and_one_solve() -> None:
    """The opening dynamics cell returns one finite population trajectory."""
    example = _run_first_cell("examples/00_hello_chip.md")
    population = np.asarray(example["result"].population("q", level=1))
    assert population.shape == (161,)
    assert np.isfinite(population).all()


@pytest.mark.examples
def test_transformations_guide_starts_with_one_active_patch() -> None:
    """The opening transformation cell replays one reduced schedule."""
    example = _run_first_cell("examples/02_reduce_and_replay.md")
    assert example["patch"].active_labels == ("q0", "q1")
    assert example["patch"].eliminated_labels == ("q2",)
    assert np.max(np.abs(example["p_full"] - example["p_small"])) < 1.0e-5


@pytest.mark.examples
@pytest.mark.optional_backend
def test_differentiability_guide_starts_with_static_shapes() -> None:
    """The opening differentiability cell returns one gradient and one Jacobian."""
    pytest.importorskip("dynamiqs")
    example = _run_first_cell("examples/03_differentiate_a_driven_chip.md")
    assert example["static_gradient"].shape == (3,)
    assert example["static_jacobian"].shape == (2, 3)


def test_sqa_snippets_are_small_and_link_once_per_topic() -> None:
    """The SQA page contains four runnable-sized snippets and four canonical links."""
    source = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    snippets = CODE_BLOCK_RE.findall(source)
    assert len(snippets) == 4
    assert all(len(snippet.splitlines()) <= 36 for snippet in snippets)
    for route in (
        "statics-and-parameter-studies",
        "dynamics-pulses-and-readout",
        "chip-transformations",
        "differentiability",
    ):
        assert source.count(f"https://docs.quchip.org/guides/{route}") == 1
    assert "github.com" not in source


@pytest.mark.examples
@pytest.mark.optional_backend
def test_sqa_snippets_execute_independently() -> None:
    """Every snippet on the SQA page runs without relying on an earlier block."""
    pytest.importorskip("dynamiqs")
    path = ROOT / "docs" / "guides" / "from-sqa-2026.md"
    snippets = CODE_BLOCK_RE.findall(path.read_text(encoding="utf-8"))
    for index, snippet in enumerate(snippets):
        namespace: dict[str, object] = {"__name__": f"__sqa_snippet_{index}__"}
        exec(compile(snippet, f"{path}#snippet-{index + 1}", "exec"), namespace)
