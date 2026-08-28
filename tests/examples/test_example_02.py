"""Contract coverage for the public chip-transformation guide."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import jupytext
import matplotlib.image as mpimg
import nbformat


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MD = ROOT / "examples" / "02_reduce_and_replay.md"
EXAMPLE_IPYNB = ROOT / "examples" / "02_reduce_and_replay.ipynb"
RESULT_RE = re.compile(r"^RESULT reduction=(\{.*\})$", re.MULTILINE)


def _code(notebook: dict) -> str:
    return "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def _stream_output(notebook: dict) -> str:
    return "".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def _markdown_output(notebook: dict) -> str:
    return "\n".join(
        "".join(output.get("data", {}).get("text/markdown", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
    )


def test_guide_code_matches_the_executed_notebook() -> None:
    """The reader-facing Markdown contains exactly the executed code."""
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    assert _code(authored) == _code(executed)
    assert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code")


def test_source_covers_the_public_transformation_families() -> None:
    """The guide shows copy, reduction, partition, and schedule-aware paths."""
    source = EXAMPLE_MD.read_text(encoding="utf-8")
    code = _code(jupytext.read(EXAMPLE_MD))
    assert "# Chip transformations" in source
    for required in (
        ".with_params(",
        ".clone()",
        "Chip.from_dict(chip.to_dict())",
        'eliminate(bridge, bus, method="sw")',
        'eliminate(bridge, bus, method="exact")',
        '"dq-dr"',
        ".partition()",
        "sequence.active_patch(hops=1, method=\"sw\")",
        ".plot_graph(",
        "patch.validity",
        "patch.simulate(tlist=times)",
        "coupling_strength = 0.012",
    ):
        assert required in code
    for forbidden in ("from quchip.engine", "from quchip.chip", "import qutip", "import dynamiqs", "patch.steps"):
        assert forbidden not in code


def test_executed_receipt_records_validity_and_forward_error() -> None:
    """The active-patch comparison records validity and a forward observable error."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])
    assert receipt["active_labels"] == ["q0", "q1"]
    assert receipt["eliminated_labels"] == ["q3", "q2"]
    assert receipt["full_dimension"] == 81 and receipt["reduced_dimension"] == 9
    assert receipt["all_folds_valid"] is True
    assert 0.03 < receipt["maximum_g_over_delta"] < 0.1
    assert receipt["same_schedule"] is True
    assert receipt["maximum_population_residual"] < receipt["residual_tolerance"]
    assert math.isfinite(receipt["maximum_population_residual"])


def test_executed_comparison_separates_declared_and_dressed_values() -> None:
    """The guide exposes the folded bare correction beside retained dressed observables."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    table = _markdown_output(executed)
    assert "| Quantity | Full model (MHz) | Active patch (MHz) | Patch - full (kHz) |" in table
    assert "| declared q1.freq |" in table
    assert "| dressed f01(q1) |" in table
    assert "| full-pull K(q0, q1) |" in table
    assert "| declared c01.g |" in table


def test_figure_and_docs_route_are_valid() -> None:
    """The comparison figure and canonical guide route are present."""
    image = mpimg.imread(ROOT / "docs" / "images" / "reduce_and_replay.png")
    assert image.shape[0] >= 900 and float(image.std()) > 0.02
    page = (ROOT / "docs" / "guides" / "chip-transformations.md").read_text(encoding="utf-8")
    sqa = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    assert "02_reduce_and_replay.md" in page
    assert "https://docs.quchip.org/guides/chip-transformations" in sqa
    for filename in ("active-patch-full.html", "active-patch-reduced.html"):
        topology = (ROOT / "docs" / "_static" / filename).read_text(encoding="utf-8")
        assert "vis-network" in topology
