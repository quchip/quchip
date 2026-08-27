"""Contract coverage for the public chip-transformation guide."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import jupytext
import matplotlib.image as mpimg
import nbformat
from jupytext.compare import compare_notebooks


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


def test_guide_is_a_strict_executed_jupytext_pair() -> None:
    """The canonical Markdown and executed notebook have identical code cells."""
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    compare_notebooks(authored, executed, fmt="md", compare_outputs=False, compare_ids=False)
    strict = subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "md", "--test-strict", str(EXAMPLE_IPYNB)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode == 0, strict.stdout + strict.stderr


def test_source_covers_the_public_transformation_families() -> None:
    """The guide shows copy, reduction, partition, fit, and schedule-aware paths."""
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
        "fit_a_dress(",
        "sequence.active_patch(hops=1, method=\"sw\")",
        "patch.validity",
        "patch.simulate(tlist=times)",
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
    assert receipt["same_schedule"] is True
    assert receipt["maximum_population_residual"] < receipt["residual_tolerance"]
    assert math.isfinite(receipt["maximum_population_residual"])


def test_figure_and_docs_route_are_valid() -> None:
    """The comparison figure and canonical guide route are present."""
    image = mpimg.imread(ROOT / "docs" / "images" / "reduce_and_replay.png")
    assert image.shape[0] >= 900 and float(image.std()) > 0.02
    page = (ROOT / "docs" / "guides" / "chip-transformations.md").read_text(encoding="utf-8")
    sqa = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    assert "02_reduce_and_replay.md" in page
    assert "https://docs.quchip.org/guides/chip-transformations" in sqa
