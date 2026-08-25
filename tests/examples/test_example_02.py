"""Contract coverage for the public reduction and replay example."""

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
FIGURE = ROOT / "docs" / "images" / "reduce_and_replay.png"
RESULT_RE = re.compile(r"^RESULT reduction=(\{.*\})$", re.MULTILINE)


def _code_cells(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _stream_output(notebook: dict) -> str:
    chunks: list[str] = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append("".join(output.get("text", [])))
    return "".join(chunks)


def test_example_is_a_strict_executed_jupytext_pair() -> None:
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    compare_notebooks(authored, executed, fmt="md", compare_outputs=False, compare_ids=False)
    assert all(cell.cell_type != "raw" for cell in executed.cells)

    strict = subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "md", "--test-strict", str(EXAMPLE_IPYNB)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode == 0, strict.stdout + strict.stderr


def test_source_uses_the_schedule_aware_public_surface() -> None:
    source = EXAMPLE_MD.read_text(encoding="utf-8")
    code = "\n\n".join(_code_cells(jupytext.read(EXAMPLE_MD)))

    assert "# Reduce and replay a chip" in source
    for required in (
        "from quchip import",
        "QuantumSequence",
        "sequence.active_patch(hops=1, method=\"sw\")",
        "patch.simulate(tlist=times)",
        "full_result.population(\"q0\", level=1)",
        "reduced_result.population(\"q0\", level=1)",
        "patch.validity",
    ):
        assert required in code
    for excluded in (
        "from quchip.chip",
        "from quchip.engine",
        "from quchip.backend",
        "import qutip",
        "import dynamiqs",
        "eliminate(",
        "patch.steps",
        "._expect_data",
        "jax.grad",
    ):
        assert excluded not in code
    assert code.count(".schedule(") == 1


def test_executed_receipt_records_validity_and_forward_error() -> None:
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])

    assert receipt.keys() == {
        "active_labels",
        "all_folds_valid",
        "eliminated_labels",
        "figure",
        "full_dimension",
        "maximum_population_residual",
        "maximum_g_over_delta",
        "minimum_block_gap_ghz",
        "original_chip_unchanged",
        "peak_full_population",
        "reduced_dimension",
        "reduction_method",
        "residual_tolerance",
        "same_schedule",
    }
    assert receipt["active_labels"] == ["q0", "q1"]
    assert receipt["eliminated_labels"] == ["q3", "q2"]
    assert receipt["full_dimension"] == 81
    assert receipt["reduced_dimension"] == 9
    assert receipt["reduction_method"] == "sw"
    assert receipt["all_folds_valid"] is True
    assert receipt["same_schedule"] is True
    assert receipt["original_chip_unchanged"] is True
    assert receipt["figure"] == "../docs/images/reduce_and_replay.png"
    assert 0.0 < receipt["maximum_g_over_delta"] < 0.1
    assert receipt["minimum_block_gap_ghz"] > 0.3
    assert receipt["peak_full_population"] > 0.7
    assert receipt["maximum_population_residual"] < receipt["residual_tolerance"]
    assert receipt["maximum_population_residual"] < 1.0e-5
    assert math.isfinite(receipt["maximum_population_residual"])


def test_executed_figure_is_valid() -> None:
    image = mpimg.imread(FIGURE)
    assert image.shape[0] >= 900
    assert image.shape[1] >= 1200
    assert image.shape[2] in (3, 4)
    assert float(image.std()) > 0.02


def test_documentation_links_the_reduction_unit() -> None:
    guide = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs" / "examples" / "reduce-and-replay.md").read_text(encoding="utf-8")

    assert "../examples/reduce-and-replay" in guide
    assert "examples/reduce-and-replay" in index
    assert "https://docs.quchip.org/examples/reduce-and-replay" in readme
    assert "02_reduce_and_replay.md" in page
    assert "02_reduce_and_replay.ipynb" in page
    assert "reduce_and_replay.png" in page
