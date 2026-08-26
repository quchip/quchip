"""Contract coverage for the public statics and sweep example."""

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
EXAMPLE_MD = ROOT / "examples" / "01_resolve_and_sweep.md"
EXAMPLE_IPYNB = ROOT / "examples" / "01_resolve_and_sweep.ipynb"
FIGURE = ROOT / "docs" / "images" / "resolve_and_sweep.png"
RESULT_RE = re.compile(r"^RESULT statics=(\{.*\})$", re.MULTILINE)


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


def test_source_stays_on_the_small_public_surface() -> None:
    source = EXAMPLE_MD.read_text(encoding="utf-8")
    code = "\n\n".join(_code_cells(jupytext.read(EXAMPLE_MD)))

    assert "# Resolve and sweep a chip" in source
    assert "from quchip import" in code
    for required in (
        "DuffingTransmon",
        "Resonator",
        "Capacitive",
        "Chip",
        "RWA",
        "Sweep",
        "SpectrumSweep",
        "chip.freq(",
        "chip.static_zz(",
        "chip.hamiltonian()",
        "chip.resolve().dropped_terms_summary()",
        "dressed_index(",
    ):
        assert required in code
    for excluded in (
        "from quchip.engine",
        "from quchip.backend",
        "import qutip",
        "import dynamiqs",
        "class ",
        "QuantumSequence",
        "jax.grad",
        "eliminate(",
    ):
        assert excluded not in code


def test_executed_receipt_matches_the_declared_physics() -> None:
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])

    assert receipt.keys() == {
        "approximation",
        "dressed_frequencies_ghz",
        "dropped_rwa_terms",
        "figure",
        "full_dimension",
        "inferred_exchange_rate_mhz",
        "minimum_at_bare_q2_ghz",
        "minimum_splitting_mhz",
        "original_chip_unchanged",
        "relative_difference_to_second_order",
        "second_order_splitting_scale_mhz",
        "static_zz_khz",
        "sweep_points",
    }
    assert receipt["approximation"] == "RWA"
    assert receipt["dropped_rwa_terms"] == 4
    assert receipt["full_dimension"] == 64
    assert receipt["original_chip_unchanged"] is True
    assert receipt["sweep_points"] == 181
    assert receipt["figure"] == "../docs/images/resolve_and_sweep.png"
    assert math.isclose(receipt["minimum_at_bare_q2_ghz"], 5.300, abs_tol=1.0e-12)
    assert math.isclose(receipt["minimum_splitting_mhz"], 4.4098, rel_tol=1.0e-3)
    assert math.isclose(receipt["inferred_exchange_rate_mhz"], 2.2049, rel_tol=1.0e-3)
    assert math.isclose(receipt["second_order_splitting_scale_mhz"], 4.0, abs_tol=1.0e-12)
    assert 0.08 < receipt["relative_difference_to_second_order"] < 0.11
    assert math.isclose(receipt["static_zz_khz"], 22.604, rel_tol=1.0e-3)
    assert all(math.isfinite(value) for value in receipt["dressed_frequencies_ghz"].values())


def test_executed_figure_is_valid() -> None:
    image = mpimg.imread(FIGURE)
    assert image.shape[0] >= 700
    assert image.shape[1] >= 1000
    assert image.shape[2] in (3, 4)
    assert float(image.std()) > 0.02


def test_documentation_exposes_the_post_talk_path() -> None:
    guide = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs" / "examples" / "resolve-and-sweep.md").read_text(encoding="utf-8")

    assert "../examples/resolve-and-sweep" in guide
    assert "../examples/hello-chip" in guide
    assert "guides/from-sqa-2026" in index
    assert "https://docs.quchip.org/guides/from-sqa-2026" in readme
    assert "01_resolve_and_sweep.md" in page
    assert "01_resolve_and_sweep.ipynb" in page
    assert "https://github.com/quchip/quchip/blob/main/examples/01_resolve_and_sweep.md" in page
    assert "resolve_and_sweep.png" in page
