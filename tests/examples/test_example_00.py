"""Contract coverage for the public dynamics and readout guide."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import jupytext
import matplotlib.image as mpimg
import nbformat
import numpy as np
from jupytext.compare import compare_notebooks


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MD = ROOT / "examples" / "00_hello_chip.md"
EXAMPLE_IPYNB = ROOT / "examples" / "00_hello_chip.ipynb"
RESULT_RE = re.compile(r"^RESULT (drive|readout)=(\{.*\})$", re.MULTILINE)


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


def test_guide_progresses_through_the_public_dynamics_surface() -> None:
    """The guide covers one solve, pulse composition, batches, result access, and readout."""
    source = EXAMPLE_MD.read_text(encoding="utf-8")
    code = _code(jupytext.read(EXAMPLE_MD))
    assert "# Dynamics, pulses, observables, and readout" in source
    for heading in (
        "## One pulse, one trace",
        "## Build a longer schedule",
        "## Compare pulse bandwidth and leakage",
        "## Read results at the right level",
        "## Part 2: Dispersive readout",
        "## What this readout model contains",
    ):
        assert heading in source
    for required in (
        "QuantumSequence",
        "GaussianDRAG",
        ".delay(",
        ".vz(",
        ".barrier(",
        ".simulate(",
        ".simulate_batch(",
        ".plot_populations(",
        ".overlap(",
        ".reduced_state(",
        ".check_truncation(",
        'chip.e_ops(r="a")',
        'readout_batch.expect("r")',
    ):
        assert required in code
    for forbidden in ("from quchip.engine", "from quchip.chip", "import qutip", "import dynamiqs", "._expect_data"):
        assert forbidden not in code


def test_executed_receipts_record_leakage_and_readout() -> None:
    """The executed notebook records the intended physical comparisons."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    receipts = {name: json.loads(payload) for name, payload in RESULT_RE.findall(_stream_output(executed))}
    assert receipts.keys() == {"drive", "readout"}
    drive = receipts["drive"]
    assert drive["final_p1"]["long"] > 0.90
    assert drive["peak_p2"]["short"] > 0.02
    assert drive["peak_p2"]["long"] < 0.25 * drive["peak_p2"]["short"]
    readout = receipts["readout"]
    assert readout["solver"] == "mesolve"
    assert readout["final_iq_separation"] > 0.01
    assert np.isfinite(readout["conditional_resonator_frequencies_ghz"]).all()


def test_committed_figures_are_valid() -> None:
    """Both selected website figures are non-empty raster images."""
    for relative in ("docs/images/hello_qubit_drive_leakage.png", "docs/images/hello_dispersive_readout_iq.png"):
        image = mpimg.imread(ROOT / relative)
        assert image.ndim == 3 and image.shape[2] in (3, 4)
        assert float(image.std()) > 0.02


def test_documentation_uses_one_canonical_guide_route() -> None:
    """Reader navigation points to the full guide on docs.quchip.org."""
    page = (ROOT / "docs" / "guides" / "dynamics-pulses-and-readout.md").read_text(encoding="utf-8")
    sqa = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "00_hello_chip.md" in page
    assert "guides/dynamics-pulses-and-readout" in index
    url = "https://docs.quchip.org/guides/dynamics-pulses-and-readout"
    assert url in sqa and url in readme
    assert "github.com/quchip/quchip/blob/main/examples" not in sqa
