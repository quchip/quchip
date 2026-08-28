"""Contract coverage for the public statics and parameter-study guide."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import jupytext
import matplotlib.image as mpimg
import nbformat


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MD = ROOT / "examples" / "01_resolve_and_sweep.md"
EXAMPLE_IPYNB = ROOT / "examples" / "01_resolve_and_sweep.ipynb"
RESULT_RE = re.compile(r"^RESULT statics=(\{.*\})$", re.MULTILINE)
PAPER_RESULT_RE = re.compile(r"^RESULT paper_statics=(\{.*\})$", re.MULTILINE)


def _code(notebook: dict) -> str:
    return "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def _stream_output(notebook: dict) -> str:
    return "".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def test_guide_code_matches_the_executed_notebook() -> None:
    """The reader-facing Markdown contains exactly the executed code."""
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    assert _code(authored) == _code(executed)
    assert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code")


def test_source_covers_static_observables_and_sweep_forms() -> None:
    """The guide stays static and uses the public analysis and sweep APIs."""
    source = EXAMPLE_MD.read_text(encoding="utf-8")
    code = _code(jupytext.read(EXAMPLE_MD))
    assert "# Statics and parameter studies" in source
    for required in (
        "chip.with_params(",
        "chip.freq(",
        "chip.transition_frequency(",
        "chip.kerr_matrix(",
        "chip.dressed_anharmonicity(",
        "chip.dispersive_shift(",
        "chip.static_zz(",
        "chip.drive_matrix_elements(",
        "SpectrumSweep",
        "Sweep.zip(",
        "Sweep.expand(",
        "dressed_index(",
        "assignment_overlaps",
        "state_components(",
        "Fluxonium(",
        "CouplingModel",
        'paper_chip.with_params({"q.phi_ext": phi_ext})',
        'point.dispersive_shift("q", "readout")',
        "np.linspace(0.5, 0.85, 351)",
        "np.interp(",
    ):
        assert required in code
    assert "QuantumSequence" not in code
    assert ".simulate(" not in code
    assert "from quchip.engine" not in code


def test_executed_receipt_matches_the_avoided_crossing() -> None:
    """The resolved crossing retains its checked splitting and RWA ledger."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])
    assert receipt["approximation"] == "RWA"
    assert receipt["dropped_rwa_terms"] == 4
    assert receipt["full_dimension"] == 64
    assert receipt["original_chip_unchanged"] is True
    assert receipt["sweep_points"] == 181
    assert math.isclose(receipt["minimum_splitting_mhz"], 4.4098, rel_tol=1.0e-3)
    assert math.isclose(receipt["static_zz_khz"], 22.604, rel_tol=1.0e-3)


def test_paper_example_reproduces_fluxonium_spectrum_and_readout() -> None:
    """The experimental example checks an independent model grid against both datasets."""
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = PAPER_RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    receipt = json.loads(matches[0])
    assert receipt["spectrum_points"] == 153
    assert receipt["spectrum_model_points"] == 351
    assert receipt["readout_points"] == 151
    assert receipt["readout_model_points"] == 351
    assert math.isclose(
        receipt["spectrum_median_absolute_error_mhz"], 1.5924, rel_tol=1.0e-3
    )
    assert math.isclose(
        receipt["spectrum_p95_absolute_error_mhz"], 6.0768, rel_tol=1.0e-3
    )
    assert math.isclose(receipt["chi_rmse_mhz"], 0.7652, rel_tol=1.0e-3)
    assert math.isclose(
        receipt["readout_frequency_rmse_mhz"], 1.0848, rel_tol=1.0e-3
    )


def test_figure_and_docs_route_are_valid() -> None:
    """The plotted crossing and its canonical guide route are present."""
    image = mpimg.imread(ROOT / "docs" / "images" / "resolve_and_sweep.png")
    assert image.shape[0] >= 700 and float(image.std()) > 0.02
    spectrum_image = mpimg.imread(
        ROOT / "docs" / "images" / "stefanski_fluxonium_spectrum.png"
    )
    readout_image = mpimg.imread(
        ROOT / "docs" / "images" / "stefanski_fluxonium_readout.png"
    )
    assert spectrum_image.shape[0] >= 700 and float(spectrum_image.std()) > 0.02
    assert readout_image.shape[0] >= 700 and float(readout_image.std()) > 0.02
    page = (ROOT / "docs" / "guides" / "statics-and-parameter-studies.md").read_text(encoding="utf-8")
    sqa = (ROOT / "docs" / "guides" / "from-sqa-2026.md").read_text(encoding="utf-8")
    assert "01_resolve_and_sweep.md" in page
    assert "https://docs.quchip.org/guides/statics-and-parameter-studies" in sqa


def test_paper_sources_are_linked_without_hidden_peak_selection() -> None:
    """The guide links the paper and public archive and compares extracted rows directly."""
    source = EXAMPLE_MD.read_text(encoding="utf-8")
    assert "https://arxiv.org/abs/2411.13437" in source
    assert "https://github.com/AndersenQubitLab/FPA-RO-experimental" in source
    assert "https://doi.org/10.4121/1092cb12-9198-4d43-8500-401c78a5dc15" in source
    assert "selected peak" not in source.lower()
