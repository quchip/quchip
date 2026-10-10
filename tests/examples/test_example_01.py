"""Contract coverage for the public statics and parameter-study guide."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import jupytext
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
    assert math.isclose(receipt["minimum_splitting_mhz"], 3.9873, rel_tol=1.0e-3)
    assert math.isclose(receipt["static_zz_khz"], 17.106, rel_tol=1.0e-3)


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
