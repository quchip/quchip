"""Contract coverage for the energy-participation quantization guide."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import jupytext
import nbformat


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MD = ROOT / "examples" / "05_energy_participation.md"
EXAMPLE_IPYNB = ROOT / "examples" / "05_energy_participation.ipynb"
RESULT_RE = re.compile(r"^RESULT energy_participation=(\{.*\})$", re.MULTILINE)


def _code(notebook: dict) -> str:
    return "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def _stream_output(notebook: dict) -> str:
    return "".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def _receipt() -> dict:
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    matches = RESULT_RE.findall(_stream_output(executed))
    assert len(matches) == 1
    return json.loads(matches[0])


def test_guide_code_matches_the_executed_notebook() -> None:
    """The reader-facing Markdown contains exactly the executed code."""
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    assert _code(authored) == _code(executed)
    assert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code")


def test_receipt_separates_first_order_error_from_truncation() -> None:
    """The cosine chip matches the circuit within its truncation bound; first order misses by percents."""
    receipt = _receipt()
    assert receipt["sweep_points"] == 13
    assert receipt["cosine_max_relative_error"] <= receipt["cosine_truncation_bound"] < 1e-3
    alpha_low, alpha_high = receipt["first_order_alpha_error"]
    assert 0.05 < alpha_low <= alpha_high < 0.2
    chi_low, chi_high = receipt["first_order_chi_error"]
    assert chi_low < 0.0 < chi_high  # the first-order dispersive-shift error changes sign across the sweep
    assert receipt["charge_dispersion_alpha_max"] > receipt["cosine_max_relative_error"]
    design = receipt["design"]
    assert math.isclose(design["cosine"]["chi_mhz"], design["circuit"]["chi_mhz"], rel_tol=1e-4)
    assert abs(design["first_order"]["chi_mhz"] / design["circuit"]["chi_mhz"] - 1) > 0.1


def test_readout_reaches_the_linear_dispersive_steady_state() -> None:
    """Both readout models settle at their linear pointer separation, set by kappa = 2 pi f_r / Q_r."""
    receipt = _receipt()
    # Readout-mode linear frequency 7.2167 GHz and Q_r = 1.5e4 give kappa / 2 pi = f_r / Q_r.
    assert math.isclose(receipt["kappa_mhz"], 1e3 * 7.2167 / 1.5e4, rel_tol=1e-4)
    for record in receipt["readout"].values():
        assert record["steady_state_residual"] <= record["steady_state_tolerance"]
    assert 0.8 < receipt["rate_ratio_first_order_to_cosine"] < 0.95


def test_published_figures_have_themed_variants() -> None:
    """Each saved figure is committed with its dark SVG and PDF variants."""
    for relative in _receipt()["figures"]:
        light = (ROOT / "examples" / relative).resolve()
        assert light.exists()
        assert light.with_name(f"{light.stem}-dark.svg").exists()
        assert light.with_suffix(".pdf").exists()
