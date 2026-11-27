"""The executed study notebooks record the comparisons that their pages state.

The studies execute again only in the release suite, so these checks read the
committed outputs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
RESULT_RE = re.compile(r"^RESULT (\w+)=(\{.*\})$", re.MULTILINE)


def _receipts(stem: str) -> dict[str, dict]:
    notebook = nbformat.read(ROOT / "examples" / f"{stem}.ipynb", as_version=4)
    printed = "".join(
        "".join(output.get("text", ""))
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    return {name: json.loads(payload) for name, payload in RESULT_RE.findall(printed)}


def test_fluxonium_model_reproduces_the_published_spectrum_and_readout() -> None:
    """The model grid is independent of the measured points and covers both datasets."""
    receipt = _receipts("fluxonium-spectroscopy")["paper_statics"]
    assert receipt["spectrum_points"] == 153
    assert receipt["spectrum_model_points"] == 351
    assert receipt["readout_points"] == 151
    assert receipt["readout_model_points"] == 351
    assert math.isclose(receipt["spectrum_median_absolute_error_mhz"], 1.5924, rel_tol=1.0e-3)
    assert math.isclose(receipt["spectrum_p95_absolute_error_mhz"], 6.0768, rel_tol=1.0e-3)
    assert math.isclose(receipt["chi_rmse_mhz"], 0.7652, rel_tol=1.0e-3)
    assert math.isclose(receipt["readout_frequency_rmse_mhz"], 1.0848, rel_tol=1.0e-3)


def test_fluxonium_fit_recovers_the_circuit_energies_on_held_out_points() -> None:
    """The fit uses every eighth measurement and reports the agreement on the others."""
    receipt = _receipts("fluxonium-spectroscopy")["experimental_statics"]
    assert receipt["fit_success"] is True
    assert receipt["training_points"] == 20
    assert receipt["holdout_points"] == 133
    assert max(abs(value) for value in receipt["relative_parameter_error"]) < 0.01
    assert receipt["holdout_median_absolute_error_mhz"] < 1.0
