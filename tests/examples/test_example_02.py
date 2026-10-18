"""Contract coverage for the public chip-transformation guide."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import jupytext
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


def test_guide_code_matches_the_executed_notebook() -> None:
    """The reader-facing Markdown contains exactly the executed code."""
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    assert _code(authored) == _code(executed)
    assert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code")


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


def _printed_value(output: str, label: str) -> float:
    match = re.search(rf"^{re.escape(label)}: (\S+)$", output, re.MULTILINE)
    assert match is not None
    return float(match.group(1))


def test_executed_outputs_record_feedline_and_excitation_space_checks() -> None:
    """The feedline reduction stays within its dispersive scale, the idle ring uses its exact space, and K = 2 holds."""
    output = _stream_output(nbformat.read(EXAMPLE_IPYNB, as_version=4))
    # Purcell dispersion of the strongest pair sets the field-error scale (g/Δ)²κ/(2π|Δ|).
    assert _printed_value(output, "Maximum complex S difference") < 10 * (0.105 / 2.0) ** 2 * 0.001 / 2.0
    assert _printed_value(output, "Return-path self-energy over width") < 0.1
    # QuTiP's default relative tolerance of 1e-6 sets the floor of the cutoff comparison.
    assert _printed_value(output, "Population error at K = 2 (15 states)") < 1e-5
    assert "Driven chain: 81 of 81 states" in output
    assert f"Idle ring: 24 of {3**24} states" in output
    # The exact 24-state solve agrees with the hopping propagator to round-off.
    assert _printed_value(output, "Maximum population difference") < 1e-10
    assert "Lossy ring: mesolve on 25 states" in output


def test_executed_pathways_give_the_exchange_shift() -> None:
    """The q1 shift of the second step is the second-order exchange through q2."""
    output = _stream_output(nbformat.read(EXAMPLE_IPYNB, as_version=4))
    pathways = re.findall(r"^(\(\d+(?:, \d+)*\)): (\S+) MHz$", output, re.MULTILINE)
    assert [state for state, _ in pathways] == ["(0, 0, 1)"]
    # The first step lowers q2 by g²/Δ, so the second step sees the detuning Δ - g²/Δ.
    coupling, detuning = 0.012, 0.35
    shifted_detuning = detuning - coupling**2 / detuning
    assert abs(float(pathways[0][1]) * 1e-3 + coupling**2 / shifted_detuning) < 5e-7
