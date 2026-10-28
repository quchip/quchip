"""Contract coverage for the public dynamics and readout guide."""

from __future__ import annotations

import json
import re
from pathlib import Path

import jupytext
import nbformat
import numpy as np


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


def test_guide_code_matches_the_executed_notebook() -> None:
    """The reader-facing Markdown contains exactly the executed code."""
    authored = jupytext.read(EXAMPLE_MD)
    executed = nbformat.read(EXAMPLE_IPYNB, as_version=4)
    nbformat.validate(executed)
    assert _code(authored) == _code(executed)
    assert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code")


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


def test_executed_outputs_read_the_pulses_as_gates() -> None:
    """The short pulse leaks most of |0⟩, and the long pulse keeps a coherent rotation error."""
    output = _stream_output(nbformat.read(EXAMPLE_IPYNB, as_version=4))
    gates = {
        name: (float(fidelity), float(leakage))
        for name, fidelity, leakage in re.findall(
            r"^(Short|Long): fidelity to −iY = (\S+); leakage from \|0⟩ = (\S+)$", output, re.MULTILINE,
        )
    }
    assert gates.keys() == {"Short", "Long"}
    short_fidelity, short_leakage = gates["Short"]
    long_fidelity, long_leakage = gates["Long"]
    assert short_leakage > 0.5
    assert long_leakage < 1e-3
    # Without leakage, the error of the long pulse stays inside the qubit states.
    assert 1.0 - long_fidelity > 10 * long_leakage
    assert short_fidelity < long_fidelity
