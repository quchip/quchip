"""Compact dispersive-readout regression matrix across frame and RWA modes."""

from __future__ import annotations

from functools import cache

from quchip.approximations import RWA, Exact

import numpy as np
import numpy.testing as npt
import pytest

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    Gaussian,
    QuantumSequence,
    Resonator,
)


def _run_readout(*, frame: str, approximation, qubit_level: int) -> tuple[np.ndarray, np.ndarray]:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    r = Resonator(freq=6.8, levels=6, label="r", internal_quality_factor=1e6)
    readout = ChargeDrive(target=r, label="readout")
    chip = Chip(
        devices=[q, r],
        couplings=[Capacitive(q, r, g=0.04)],
        control_equipment=ControlEquipment(lines=[readout]),
        frame=frame,
        approximation=approximation,
    )
    chip.dress()

    readout_freq = 0.5 * (chip.freq(r, {q: 0}) + chip.freq(r, {q: 1}))
    tlist = np.linspace(0.0, 80.0, 81)

    seq = QuantumSequence(chip)
    seq.schedule(
        readout,
        envelope=Gaussian(duration=80.0, amplitude=0.01, sigmas=4),
        freq=readout_freq,
    )
    result = seq.simulate(
        tlist=tlist,
        initial_state=chip.state(q=qubit_level, r=0),
        e_ops=chip.e_ops(r="a"),
    )
    expect = np.asarray(result.expect_values("r"), dtype=complex)
    if frame == "lab":
        expect = expect * np.exp(+1j * 2 * np.pi * readout_freq * tlist)
    return tlist, expect


@pytest.fixture(scope="module")
def readout_response():
    """Reuse each fixed frame/approximation/state solve across numerical assertions."""
    @cache
    def response(frame, approximation_type, qubit_level):
        arrays = _run_readout(frame=frame, approximation=approximation_type(), qubit_level=qubit_level)
        for array in arrays:
            array.setflags(write=False)
        return arrays

    return response


@pytest.mark.parametrize("frame, approximation", [("rotating", RWA()), ("lab", Exact())],
                         ids=["rotating-rwa", "lab-exact"])
def test_dispersive_readout_matrix_has_state_dependent_response(frame: str, approximation, readout_response) -> None:
    """Dispersive readout response is strong and qubit-state dependent in the rotating RWA and lab Exact configs."""
    _, response_g = readout_response(frame, type(approximation), 0)
    _, response_e = readout_response(frame, type(approximation), 1)

    final_separation = abs(response_g[-1] - response_e[-1])
    # Measured across all four frame/RWA configs: max|response| ~= 0.785-0.787 (>0.2 keeps a
    # ~4x margin), final_separation ~= 0.048-0.050 (>0.02 keeps a ~2.4x margin).
    assert np.max(np.abs(response_g)) > 0.2
    assert np.max(np.abs(response_e)) > 0.2
    assert final_separation > 0.02


def test_dispersive_readout_matrix_configs_agree(readout_response) -> None:
    """Readout state separation in the lab frame with Exact() agrees with the rotating+RWA reference."""
    _, reference_g = readout_response("rotating", RWA, 0)
    _, reference_e = readout_response("rotating", RWA, 1)
    reference_sep = abs(reference_g[-1] - reference_e[-1])

    _, response_g = readout_response("lab", Exact, 0)
    _, response_e = readout_response("lab", Exact, 1)
    separation = abs(response_g[-1] - response_e[-1])
    # Measured: the largest cross-config drift is approximation=Exact() vs the rotating/approximation=RWA()
    # reference (~3.0% relative, ~1.45e-3 absolute, dominated by the counter-rotating
    # coupling terms approximation=Exact() retains); rtol=0.15/atol=5e-3 keep >=5x margin over that.
    npt.assert_allclose(separation, reference_sep, rtol=0.15, atol=5e-3)
