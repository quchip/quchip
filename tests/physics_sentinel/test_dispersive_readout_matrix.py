"""Compact dispersive-readout regression matrix across frame and RWA modes."""

from __future__ import annotations

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


def _run_readout(*, frame: str, rwa: bool, qubit_level: int) -> tuple[np.ndarray, np.ndarray]:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    r = Resonator(freq=6.8, levels=6, label="r", quality_factor=1e6)
    readout = ChargeDrive(target=r, label="readout")
    chip = Chip(
        devices=[q, r],
        couplings=[Capacitive(q, r, g=0.04, rwa=rwa)],
        control_equipment=ControlEquipment(lines=[readout]),
        frame=frame,
        rwa=rwa,
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


@pytest.mark.parametrize("rwa", [True, False], ids=["rwa", "nonrwa"])
@pytest.mark.parametrize("frame", ["lab", "rotating"], ids=["lab", "rotating"])
def test_dispersive_readout_matrix_has_state_dependent_response(frame: str, rwa: bool) -> None:
    """Dispersive readout response is strong and qubit-state dependent across all frame/RWA configs."""
    _, response_g = _run_readout(frame=frame, rwa=rwa, qubit_level=0)
    _, response_e = _run_readout(frame=frame, rwa=rwa, qubit_level=1)

    final_separation = abs(response_g[-1] - response_e[-1])
    # Measured across all four frame/RWA configs: max|response| ~= 0.785-0.787 (>0.2 keeps a
    # ~4x margin), final_separation ~= 0.048-0.050 (>0.02 keeps a ~2.4x margin).
    assert np.max(np.abs(response_g)) > 0.2
    assert np.max(np.abs(response_e)) > 0.2
    assert final_separation > 0.02


def test_dispersive_readout_matrix_configs_agree() -> None:
    """Readout state separation agrees across all frame/RWA configs against the rotating+RWA reference."""
    _, reference_g = _run_readout(frame="rotating", rwa=True, qubit_level=0)
    _, reference_e = _run_readout(frame="rotating", rwa=True, qubit_level=1)
    reference_sep = abs(reference_g[-1] - reference_e[-1])

    for rwa in (True, False):
        for frame in ("lab", "rotating"):
            _, response_g = _run_readout(frame=frame, rwa=rwa, qubit_level=0)
            _, response_e = _run_readout(frame=frame, rwa=rwa, qubit_level=1)
            separation = abs(response_g[-1] - response_e[-1])
            # Measured: the largest cross-config drift is rwa=False vs the rotating/rwa=True
            # reference (~3.0% relative, ~1.45e-3 absolute, dominated by the counter-rotating
            # coupling terms rwa=False retains); rtol=0.15/atol=5e-3 keep >=5x margin over that.
            npt.assert_allclose(
                separation,
                reference_sep,
                rtol=0.15,
                atol=5e-3,
                err_msg=f"Dispersive readout separation drifted for frame={frame}, rwa={rwa}",
            )
