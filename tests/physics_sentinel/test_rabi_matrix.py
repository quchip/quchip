"""Compact Rabi regression matrix across frame and RWA modes."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip import ChargeDrive, Chip, ControlEquipment, DuffingTransmon, Square
from quchip.engine import simulate
from quchip.engine.ir import DriveOp


def _run_rabi(*, frame: str, rwa: bool) -> tuple[np.ndarray, np.ndarray]:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    drive = ChargeDrive(target=q)
    chip = Chip([q], frame=frame, rwa=rwa)
    chip.connect(ControlEquipment(lines=[drive]))
    tlist = np.linspace(0.0, 50.0, 201)
    drive_op = DriveOp(
        target_label="q",
        envelope=Square(duration=50.0, amplitude=0.02),
        freq=q.freq,
        start_time=0.0,
        drive_label=drive.label,
    )
    result = simulate(chip, [drive_op], tlist)
    return tlist, np.asarray(result.population("q", 1))


@pytest.mark.parametrize("rwa", [True, False], ids=["rwa", "nonrwa"])
@pytest.mark.parametrize("frame", ["lab", "rotating"], ids=["lab", "rotating"])
def test_rabi_matrix_hits_pi_and_two_pi_points(frame: str, rwa: bool) -> None:
    """Rabi drive reaches near-full inversion at the pi time and returns near zero at the 2pi time."""
    tlist, p1 = _run_rabi(frame=frame, rwa=rwa)
    idx_25 = int(np.argmin(np.abs(tlist - 25.0)))
    idx_50 = int(np.argmin(np.abs(tlist - 50.0)))

    # Measured: max P(1) = P(1)[t=25] ~= 0.9955-0.9958 across all four configs, converged
    # w.r.t. Hilbert truncation (identical to 4 decimals at levels=5,8). The ~0.4% shortfall
    # from unity is genuine leakage to |2> from the non-selective resonant square pulse
    # (measured max P(2) ~= 0.34%), not a numerical artifact; 0.99 keeps a margin below that
    # floor. P(1)[t=50] (one full Rabi period) measures ~6e-6, far under the 1e-3 bound.
    assert np.max(p1) > 0.99
    assert p1[idx_25] > 0.99
    assert p1[idx_50] < 1e-3


def test_rabi_matrix_configs_agree() -> None:
    """Rabi population trajectories agree across all frame/RWA configs against the rotating+RWA reference."""
    reference_time, reference = _run_rabi(frame="rotating", rwa=True)

    for rwa in (True, False):
        for frame in ("lab", "rotating"):
            tlist, p1 = _run_rabi(frame=frame, rwa=rwa)
            npt.assert_allclose(tlist, reference_time)
            # Measured: max deviation from the rotating/rwa=True reference is ~3.4e-4 (lab
            # frame, rwa=False), dominated by the Bloch-Siegert-scale counter-rotating shift;
            # atol=5e-3 keeps a ~15x margin over that.
            npt.assert_allclose(
                p1,
                reference,
                atol=5e-3,
                err_msg=f"Rabi population drifted for frame={frame}, rwa={rwa}",
            )
