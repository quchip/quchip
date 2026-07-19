"""End-to-end pipeline tests for the single-qubit Rabi path.

Verifies the simulation pipeline from `Chip` through `simulate` for a
resonant charge drive on a single transmon.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quchip.backend.qutip import QuTiPBackend
from quchip.chip.chip import Chip
from quchip.control.drive import ChargeDrive
from quchip.control.equipment import ControlEquipment
from quchip.control.envelopes import Square
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate
from quchip.engine.ir import DriveOp


# ── Rabi oscillation ────────────────────────────────────────────────


class TestRabiOscillation:
    """Verify Rabi oscillation P(|1⟩) = sin²(π·Ω·t) end-to-end."""

    FREQ = 5.0  # transmon frequency (GHz)
    ANHARMONICITY = -0.25  # transmon anharmonicity (GHz)
    LEVELS = 3  # Fock space truncation
    OMEGA = 0.02  # drive amplitude (GHz) → Rabi frequency
    #   Ω/|α| = 0.08, so leakage to |2⟩ is about 0.6%, within tolerance.
    DURATION = 50.0  # pulse duration (ns)
    N_POINTS = 501  # time grid points

    def _run_rabi(self) -> tuple[np.ndarray, np.ndarray]:
        """Set up and run a single-transmon Rabi oscillation.

        Returns (tlist, p1) where p1 is the |1⟩ population at each time.
        """
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ANHARMONICITY,
            levels=self.LEVELS,
            label="q",
        )

        drive = ChargeDrive(target=q)

        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[drive]))
        chip.set_frame("rotating")

        envelope = Square(duration=self.DURATION, amplitude=self.OMEGA)
        drive_op = DriveOp(
            target_label="q",
            envelope=envelope,
            freq=self.FREQ,
            start_time=0.0,
            drive_label=drive.label,
        )

        tlist = np.linspace(0, self.DURATION, self.N_POINTS)

        result = simulate(
            chip,
            [drive_op],
            tlist,
        )

        p1 = result.population("q", 1)
        return tlist, p1

    def test_rabi_matches_analytic(self, backend: QuTiPBackend) -> None:
        """P(|1⟩, t) matches sin²(π·Ω·t) within 1% at all time points."""
        tlist, p1 = self._run_rabi()

        p1_analytic = np.sin(np.pi * self.OMEGA * tlist) ** 2

        npt.assert_allclose(
            p1,
            p1_analytic,
            atol=0.01,
            err_msg=(
                "Rabi oscillation P(|1⟩) deviates from sin²(π·Ω·t) by >1%. "
                f"Max deviation: {np.max(np.abs(p1 - p1_analytic)):.4f}"
            ),
        )

    def test_pi_pulse_excitation(self, backend: QuTiPBackend) -> None:
        """At t=25 ns (π-pulse), population should be ≈1.0."""
        tlist, p1 = self._run_rabi()

        # t=25 ns → π·Ω·t = π·0.02·25 = π/2 → sin²(π/2) = 1.0
        idx_25 = np.argmin(np.abs(tlist - 25.0))
        assert abs(p1[idx_25] - 1.0) < 0.01, f"π-pulse at t=25 ns: P(|1⟩) = {p1[idx_25]:.4f}, expected ≈1.0"

    def test_two_pi_pulse_return(self, backend: QuTiPBackend) -> None:
        """At t=50 ns (2π-pulse), population should return to ≈0.0."""
        tlist, p1 = self._run_rabi()

        # t=50 ns → π·Ω·t = π → sin²(π) = 0.0
        idx_50 = np.argmin(np.abs(tlist - 50.0))
        assert abs(p1[idx_50]) < 0.01, f"2π-pulse at t=50 ns: P(|1⟩) = {p1[idx_50]:.4f}, expected ≈0.0"

    def test_populations_sum_to_one(self) -> None:
        """Total population across all levels sums to 1 at every time step."""
        q = DuffingTransmon(
            freq=self.FREQ,
            anharmonicity=self.ANHARMONICITY,
            levels=self.LEVELS,
            label="q",
        )
        drive = ChargeDrive(target=q)
        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[drive]))
        chip.set_frame("rotating")

        envelope = Square(duration=self.DURATION, amplitude=self.OMEGA)
        drive_op = DriveOp(
            target_label="q",
            envelope=envelope,
            freq=self.FREQ,
            start_time=0.0,
            drive_label=drive.label,
        )
        tlist = np.linspace(0, self.DURATION, self.N_POINTS)

        result = simulate(chip, [drive_op], tlist)

        pops = result.populations
        total = sum(pops.values())
        npt.assert_allclose(
            total,
            np.ones(len(tlist)),
            atol=1e-6,
            err_msg="Total population across all levels deviates from 1.0",
        )
