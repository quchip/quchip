"""Tests for per-device ``reference_freq`` (the readout / rotating-frame reference, LO).

Covers the co-rotating readout contract: reference vs. drive frequency, idle Ramsey
precession under detuning, expect/states frame agreement, and frame-invariant observables.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.chip.chip import Chip
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate


def _plus(chip: Chip, q: DuffingTransmon) -> object:
    """|+> = (|0> + |1>)/sqrt(2) on the chip Hilbert space."""
    return (chip.bare_state({q: 0}) + chip.bare_state({q: 1})) / np.sqrt(2)


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


class TestReferenceFreqAttribute:
    def test_defaults_to_drive_freq(self) -> None:
        """An unset reference_freq inherits the dressed drive frequency."""
        q = DuffingTransmon(freq=5.01, anharmonicity=-0.30, levels=3, label="q")
        Chip([q], couplings=[])
        assert float(q.reference_freq) == pytest.approx(float(q.drive_freq))

    def test_setter_bumps_state_version_and_none_restores(self) -> None:
        """Setting reference_freq invalidates caches; None restores the default."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=3, label="q")
        Chip([q], couplings=[])
        v0 = q.state_version
        q.reference_freq = 4.99
        assert q.state_version > v0, "setting reference_freq must bump state_version"
        assert float(q.reference_freq) == pytest.approx(4.99)
        q.reference_freq = None
        assert float(q.reference_freq) == pytest.approx(float(q.drive_freq))

    def test_override_survives_serialization_round_trip(self) -> None:
        """A set reference_freq round-trips through to_dict/from_dict; unset stays default."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=3, label="q")
        q.reference_freq = 4.97
        restored = DuffingTransmon.from_dict(q.to_dict())
        assert float(restored.reference_freq) == pytest.approx(4.97), "override must persist"

        plain = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=3, label="q")
        d = plain.to_dict()
        assert "reference_freq" not in d, "the drive_freq default must not be serialized as an override"
        restored_plain = DuffingTransmon.from_dict(d)
        assert restored_plain._reference_freq_override is None


# ---------------------------------------------------------------------------
# Physics: co-rotating readout
# ---------------------------------------------------------------------------


class TestCoRotatingReadout:
    LEVELS = 3
    ALPHA = -0.30

    def test_detuned_reference_shows_idle_ramsey(self) -> None:
        """Qubit at ω, LO at ω−Δ → <σ_x>(t) = cos(2π·Δ·t) (idle Ramsey)."""
        omega, delta = 5.01, 0.01  # GHz  → 10 MHz detuning, 100 ns period
        q = DuffingTransmon(freq=omega, anharmonicity=self.ALPHA, levels=self.LEVELS, label="q")
        chip = Chip([q], couplings=[])
        q.reference_freq = omega - delta
        chip.set_frame("rotating")

        t = np.linspace(0.0, 100.0, 201)
        res = simulate(chip, [], t, initial_state=_plus(chip, q), e_ops={q: [q.sigma_x]})
        sx = np.real(np.asarray(res.expect(q, index=0)))

        npt.assert_allclose(
            sx, np.cos(2 * np.pi * delta * t), atol=0.02,
            err_msg="detuned reference_freq should make <σ_x> precess at Δ = ω − reference_freq",
        )

    def test_states_equal_expect_in_rotating_frame(self) -> None:
        """result.expect == Tr(O·ρ) over the same result.states (issue #101)."""
        q = DuffingTransmon(freq=5.01, anharmonicity=self.ALPHA, levels=self.LEVELS, label="q")
        chip = Chip([q], couplings=[])
        q.reference_freq = 5.00
        chip.set_frame("rotating")

        t = np.linspace(0.0, 100.0, 101)
        res = simulate(chip, [], t, initial_state=_plus(chip, q),
                       e_ops={q: [q.sigma_x]}, options={"store_states": True})
        sx_eop = np.real(np.asarray(res.expect(q, index=0)))
        sx_op = np.asarray(q.sigma_x.full())
        sx_manual = np.array([np.real(np.trace(sx_op @ np.asarray(res.dm_at(tt).full()))) for tt in t])

        npt.assert_allclose(sx_eop, sx_manual, atol=1e-9,
                            err_msg="e_ops and states must be reported in the same (reference) frame")

    def test_calibrated_lowering_operator_is_non_oscillatory(self) -> None:
        """<a> is stationary when the device sits at its reference (calibrated)."""
        q = DuffingTransmon(freq=5.0, anharmonicity=self.ALPHA, levels=self.LEVELS, label="q")
        chip = Chip([q], couplings=[])  # reference_freq defaults to drive_freq = 5.0
        chip.set_frame("rotating")

        t = np.linspace(0.0, 200.0, 201)
        res = simulate(chip, [], t, initial_state=_plus(chip, q), e_ops={q: [q.lowering_operator()]})
        a = np.abs(np.asarray(res.expect(q, index=0)))
        assert a.max() - a.min() < 1e-6, "calibrated <a> must be non-oscillatory in the co-rotating readout"
        assert a.mean() == pytest.approx(0.5, abs=1e-3)

    def test_populations_are_frame_invariant(self) -> None:
        """Diagonal observables (populations) are unaffected by the reference detuning."""
        q = DuffingTransmon(freq=5.0, anharmonicity=self.ALPHA, levels=self.LEVELS, label="q")
        chip = Chip([q], couplings=[])
        q.reference_freq = 4.97  # 30 MHz detuning
        chip.set_frame("rotating")

        t = np.linspace(0.0, 100.0, 101)
        res = simulate(chip, [], t, initial_state=_plus(chip, q))
        p1 = res.population("q", 1)
        # A diagonal rotating-frame H conserves Fock populations exactly; the
        # residual is solver discretization, not a reference-frame effect.
        assert p1.max() - p1.min() < 1e-4, "populations must not depend on the readout reference"
        npt.assert_allclose(p1, 0.5, atol=1e-3)

    def test_expect_independent_of_integration_frame(self) -> None:
        """<σ_x> is the same reported in a 'lab' vs 'rotating' integration frame."""
        q = DuffingTransmon(freq=5.0, anharmonicity=self.ALPHA, levels=self.LEVELS, label="q")
        chip = Chip([q], couplings=[])
        q.reference_freq = 4.995  # 5 MHz detuning; readout reference fixed

        t = np.linspace(0.0, 200.0, 201)
        tol = {"atol": 1e-10, "rtol": 1e-8}  # tighten so lab-frame carrier is resolved
        chip.set_frame("rotating")
        sx_rot = np.real(np.asarray(simulate(chip, [], t, initial_state=_plus(chip, q),
                                             e_ops={q: [q.sigma_x]}, options=tol).expect(q, index=0)))
        chip.set_frame("lab")
        sx_lab = np.real(np.asarray(simulate(chip, [], t, initial_state=_plus(chip, q),
                                             e_ops={q: [q.sigma_x]}, options=tol).expect(q, index=0)))

        npt.assert_allclose(sx_rot, sx_lab, atol=1e-3,
                            err_msg="reported expect must be frame-invariant (co-rotating at reference_freq)")

    def test_per_device_references_demodulate_independently(self) -> None:
        """Each device demodulates at its own reference_freq (the changed line iterates all devices)."""
        qa = DuffingTransmon(freq=5.0, anharmonicity=self.ALPHA, levels=self.LEVELS, label="qa")
        qb = DuffingTransmon(freq=6.0, anharmonicity=self.ALPHA, levels=self.LEVELS, label="qb")
        chip = Chip([qa, qb], couplings=[])
        qa.reference_freq = 4.99  # Δ_a = 10 MHz
        qb.reference_freq = 5.98  # Δ_b = 20 MHz
        chip.set_frame("rotating")

        plus_ab = (chip.bare_state({qa: 0, qb: 0}) + chip.bare_state({qa: 1, qb: 0})
                   + chip.bare_state({qa: 0, qb: 1}) + chip.bare_state({qa: 1, qb: 1})) / 2.0
        t = np.linspace(0.0, 100.0, 201)
        res = simulate(chip, [], t, initial_state=plus_ab, e_ops={qa: [qa.sigma_x], qb: [qb.sigma_x]})
        sx_a = np.real(np.asarray(res.expect(qa, index=0)))
        sx_b = np.real(np.asarray(res.expect(qb, index=0)))

        npt.assert_allclose(sx_a, np.cos(2 * np.pi * 0.01 * t), atol=0.03,
                            err_msg="device qa must demodulate at its own reference (Δ=10 MHz)")
        npt.assert_allclose(sx_b, np.cos(2 * np.pi * 0.02 * t), atol=0.03,
                            err_msg="device qb must demodulate at its own reference (Δ=20 MHz)")
