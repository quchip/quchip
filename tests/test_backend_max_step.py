"""Regression coverage for QuTiPBackend.resolve_solver_options' max_step consumption."""

from __future__ import annotations

import numpy as np
import pytest

from quchip.chip.chip import Chip
from quchip.control import ChargeDrive
from quchip.control.envelopes import Gaussian
from quchip.control.equipment import ControlEquipment
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate
from quchip.engine.ir import DriveOp


def _resonant_gaussian_pulse_result(tlist: np.ndarray):
    """Simulate a resonant, finite-support Gaussian pulse inside a long idle span."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    drive = ChargeDrive(target=q)
    chip = Chip(devices=[q], control_equipment=ControlEquipment(lines=[drive]), backend="qutip")

    drive_op = DriveOp(
        target_label="q",
        envelope=Gaussian(duration=8.0, amplitude=1.0, sigmas=3),
        freq=5.0,
        start_time=150.0,
        drive_label=drive.label,
    )

    return simulate(chip, [drive_op], tlist)


class TestMaxStepConsumption:
    """A short pulse inside a long idle span must not be stepped over by the adaptive integrator."""

    def test_pulse_resolved_with_two_point_tlist(self) -> None:
        """A 2-point [t0, t_end] tlist still resolves the 8 ns pulse with no manual options."""
        tlist = np.array([0.0, 308.0])
        result = _resonant_gaussian_pulse_result(tlist)
        final_p0 = result.population("q", level=0)[-1]
        assert final_p0 < 0.99, (
            f"Ground population {final_p0:.4f} at t=308 indicates the pulse was stepped over."
        )

    def test_pulse_resolved_with_dense_tlist(self) -> None:
        """A dense 10-points-per-ns tlist resolves the same pulse (consistency reference)."""
        tlist = np.linspace(0.0, 308.0, 308 * 10 + 1)
        result = _resonant_gaussian_pulse_result(tlist)
        final_p0 = result.population("q", level=0)[-1]
        assert final_p0 < 0.99, (
            f"Ground population {final_p0:.4f} at t=308 indicates the pulse was stepped over."
        )


class TestMaxStepGuardPolicy:
    """resolve_solver_options' max_step insertion/authority policy, tested directly on the dict contract."""

    def _backend(self):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        return Chip(devices=[q], backend="qutip").backend

    def _resolve(self, options: dict, max_step_ns) -> dict:
        return self._backend().resolve_solver_options(
            options, metadata={"max_step_ns": max_step_ns}, tlist=np.array([0.0, 308.0])
        )

    @pytest.mark.parametrize("max_step_ns", [4.0, 0.5, 100.0], ids=["typical", "sub-ns", "large"])
    def test_concrete_positive_finite_metadata_is_inserted(self, max_step_ns: float) -> None:
        """A concrete, positive, finite max_step_ns hint is inserted as max_step when unset."""
        resolved = self._resolve({}, max_step_ns)
        assert resolved["max_step"] == max_step_ns

    @pytest.mark.parametrize(
        "max_step_ns", [0.0, -1.0, float("nan"), float("inf")], ids=["zero", "negative", "nan", "inf"]
    )
    def test_non_positive_or_non_finite_metadata_is_ignored(self, max_step_ns: float) -> None:
        """Zero, negative, NaN, and infinite max_step_ns hints never insert a max_step."""
        resolved = self._resolve({}, max_step_ns)
        assert "max_step" not in resolved

    @pytest.mark.parametrize("user_max_step", [1.0, 2.5, 0], ids=["typical", "fractional", "zero-unbounded"])
    def test_explicit_user_max_step_is_always_authoritative(self, user_max_step) -> None:
        """An explicit user max_step -- including QuTiP's own unbounded 0 -- is never overridden by the hint."""
        resolved = self._backend().resolve_solver_options(
            {"max_step": user_max_step}, metadata={"max_step_ns": 4.0}, tlist=np.array([0.0, 308.0])
        )
        assert resolved["max_step"] == user_max_step
