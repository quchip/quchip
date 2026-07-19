"""Cross-mode physics equivalence and optimal-frame objective tests."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.equipment import ControlEquipment
from quchip.control.drive import ChargeDrive
from quchip.control.envelopes import Gaussian, Square
from quchip.utils.labeling import reset_label_counters
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate
from quchip.engine.ir import DriveOp

FRAME_MODES: list[tuple[str, str | float | dict[str, float]]] = [
    ("lab", "lab"),
    ("rotating", "rotating"),
    ("float", 6.0),
    ("dict", {"q": 5.0, "r": 7.0}),
]
SOLVER_OPTS = {"atol": 1e-10, "rtol": 1e-8}


@pytest.fixture(autouse=True)
def _reset_labels():
    reset_label_counters()
    yield
    reset_label_counters()


def _build_dispersive_system(
    frame_mode: str | float | dict[str, float],
    *,
    rwa: bool,
) -> tuple[Chip, DuffingTransmon, Resonator, ChargeDrive, ChargeDrive]:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=0.1, rwa=rwa)
    chip = Chip([q, r], [coupling])

    drive_q = ChargeDrive(target=q)
    drive_r = ChargeDrive(target=r)
    chip.connect(ControlEquipment(lines=[drive_q, drive_r]))
    chip.set_frame(frame_mode)
    return chip, q, r, drive_q, drive_r


def test_cross_mode_populations_match():
    """All frame modes produce identical qubit populations."""
    p0_results: dict[str, np.ndarray] = {}
    p1_results: dict[str, np.ndarray] = {}

    for mode_name, mode_spec in FRAME_MODES:
        reset_label_counters()
        chip, q, _, drive_q, _ = _build_dispersive_system(mode_spec, rwa=True)
        tlist = np.linspace(0.0, 30.0, 301)
        dop = DriveOp(
            target_label="q",
            envelope=Square(duration=30.0, amplitude=0.02),
            freq=q.drive_freq,
            drive_label=drive_q.label,
        )
        result = simulate(chip, [dop], tlist, options=SOLVER_OPTS)
        p0_results[mode_name] = result.population("q", 0)
        p1_results[mode_name] = result.population("q", 1)

    p0_ref = p0_results["rotating"]
    p1_ref = p1_results["rotating"]
    for mode_name, _ in FRAME_MODES:
        npt.assert_allclose(
            p0_results[mode_name],
            p0_ref,
            rtol=1e-5,
            atol=2e-5,
            err_msg=f"Frame mode {mode_name} p0 differs from rotating",
        )
        npt.assert_allclose(
            p1_results[mode_name],
            p1_ref,
            rtol=1e-5,
            atol=2e-5,
            err_msg=f"Frame mode {mode_name} p1 differs from rotating",
        )


def test_cross_mode_demodulated_amplitude_match():
    """Demodulated <a> envelope matches across all frame modes."""
    expect_results: dict[str, np.ndarray] = {}

    for mode_name, mode_spec in FRAME_MODES:
        reset_label_counters()
        chip, q, r, _, drive_r = _build_dispersive_system(mode_spec, rwa=True)

        readout_freq = 0.5 * (chip.freq(r, {q: 0}) + chip.freq(r, {q: 1}))
        tlist = np.linspace(0.0, 50.0, 401)
        dop = DriveOp(
            target_label="r",
            envelope=Gaussian(duration=50.0, amplitude=0.005, sigmas=4),
            freq=readout_freq,
            drive_label=drive_r.label,
        )
        e_ops = {"r": r.lowering_operator()}
        result = simulate(chip, [dop], tlist, e_ops=e_ops, options=SOLVER_OPTS)
        expect_results[mode_name] = np.asarray(result.expect_values("r"))

    ref = expect_results["rotating"]
    for mode_name, _ in FRAME_MODES:
        npt.assert_allclose(
            np.abs(expect_results[mode_name]),
            np.abs(ref),
            rtol=1e-5,
            atol=1e-3,
            err_msg=f"Frame mode {mode_name} |<a>| envelope differs from rotating",
        )
