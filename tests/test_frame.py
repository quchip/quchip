"""Frame resolution and simulation-time frame integration tests.

These tests validate the frame abstraction:
- ``resolve_frame`` converts user-facing frame specs into ``ResolvedFrame``.
- ``simulate`` consumes resolved frames through ``SolveProblem`` state.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.drive import ChargeDrive
from quchip.control.envelopes import Square
from quchip.control.equipment import ControlEquipment
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate
from quchip.engine.ir import DriveOp
from quchip.engine.stage1_frames import resolve_frame


@pytest.fixture
def coupled_chip():
    """Return a coupled transmon-resonator chip plus device handles."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    coupling = Capacitive(q, r, g=0.02, rwa=False)
    chip = Chip([q, r], [coupling])
    return chip, q, r


def test_resolve_frame_lab_returns_zeros(coupled_chip) -> None:
    """The lab frame resolves to zero reference frequency for every device."""
    chip, _, _ = coupled_chip
    resolved = resolve_frame(chip, "lab")
    assert resolved.mode == "lab"
    assert resolved.frequencies == {"q": 0.0, "r": 0.0}


def test_resolve_frame_rotating_uses_dressed_frequencies(coupled_chip) -> None:
    """The rotating frame resolves each device's reference to its own drive frequency."""
    chip, q, r = coupled_chip
    resolved = resolve_frame(chip, "rotating")
    assert resolved.mode == "rotating"
    assert resolved.frequencies["q"] == pytest.approx(q.drive_freq)
    assert resolved.frequencies["r"] == pytest.approx(r.drive_freq)


def test_resolve_frame_float_applies_shared_reference(coupled_chip) -> None:
    """A float frame spec applies the same reference frequency to every device."""
    chip, _, _ = coupled_chip
    resolved = resolve_frame(chip, 5.2)
    assert resolved.mode == "float"
    assert resolved.frequencies == {"q": 5.2, "r": 5.2}


def test_resolve_frame_dict_supports_device_keys_and_missing_defaults(coupled_chip) -> None:
    """A dict frame spec keys by device object and defaults omitted devices to zero."""
    chip, q, _ = coupled_chip
    resolved = resolve_frame(chip, {q: 5.1})
    assert resolved.mode == "dict"
    assert resolved.frequencies["q"] == pytest.approx(5.1)
    assert resolved.frequencies["r"] == pytest.approx(0.0)


def test_resolve_frame_rejects_unknown_strings(coupled_chip) -> None:
    """An unrecognized frame string raises ValueError."""
    chip, _, _ = coupled_chip
    with pytest.raises(ValueError, match="Unknown frame string"):
        resolve_frame(chip, "foo")


def test_chip_no_longer_exposes_resolved_frame_state(coupled_chip) -> None:
    """A Chip carries no persistent resolved-frame attribute."""
    chip, _, _ = coupled_chip
    assert not hasattr(chip, "resolved_frame")


def test_simulation_with_rotating_frame_matches_rabi_analytic() -> None:
    """A resonant drive in the rotating frame reproduces the analytic Rabi oscillation."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    drive = ChargeDrive(target=q)
    chip = Chip([q], frame="rotating")
    chip.connect(ControlEquipment(lines=[drive]))
    tlist = np.linspace(0.0, 50.0, 201)
    omega = 0.02
    op = DriveOp(
        target_label="q",
        envelope=Square(duration=50.0, amplitude=omega),
        freq=5.0,
        start_time=0.0,
        drive_label=drive.label,
    )

    result = simulate(chip, [op], tlist)
    p1 = result.population("q", 1)
    p1_expected = np.sin(np.pi * omega * tlist) ** 2
    npt.assert_allclose(p1, p1_expected, atol=0.01)


@pytest.mark.parametrize(
    ("frame_spec", "expected_mode"),
    [(5.0, "float"), ("rotating", "rotating")],
)
def test_build_problem_stores_resolved_frame(frame_spec: str | float, expected_mode: str) -> None:
    """The built SolveProblem carries the resolved frame with the expected mode."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    drive = ChargeDrive(target=q)
    chip = Chip([q], frame=frame_spec)
    chip.connect(ControlEquipment(lines=[drive]))

    tlist = np.linspace(0.0, 20.0, 101)
    op = DriveOp(
        target_label="q",
        envelope=Square(duration=20.0, amplitude=0.01),
        freq=5.0,
        start_time=0.0,
        drive_label=drive.label,
    )

    from quchip.engine.stage4_problem import build_solve_problem

    problem = build_solve_problem(chip, [op], tlist)
    result = simulate(chip, [op], tlist)
    assert result is not None
    assert problem.resolved_frame is not None
    assert problem.resolved_frame.mode == expected_mode
