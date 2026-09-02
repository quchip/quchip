"""Dressing the resolved engine snapshot without changing ``Chip.dress()``."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import ChargeDrive, Chip, ControlEquipment, DuffingTransmon, QuantumSequence, Square


def _driven_engine():
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    drive = ChargeDrive(qubit, label="charge")
    chip = Chip([qubit], control_equipment=ControlEquipment(lines=[drive]), frame="rotating")
    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Square(duration=4.0, amplitude=0.04),
        freq=5.0,
        phase=0.0,
    )
    return chip, sequence.build_problem(tlist=np.linspace(0.0, 4.0, 9)).engine_result


def test_resolved_static_snapshot_dresses_its_selected_frame() -> None:
    """Resolved dressing analyzes its selected frame rather than intrinsic lab statics."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([qubit], frame="rotating")

    intrinsic = chip.dress()
    resolved = chip.resolve()
    dressed = resolved.dress()

    np.testing.assert_allclose(intrinsic.eigenvalues, [0.0, 5.0, 9.75])
    np.testing.assert_allclose(dressed.eigenvalues, [-0.25, 0.0, 0.0], atol=1e-12)


def test_dynamic_resolved_snapshot_requires_at_time() -> None:
    """A dynamic snapshot cannot imply an arbitrary dressing instant."""
    _, resolved = _driven_engine()

    with pytest.raises(ValueError, match="at_time"):
        resolved.dress()


def test_dynamic_resolved_snapshot_dresses_the_instantaneous_hamiltonian() -> None:
    """Dynamic dressing diagonalizes the resolved Hamiltonian at the requested instant."""
    _, resolved = _driven_engine()

    at_start = resolved.dress(at_time=0.0)
    after_pulse = resolved.dress(at_time=5.0)

    assert not np.allclose(at_start.eigenvalues, after_pulse.eigenvalues)
    np.testing.assert_allclose(after_pulse.eigenvalues, [-0.25, 0.0, 0.0], atol=1e-12)


def test_resolved_dressing_is_a_snapshot_after_chip_mutation() -> None:
    """Resolved dressing retains frozen basis data after source-chip mutation."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([qubit])
    resolved = chip.resolve()

    qubit.freq = 6.0

    np.testing.assert_allclose(resolved.dress().eigenvalues, [0.0, 5.0, 9.75])
    np.testing.assert_allclose(chip.dress().eigenvalues, [0.0, 6.0, 11.75])
