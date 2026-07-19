"""Dense crosstalk matrix view on ``ControlEquipment``: extract/perturb/rehydrate/simulate."""

from __future__ import annotations

import numpy as np
import pytest

from quchip.chip.chip import Chip
from quchip.control import ChargeDrive, ControlEquipment, Crosstalk, CrosstalkMatrix
from quchip.control.envelopes import Square
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate
from quchip.engine.ir import Constant, DriveOp


def _build_two_drive_chip(beta_xt: float = 0.1):
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.5, anharmonicity=-0.2, levels=3, label="q2")
    d1 = ChargeDrive(target=q1, label="d1")
    d2 = ChargeDrive(target=q2, label="d2")
    equip = ControlEquipment(
        lines=[d1, d2],
        signal_chain=[Crosstalk(source=d1.label, victim=d2.label, beta=beta_xt)],
    )
    chip = Chip([q1, q2])
    chip.set_frame("lab")
    chip.connect(equip)
    return chip, equip, d1, d2


def test_crosstalk_matrix_extract_shape_and_labels() -> None:
    """Matrix view reports wiring-order labels and populates the right cell."""
    _, equip, d1, d2 = _build_two_drive_chip(beta_xt=0.1)

    m = equip.crosstalk_matrix()

    assert m.labels == (d1.label, d2.label)
    assert m.beta.shape == (2, 2)
    assert m.theta.shape == (2, 2)
    assert m.delay.shape == (2, 2)
    # Diagonal convention: beta=1, theta=0, delay=0.
    assert m.beta[0, 0] == pytest.approx(1.0)
    assert m.beta[1, 1] == pytest.approx(1.0)
    # Off-diagonal: Crosstalk(source=d1, victim=d2) -> beta[victim=1, source=0]
    assert m.beta[1, 0] == pytest.approx(0.1)
    assert m.beta[0, 1] == pytest.approx(0.0)


def test_reciprocal_matrix_edges_do_not_recursively_leak() -> None:
    """Reciprocal edges read the same input map rather than one another's output."""
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q1")
    q2 = DuffingTransmon(freq=5.2, anharmonicity=-0.2, levels=2, label="q2")
    d1 = ChargeDrive(target=q1, label="d1")
    d2 = ChargeDrive(target=q2, label="d2")
    equipment = ControlEquipment([d1, d2])
    equipment.set_crosstalk_matrix(np.array([[1.0, 0.16], [0.14, 1.0]]))

    transformed = equipment.apply_signal_chain({
        ("d1", 0): Constant(1.0),
        ("d2", 1): Constant(2.0),
    })

    assert transformed[("d1", 0)].evaluate(0.0, xp=np) == pytest.approx(1.0)
    assert transformed[("d2", 1)].evaluate(0.0, xp=np) == pytest.approx(2.0)
    assert transformed[("d2", 0)].evaluate(0.0, xp=np) == pytest.approx(0.14)
    assert transformed[("d1", 1)].evaluate(0.0, xp=np) == pytest.approx(0.32)


def test_set_crosstalk_matrix_installs_one_signal_transform() -> None:
    """The matrix owns parallel mixing instead of special-casing equipment dispatch."""
    _, equipment, *_ = _build_two_drive_chip()

    equipment.set_crosstalk_matrix(np.array([[1.0, 0.16], [0.14, 1.0]]))

    assert len(equipment.signal_chain) == 1
    assert isinstance(equipment.signal_chain[0], CrosstalkMatrix)


def test_set_crosstalk_matrix_roundtrip_changes_simulation() -> None:
    """Perturbing a matrix entry and re-injecting it changes simulate() output."""
    chip, equip, d1, d2 = _build_two_drive_chip(beta_xt=0.1)

    envelope = Square(duration=40.0, amplitude=0.1)
    drive_op = DriveOp(
        target_label="q1",
        envelope=envelope,
        freq=5.0,
        start_time=0.0,
        drive_label=d1.label,
    )
    tlist = np.linspace(0.0, 40.0, 201)

    baseline = simulate(chip, [drive_op], tlist)
    p_q2_baseline = baseline.population("q2", 1)

    matrix = equip.crosstalk_matrix()
    perturbed_beta = np.array(matrix.beta, dtype=float, copy=True)
    perturbed_beta[1, 0] = 0.35  # was 0.1

    equip.set_crosstalk_matrix(perturbed_beta)

    perturbed = simulate(chip, [drive_op], tlist)
    p_q2_perturbed = perturbed.population("q2", 1)

    assert np.max(np.abs(p_q2_perturbed - p_q2_baseline)) > 1e-3, (
        "Perturbing the off-diagonal crosstalk amplitude should change "
        "the victim population trajectory."
    )
    edges = equip.crosstalks
    beta_12 = next(
        e.beta for e in edges if e.source == d1.label and e.victim == d2.label
    )
    assert float(beta_12) == pytest.approx(0.35)


def test_set_crosstalk_matrix_preserves_non_crosstalk_transforms() -> None:
    """Non-:class:`Crosstalk` signal-chain entries survive a matrix rehydrate."""
    from quchip.control import Gain

    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.5, anharmonicity=-0.2, levels=3, label="q2")
    d1 = ChargeDrive(target=q1, label="d1")
    d2 = ChargeDrive(target=q2, label="d2")
    gain = Gain(line=d1.label, factor=0.9)
    equip = ControlEquipment(
        lines=[d1, d2],
        signal_chain=[gain, Crosstalk(source=d1.label, victim=d2.label, beta=0.1)],
    )
    Chip([q1, q2]).connect(equip)

    beta = np.zeros((2, 2), dtype=float)
    beta[1, 0] = 0.2
    equip.set_crosstalk_matrix(beta)

    assert any(t is gain for t in equip.signal_chain), (
        "Gain transform should be preserved when rehydrating the crosstalk matrix"
    )
    matrices = [t for t in equip.signal_chain if isinstance(t, CrosstalkMatrix)]
    assert len(matrices) == 1
    assert len(equip.crosstalks) == 2


def test_matrix_transform_round_trips_through_equipment_serialization() -> None:
    """The matrix remains one transform after equipment serialization."""
    chip, equipment, *_ = _build_two_drive_chip()
    equipment.set_crosstalk_matrix(
        np.array([[1.0, 0.16], [0.14, 1.0]]),
        np.array([[0.0, 0.3], [0.4, 0.0]]),
    )

    restored = ControlEquipment.from_dict(
        equipment.to_dict(),
        chip.device_map,
        chip.coupling_map,
    )

    assert len(restored.signal_chain) == 1
    assert isinstance(restored.signal_chain[0], CrosstalkMatrix)
    np.testing.assert_allclose(restored.crosstalk_matrix().beta, [[1.0, 0.16], [0.14, 1.0]])


def test_unwire_restricts_matrix_to_remaining_lines() -> None:
    """Removing one line preserves the matrix entries among surviving lines."""
    qubits = [
        DuffingTransmon(freq=5.0 + 0.2 * i, anharmonicity=-0.2, levels=2, label=f"q{i}")
        for i in range(3)
    ]
    drives = [ChargeDrive(target=qubit, label=f"d{i}") for i, qubit in enumerate(qubits)]
    equipment = ControlEquipment(drives)
    equipment.set_crosstalk_matrix(np.array([
        [1.0, 0.1, 0.2],
        [0.3, 1.0, 0.4],
        [0.5, 0.6, 1.0],
    ]))
    chip = Chip(qubits, control_equipment=equipment)

    chip.unwire(drives[1])

    reduced = chip.control_equipment.crosstalk_matrix()
    assert reduced.labels == ("d0", "d2")
    np.testing.assert_allclose(reduced.beta, [[1.0, 0.2], [0.5, 1.0]])


def test_crosstalk_matrix_rejects_wrong_shape() -> None:
    """``set_crosstalk_matrix`` guards against shape mismatches."""
    _, equip, *_ = _build_two_drive_chip()
    with pytest.raises(ValueError):
        equip.set_crosstalk_matrix(np.zeros((3, 3)))
