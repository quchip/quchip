"""FluxDrive channel generation, signal-chain composition, and population preservation."""

from __future__ import annotations


import numpy as np
import pytest

from quchip import Chip, ControlEquipment, DuffingTransmon, Square
from quchip.control.drive import FluxDrive
from quchip.control.signal import AnalyticSignal, Crosstalk, Delay, Gain
from quchip.engine import simulate
from quchip.control.drive import ChargeDrive
from quchip.engine.ir import Constant, DriveOp, Shift


def test_control_equipment_applies_signal_chain_in_order() -> None:
    """Signal chain transforms are applied in order and can create new signals."""
    equipment = ControlEquipment(
        lines=[],
        signal_chain=[
            Delay(line="charge_0", delta_t=1.5),
            Gain(line="charge_0", factor=0.5),
            Crosstalk(source="charge_0", victim="charge_1", beta=0.2, delay=0.25),
        ],
    )

    built = equipment.apply_signal_chain({("charge_0", 0): AnalyticSignal(Constant(1.0 + 0.0j))})
    assert any(k[0] == "charge_1" for k in built)


def test_flux_drive_authors_the_delivered_signal_hamiltonian():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    d = FluxDrive(target=q)
    authored = d.hamiltonian(q, AnalyticSignal(Constant(1.0 + 0.0j)))
    assert authored.labels == ("q",)


def test_equipment_signal_chain_composes_transforms():
    """Signal-chain transforms should compose on the signal map."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    drive = FluxDrive(target=q)
    drive_label = drive.label
    equipment = ControlEquipment(
        lines=[drive],
        signal_chain=[
            Delay(line=drive_label, delta_t=3.0),
            Gain(line=drive_label, factor=0.5 + 0.25j),
        ],
    )
    drive_op = DriveOp(
        target_label="q",
        envelope=Square(amplitude=1.0, duration=10.0),
        freq=5.0,
        start_time=1.0,
        drive_label=drive.label,
    )

    raw_signal = drive.signal(drive_op, q)
    transformed = equipment.apply_signal_chain({(drive_label, 0): raw_signal})
    times = np.asarray([0.0, 4.0, 12.0, 15.0])
    values = transformed[(drive_label, 0)].evaluate(times)

    np.testing.assert_allclose(
        values,
        np.asarray([0.0 + 0.0j, 0.5 + 0.25j, 0.5 + 0.25j, 0.0 + 0.0j]),
    )


def test_drive_has_no_per_line_transform_collection():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    drive = FluxDrive(target=q)
    assert not hasattr(drive, "signal_transforms")
    assert not hasattr(drive, "build_signal")
    assert not hasattr(drive, "signal_spec")


def test_signal_transforms_accept_drive_objects():
    """Delay, Gain, and Crosstalk accept BaseDrive instances for line identification."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.5, anharmonicity=-0.25, levels=3, label="q1")
    d0 = ChargeDrive(target=q0)
    d1 = ChargeDrive(target=q1)
    delay = Delay(line=d0, delta_t=1.0)
    gain = Gain(line=d0, factor=0.5)
    xt = Crosstalk(source=d0, victim=d1, beta=0.1)
    assert delay.line == d0.label
    assert gain.line == d0.label
    assert xt.source == d0.label
    assert xt.victim == d1.label


def test_signal_chain_handles_multiple_drive_ops_on_same_line():
    """Multiple drive ops on the same drive each get their own signal in the chain."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q0")
    d0 = ChargeDrive(target=q0)
    drive_label = d0.label
    delay = Delay(line=drive_label, delta_t=2.0)
    sig_a = AnalyticSignal(Constant(1.0 + 0j))
    sig_b = AnalyticSignal(Constant(2.0 + 0j))
    signals = {(drive_label, 0): sig_a, (drive_label, 1): sig_b}
    result = delay.apply(signals)
    assert (drive_label, 0) in result
    assert (drive_label, 1) in result
    assert isinstance(result[(drive_label, 0)].program, Shift)
    assert isinstance(result[(drive_label, 1)].program, Shift)


def test_drives_have_no_rwa_constructor_field():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    with pytest.raises(TypeError, match="rwa"):
        FluxDrive(target=q, rwa=True)


def test_flux_drive_rabi():
    """An on-resonance flux drive causes no transitions from the ground state (n_hat|0>=0)."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=4, label="q")
    d = FluxDrive(target=q)
    chip = Chip(devices=[q], control_equipment=ControlEquipment(lines=[d]))
    tlist = np.linspace(0, 50, 301)

    drive_op = DriveOp(
        target_label="q",
        envelope=Square(amplitude=0.02, duration=50),
        freq=5.0,
        start_time=0.0,
        drive_label=d.label,
    )

    result = simulate(chip, [drive_op], tlist)

    pops = result.populations
    p0 = pops[(0,)]
    assert p0[-1] > 0.99, f"FluxDrive on ground state should not cause transitions, but P(|0>) at t=50 is {p0[-1]:.4f}"


def test_flux_drive_nonzero_initial_state():
    """A flux drive preserves |1> population since n_hat is diagonal (phase-only effect)."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=4, label="q")
    d = FluxDrive(target=q)
    chip = Chip(devices=[q], control_equipment=ControlEquipment(lines=[d]))
    tlist = np.linspace(0, 50, 301)

    drive_op = DriveOp(
        target_label="q",
        envelope=Square(amplitude=0.02, duration=50),
        freq=5.0,
        start_time=0.0,
        drive_label=d.label,
    )

    initial = chip.bare_state(q=1)
    result = simulate(chip, [drive_op], tlist, initial_state=initial)

    pops = result.populations
    p1 = pops[(1,)]
    assert p1[-1] > 0.98, (
        f"FluxDrive diagonal coupling should preserve |1> population, but P(|1>) at t=50 is {p1[-1]:.4f}"
    )
