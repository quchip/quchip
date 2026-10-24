"""FluxDrive channel generation, signal-chain composition, and population preservation."""

from __future__ import annotations


import numpy as np
import pytest

from quchip import Chip, ControlEquipment, DuffingTransmon, Square
from quchip.control.drive import FluxDrive
from quchip.control.signal import AnalyticSignal, Crosstalk, Delay, Gain
from quchip.engine import simulate
from quchip.control.drive import ChargeDrive
from quchip.engine.ir import Constant, DriveOp


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


@pytest.mark.parametrize("frequency", [None, 0.0, 0.1])
def test_longitudinal_drive_retains_dc_and_oscillating_phase(frequency):
    """A longitudinal tone changes level energies by its full real waveform."""
    from quchip import QuantumSequence

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    drive = FluxDrive(q)
    chip = Chip([q], frame="rotating")
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Square(duration=10.0, amplitude=0.02), freq=frequency)
    static = np.asarray(chip.hamiltonian().matrix(t=0.0))
    driven = sequence.hamiltonian()
    for time in (0.0, 1.0, 2.0):
        modulation = 0.02 * np.cos(2 * np.pi * (frequency or 0.0) * time)
        expected = static + np.diag(np.arange(3) * modulation)
        np.testing.assert_allclose(driven.matrix(t=time), expected, atol=1e-13)


@pytest.mark.validation
def test_longitudinal_phase_matches_integrated_frequency():
    """A longitudinal tone gives the analytic relative phase without changing populations."""
    from quchip import QuantumSequence

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    drive = FluxDrive(q)
    chip = Chip([q], frame={"q": 5.0})
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    amplitude, frequency, duration = 0.02, 0.1, 3.0
    sequence.schedule(drive, envelope=Square(duration=duration, amplitude=amplitude), freq=frequency)
    initial = (chip.backend.basis(2, 0) + chip.backend.basis(2, 1)) / np.sqrt(2)
    options = {"atol": 1e-10, "rtol": 1e-9}
    result = sequence.simulate(tlist=[0.0, duration], initial_state=initial, states="final", options=options)
    state = np.asarray(chip.backend.to_array(result.final_state)).ravel()
    phase = amplitude * np.sin(2 * np.pi * frequency * duration) / frequency
    expected = np.array([1.0, np.exp(-1j * phase)]) / np.sqrt(2)
    # Integrating the sinusoidal frequency gives this phase; tolerance exceeds solver settings.
    np.testing.assert_allclose(np.outer(state, state.conj()), np.outer(expected, expected.conj()), atol=2e-8)
