"""Scheduling edge pumps through QuantumSequence (spec §5)."""

from __future__ import annotations

import pytest

from quchip import (
    Chip, ControlEquipment, DuffingTransmon, ParametricDrive, QuantumSequence, Square, TunableCapacitive,
)


def _wired_chip():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
    pump = ParametricDrive(tc, label="pump")
    chip = Chip([q0, q1], couplings=[tc])
    chip.connect(ControlEquipment([pump]))
    return chip, tc, pump


def test_pump_baseband_emits_drive_op_with_coupling_target():
    """Pumping a coupling without freq emits one baseband op, labeled by the pump line and targeted at the coupling."""
    chip, tc, pump = _wired_chip()
    seq = QuantumSequence(chip)
    seq.pump(tc, envelope=Square(duration=100.0, amplitude=0.005))
    ops = seq.scheduled_ops
    assert len(ops) == 1
    assert ops[0].target_label == "tc"
    assert ops[0].drive_label == "pump"
    assert ops[0].freq is None  # baseband form


def test_pump_single_tone_carries_freq_and_phase():
    """Pumping a coupling by its string label carries the supplied carrier frequency and phase into the drive op."""
    chip, tc, _ = _wired_chip()
    seq = QuantumSequence(chip)
    seq.pump("tc", envelope=Square(duration=100.0, amplitude=0.005), freq=0.2, phase=0.3)
    (op,) = seq.scheduled_ops
    assert op.freq == 0.2 and op.phase_offset == 0.3


def test_schedule_accepts_pump_drive_and_coupling_label():
    """A ParametricDrive object and its coupling's string label resolve through schedule() to the same edge target."""
    chip, tc, pump = _wired_chip()
    seq = QuantumSequence(chip)
    seq.schedule(pump, envelope=Square(duration=50.0, amplitude=0.001))
    seq.schedule("tc", envelope=Square(duration=50.0, amplitude=0.001))
    assert len(seq.scheduled_ops) == 2
    assert {op.target_label for op in seq.scheduled_ops} == {"tc"}


def test_pump_without_line_raises_with_guidance():
    """Pumping a coupling with no ParametricDrive line wired raises a ValueError that names the missing drive type."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
    chip = Chip([q0, q1], couplings=[tc])
    seq = QuantumSequence(chip)
    with pytest.raises(ValueError, match="ParametricDrive"):
        seq.pump(tc, envelope=Square(duration=50.0, amplitude=0.001))


def test_pump_handle_supports_batch_axis():
    """A pump() handle supports .vary() to create a batch axis, like any other scheduled-pulse handle."""
    chip, tc, _ = _wired_chip()
    seq = QuantumSequence(chip)
    handle = seq.pump(tc, envelope=Square(duration=100.0, amplitude=0.005))
    axis = handle.vary("amplitude", [0.001, 0.002, 0.003])
    assert axis.entry_index is not None


def test_device_scheduling_is_unchanged():
    """Wiring an edge pump line alongside device drives leaves charge-drive scheduling on devices unaffected."""
    chip, _, _ = _wired_chip()
    # Re-wire with a charge line as well to check device path end-to-end.
    from quchip import ChargeDrive

    dq = ChargeDrive(chip["q0"], label="dq")
    chip.connect(ControlEquipment([dq, *chip.control_equipment.lines]))
    seq = QuantumSequence(chip)
    seq.charge("q0", envelope=Square(duration=20.0, amplitude=0.01))
    (op,) = seq.scheduled_ops
    assert op.target_label == "q0"
