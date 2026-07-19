"""Unit tests for QuantumSequence scheduling: cursors, timing, barriers, errors.

Exercises the scheduling API using pure data-manipulation tests:
no simulation engine or backend needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from quchip.chip.chip import Chip
from quchip.control.equipment import ControlEquipment
from quchip.control.drive import ChargeDrive, FluxDrive
from quchip.control.envelopes import Square
from quchip.control.sequence import QuantumSequence
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine.ir import DriveOp


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def single_qubit_chip() -> Chip:
    """Single-transmon chip with a ChargeDrive for basic scheduling tests."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    drive = ChargeDrive(target=q0, label="charge_0")
    chip = Chip(devices=[q0])
    chip.connect(ControlEquipment(lines=[drive]))
    return chip


@pytest.fixture
def two_qubit_chip() -> Chip:
    """Two-transmon chip with ChargeDrives for multi-device and barrier tests."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.5, anharmonicity=-0.22, levels=3, label="q1")
    d0 = ChargeDrive(target=q0, label="charge_q0")
    d1 = ChargeDrive(target=q1, label="charge_q1")
    chip = Chip(devices=[q0, q1])
    chip.connect(ControlEquipment(lines=[d0, d1]))
    return chip


@pytest.fixture
def three_qubit_chip() -> Chip:
    """Three-transmon chip with ChargeDrives for selective barrier tests."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.5, anharmonicity=-0.22, levels=3, label="q1")
    q2 = DuffingTransmon(freq=6.0, anharmonicity=-0.20, levels=3, label="q2")
    d0 = ChargeDrive(target=q0, label="charge_q0")
    d1 = ChargeDrive(target=q1, label="charge_q1")
    d2 = ChargeDrive(target=q2, label="charge_q2")
    chip = Chip(devices=[q0, q1, q2])
    chip.connect(ControlEquipment(lines=[d0, d1, d2]))
    return chip


# ── charge() basics ────────────────────────────────────────────────────


class TestChargeBasic:
    """Basic charge() scheduling and cursor advancement."""

    def test_drive_alias_removed(self, single_qubit_chip: Chip) -> None:
        """QuantumSequence no longer exposes a `drive` alias."""
        seq = QuantumSequence(single_qubit_chip)
        assert not hasattr(seq, "drive")

    def test_single_charge_schedules_one_op(self, single_qubit_chip: Chip) -> None:
        """A single charge() call schedules exactly one DriveOp with the given parameters."""
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=20.0, amplitude=0.5), freq=5.0)

        assert len(seq.scheduled_ops) == 1
        op = seq.scheduled_ops[0]
        assert isinstance(op, DriveOp)
        assert op.target_label == "q0"
        assert op.freq == 5.0
        assert op.start_time == 0.0
        assert op.drive_label == "charge_0"
        assert op.envelope.duration == 20.0

    def test_charge_advances_cursor(self, single_qubit_chip: Chip) -> None:
        """charge() advances the channel cursor and total_duration by the pulse duration."""
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=20.0), freq=5.0)

        assert seq.channel_cursors[("q0", "charge_0")] == 20.0
        assert seq.total_duration == 20.0

    def test_two_sequential_charges_append(self, single_qubit_chip: Chip) -> None:
        """Two sequential charge() calls on the same channel append back-to-back."""
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.charge("q0", envelope=Square(duration=15.0), freq=5.0)

        assert len(seq.scheduled_ops) == 2
        assert seq.scheduled_ops[0].start_time == 0.0
        assert seq.scheduled_ops[1].start_time == 10.0
        assert seq.total_duration == 25.0

    def test_charge_uses_drive_freq_when_freq_omitted(self, single_qubit_chip: Chip) -> None:
        """charge() defaults freq to the device's drive_freq when omitted."""
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0))

        op = seq.scheduled_ops[0]
        assert op.freq == 5.0  # DuffingTransmon.drive_freq == freq

    def test_charge_accepts_device_object(self, single_qubit_chip: Chip) -> None:
        """charge() accepts a device object in place of its string label."""
        q0 = single_qubit_chip.devices[0]
        seq = QuantumSequence(single_qubit_chip)
        seq.charge(q0, envelope=Square(duration=10.0))

        assert seq.scheduled_ops[0].target_label == "q0"

    def test_multiple_charge_drives_keep_independent_cursors(self, single_qubit_chip: Chip) -> None:
        """Multiple charge drives on the same device keep independent channel cursors."""
        q0 = single_qubit_chip.devices[0]
        drive_a = ChargeDrive(target=q0, label="charge_a")
        drive_b = ChargeDrive(target=q0, label="charge_b")
        single_qubit_chip.connect(ControlEquipment(lines=[drive_a, drive_b]))

        seq = QuantumSequence(single_qubit_chip)
        seq.schedule(drive_a, envelope=Square(duration=10.0), freq=5.0)
        seq.schedule(drive_b, envelope=Square(duration=6.0), freq=5.0)
        seq.schedule(drive_a, envelope=Square(duration=4.0), freq=5.0)

        assert [op.drive_label for op in seq.scheduled_ops] == ["charge_a", "charge_b", "charge_a"]
        assert [op.start_time for op in seq.scheduled_ops] == [0.0, 0.0, 10.0]
        assert seq.channel_cursors[("q0", "charge_a")] == 14.0
        assert seq.channel_cursors[("q0", "charge_b")] == 6.0
        assert seq.total_duration == 14.0

    def test_charge_phase_argument_is_recorded_on_drive_op(self, single_qubit_chip: Chip) -> None:
        """charge()'s phase argument is recorded as the DriveOp's phase_offset."""
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0, phase=0.3)

        assert seq.scheduled_ops[0].phase_offset == pytest.approx(0.3)

    def test_vz_accumulates_into_later_microwave_pulses(self, single_qubit_chip: Chip) -> None:
        """vz() phase accumulates into the phase_offset of later charge() pulses on the same device."""
        seq = QuantumSequence(single_qubit_chip)
        seq.vz("q0", 0.25)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.vz("q0", -0.10)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0, phase=0.05)

        assert seq.scheduled_ops[0].phase_offset == pytest.approx(0.25)
        assert seq.scheduled_ops[1].phase_offset == pytest.approx(0.20)

    def test_vz_does_not_advance_time_or_schedule_ops(self, single_qubit_chip: Chip) -> None:
        """vz() advances neither the cursor nor total_duration and schedules no op."""
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.vz("q0", 0.5)

        assert len(seq.scheduled_ops) == 1
        assert seq.total_duration == 10.0
        assert seq.channel_cursors[("q0", "charge_0")] == 10.0


# ── delay() ─────────────────────────────────────────────────────────────


class TestDelay:
    """delay() advances cursors without scheduling operations."""

    def test_delay_advances_cursor_no_op(self, single_qubit_chip: Chip) -> None:
        """delay() advances the cursor and total_duration without scheduling an op."""
        seq = QuantumSequence(single_qubit_chip)
        seq.delay("q0", 30.0)

        assert seq.channel_cursors[("q0", "charge_0")] == 30.0
        assert len(seq.scheduled_ops) == 0
        assert seq.total_duration == 30.0

    def test_delay_then_charge_starts_after_delay(self, single_qubit_chip: Chip) -> None:
        """A charge() after delay() starts at the delayed cursor position."""
        seq = QuantumSequence(single_qubit_chip)
        seq.delay("q0", 10.0)
        seq.charge("q0", envelope=Square(duration=20.0), freq=5.0)

        assert seq.scheduled_ops[0].start_time == 10.0
        assert seq.total_duration == 30.0

    def test_charge_then_delay_then_charge(self, single_qubit_chip: Chip) -> None:
        """delay() between two charge() calls shifts the second op's start time."""
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.delay("q0", 5.0)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)

        assert seq.scheduled_ops[0].start_time == 0.0
        assert seq.scheduled_ops[1].start_time == 15.0
        assert seq.total_duration == 25.0

    def test_delay_negative_raises(self, single_qubit_chip: Chip) -> None:
        """delay() with a negative duration raises ValueError."""
        seq = QuantumSequence(single_qubit_chip)
        with pytest.raises(ValueError, match="must be > 0"):
            seq.delay("q0", -5.0)

    def test_delay_zero_raises(self, single_qubit_chip: Chip) -> None:
        """delay() with zero duration raises ValueError."""
        seq = QuantumSequence(single_qubit_chip)
        with pytest.raises(ValueError, match="must be > 0"):
            seq.delay("q0", 0.0)

    def test_delay_accepts_device_object(self, single_qubit_chip: Chip) -> None:
        """delay() accepts a device object in place of its string label."""
        q0 = single_qubit_chip.devices[0]
        seq = QuantumSequence(single_qubit_chip)
        seq.delay(q0, 15.0)

        assert seq.channel_cursors[("q0", "charge_0")] == 15.0


# ── barrier() ────────────────────────────────────────────────────────────


class TestBarrier:
    """barrier() synchronizes channel cursors."""

    def test_global_barrier_syncs_all(self, two_qubit_chip: Chip) -> None:
        """barrier() with no arguments syncs all channel cursors to the latest one."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=20.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=10.0), freq=5.5)
        seq.barrier()

        assert seq.channel_cursors[("q0", "charge_q0")] == 20.0
        assert seq.channel_cursors[("q1", "charge_q1")] == 20.0

    def test_barrier_then_charge_starts_at_synced_time(self, two_qubit_chip: Chip) -> None:
        """A charge() after barrier() starts at the barrier-synced cursor time."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=30.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=10.0), freq=5.5)
        seq.barrier()
        seq.charge("q1", envelope=Square(duration=5.0), freq=5.5)

        assert seq.scheduled_ops[-1].start_time == 30.0
        assert seq.total_duration == 35.0

    def test_selective_barrier_syncs_named_only(self, three_qubit_chip: Chip) -> None:
        """barrier() with named devices syncs only those cursors, leaving others untouched."""
        seq = QuantumSequence(three_qubit_chip)
        seq.charge("q0", envelope=Square(duration=30.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=10.0), freq=5.5)
        seq.charge("q2", envelope=Square(duration=5.0), freq=6.0)

        seq.barrier("q0", "q1")

        assert seq.channel_cursors[("q0", "charge_q0")] == 30.0
        assert seq.channel_cursors[("q1", "charge_q1")] == 30.0
        assert seq.channel_cursors[("q2", "charge_q2")] == 5.0

    def test_barrier_on_empty_sequence(self, two_qubit_chip: Chip) -> None:
        """barrier() on an empty sequence leaves total_duration at zero."""
        seq = QuantumSequence(two_qubit_chip)
        seq.barrier()
        assert seq.total_duration == 0.0

    def test_barrier_accepts_device_objects(self, two_qubit_chip: Chip) -> None:
        """barrier() accepts device objects in place of string labels."""
        q0, q1 = two_qubit_chip.devices
        seq = QuantumSequence(two_qubit_chip)
        seq.charge(q0, envelope=Square(duration=20.0))
        seq.charge(q1, envelope=Square(duration=10.0))
        seq.barrier(q0, q1)

        assert seq.channel_cursors[("q0", "charge_q0")] == 20.0
        assert seq.channel_cursors[("q1", "charge_q1")] == 20.0


# ── schedule() with explicit start_time ──────────────────────────────────


class TestExplicitStartTime:
    """schedule() with explicit start_time overrides cursor."""

    def test_start_time_overrides_cursor(self, single_qubit_chip: Chip) -> None:
        """An explicit start_time overrides the cursor and advances it past the op's end."""
        q0 = single_qubit_chip.devices[0]
        drive = q0.connected_drives[0]
        seq = QuantumSequence(single_qubit_chip)
        seq.schedule(
            drive,
            envelope=Square(duration=10.0),
            freq=5.0,
            start_time=50.0,
        )

        assert seq.scheduled_ops[0].start_time == 50.0
        assert seq.channel_cursors[("q0", drive.label)] == 60.0

    def test_start_time_before_cursor_raises(self, single_qubit_chip: Chip) -> None:
        """An explicit start_time earlier than the current cursor raises ValueError."""
        q0 = single_qubit_chip.devices[0]
        drive = q0.connected_drives[0]
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=20.0), freq=5.0)
        seq.schedule(
            drive,
            envelope=Square(duration=5.0),
            freq=5.0,
            start_time=10.0,
        )

        with pytest.raises(ValueError, match="cannot be earlier than current cursor"):
            seq.scheduled_ops

    def test_start_time_at_cursor_is_valid(self, single_qubit_chip: Chip) -> None:
        """An explicit start_time equal to the current cursor is accepted."""
        q0 = single_qubit_chip.devices[0]
        drive = q0.connected_drives[0]
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=20.0), freq=5.0)

        seq.schedule(
            drive,
            envelope=Square(duration=5.0),
            freq=5.0,
            start_time=20.0,
        )

        assert seq.scheduled_ops[1].start_time == 20.0

    def test_start_time_creates_gap(self, single_qubit_chip: Chip) -> None:
        """An explicit start_time beyond the cursor creates an idle gap before the op."""
        q0 = single_qubit_chip.devices[0]
        drive = q0.connected_drives[0]
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.schedule(
            drive,
            envelope=Square(duration=10.0),
            freq=5.0,
            start_time=30.0,
        )

        assert seq.scheduled_ops[0].start_time == 0.0
        assert seq.scheduled_ops[1].start_time == 30.0
        assert seq.total_duration == 40.0


# ── Multi-device independent cursors ─────────────────────────────────────


class TestMultiDevice:
    """Multi-device sequences with independent per-device cursors."""

    def test_independent_cursors(self, two_qubit_chip: Chip) -> None:
        """Different devices maintain independent channel cursors."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=20.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=10.0), freq=5.5)

        assert seq.channel_cursors[("q0", "charge_q0")] == 20.0
        assert seq.channel_cursors[("q1", "charge_q1")] == 10.0
        assert seq.total_duration == 20.0

    def test_interleaved_ops_preserve_independence(self, two_qubit_chip: Chip) -> None:
        """Interleaved charge() calls on different devices preserve independent start times."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=20.0), freq=5.5)
        seq.charge("q0", envelope=Square(duration=5.0), freq=5.0)

        assert seq.scheduled_ops[0].start_time == 0.0  # q0 first
        assert seq.scheduled_ops[1].start_time == 0.0  # q1 first (independent)
        assert seq.scheduled_ops[2].start_time == 10.0  # q0 second

        assert seq.channel_cursors[("q0", "charge_q0")] == 15.0
        assert seq.channel_cursors[("q1", "charge_q1")] == 20.0

    def test_delay_only_affects_target_device(self, two_qubit_chip: Chip) -> None:
        """delay() on one device leaves other devices' cursors untouched."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.delay("q0", 20.0)

        assert seq.channel_cursors[("q0", "charge_q0")] == 30.0
        assert seq.channel_cursors[("q1", "charge_q1")] == 0.0


# ── schedule() with drive objects ────────────────────────────────────────


class TestScheduleWithDriveObject:
    """schedule() accepting BaseDrive (ChargeDrive) as target argument."""

    def test_schedule_with_connected_drive(self, single_qubit_chip: Chip) -> None:
        """schedule() accepts a connected BaseDrive object as the target."""
        q0 = single_qubit_chip.devices[0]
        drive = q0.connected_drives[0]
        seq = QuantumSequence(single_qubit_chip)
        seq.schedule(drive, envelope=Square(duration=15.0), freq=5.0)

        assert len(seq.scheduled_ops) == 1
        op = seq.scheduled_ops[0]
        assert op.target_label == "q0"
        assert op.drive_label == "charge_0"
        assert op.start_time == 0.0

    def test_schedule_with_unconnected_drive_raises(self, single_qubit_chip: Chip) -> None:
        """schedule() with an unconnected drive raises ValueError."""
        drive = ChargeDrive()
        seq = QuantumSequence(single_qubit_chip)

        with pytest.raises(ValueError, match="not connected"):
            seq.schedule(drive, envelope=Square(duration=10.0), freq=5.0)


# ── schedule() with device object ──────────────────────────────────────


class TestScheduleWithDevice:
    """schedule() accepting a BaseDevice as the target argument."""

    def test_schedule_with_device_object(self, single_qubit_chip: Chip) -> None:
        """schedule() accepts a BaseDevice object as the target, using its first connected drive."""
        q0 = single_qubit_chip.devices[0]
        seq = QuantumSequence(single_qubit_chip)
        seq.schedule(q0, envelope=Square(duration=10.0), freq=5.0)

        op = seq.scheduled_ops[0]
        assert op.target_label == "q0"
        assert op.drive_label == "charge_0"

    def test_schedule_with_device_string_label(self, single_qubit_chip: Chip) -> None:
        """schedule() accepts a device's string label as the target."""
        seq = QuantumSequence(single_qubit_chip)
        seq.schedule("q0", envelope=Square(duration=10.0), freq=5.0)

        op = seq.scheduled_ops[0]
        assert op.target_label == "q0"
        assert op.drive_label == "charge_0"


# ── Error paths ──────────────────────────────────────────────────────────


class TestErrorPaths:
    """Validation and error handling in scheduling."""

    def test_charge_unknown_device_raises(self, single_qubit_chip: Chip) -> None:
        """charge() on an unknown device label raises ValueError."""
        seq = QuantumSequence(single_qubit_chip)
        with pytest.raises(ValueError, match="not found on chip"):
            seq.charge("q_missing", envelope=Square(duration=10.0), freq=5.0)

    def test_delay_unknown_device_raises(self, single_qubit_chip: Chip) -> None:
        """delay() on an unknown device label raises ValueError."""
        seq = QuantumSequence(single_qubit_chip)
        with pytest.raises(ValueError, match="not found on chip"):
            seq.delay("q_missing", 10.0)

    def test_barrier_unknown_device_raises(self, single_qubit_chip: Chip) -> None:
        """barrier() on an unknown device label raises ValueError."""
        seq = QuantumSequence(single_qubit_chip)
        with pytest.raises(ValueError, match="not found on chip"):
            seq.barrier("q_missing")

    def test_charge_no_charge_drive_raises(self) -> None:
        """charge() on a device with no ChargeDrive raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q_no_drive")
        chip = Chip(devices=[q])
        seq = QuantumSequence(chip)
        with pytest.raises(ValueError, match="No ChargeDrive"):
            seq.charge("q_no_drive", envelope=Square(duration=10.0), freq=5.0)

    def test_flux_no_flux_drive_raises(self) -> None:
        """flux() on a device with no FluxDrive raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q_no_flux")
        chip = Chip(devices=[q])
        seq = QuantumSequence(chip)
        with pytest.raises(ValueError, match="No FluxDrive"):
            seq.flux("q_no_flux", envelope=Square(duration=10.0))


# ── Complex multi-step sequences ────────────────────────────────────────


class TestComplexSequence:
    """Integration-style tests combining multiple operations."""

    def test_charge_barrier_charge_pattern(self, two_qubit_chip: Chip) -> None:
        """Parallel ops followed by barrier() resynchronize both channels before further ops."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=20.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=15.0), freq=5.5)
        seq.barrier()
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.charge("q1", envelope=Square(duration=10.0), freq=5.5)

        assert seq.scheduled_ops[2].start_time == 20.0
        assert seq.scheduled_ops[3].start_time == 20.0
        assert seq.total_duration == 30.0

    def test_delay_barrier_interaction(self, two_qubit_chip: Chip) -> None:
        """barrier() after a delay on one device pushes the other device's next op up to match."""
        seq = QuantumSequence(two_qubit_chip)
        seq.delay("q0", 50.0)
        seq.barrier()
        seq.charge("q1", envelope=Square(duration=10.0), freq=5.5)

        assert seq.scheduled_ops[0].start_time == 50.0
        assert seq.total_duration == 60.0

    def test_multiple_barriers(self, two_qubit_chip: Chip) -> None:
        """Multiple sequential barrier() calls each resynchronize cursors to the latest."""
        seq = QuantumSequence(two_qubit_chip)
        seq.charge("q0", envelope=Square(duration=10.0), freq=5.0)
        seq.barrier()
        seq.charge("q1", envelope=Square(duration=20.0), freq=5.5)
        seq.barrier()
        seq.charge("q0", envelope=Square(duration=5.0), freq=5.0)

        assert seq.scheduled_ops[2].start_time == 30.0
        assert seq.total_duration == 35.0


# ── total_duration edge cases ────────────────────────────────────────────


class TestTotalDuration:
    """total_duration property edge cases."""

    def test_empty_sequence_duration_is_zero(self, single_qubit_chip: Chip) -> None:
        """An empty sequence has total_duration zero."""
        seq = QuantumSequence(single_qubit_chip)
        assert seq.total_duration == 0.0

    def test_delay_only_contributes_to_duration(self, single_qubit_chip: Chip) -> None:
        """A delay-only sequence contributes to total_duration without scheduling ops."""
        seq = QuantumSequence(single_qubit_chip)
        seq.delay("q0", 100.0)
        assert seq.total_duration == 100.0
        assert len(seq.scheduled_ops) == 0


# ── Tracer-safe max under jax.jit ────────────────────────────────────────


class TestTracerSafeMax:
    """total_duration and barrier() reduce cursors without a Python max() under jax.jit."""

    def test_total_duration_traced_under_jit(self, two_qubit_chip: Chip) -> None:
        """A traced envelope duration flows through total_duration under jax.jit."""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def total_duration(duration):
            seq = QuantumSequence(two_qubit_chip)
            seq.charge("q0", envelope=Square(duration=duration, amplitude=0.5), freq=5.0)
            return seq.total_duration

        result = total_duration(jnp.asarray(20.0))
        assert float(result) == 20.0

    def test_global_barrier_traced_under_jit(self, two_qubit_chip: Chip) -> None:
        """A global barrier() syncs a traced cursor against concrete ones under jax.jit."""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def synced_cursors(duration):
            seq = QuantumSequence(two_qubit_chip)
            seq.charge("q0", envelope=Square(duration=duration, amplitude=0.5), freq=5.0)
            seq.charge("q1", envelope=Square(duration=10.0, amplitude=0.5), freq=5.5)
            seq.barrier()
            cursors = seq.channel_cursors
            return cursors[("q0", "charge_q0")], cursors[("q1", "charge_q1")]

        c0, c1 = synced_cursors(jnp.asarray(20.0))
        assert float(c0) == 20.0
        assert float(c1) == 20.0

    def test_selective_barrier_traced_under_jit(self, three_qubit_chip: Chip) -> None:
        """A selective barrier(device) syncs a traced cursor against concrete ones under jax.jit."""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def synced_cursors(duration):
            seq = QuantumSequence(three_qubit_chip)
            seq.charge("q0", envelope=Square(duration=duration, amplitude=0.5), freq=5.0)
            seq.charge("q1", envelope=Square(duration=10.0, amplitude=0.5), freq=5.5)
            seq.charge("q2", envelope=Square(duration=3.0, amplitude=0.5), freq=6.0)
            seq.barrier("q0", "q1")
            cursors = seq.channel_cursors
            return cursors[("q0", "charge_q0")], cursors[("q1", "charge_q1")], cursors[("q2", "charge_q2")]

        c0, c1, c2 = synced_cursors(jnp.asarray(20.0))
        assert float(c0) == 20.0
        assert float(c1) == 20.0
        assert float(c2) == 3.0  # q2 was excluded from the barrier group


# ── repr ──────────────────────────────────────────────────────────────────


class TestRepr:
    """QuantumSequence repr shows useful info."""

    def test_repr_contains_ops_and_duration(self, single_qubit_chip: Chip) -> None:
        """repr(sequence) reports the op count and total duration."""
        seq = QuantumSequence(single_qubit_chip)
        seq.charge("q0", envelope=Square(duration=20.0), freq=5.0)

        r = repr(seq)
        assert "ops=1" in r
        assert "20.0 ns" in r


# ── Batch-axis regression tests ───────────────────────────────────────────


@pytest.fixture
def qubit() -> DuffingTransmon:
    return DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="bq")


@pytest.fixture
def envelope() -> Square:
    return Square(duration=10.0, amplitude=0.02)


@pytest.fixture
def sequence(qubit: DuffingTransmon, envelope: Square) -> tuple[QuantumSequence, object]:
    drive = ChargeDrive(target=qubit)
    chip = Chip([qubit])
    chip.connect(ControlEquipment(lines=[drive]))
    seq = QuantumSequence(chip)
    pulse = seq.schedule(drive, envelope=envelope, freq=5.0)
    return seq, pulse


def test_build_batch_expands_axes_into_coordinate_override_pairs(
    sequence: tuple[QuantumSequence, object],
) -> None:
    """build_batch() expands a pulse-parameter axis into a batch of the matching shape."""
    seq, pulse = sequence
    axis = pulse.vary("freq", [4.8, 4.9], name="freq")
    batch = seq.build_batch(axis, tlist=np.linspace(0.0, 10.0, 11))
    assert batch.shape == (2,)


def test_flux_rejects_bias_point_argument(
    qubit: DuffingTransmon,
    envelope: Square,
) -> None:
    """flux() rejects an unsupported bias_point keyword argument with TypeError."""
    flux = FluxDrive(target=qubit)
    chip = Chip([qubit])
    chip.connect(ControlEquipment(lines=[flux]))
    seq = QuantumSequence(chip)
    with pytest.raises(TypeError):
        seq.flux(qubit, envelope=envelope, bias_point=0.1)


def test_schedule_disconnected_drive_raises_clean_error(
    single_qubit_chip: Chip,
) -> None:
    """schedule() on a disconnected ChargeDrive raises ValueError, not AttributeError."""
    # ChargeDrive defaults its carrier to the target device's drive_freq; the carrier resolver
    # used to dereference `drive._target.label` before the None-guard, crashing with AttributeError.
    seq = QuantumSequence(single_qubit_chip)
    orphan = ChargeDrive(label="orphan")
    with pytest.raises(ValueError, match="not connected"):
        seq.schedule(orphan, envelope=Square(duration=10.0))


def test_coarse_output_tlist_preserves_offgrid_pulse(single_qubit_chip: Chip) -> None:
    """A short pulse deep inside a long idle span survives a coarse final-state-only tlist."""
    import numpy as np

    from quchip import Gaussian

    def run(tlist):
        seq = QuantumSequence(single_qubit_chip)
        q = single_qubit_chip.devices[0]
        seq.delay(q, duration=90.0)
        seq.charge(q, envelope=Gaussian(duration=10.0, amplitude=0.05), freq=single_qubit_chip.freq(q))
        return seq.simulate(tlist=tlist)

    dense = run(np.linspace(0.0, 100.0, 2001))
    coarse = run(np.array([0.0, 100.0]))
    p_dense = float(np.asarray(dense.populations[(0,)])[-1])
    p_coarse = float(np.asarray(coarse.populations[(0,)])[-1])
    assert p_dense < 0.99  # the pulse actually did something
    np.testing.assert_allclose(p_coarse, p_dense, atol=5e-3)
