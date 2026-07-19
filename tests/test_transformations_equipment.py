"""eliminate() carries the control plane into the reduced chip (spec §6.4)."""

from __future__ import annotations

import pytest

from quchip import Capacitive, ChargeDrive, Chip, ControlEquipment, DuffingTransmon, Resonator
from quchip.chip.transformations import eliminate
from quchip.control.signal import Crosstalk, Delay


def _chip_with_equipment():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    q2 = DuffingTransmon(freq=5.3, anharmonicity=-0.24, levels=3, label="q2")
    r = Resonator(freq=7.0, levels=4, label="r")
    dq = ChargeDrive(q, label="dq")
    dq2 = ChargeDrive(q2, label="dq2")
    chain = [Delay("dq", delta_t=1.5), Crosstalk("dq", "dq2", beta=0.02)]
    chip = Chip(
        [q, q2, r],
        couplings=[Capacitive(q, r, g=0.05), Capacitive(q, q2, g=0.004)],
        control_equipment=ControlEquipment([dq, dq2], signal_chain=chain),
    )
    return chip


def test_eliminate_carries_survivor_lines_and_signal_chain():
    """eliminate() rebinds surviving control lines onto the reduced chip's devices and keeps the signal chain intact."""
    chip = _chip_with_equipment()
    reduced = eliminate(chip, "r").chip
    ce = reduced.control_equipment
    assert ce is not None
    assert [line.label for line in ce.lines] == ["dq", "dq2"]
    # Lines rebind to the *reduced* chip's canonical device instances.
    assert ce.lines[0]._target is reduced["q"]
    assert ce.lines[1]._target is reduced["q2"]
    # Signal chain survives verbatim (same transform count and keying).
    kinds = [type(t).__name__ for t in ce.signal_chain]
    assert kinds == ["Delay", "Crosstalk"]
    assert ce.signal_chain[0].line == "dq"


def test_eliminate_without_equipment_still_returns_none():
    """eliminate() leaves control_equipment as None on the reduced chip when the input chip carries none."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05)])
    reduced = eliminate(chip, "r").chip
    assert reduced.control_equipment is None


def test_eliminate_raises_on_line_targeting_eliminated_mode():
    """eliminate() raises ValueError naming the eliminated mode and pointing to register_retarget_rule for that line."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    probe = ChargeDrive(r, label="probe")
    chip = Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=0.05)],
        control_equipment=ControlEquipment([probe]),
    )
    with pytest.raises(ValueError, match="targets the eliminated mode 'r'") as exc_info:
        eliminate(chip, "r")
    assert "register_retarget_rule" in str(exc_info.value)


def test_input_chip_equipment_is_not_mutated():
    """eliminate() does not mutate the input chip's control-equipment lines or their device targets."""
    chip = _chip_with_equipment()
    original_lines = chip.control_equipment.lines
    eliminate(chip, "r")
    # Original lines still target the original chip's devices.
    assert chip.control_equipment.lines[0]._target is chip["q"]
    assert [line.label for line in chip.control_equipment.lines] == [line.label for line in original_lines]


def test_eliminate_does_not_duplicate_connected_drives():
    """Elimination's clone-then-reattach equipment wiring does not duplicate connected drives on surviving devices."""
    # Regression: eliminate()'s reduction path builds ``reduced = chip.clone()``
    # (which already connects the full cloned equipment onto the clone's
    # devices) and then reattach_equipment() connects a fresh
    # ControlEquipment(...).copy(...) onto those same surviving device
    # objects — the same clone-then-re-copy-equipment pattern that duplicates
    # drives in chip.partition(). Each surviving device must end up with
    # exactly one drive per label.
    chip = _chip_with_equipment()
    reduced = eliminate(chip, "r", method="exact").chip
    for label in ("q", "q2"):
        device = reduced[label]
        drive_labels = [d.label for d in device.connected_drives]
        assert len(drive_labels) == len(set(drive_labels)), (
            f"device '{label}' has duplicate-label connected drives: {drive_labels}"
        )
