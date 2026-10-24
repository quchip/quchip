"""Explicit model context and stable device identities."""

import numpy as np
import pytest

from quchip import Capacitive, ChargeDrive, Chip, DuffingTransmon, QuantumSequence, Resonator, Square


pytestmark = pytest.mark.unit


def test_local_hamiltonian_is_independent_of_chip_membership() -> None:
    """Sharing a device between differently framed chips never changes its local Hamiltonian."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    local = q.hamiltonian().matrix()
    first = Chip([q], frame="rotating")
    second = Chip([q], frame=4.8)

    np.testing.assert_allclose(q.hamiltonian().matrix(), local, atol=1e-12)
    assert first.freq(q) == pytest.approx(5.0)
    assert second.freq(q) == pytest.approx(5.0)
    np.testing.assert_allclose(np.diag(q.resolve(frame=4.8).hamiltonian().matrix()), [0.0, 0.2, 0.2], atol=1e-12)


def test_default_pulse_carrier_uses_the_sequence_chip_and_is_captured() -> None:
    """A shared device gets chip-specific default carriers that later edits do not retune."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    r = Resonator(freq=6.0, levels=3, label="r")
    first = Chip([q])
    second = Chip([q, r], [Capacitive(q, r, g=0.08, label="qr")])
    drive = ChargeDrive(q, label="xy")
    first.wire(drive)
    second.wire(drive)
    seq_first, seq_second = QuantumSequence(first), QuantumSequence(second)
    for seq in (seq_first, seq_second):
        with pytest.warns(UserWarning, match="A pulse without freq"):
            seq.charge(q, envelope=Square(duration=10.0, amplitude=0.02))
    f_first, f_second = first.freq(q), second.freq(q)
    assert seq_first.parameters["pulse.0.freq"] == pytest.approx(f_first)
    assert seq_second.parameters["pulse.0.freq"] == pytest.approx(f_second)
    assert abs(float(f_first - f_second)) > 1e-3

    q.freq = 5.1
    assert seq_second.parameters["pulse.0.freq"] == pytest.approx(f_second)


def test_unset_reference_is_authored_none_and_resolves_for_each_calculation() -> None:
    """References resolve in the chip while old frame records retain their values."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    chip = Chip([q], frame="rotating")
    assert q.reference_freq is None
    first = chip.resolve()
    q.freq = 5.1
    second = chip.resolve()
    assert first.resolved_frame.frequencies["q"] == pytest.approx(5.0)
    assert second.resolved_frame.frequencies["q"] == pytest.approx(5.1)
    q.reference_freq = 4.9
    q.freq = 5.2
    assert chip.resolve().resolved_frame.frequencies["q"] == pytest.approx(4.9)


def test_device_label_is_fixed_after_construction() -> None:
    """Changing a label requires a new device rather than invalidating existing membership."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, label="q")
    chip = Chip([q])
    with pytest.raises(AttributeError, match="label.*fixed"):
        q.label = "other"
    assert chip["q"] is q
    assert q.label == "q"


def test_coupling_object_endpoints_must_match_the_declared_devices() -> None:
    """A same-label foreign endpoint cannot supply physics outside the declared chip."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, label="q")
    foreign = DuffingTransmon(freq=5.5, anharmonicity=-0.3, label="q")
    r = Resonator(freq=6.0, levels=3, label="r")
    coupling = Capacitive(foreign, r, g=0.02)
    with pytest.raises(ValueError, match="different device"):
        Chip([q, r], [coupling])
    assert coupling.device_a is foreign


def test_coupling_label_binding_validates_all_endpoints_before_binding() -> None:
    """A failed chip construction leaves an unresolved coupling unchanged."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, label="q")
    coupling = Capacitive("q", "missing", g=0.02)
    with pytest.raises(ValueError, match="missing"):
        Chip([q], [coupling])
    assert coupling.device_a == "q"
    assert coupling.device_b == "missing"


def test_shared_device_scheduling_uses_only_the_current_chips_wiring() -> None:
    """Shared devices do not merge the schedules or line choices of their chips."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, label="q")
    first, second = Chip([q]), Chip([q])
    first.wire(ChargeDrive(q, label="first"))
    second.wire(ChargeDrive(q, label="second"))
    for chip, label in ((first, "first"), (second, "second")):
        seq = QuantumSequence(chip)
        seq.charge(q, envelope=Square(duration=10.0), freq=5.0)
        seq.schedule(q, envelope=Square(duration=5.0), freq=5.0)
        assert [op.drive_label for op in seq.scheduled_ops] == [label, label]
        assert seq.channel_cursors == {("q", label): 15.0}


def test_detached_wiring_is_not_available_for_new_scheduling() -> None:
    """Unwiring a line removes it from subsequent implicit scheduling and cursors."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, label="q")
    chip = Chip([q])
    chip.wire(ChargeDrive(q, label="xy"))
    chip.unwire("xy")
    seq = QuantumSequence(chip)
    with pytest.raises(ValueError, match="No ChargeDrive"):
        seq.charge(q, envelope=Square(duration=10.0))
    assert seq.channel_cursors == {}
