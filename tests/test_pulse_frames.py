"""Pulse frames: which device's virtual-Z phase a scheduled carrier follows."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    FluxDrive,
    QuantumSequence,
    RWA,
    Square,
    SquareWithGaussianEdges,
)
from quchip.analysis import analyze_cr_susceptibility


def _cross_resonance_chip() -> Chip:
    """Control q0 and target q1, each with its own line, plus a CR line on q0."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q1")
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003)], frame="rotating",
                approximation=RWA(), backend="qutip")
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"), ChargeDrive(q0, label="cr"))
    return chip


@pytest.mark.unit
def test_vz_shifts_every_later_pulse_in_its_frame_on_any_line():
    """vz() follows each pulse's frame: a CR tone in q1's frame follows q1, not the q0 line it runs on."""
    chip = _cross_resonance_chip()
    pulse = Square(duration=10.0, amplitude=0.01)
    seq = QuantumSequence(chip)
    seq.vz("q1", 0.3)
    seq.schedule("cr", envelope=pulse, freq=5.1, phase=0.1, frame="q1")
    seq.schedule("xy1", envelope=pulse, freq=5.1)
    seq.schedule("xy0", envelope=pulse, freq=5.0)
    seq.vz("q0", 0.7)
    seq.schedule("cr", envelope=pulse, freq=5.1, frame=chip["q1"])
    seq.schedule("xy0", envelope=pulse, freq=5.0)

    ops = seq.scheduled_ops
    assert [op.frame for op in ops] == ["q1", "q1", "q0", "q1", "q0"]
    np.testing.assert_allclose([op.phase_offset for op in ops], [0.4, 0.3, 0.0, 0.3, 0.7])


@pytest.mark.unit
def test_charge_names_the_frame_of_a_tone_at_another_devices_frequency():
    """charge() on q1's own line can drive a CR tone at q0's frequency in q0's frame."""
    chip = _cross_resonance_chip()
    seq = QuantumSequence(chip)
    seq.vz("q0", 0.5)
    seq.charge("q1", envelope=Square(duration=10.0, amplitude=0.01), freq=5.0, frame=chip["q0"])
    seq.charge("q1", envelope=Square(duration=10.0, amplitude=0.01))
    assert [(op.frame, op.phase_offset) for op in seq.scheduled_ops] == [("q0", 0.5), ("q1", 0.0)]


@pytest.mark.unit
def test_baseband_pulses_have_no_frame():
    """A flux pulse keeps its phase under vz(), and an explicit frame without a carrier is rejected."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])
    chip.wire(FluxDrive(q, label="z"))
    seq = QuantumSequence(chip)
    seq.vz(q, 0.5)
    seq.flux(q, envelope=Square(duration=10.0, amplitude=0.01))
    (op,) = seq.scheduled_ops
    assert (op.frame, op.phase_offset) == (None, 0.0)
    with pytest.raises(ValueError, match="carrier"):
        seq.schedule("z", envelope=Square(duration=10.0, amplitude=0.01), frame=q)


@pytest.mark.unit
def test_frame_must_name_a_device_on_the_chip():
    """A frame that names a line or an unknown label raises with the available devices."""
    chip = _cross_resonance_chip()
    seq = QuantumSequence(chip)
    with pytest.raises(ValueError, match=r"Frame 'xy1' is not a device.*\['q0', 'q1'\]"):
        seq.schedule("cr", envelope=Square(duration=10.0, amplitude=0.01), freq=5.1, frame="xy1")


def test_vz_on_the_cross_resonance_target_keeps_basis_state_populations():
    """With both q1-frequency tones in q1's frame, vz(q1) before a CR pulse keeps every basis-state population."""
    # Under RWA with an exchange coupling, a common phase shift θ of every tone conjugates the
    # propagator by exp(-iθN). This is diagonal in the product basis, so populations agree to
    # solver tolerance. Before frames, vz(q1) shifted only the cancellation tone (0.26 change).
    chip = _cross_resonance_chip()
    duration, target_freq = 160.0, float(chip.freq("q1"))

    def envelope(amplitude):
        return SquareWithGaussianEdges(duration=duration, amplitude=amplitude, edge_frac=0.375)

    # Weak-drive ZX90 seed with an IX-cancellation tone on q1's own line, as in issue #92.
    rates = analyze_cr_susceptibility(chip, "q0", "q1", drive="cr")
    zx, ix = complex(rates.ZX_per_amplitude), complex(rates.IX_per_amplitude)
    times = np.linspace(0.0, duration, 20001)
    amplitude = 1 / (2 * np.trapezoid(envelope(1.0).value(times).real, times) * abs(zx))
    phase = -np.angle(zx)
    cancel = -ix * amplitude * np.exp(1j * phase) / (2 * complex(chip.drive_matrix_elements("q1")["xy1"]))

    def populations(vz_device):
        rows = []
        for control in (0, 1):
            for target in (0, 1):
                seq = QuantumSequence(chip)
                if vz_device is not None:
                    seq.vz(vz_device, np.pi / 2)
                seq.schedule("cr", envelope=envelope(amplitude), freq=target_freq, phase=phase, frame="q1")
                seq.schedule("xy1", envelope=envelope(abs(cancel)), freq=target_freq,
                             phase=float(np.angle(cancel)), frame="q1")
                result = seq.simulate(tlist=[0.0, duration], initial_state={"q0": control, "q1": target},
                                      states="final", options={"atol": 1e-10, "rtol": 1e-8})
                rows.append(np.abs(result.final_state.full().ravel()) ** 2)
        return np.array(rows)

    reference = populations(None)
    assert reference[0, 1] > 0.1  # the pulses rotate the target out of |00>
    np.testing.assert_allclose(populations("q1"), reference, atol=1e-6)
