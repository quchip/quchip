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
    ParametricDrive,
    PhaseDrive,
    PortNetwork,
    QuantumSequence,
    RWA,
    Square,
    SquareWithGaussianEdges,
    TunableCapacitive,
)
from quchip.analysis import analyze_cr_susceptibility
from quchip.control.field import CoherentInput


@pytest.mark.unit
def test_vz_shifts_each_carrier_in_its_frame_and_no_other_pulse():
    """vz(q1) shifts every carrier in q1's frame on any line, but no unframed or baseband pulse. Bad frames raise."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    coupling = TunableCapacitive(q0, q1, g_0=0.01, label="tc")
    network = PortNetwork(label="line")
    network.port("feed", target=q1, rate=0.001)
    chip = Chip([q0, q1], couplings=[coupling], port_network=network)
    chip.wire(ChargeDrive(q0, label="xy0"), PhaseDrive(q0, label="ph0"), FluxDrive(q1, label="z1"),
              ParametricDrive(coupling, label="pump"))
    seq = QuantumSequence(chip)
    seq.vz("q1", 0.5)
    pulse = Square(duration=10.0, amplitude=0.01)
    seq.charge("q0", envelope=pulse, freq=5.2, frame=q1)
    seq.phase("q0", envelope=pulse, freq=5.2, frame="q1")
    seq.flux("q1", envelope=pulse)
    for line in ("pump", CoherentInput("feed")):
        seq.schedule(line, envelope=pulse, freq=0.2)
        seq.schedule(line, envelope=pulse, freq=0.2, frame="q1")
    shifted, unshifted = ("q1", 0.5), (None, 0.0)
    expected = [shifted, shifted, unshifted] + [unshifted, shifted] * 2
    assert [(op.frame, op.phase_offset) for op in seq.scheduled_ops] == expected
    with pytest.raises(ValueError, match="A frame requires a carrier"):
        seq.schedule("z1", envelope=pulse, frame="q1")
    with pytest.raises(ValueError, match="Frame 'xy0' is not a device"):
        seq.schedule("xy0", envelope=pulse, freq=5.2, frame="xy0")


def test_vz_before_a_cross_resonance_gate_keeps_basis_state_populations():
    """With both q1-frequency tones in q1's frame, vz() on either qubit before a CR pulse keeps every population."""
    # vz(q0) shifts neither tone. vz(q1) shifts both by θ. Under RWA with an exchange coupling, this
    # conjugates the propagator by exp(-iθN), which is diagonal in the product basis. Populations
    # therefore agree to solver tolerance. Before frames, each vz() shifted one tone (0.26 change).
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q1")
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003)], frame="rotating",
                approximation=RWA(), backend="qutip")
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"), ChargeDrive(q0, label="cr"))
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
    for device in ("q0", "q1"):
        np.testing.assert_allclose(populations(device), reference, atol=1e-6)
