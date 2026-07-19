"""Scheduled edge pumps reproduce analytic exchange physics (spec Sec. 9)."""

from __future__ import annotations

import numpy as np

from quchip import (
    Chip,
    ControlEquipment,
    DuffingTransmon,
    ParametricDrive,
    QuantumSequence,
    Square,
    TunableCapacitive,
)


def _wire(freq0: float, freq1: float, *, rwa: bool, frame: str):
    q0 = DuffingTransmon(freq=freq0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=freq1, anharmonicity=-0.24, levels=3, label="q1")
    tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
    pump = ParametricDrive(tc, label="pump")
    chip = Chip([q0, q1], couplings=[tc], frame=frame, rwa=rwa)
    chip.connect(ControlEquipment([pump]))
    return chip, q0, q1, tc


def test_baseband_pump_drives_full_exchange_swap():
    """A resonant baseband pump completes a full |10>-|01> excitation swap at the analytic swap time."""
    # Resonant pair, rotating frame, RWA: H = 2 pi A (a^dag b + a b^dag) while the pump is on.
    # |10> <-> |01> swaps fully with period 1/(2A); at t = 1/(4A) the transfer is complete.
    A = 0.005
    t_swap = 1.0 / (4.0 * A)  # 50 ns
    chip, q0, q1, tc = _wire(5.0, 5.0, rwa=True, frame="rotating")
    seq = QuantumSequence(chip)
    seq.pump(tc, envelope=Square(duration=t_swap, amplitude=A))
    result = seq.simulate(
        tlist=np.linspace(0.0, t_swap, 201),
        initial_state={"q0": 1, "q1": 0},
    )
    # sin^2(2*pi*A*t_swap) = sin^2(2*pi*0.005*50) = sin^2(pi/2) = 1.0; measured p_final =
    # 0.99999999999844 (diff ~1.6e-12), so >0.98 keeps a wide margin over solver precision.
    assert float(result.population(q1, 1)[-1]) > 0.98


def test_parametric_resonance_activates_detuned_exchange():
    """A pump tone at the detuning frequency activates near-complete exchange between detuned qubits."""
    # Detuned pair (delta = 0.2 GHz), tone at the difference frequency: the
    # co-rotating half of A cos(2 pi delta t) cancels the detuning; effective exchange A/2.
    A = 0.01
    j_eff = A / 2.0
    t_swap = 1.0 / (4.0 * j_eff)  # 50 ns
    chip, q0, q1, tc = _wire(5.0, 5.2, rwa=True, frame="rotating")
    seq = QuantumSequence(chip)
    seq.pump(tc, envelope=Square(duration=t_swap, amplitude=A), freq=0.2)
    result = seq.simulate(
        tlist=np.linspace(0.0, t_swap, 401),
        initial_state={"q0": 1, "q1": 0},
    )
    # sin^2(2*pi*j_eff*t_swap) = sin^2(2*pi*0.005*50) = sin^2(pi/2) = 1.0; measured max
    # p1 = 0.99984 (the counter-rotating micromotion the co-rotating cancellation leaves
    # behind costs ~1.6e-4), so >0.9 keeps a wide margin over that residual.
    assert float(np.max(result.population(q1, 1))) > 0.9


def test_pump_amplitude_batch_axis_sweeps():
    """Batched pump-amplitude sweeps reach full transfer at A and partial transfer at half A."""
    A = 0.005
    chip, q0, q1, tc = _wire(5.0, 5.0, rwa=True, frame="rotating")
    seq = QuantumSequence(chip)
    handle = seq.pump(tc, envelope=Square(duration=50.0, amplitude=A))
    batch = seq.simulate_batch(
        handle.vary("amplitude", [0.0025, 0.005]),
        tlist=np.linspace(0.0, 50.0, 101),
        initial_state={"q0": 1, "q1": 0},
        progress=False,
    )
    p_half, p_full = (float(el.population(q1, 1)[-1]) for el in batch)
    # sin^2(2*pi*A*50) predicts 1.0 at A=0.005 (measured p_full = 0.99999999999929) and
    # 0.5 at A=0.0025 (measured p_half = 0.50000024), the quarter-period point; 0.3-0.8
    # keeps a wide margin around that analytic 0.5.
    assert p_full > 0.98
    assert 0.3 < p_half < 0.8  # half amplitude -> quarter period reached, partial transfer
