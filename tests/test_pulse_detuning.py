"""Start-referenced carrier detuning of scheduled pulses."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    FluxDrive,
    Gaussian,
    GaussianDRAG,
    QuantumSequence,
    RWA,
    Square,
)
from quchip.control.signal import AnalyticSignal
from quchip.engine.ir import DriveOp, decompose_carrier_bands, evaluate_signal_program


@pytest.mark.unit
@pytest.mark.parametrize("start", [0.0, 7.3, 50.0])
def test_detuned_signal_is_the_same_in_its_frame_at_every_start(start):
    """A detuning multiplies the envelope by exp(-2πiδ(t - t0)), so the frame-referenced pulse ignores t0."""
    freq, detuning, phase = 5.0, 4.7e-3, 0.3
    envelope = Gaussian(duration=20.0, sigmas=3, amplitude=0.02)
    op = DriveOp(target_label="q", drive_label="xy", envelope=envelope, freq=freq, start_time=start,
                 phase_offset=phase, detuning=detuning)
    signal = AnalyticSignal.from_pulse(op)
    local = np.linspace(0.0, 20.0, 41)
    t = start + local

    expected = (envelope.value(local) * np.exp(1j * phase) * np.exp(-2j * np.pi * freq * t)
                * np.exp(-2j * np.pi * detuning * local))
    np.testing.assert_allclose(signal.evaluate(t), expected, rtol=1e-12, atol=1e-15)
    # Each carrier band reproduces the same signal analytically, as the backends sample it.
    (band,) = decompose_carrier_bands(signal.program)
    np.testing.assert_allclose(band.freq, -2 * np.pi * (freq + detuning), rtol=1e-15)
    np.testing.assert_allclose(evaluate_signal_program(band.envelope, t) * np.exp(1j * band.freq * t), expected,
                               rtol=1e-11, atol=1e-15)
    assert signal.carrier == pytest.approx(freq + detuning, rel=1e-15)
    # In the frame at freq, the pulse depends on local time only.
    np.testing.assert_allclose(signal.evaluate(t) * np.exp(2j * np.pi * freq * t),
                               envelope.value(local) * np.exp(1j * phase) * np.exp(-2j * np.pi * detuning * local),
                               rtol=1e-11, atol=1e-15)


def _single_qubit_chip(levels: int = 3, **kwargs) -> Chip:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=levels, label="q")
    chip = Chip([q], frame="rotating", approximation=RWA(), **kwargs)
    chip.wire(ChargeDrive(q, label="xy"))
    return chip


@pytest.mark.unit
def test_detuning_rebinds_as_a_pulse_parameter():
    """pulse.<i>.detuning is None for an undetuned pulse, and rebinding a number sets the offset."""
    seq = QuantumSequence(_single_qubit_chip())
    seq.charge("q", envelope=Square(duration=10.0, amplitude=0.01), detuning=0.002)
    seq.charge("q", envelope=Square(duration=10.0, amplitude=0.01))

    assert seq.parameters["pulse.0.detuning"] == 0.002
    assert seq.parameters["pulse.1.detuning"] is None
    rebound = seq.with_params({"pulse.0.detuning": 0.003, "pulse.1.detuning": 0.0})
    assert [op.detuning for op in rebound.scheduled_ops] == [0.003, 0.0]
    assert [op.detuning for op in seq.scheduled_ops] == [0.002, None]
    assert "detuning 0.002 GHz" in seq.describe()


@pytest.mark.unit
def test_detuning_requires_a_carrier():
    """A baseband pulse has no carrier to offset, so a scheduled or rebound detuning without freq is rejected."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])
    chip.wire(FluxDrive(q, label="z"))
    seq = QuantumSequence(chip)
    with pytest.raises(ValueError, match="carrier"):
        seq.schedule("z", envelope=Square(duration=10.0, amplitude=0.01), detuning=0.001)
    seq.schedule("z", envelope=Square(duration=10.0, amplitude=0.01))
    with pytest.raises(ValueError, match="carrier"):
        seq.with_params({"pulse.0.detuning": 0.001}).scheduled_ops


def test_detuning_sweep_matches_rebound_sequences():
    """A batch over pulse.vary("detuning") on an undetuned pulse matches rebinding each value."""
    # Two levels have no Stark shift, so a resonant pi pulse reaches P(1) = 1 and detuning lowers it.
    chip = _single_qubit_chip(levels=2)
    seq = QuantumSequence(chip)
    element = abs(complex(chip.drive_matrix_elements("q")["xy"]))
    amplitude = 0.5 / element / (20.0 / 6 * np.sqrt(2 * np.pi) * erf(3 / np.sqrt(2)))
    pulse = seq.charge("q", envelope=Gaussian(duration=20.0, sigmas=3, amplitude=amplitude))
    values = [0.0, 0.004, 0.012]
    options = {"atol": 1e-10, "rtol": 1e-8}
    batch = seq.simulate_batch(pulse.vary("detuning", values), tlist=[0.0, 20.0], options=options, progress=False)
    for index, value in enumerate(values):
        single = seq.with_params({"pulse.0.detuning": value}).simulate(tlist=[0.0, 20.0], options=options)
        np.testing.assert_allclose(batch[index].final_state.full(), single.final_state.full(), atol=1e-8)
    populations = np.asarray(batch.population("q", 1, reduce="last"))
    assert populations[0] == pytest.approx(1.0, abs=1e-6)
    assert populations[0] > populations[1] > populations[2]


def _detuned_sx_chip() -> tuple[Chip, float, complex]:
    """Return the two-transmon chip of issue #93 with the frequency and drive element of q0."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q1")
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003)], frame="rotating",
                approximation=RWA(), backend="qutip")
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    return chip, float(chip.freq("q0")), complex(chip.drive_matrix_elements("q0")["xy0"])


def test_detuned_sx_pair_makes_an_x_at_every_gap():
    """Two start-referenced detuned SX pulses make an X at any gap, and a fixed oscillator keeps its meaning."""
    chip, f0, element = _detuned_sx_chip()
    # 20 ns DRAG SX with a 4.69 MHz detuning, calibrated by least squares (1 - F_avg = 1.2e-5).
    duration, detuning, phase = 20.0, 4.68882e-3, -np.angle(element)
    amplitude = 1.00432 * 0.25 / abs(element) / (duration / 6 * np.sqrt(2 * np.pi) * erf(3 / np.sqrt(2)))
    sx = GaussianDRAG(duration=duration, sigmas=3, amplitude=amplitude, beta=-2.11643)
    options = {"atol": 1e-10, "rtol": 1e-8}

    def sx_pair(gap, carrier):
        seq = QuantumSequence(chip)
        for start in (0.0, duration + gap):
            if carrier == "fixed":
                seq.schedule("xy0", envelope=sx, freq=f0 + detuning, phase=phase, start_time=start)
            elif carrier == "fixed by hand":
                seq.schedule("xy0", envelope=sx, freq=f0, detuning=detuning,
                             phase=phase - 2 * np.pi * detuning * start, start_time=start)
            else:
                if start > 0:
                    seq.vz("q0", -2 * np.pi * detuning * duration)  # the calibration's post-pulse Z
                seq.schedule("xy0", envelope=sx, freq=f0, detuning=detuning, phase=phase, start_time=start)
        return seq.simulate(tlist=[0.0, 2 * duration + gap], options=options)

    # population() reads the bare level. The dressed |10> holds (g/Δ)² of its weight in |01>.
    leakage = (0.003 / 0.1) ** 2
    populations = [float(sx_pair(gap, "start").population("q0", 1)[-1]) for gap in (0.0, 10.0, 25.0, 50.0)]
    np.testing.assert_allclose(populations, 1 - leakage, atol=5e-5)
    # A carrier at f0 + δ is the detuned pulse with phase φ - 2πδ t0 and no post-pulse Z.
    fixed = sx_pair(50.0, "fixed")
    np.testing.assert_allclose(fixed.final_state.full(), sx_pair(50.0, "fixed by hand").final_state.full(), atol=1e-6)
    # Its second pulse turns its axis by 2πδ·gap, so P(1) follows cos²(πδ·gap) up to the SX error.
    np.testing.assert_allclose(fixed.population("q0", 1)[-1], np.cos(np.pi * detuning * 50.0) ** 2, atol=1e-2)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_detuning_gradient_matches_finite_differences():
    """The JAX derivative of P(1) with respect to a rebound pulse.<i>.detuning matches central differences."""
    pytest.importorskip("dynamiqs")
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp

    chip = _single_qubit_chip(backend="dynamiqs")
    seq = QuantumSequence(chip)
    envelope = Gaussian(duration=20.0, sigmas=3, amplitude=0.0125)
    seq.charge("q", envelope=envelope, detuning=0.0)
    seq.charge("q", envelope=envelope, detuning=0.0, phase=0.4)
    tlist = jnp.linspace(0.0, 65.0, 3)
    method = dq.method.Tsit5(rtol=1e-10, atol=1e-12)

    def excited_population(detuning):
        rebound = seq.with_params({"pulse.0.detuning": detuning, "pulse.1.detuning": detuning,
                                   "pulse.1.start_time": 45.0})
        result = rebound.simulate(tlist=tlist, initial_state=chip.bare_state({"q": 0}), options={"method": method})
        return result.population("q", 1)[-1]

    detuning = jnp.asarray(0.006)
    gradient = jax.grad(excited_population)(detuning)
    step = 1e-5
    reference = (excited_population(detuning + step) - excited_population(detuning - step)) / (2 * step)
    assert abs(float(reference)) > 1.0
    np.testing.assert_allclose(float(gradient), float(reference), rtol=1e-6)
