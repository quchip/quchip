"""Start-referenced carrier detuning of scheduled pulses."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from quchip import (
    ChargeDrive,
    Chip,
    DuffingTransmon,
    FluxDrive,
    Gaussian,
    QuantumSequence,
    Qubit,
    RWA,
    Square,
)
from quchip.engine.ir import evaluate_signal_program


@pytest.mark.unit
def test_detuning_sweeps_from_its_default_and_needs_a_carrier():
    """pulse.vary("detuning") from an undetuned pulse builds each rebound Hamiltonian. A baseband pulse rejects it."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", approximation=RWA())
    chip.wire(ChargeDrive(q, label="xy"), FluxDrive(q, label="z"))
    seq = QuantumSequence(chip)
    pulse = seq.charge("q", envelope=Square(duration=20.0, amplitude=0.01))
    values, times = [0.0, 0.004], np.linspace(0.0, 20.0, 5)

    def hamiltonian(problem, t):
        return sum(np.asarray(term.operator.to_dense()) * evaluate_signal_program(term.time_dependence.signal, t)
                   for term in problem.engine_result.dynamic_terms)

    batch = seq.build_batch(pulse.vary("detuning", values), tlist=times)
    for problem, value in zip(batch.problems, values):
        rebound = seq.with_params({"pulse.0.detuning": value}).build_problem(times)
        for t in times:
            np.testing.assert_allclose(hamiltonian(problem, t), hamiltonian(rebound, t), rtol=1e-12, atol=1e-15)
    # At t = 10 ns, the detuning turns the drive phase by 2π (0.004 GHz)(10 ns), about 0.25 rad.
    assert not np.allclose(hamiltonian(batch.problems[0], 10.0), hamiltonian(batch.problems[1], 10.0))

    seq.flux("q", envelope=Square(duration=10.0, amplitude=0.01))
    with pytest.raises(ValueError, match="A detuning requires a carrier"):
        seq.with_params({"pulse.1.detuning": 0.001}).scheduled_ops
    with pytest.raises(ValueError, match="A detuning requires a carrier"):
        seq.schedule("z", envelope=Square(duration=10.0, amplitude=0.01), detuning=0.001)


def test_detuned_pulse_pair_matches_analytic_evolution_at_every_gap():
    """Start-referenced square pulses have the same analytic propagator at each start time."""
    q = Qubit(freq=5.0, label="q")
    chip = Chip([q], frame="rotating", approximation=RWA(), backend="qutip")
    chip.wire(ChargeDrive(q, label="xy"))
    duration, amplitude, detuning = 20.0, 0.017, 0.006
    pulse = Square(duration=duration, amplitude=amplitude)
    options = {"atol": 1e-11, "rtol": 1e-9}

    # In the qubit frame, H01 = iπ A exp(2πi δτ). Transforming with
    # R(τ) = diag(1, exp(-2πi δτ)) gives Hc = π A i(a-a†) - 2π δ n.
    # The propagator R(T) exp(-i Hc T) is exact for the declared two-level RWA.
    number = np.diag([0.0, 1.0])
    rotating = np.diag([1.0, np.exp(-2j * np.pi * detuning * duration)])
    constant = np.pi * amplitude * np.array([[0, 1j], [-1j, 0]]) - 2 * np.pi * detuning * number
    propagator = rotating @ expm(-1j * constant * duration)
    reference = propagator @ propagator @ np.array([1.0, 0.0])

    def final_state(gap, carrier):
        seq = QuantumSequence(chip)
        for start in (0.0, duration + gap):
            if carrier == "fixed":
                seq.schedule("xy", envelope=pulse, freq=q.freq + detuning, start_time=start)
            else:
                phase = -2 * np.pi * detuning * start if carrier == "fixed by hand" else 0.0
                seq.schedule("xy", envelope=pulse, freq=q.freq, detuning=detuning,
                             phase=phase, start_time=start)
        result = seq.simulate(tlist=[0.0, 2 * duration + gap], states="final", options=options)
        return result.final_state.full().ravel()

    # Idle evolution vanishes in this frame. The bound allows accumulated solver error.
    for gap in (0.0, 13.0, 50.0):
        np.testing.assert_allclose(final_state(gap, "start"), reference, atol=1e-7, rtol=0)
    # A fixed oscillator has phase -2πδ t0 relative to a start-referenced pulse.
    np.testing.assert_allclose(final_state(50.0, "fixed"), final_state(50.0, "fixed by hand"),
                               atol=1e-7, rtol=0)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_detuning_gradient_matches_finite_differences():
    """The JAX derivative of P(1) with respect to a rebound pulse.<i>.detuning matches central differences."""
    pytest.importorskip("dynamiqs")
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", approximation=RWA(), backend="dynamiqs")
    chip.wire(ChargeDrive(q, label="xy"))
    seq = QuantumSequence(chip)
    envelope = Gaussian(duration=20.0, sigmas=3, amplitude=0.0125)
    seq.charge("q", envelope=envelope)
    seq.charge("q", envelope=envelope, phase=0.4)
    tlist = jnp.linspace(0.0, 65.0, 3)
    method = dq.method.Tsit5(rtol=1e-10, atol=1e-12)

    def excited_population(detuning):
        # Both pulses start without a detuning, so rebinding adds the offset.
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
