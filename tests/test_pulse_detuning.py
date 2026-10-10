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


def test_detuned_sx_pair_makes_an_x_at_every_gap():
    """Two start-referenced detuned SX pulses make an X at any gap, and a fixed oscillator keeps its meaning."""
    # The two-transmon chip of issue #93.
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q1")
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003)], frame="rotating",
                approximation=RWA(), backend="qutip")
    chip.wire(ChargeDrive(q0, label="xy0"))
    f0, element = float(chip.freq("q0")), complex(chip.drive_matrix_elements("q0")["xy0"])
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
    np.testing.assert_allclose(sx_pair(50.0, "fixed").final_state.full(),
                               sx_pair(50.0, "fixed by hand").final_state.full(), atol=1e-6)


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
