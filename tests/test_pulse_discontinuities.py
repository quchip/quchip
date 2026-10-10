"""Pulse edges and envelope steps stay sharp, and short pulses retain their area and
gradients, independently of output sampling."""

import numpy as np
import pytest

from quchip import Envelope, Scalar, parameter, qnp

# Integration error at these tolerances stays below 1e-8 in the solves below.
_TIGHT_QUTIP_OPTIONS = {"atol": 1e-12, "rtol": 1e-10}


class _SteppedEnvelope(Envelope):
    """Amplitude a1 before t_step and a2 from t_step on, with t_step listed as a feature time."""

    duration: Scalar = parameter(positive=True, unit="ns")
    a1: Scalar = parameter(default=0.0)
    a2: Scalar = parameter(default=0.0)
    t_step: Scalar = parameter(default=0.0, unit="ns")

    def value(self, t):
        # Arithmetic on the comparison stays traceable and avoids a JAX compile per NumPy grid shape.
        return self.a2 + (self.a1 - self.a2) * (t < self.t_step) + 0j

    def sampling_times(self):
        return qnp.asarray([0.0, self.t_step, self.duration])


def _driven_transmon():
    from quchip import RWA, ChargeDrive, Chip, DuffingTransmon

    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q")
    chip = Chip([qubit], frame="rotating", approximation=RWA(), backend="qutip")
    chip.wire(ChargeDrive(qubit, label="xy"))
    return chip


def _problem(backend, amplitude=5.0, width=0.1, start=150.17):
    from quchip.engine.ir import (
        CanonicalOperator, Constant, DynamicTerm, EngineResult, ResolvedSLH,
        ScalarModulation, Shift, SolveProblem, Window,
    )
    operator = CanonicalOperator.from_dense(np.array([[0, 1], [1, 0]], dtype=complex),
                                           dims=(2,), basis="fock", subsystem_labels=("q",))
    signal = Shift(Window(Constant(amplitude), start=0.0, stop=width), delta_t=start)
    engine = EngineResult(slh=ResolvedSLH.from_terms(static_terms=(), collapse_terms=(), dynamic_terms=(
        DynamicTerm(operator=operator, time_dependence=ScalarModulation(signal), origin="drive"),
    )), dims=(2,), metadata={"max_step_ns": 0.025})
    options = {"rtol": 1e-9, "atol": 1e-11}
    from quchip.backend.dynamiqs import DynamiqsBackend

    if isinstance(backend, DynamiqsBackend):
        import dynamiqs as dq
        options = {"method": dq.method.Tsit5(rtol=1e-9, atol=1e-11)}
    return SolveProblem(chip=None, backend=backend, engine_result=engine,
                        initial_state=backend.basis(2, 0), tlist=np.array([0.0, 308.0]),
                        states="final", options=options)


@pytest.mark.parametrize("backend_name", ["qutip", "dynamiqs"])
def test_short_square_pulse_is_resolved_on_two_point_grid(backend_name):
    from quchip import Chip
    backend = Chip([], backend=backend_name).backend
    problem = _problem(backend)
    result = backend.solve_problem(problem)
    excited = abs(backend.to_array(result.final_state)[1, 0]) ** 2
    np.testing.assert_allclose(excited, np.sin(0.5) ** 2, atol=2e-7)
    np.testing.assert_array_equal(result.times, [0.0, 308.0])


@pytest.mark.validation
def test_short_pulse_width_and_amplitude_gradients_survive_jit():
    import jax
    import jax.numpy as jnp
    from quchip.backend.dynamiqs import DynamiqsBackend

    backend = DynamiqsBackend()

    @jax.jit
    @jax.value_and_grad
    def objective(parameters):
        problem = _problem(backend, amplitude=parameters[0], width=parameters[1])
        result = backend.solve_problem(problem)
        return jnp.abs(backend.to_array(result.final_state)[1, 0]) ** 2

    value, gradient = objective(jnp.array([5.0, 0.1]))
    np.testing.assert_allclose(value, np.sin(0.5) ** 2, atol=2e-7)
    np.testing.assert_allclose(gradient, np.sin(1.0) * np.array([0.1, 5.0]), atol=2e-6)


@pytest.mark.validation
def test_native_batch_preserves_distinct_pulse_edges_and_gradients():
    import jax
    import jax.numpy as jnp
    from quchip.backend.dynamiqs import DynamiqsBackend
    from quchip.engine.ir import SolveBatch

    backend = DynamiqsBackend()

    @jax.jit
    @jax.value_and_grad
    def objective(widths):
        problems = tuple(_problem(backend, width=widths[i], start=start)
                         for i, start in enumerate([150.17, 200.17]))
        results = backend.solve_batch(SolveBatch(chip=None, problems=problems), progress=False)
        return sum(jnp.abs(backend.to_array(result.final_state)[1, 0]) ** 2 for result in results)

    value, gradient = objective(jnp.array([0.1, 0.2]))
    np.testing.assert_allclose(value, np.sum(np.sin([0.5, 1.0]) ** 2), atol=2e-7)
    np.testing.assert_allclose(gradient, 5 * np.sin([1.0, 2.0]), atol=2e-6)


def test_qutip_square_pulse_has_no_interpolation_area_outside_support():
    from quchip.backend.qutip import _envelope_coefficient
    from quchip.engine.ir import Constant, Shift, Window

    start, width = 150.17, 0.1
    signal = Shift(Window(Constant(1.0), start=0.0, stop=width), delta_t=start)
    coefficient = _envelope_coefficient(signal, [0.0, 308.0])
    assert abs(coefficient(start - 0.01)) < 1e-12
    assert abs(coefficient(start + width + 0.01)) < 1e-12
    times = np.linspace(start, start + width, 1001)
    area = np.trapezoid([coefficient(time).real for time in times], times)
    assert area == pytest.approx(width, abs=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize("delay", [None, -0.2])
def test_qutip_coefficient_keeps_a_listed_envelope_step_sharp_at_any_start(delay):
    """The QuTiP coefficient equals a stepped envelope 1 fs on each side of its listed step, for every start time."""
    from quchip.backend.qutip import _envelope_coefficient
    from quchip.engine.ir import EnvelopeRef, Shift, Window

    envelope = _SteppedEnvelope(duration=20.0, a1=1.0, a2=0.5, t_step=10.0)
    # Without a delay, the adjacent float below the shifted step maps back onto the step at 22 of these
    # starts, e.g. 2.3 ns.
    for start in np.arange(300) / 100:
        signal = Shift(Window(EnvelopeRef(envelope), 0.0, 20.0), start)
        step = start + 10.0
        if delay is not None:  # A line delay shifts the pulse a second time.
            signal, step = Shift(signal, delay), step + delay
        coefficient = _envelope_coefficient(signal, [0.0, step + 11.0])
        # A step without close knots ramps across a 25 ps interval, so 1 fs away it is near the other level.
        assert coefficient(step - 1e-6) == pytest.approx(1.0, abs=1e-12)
        assert coefficient(step + 1e-6) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("start", [0.0, 0.5, 2.3, 37.3, 100.0])
def test_qutip_listed_envelope_step_evolves_like_two_abutting_squares(start):
    """A pulse with a listed internal step reaches the final state of two abutting squares at any start time."""
    from quchip import QuantumSequence, Square

    chip = _driven_transmon()
    freq = chip.freq("q")
    amplitude = 0.25 / (15.0 * abs(complex(chip.drive_matrix_elements("q")["xy"])))

    def final_state(*envelopes):
        sequence = QuantumSequence(chip)
        sequence.schedule("xy", envelope=envelopes[0], freq=freq, start_time=start)
        for envelope in envelopes[1:]:
            sequence.schedule("xy", envelope=envelope, freq=freq)
        result = sequence.simulate(tlist=[0.0, start + 20.0], states="final", options=_TIGHT_QUTIP_OPTIONS)
        return result.final_state.full().ravel()

    stepped = final_state(_SteppedEnvelope(duration=20.0, a1=amplitude, a2=amplitude / 2, t_step=10.0))
    reference = final_state(Square(duration=10.0, amplitude=amplitude),
                            Square(duration=10.0, amplitude=amplitude / 2))
    # A 25 ps ramp across the step moves the final state by 2.3e-4. The integration error stays below 1e-8.
    np.testing.assert_allclose(stepped, reference, rtol=0.0, atol=1e-6)


def test_qutip_pulse_at_zero_solves_from_negative_time_without_warnings():
    """A pulse edge at t = 0 adds no subnormal knot, so a solve from -1 ns gives no RuntimeWarning."""
    import warnings

    from quchip import QuantumSequence, Square

    chip = _driven_transmon()
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Square(duration=10.0, amplitude=0.02), freq=chip.freq("q"), start_time=0.0)
    excited = []
    for t0 in (0.0, -1.0):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = sequence.simulate(tlist=[t0, 10.0], states="all", options=_TIGHT_QUTIP_OPTIONS)
        excited.append(result.population("q", 1)[-1])
    # The idle ground state keeps its population before the pulse, so only integration error separates the solves.
    np.testing.assert_allclose(excited[1], excited[0], rtol=0.0, atol=1e-9)


@pytest.mark.validation
def test_native_batch_can_mix_absent_and_dense_static_terms():
    from dataclasses import replace
    from quchip.backend.dynamiqs import DynamiqsBackend
    from quchip.engine.ir import CanonicalOperator, ResolvedSLH, SolveBatch, StaticTerm

    backend = DynamiqsBackend()
    first = _problem(backend)
    detuning = 0.3
    operator = CanonicalOperator.from_dense(np.diag([0.0, detuning]).astype(complex),
                                           dims=(2,), basis="fock", subsystem_labels=("q",))
    second = replace(first, engine_result=replace(first.engine_result, slh=ResolvedSLH.from_terms(
        static_terms=(StaticTerm(operator=operator, coefficient=1.0),),
        dynamic_terms=first.engine_result.dynamic_terms, collapse_terms=(),
    )))
    results = backend.solve_batch(SolveBatch(chip=None, problems=(first, second)), progress=False)
    population = [abs(backend.to_array(result.final_state)[1, 0]) ** 2 for result in results]
    omega = np.sqrt(25 + (detuning / 2) ** 2)
    np.testing.assert_allclose(population, [np.sin(0.5) ** 2, 25 / omega**2 * np.sin(0.1 * omega)**2],
                               atol=2e-7)
