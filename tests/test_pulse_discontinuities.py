"""Pulse edges and envelope steps stay sharp, and short pulses retain their area and
gradients, independently of output sampling."""

import numpy as np
import pytest

from quchip import Envelope, Scalar, parameter, qnp


class _SteppedEnvelope(Envelope):
    """Amplitude 1 before 10 ns and 0.5 from 10 ns on, with 10 ns listed as a feature time."""

    duration: Scalar = parameter(positive=True, unit="ns")

    def value(self, t):
        return 1.0 - 0.5 * (t >= 10.0) + 0j

    def sampling_times(self):
        return qnp.asarray([0.0, 10.0, self.duration])


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
def test_qutip_coefficient_keeps_a_listed_envelope_step_sharp():
    """The QuTiP coefficient equals a stepped envelope 1 fs on each side of its listed step."""
    from quchip.backend.qutip import _envelope_coefficient
    from quchip.engine.ir import EnvelopeRef, Shift, Window

    # At a 2.3 ns start, the adjacent float below the shifted step maps back onto the step.
    start = 2.3
    signal = Shift(Window(EnvelopeRef(_SteppedEnvelope(duration=20.0)), 0.0, 20.0), start)
    coefficient = _envelope_coefficient(signal, [0.0, start + 20.0])
    assert coefficient(start + 10.0 - 1e-6) == pytest.approx(1.0, abs=1e-12)
    assert coefficient(start + 10.0 + 1e-6) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.unit
def test_qutip_coefficient_of_a_pulse_at_zero_builds_from_negative_time_without_warnings():
    """A pulse edge at t = 0 stays sharp on a grid from -1 ns, and the coefficient builds without a RuntimeWarning."""
    import warnings

    from quchip.backend.qutip import _envelope_coefficient
    from quchip.engine.ir import Constant, Window

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        coefficient = _envelope_coefficient(Window(Constant(1.0), start=0.0, stop=10.0), [-1.0, 10.0])
    assert coefficient(-1e-6) == pytest.approx(0.0, abs=1e-12)
    assert coefficient(1e-6) == pytest.approx(1.0, abs=1e-12)


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
