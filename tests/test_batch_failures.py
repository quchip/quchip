"""Numerical batch failures retain point coordinates without retrying failed work."""

import warnings

import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence, ChargeDrive, Square, solve_many
from quchip.engine import solve_batch


def _batch(*, grid=False, backend="qutip", amplitude=100.0):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    chip = Chip([q], frame="rotating", backend=backend)
    drive = ChargeDrive(q)
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(drive, envelope=Square(duration=1.0, amplitude=0.0), freq=5.0)
    axes = [pulse.vary("amplitude", [0.0, amplitude] if grid else [0.0, amplitude, 0.0], name="amp")]
    if grid:
        axes.append(pulse.vary("freq", [5.0, 5.1]))
    return sequence.build_batch(*axes, tlist=[0.0, 0.1, 1.0], e_ops={"q": q.number_operator()},
                                options={"nsteps" if backend == "qutip" else "max_steps": 50},
                                states="none")


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_native_failure_reports_flat_index_parameters_and_cause(backend):
    batch = _batch(grid=True, backend=backend)
    with pytest.raises(RuntimeError, match="Batch point 2 failed") as caught:
        solve_batch(batch, progress=False)
    error = caught.value
    assert error.index == 2
    assert error.parameters == {"amp": 100.0, "freq": 5.0}
    assert ("Excess work" if backend == "qutip" else "maximum number of solver steps") in str(error)
    assert error.__cause__ is not None


def test_numerical_worker_failure_is_not_retried_sequentially(monkeypatch):
    batch = _batch()
    backend = batch.problems[0].backend
    calls = []
    solve = backend.sesolve

    def counted(**kwargs):
        calls.append(len(calls))
        return solve(**kwargs)

    class Executor:
        def map(self, task, items):
            return map(task, items)

    monkeypatch.setattr(backend, "sesolve", counted)
    monkeypatch.setattr(backend, "_PARALLEL_MIN_BATCH", 1)
    monkeypatch.setattr(backend, "_get_executor", lambda _: Executor())
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="Batch point 1 failed"):
            solve_batch(batch, progress=False)
    assert len(calls) == 2
    assert not any("falling back" in str(message.message) for message in messages)


def test_failure_index_survives_heterogeneous_backend_grouping():
    from tests.test_heterogeneous_solves import _problem

    independent = _problem("dynamiqs", 2, 10.0)
    batch = _batch()
    problems = [independent, batch[0], batch[2], batch[1]]
    with pytest.raises(RuntimeError, match="Batch point 3 failed") as caught:
        solve_many(problems, progress=False)
    assert caught.value.index == 3
    assert "Excess work" in str(caught.value)


def test_actual_worker_error_preserves_point_context_and_pool_usability(monkeypatch):
    batch = _batch()
    backend = batch.problems[0].backend
    executor = backend._get_executor(2)
    monkeypatch.setattr(backend, "_PARALLEL_MIN_BATCH", 2)
    monkeypatch.setattr(type(backend), "_get_executor", staticmethod(lambda _: executor))
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="Batch point 1 failed") as caught:
            solve_batch(batch, progress=False)
    assert caught.value.parameters == {"amp": 100.0}
    assert not any("falling back" in str(message.message) for message in messages)
    results = solve_many([batch[0], batch[2]], progress=False)
    assert len(results) == 2
    assert all(abs(point.expect("q")[-1]) < 1e-12 for point in results)


def test_incomplete_backend_output_is_never_a_partial_batch(monkeypatch):
    batch = _batch()
    monkeypatch.setattr(batch.problems[0].backend, "solve_batch", lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match="returned 0 results for 3 batch points"):
        solve_batch(batch, progress=False)


@pytest.mark.parametrize("consumer", ["value_and_grad", "grad", "vmap"])
def test_dynamiqs_failure_keeps_runtime_parameters_under_jit_and_gradient(consumer):
    import jax
    import jax.numpy as jnp

    def objective(amplitude):
        batch = _batch(backend="dynamiqs", amplitude=amplitude)
        result = solve_batch(batch, progress=False)
        return jnp.real(result.expect("q", reduce="last").sum())

    evaluate = jax.jit(jax.value_and_grad(objective))
    value, derivative = evaluate(jnp.asarray(0.1))
    import numpy as np
    assert value == pytest.approx(np.sin(np.pi * 0.1) ** 2, abs=2e-6)
    assert derivative == pytest.approx(np.pi * np.sin(2 * np.pi * 0.1), abs=2e-6)
    if consumer == "grad":
        evaluate = jax.jit(jax.grad(objective))
        arguments = jnp.asarray(100.0)
    elif consumer == "vmap":
        evaluate = jax.jit(jax.vmap(jax.value_and_grad(objective)))
        np.testing.assert_allclose(evaluate(jnp.asarray([0.1, 0.2]))[0],
                                   np.sin(np.pi * np.asarray([0.1, 0.2])) ** 2, atol=2e-6)
        arguments = jnp.asarray([0.1, 100.0])
    else:
        arguments = jnp.asarray(100.0)
    with pytest.raises(Exception, match="Batch point 1 failed.*amp.*100.0.*maximum number of solver steps"):
        jax.block_until_ready(evaluate(arguments))


@pytest.mark.parametrize("consumer", ["value", "grad", "constant"])
def test_compiled_heterogeneous_failure_uses_original_collection_index(consumer):
    import jax
    import jax.numpy as jnp
    from tests.test_heterogeneous_solves import _problem

    def objective(amplitude):
        batch = _batch(backend="dynamiqs", amplitude=amplitude)
        independent = _problem("dynamiqs", 3, 10.0)
        results = solve_many([independent, batch[0], batch[2], batch[1]],
                              progress=False)
        return jnp.asarray(7.0) if consumer == "constant" else jnp.real(results[0].expect("q")[-1])

    evaluate = jax.jit(jax.grad(objective) if consumer == "grad" else objective)
    evaluate(jnp.asarray(0.1)).block_until_ready()
    with pytest.raises(Exception, match="Batch point 3 failed.*maximum number of solver steps"):
        evaluate(jnp.asarray(100.0)).block_until_ready()


def test_compiled_varying_grid_failure_keeps_sweep_parameters():
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    import numpy as np

    @jax.jit
    def objective(amplitude):
        batch = _batch(backend="dynamiqs", amplitude=amplitude)
        batch = replace(batch, problems=(batch[0], replace(batch[1], tlist=np.array([0.0, 0.1, 2.0])), batch[2]))
        result = solve_batch(batch, progress=False)
        return jnp.real(result.expect("q", reduce="last").sum())

    objective(jnp.asarray(0.1)).block_until_ready()
    with pytest.raises(Exception, match="Batch point 1 failed.*amp.*100.0.*maximum number of solver steps"):
        objective(jnp.asarray(100.0)).block_until_ready()
