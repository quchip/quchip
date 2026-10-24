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


def test_native_failure_reports_flat_index_parameters_and_cause():
    batch = _batch(grid=True, backend="qutip")
    with pytest.raises(RuntimeError, match="Batch point 2 failed") as caught:
        solve_batch(batch, progress=False)
    error = caught.value
    assert error.index == 2
    assert error.parameters == {"amp": 100.0, "freq": 5.0}
    assert "Excess work" in str(error)
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


def test_incomplete_backend_output_is_never_a_partial_batch(monkeypatch):
    batch = _batch()
    monkeypatch.setattr(batch.problems[0].backend, "solve_batch", lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match="returned 0 results for 3 batch points"):
        solve_batch(batch, progress=False)
