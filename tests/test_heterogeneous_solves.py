"""Independent captured calculations can share a request collection."""

import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence, solve_many


def _problem(backend, levels, t1, *, states="all", times=(0.0, 1.0, 3.0)):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=levels, T1=t1, label="q")
    chip = Chip([q], backend=backend, frame="rotating")
    return QuantumSequence(chip).build_problem(times, initial_state={"q": 1},
                                               e_ops={"q": q.number_operator()}, states=states)


@pytest.mark.validation
@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_independent_models_preserve_dimensions_physics_and_order(backend):
    problems = [_problem(backend, levels, t1) for levels, t1 in [(4, 20.0), (2, 10.0), (3, 30.0)]]
    result = solve_many(problems, progress=False)
    for point, levels, t1 in zip(result, (4, 2, 3), (20.0, 10.0, 30.0)):
        assert point.dims == [levels]
        np.testing.assert_allclose(point.expect("q"), np.exp(-np.asarray(point.times) / t1), atol=2e-6)
    assert result.expect("q").shape == (3, 3)


def test_mixed_native_array_backends_are_preserved_without_implicit_conversion():
    problems = [_problem(backend, 2, 10.0) for backend in ("qutip", "dynamiqs", "qutip")]
    results = solve_many(problems, progress=False)
    for problem, result in zip(problems, results):
        assert result._backend is problem.backend
        np.testing.assert_allclose(result.expect("q"), np.exp(-np.asarray(result.times) / 10), atol=2e-6)
    with pytest.raises(ValueError, match="different native array backends"):
        results.expect("q")
    with pytest.raises(ValueError, match="different native array backends"):
        _ = results.backend


def test_independent_requests_keep_storage_and_solver_options():
    problems = [_problem("qutip", 2, 10.0, states=states) for states in ("none", "final", "all")]
    results = solve_many(problems, progress=False)
    assert [point.stats["states"] for point in results] == ["none", "final", "all"]
    assert results.expect("q").shape == (3, 3)
    with pytest.raises(RuntimeError, match="final state"):
        _ = results[0].final_state
    assert results[1].final_state is not None
    assert len(results[2].states) == 3
