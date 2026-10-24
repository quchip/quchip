"""Point-specific observables must not require temporary state histories."""

from dataclasses import replace

import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence
from quchip.engine import solve_batch
from quchip.engine.ir import CanonicalOperator, SolveBatch, StaticTerm


@pytest.mark.validation
def test_batch_keeps_distinct_applied_static_hamiltonians():
    q = DuffingTransmon(freq=1 / (2 * np.pi), anharmonicity=-0.2, levels=2, label="q")
    # The applied X term and the expected detuned Rabi oscillation are lab-frame quantities.
    chip = Chip([q], frame="lab")
    times = np.linspace(0.0, 1.0, 11)
    problem = QuantumSequence(chip).build_problem(times, e_ops={"q": q.number_operator()}, states="none")
    x = CanonicalOperator.from_dense(np.array([[0.0, 1.0], [1.0, 0.0]]),
        dims=(2,), basis="solver", subsystem_labels=("q",))
    strengths = (0.2, 0.8)
    problems = tuple(replace(problem, engine_result=problem.engine_result.with_applied_hamiltonian_terms(
        static_terms=(StaticTerm(x, strength),))) for strength in strengths)
    results = solve_batch(SolveBatch(chip=chip, problems=problems), progress=False)
    for strength, result in zip(strengths, results):
        frequency = np.sqrt(strength**2 + 0.25)
        expected = (strength / frequency * np.sin(frequency * times)) ** 2
        np.testing.assert_allclose(result.expect("q"), expected, atol=2e-6)


def _batch(states, noisy=True, scales=(1.0, 2.0)):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0 if noisy else None, label="q")
    chip = Chip([q], backend="dynamiqs", frame="rotating")
    problem = QuantumSequence(chip).build_problem(np.linspace(0.0, 3.0, 41),
        initial_state={"q": 1}, e_ops={"q": q.number_operator()}, states=states)
    problems = tuple(replace(problem, e_ops=[scale * op for op in problem.e_ops]) for scale in scales)
    return SolveBatch(chip=chip, problems=problems)


@pytest.mark.parametrize("noisy, states", [(False, "all"), (True, "final"), (True, "none")])
def test_point_observables_obey_native_storage_choice(states, noisy):
    result = solve_batch(_batch(states, noisy), progress=False)
    for scale, point in zip((1.0, 2.0), result):
        expected = scale * np.exp(-np.asarray(point.times) / 10) if noisy else scale
        np.testing.assert_allclose(point.expect("q"), expected, atol=3e-6)
        assert bool(point.stats["options"]["save_states"]) == (states == "all")
        if states != "all":
            with pytest.raises(RuntimeError, match="Full state history"):
                _ = point.states
        if states != "none":
            assert point.final_state is not None
