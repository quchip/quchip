"""Truncation checks measure sampled physical boundaries, independently of storage."""

import warnings

import numpy as np
import pytest

from quchip import Chip, QuantumSequence, check_truncation, with_truncation
from quchip.engine import solve_problem, solve_many
from quchip.declarative import DeviceModel, parameter


class DrivenLadder(DeviceModel):
    rate: float = parameter(default=0.25)

    def local_hamiltonian(self, op, p):
        return p.rate * (op.a + op.adag)


class FiniteQubit(DrivenLadder):
    def truncation_boundary(self):
        return None


@pytest.mark.validation
@pytest.mark.parametrize("storage", ["all", "final", "none"])
def test_boundary_excursion_is_checked_before_return_to_ground(storage):
    backend = "qutip"
    q = DrivenLadder(levels=2, label="q")
    q.reference_freq = 0.0
    chip = Chip([q], backend=backend, frame="lab")
    times = np.linspace(0.0, 2.0, 5)
    with warnings.catch_warnings(record=True) as caught:
        result = solve_problem(with_truncation(QuantumSequence(chip).build_problem(
            times, initial_state=chip.backend.basis(2, 0), states=storage,
            e_ops={"q": q.number_operator()},
        )))
    assert not any("boundary" in str(w.message) for w in caught)
    with pytest.warns(UserWarning, match="q.*sampled"):
        check_truncation(result)
    observed = check_truncation(result, threshold=2.0)
    assert observed["q"] == pytest.approx(1.0, abs=2e-6)
    np.testing.assert_allclose(result.expect("q"), np.sin(0.5 * np.pi * times) ** 2, atol=2e-6)
    assert result.observable_traces.keys() == {"q"}


@pytest.mark.validation
def test_intrinsically_finite_qubit_has_no_cutoff_warning():
    chip = Chip([FiniteQubit(label="q")], frame="lab")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solve_problem(with_truncation(QuantumSequence(chip).build_problem(
            [0.0, 1.0], initial_state=chip.backend.basis(2, 0), states="none",
        )))
    assert not [warning for warning in caught if "truncation" in str(warning.message).lower()]
    assert check_truncation(result) == {}


def test_result_method_warns_and_returns_the_function_report():
    """The deprecated method calls quchip.check_truncation, which rejects a request that has no solve."""
    chip = Chip([DrivenLadder(label="q")], frame="lab")
    sequence = QuantumSequence(chip)
    result = sequence.simulate([0.0, 1.0], initial_state=chip.backend.basis(2, 0))
    with pytest.warns(DeprecationWarning, match=r"quchip\.check_truncation\(result\).*quchip 0\.6"):
        assert result.check_truncation(threshold=2.0) == check_truncation(result, threshold=2.0)
    with pytest.raises(TypeError, match="not SolveProblem"):
        check_truncation(sequence.build_problem([0.0, 1.0]))


def test_unretained_diagnostic_explains_what_to_request():
    chip = Chip([DrivenLadder(label="q")], frame="lab")
    result = QuantumSequence(chip).simulate([0.0, 1.0], states="none")
    with pytest.raises(RuntimeError, match="with_truncation"):
        check_truncation(result)


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.validation)])
def test_batched_diagnostics_do_not_need_state_histories(backend):
    q = DrivenLadder(label="q")
    q.reference_freq = 0.0
    chip = Chip([q], backend=backend, frame="lab")
    sequence = QuantumSequence(chip)
    axis = sequence.vary("initial_state", [chip.backend.basis(2, 0), chip.backend.basis(2, 1)])
    with pytest.warns(UserWarning, match="maximum sampled"):
        batch = solve_many(with_truncation(sequence.build_batch(
            axis, tlist=[0.0, 1.0, 2.0], states="none", e_ops={"q": q.number_operator()})), progress=False)
        for result in batch:
            check_truncation(result)
    with pytest.raises(TypeError, match="each result of the batch"):
        check_truncation(batch)
    for result in batch:
        assert check_truncation(result, threshold=2.0)["q"] == pytest.approx(1.0, abs=2e-6)
        with pytest.raises(RuntimeError, match='states="all"'):
            _ = result.states
        if backend == "dynamiqs":
            assert not bool(result.stats["options"]["save_states"])


@pytest.mark.parametrize("frame", ["lab", "rotating"])
@pytest.mark.parametrize("reference", [0.0, 0.123])
def test_cutoff_population_is_independent_of_frame_and_readout_reference(frame, reference):
    from quchip import Exact

    q = DrivenLadder(label="q")
    q.reference_freq = reference
    chip = Chip([q], frame=frame, approximation=Exact())
    times = np.linspace(0.0, 2.0, 9)
    result = solve_problem(
        with_truncation(
            QuantumSequence(chip).build_problem(times, initial_state=chip.backend.basis(2, 0), states="none")
        )
    )
    np.testing.assert_allclose(result._boundary_traces[0], np.sin(0.5 * np.pi * times) ** 2, atol=2e-6)


@pytest.mark.parametrize("population", [1e-6, 0.01])
def test_warning_threshold_for_small_boundary_populations(population):
    from quchip import DuffingTransmon

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=4, label="q")
    chip = Chip([q], frame="lab")
    initial = np.array([np.sqrt(1 - population), 0.0, 0.0, np.sqrt(population)]).reshape(-1, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solve_problem(
            with_truncation(QuantumSequence(chip).build_problem([0.0, 0.01], initial_state=initial, states="none"))
        )
        check_truncation(result)
    matching = [warning for warning in caught if "maximum sampled" in str(warning.message)]
    assert bool(matching) == (population > 1e-3)
    assert check_truncation(result, threshold=2.0)["q"] == pytest.approx(population, abs=1e-10)


@pytest.mark.parametrize("space", ["charge", "phase"])
@pytest.mark.parametrize("edge", [0, 4])
def test_native_coordinate_cutoffs_cover_both_edges(space, edge):
    from quchip import ChargeSpace, PhaseGridSpace

    class CoordinateModel(DeviceModel):
        def local_space(self):
            return ChargeSpace(5) if space == "charge" else PhaseGridSpace(5, 3.0)

        def local_hamiltonian(self, op, p):
            return op.n if space == "charge" else op.phi

    chip = Chip([CoordinateModel(levels=5, label="q")], frame="lab")
    with pytest.warns(UserWarning, match=f"{space}.*edges"):
        result = solve_problem(
            with_truncation(
                QuantumSequence(chip).build_problem(
                    [0.0, 0.1], initial_state=chip.backend.basis(5, edge), states="none"
                )
            )
        )
        check_truncation(result)
    assert check_truncation(result, threshold=2.0)["q"] == pytest.approx(1.0)


def test_projected_space_checks_its_retained_energy_boundary():
    from quchip import ChargeBasisTransmon

    q = ChargeBasisTransmon(E_J=20.0, E_C=0.3, num_basis=15, levels=3, label="q")
    chip = Chip([q], basis="eigen", frame="lab")
    with pytest.warns(UserWarning, match="retained energy boundary"):
        result = solve_problem(
            with_truncation(QuantumSequence(chip).build_problem([0.0, 0.1], initial_state={"q": 2}, states="none"))
        )
        check_truncation(result)
    assert check_truncation(result, threshold=2.0)["q"] == pytest.approx(1.0)


def test_request_captures_boundary_before_source_changes():
    from quchip.engine import solve_problem

    q = DrivenLadder(label="q")
    chip = Chip([q], frame="lab")
    sequence = QuantumSequence(chip)
    problem = sequence.build_problem([0.0, 1.0, 2.0], initial_state=chip.backend.basis(2, 0), states="none")
    q.truncation_boundary = lambda: None
    with pytest.warns(UserWarning, match="maximum sampled"):
        result = solve_problem(with_truncation(problem))
        check_truncation(result)
    assert check_truncation(result, threshold=2.0)["q"] == pytest.approx(1.0)
    fresh = sequence.build_problem([0.0, 1.0], states="none")
    assert check_truncation(solve_problem(fresh)) == {}


def test_lazy_boundary_query_does_not_cache_a_tracer():
    import jax
    from quchip.utils.jax_utils import contains_tracer

    chip = Chip([DrivenLadder(label="q")], backend="dynamiqs", frame="lab")
    result = QuantumSequence(chip).simulate([0.0, 1.0, 2.0], initial_state=chip.backend.basis(2, 0))
    with pytest.warns(UserWarning, match="traced"):
        maximum = jax.jit(lambda: check_truncation(result, threshold=2.0)["q"])()
    assert float(maximum) == pytest.approx(1.0, abs=2e-6)
    assert not contains_tracer(result._boundary_traces)
    assert check_truncation(result, threshold=2.0)["q"] == pytest.approx(1.0, abs=2e-6)


def test_unknown_custom_cutoff_explains_unavailable_diagnostic():
    from quchip import CustomSpace

    class UnknownCutoff(DeviceModel):
        def local_space(self):
            return CustomSpace(2, {"n": np.diag([0.0, 1.0])})

        def local_hamiltonian(self, op, p):
            return op.n

    chip = Chip([UnknownCutoff(label="q")], frame="lab")
    with pytest.warns(UserWarning, match="diagnostic unavailable.*CustomSpace"):
        result = solve_problem(with_truncation(QuantumSequence(chip).build_problem([0.0, 1.0], states="none")))
        check_truncation(result)
    with pytest.warns(UserWarning, match="diagnostic unavailable"):
        assert check_truncation(result) == {}


def test_boundary_indices_capture_source_sequence_and_reject_double_counting():
    from quchip import TruncationBoundary

    indices = [1]
    boundary = TruncationBoundary(indices, "custom cutoff", "Compare larger cutoffs.")
    indices[0] = 0
    assert boundary.indices == (1,)
    with pytest.raises(ValueError, match="distinct"):
        TruncationBoundary((1, 1), "custom cutoff", "Compare larger cutoffs.")
