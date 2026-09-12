"""Saved observables and state retention have independent public contracts."""

import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence


def _sequence(backend):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0, label="q")
    return QuantumSequence(Chip([q], backend=backend, frame="rotating")), q


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("storage", ["all", "final", "none"])
@pytest.mark.parametrize("batch", [False, True])
def test_observable_history_is_independent_of_state_storage(backend, storage, batch):
    seq, q = _sequence(backend)
    times = np.linspace(0.0, 2.0, 5)
    kwargs = dict(tlist=times, states=storage, e_ops={"q": q.number_operator()})
    if batch:
        results = seq.simulate_batch(seq.vary("initial_state", [{"q": 1}, {"q": 0}]), **kwargs)
    else:
        results = [seq.simulate(initial_state={"q": 1}, **kwargs)]
    for index, result in enumerate(results):
        assert result.stats["states"] == storage
        expected = np.exp(-times / 10.0) if index == 0 else np.zeros_like(times)
        np.testing.assert_allclose(result.expect("q"), expected, atol=2e-6)
        if storage == "all":
            assert len(result.states) == len(times)
            np.testing.assert_allclose(result._backend.to_array(result.final_state),
                                       result._backend.to_array(result.states[-1]))
        else:
            with pytest.raises(RuntimeError, match='states="all"'):
                _ = result.states
        if storage == "none":
            with pytest.raises(RuntimeError, match="final state"):
                _ = result.final_state
        else:
            assert np.asarray(result._backend.to_array(result.final_state))[1, 1].real == pytest.approx(
                expected[-1], abs=2e-6
            )


@pytest.mark.parametrize("storage", ["auto", True])
def test_invalid_state_storage_is_rejected(storage):
    seq, _ = _sequence("qutip")
    with pytest.raises(ValueError, match="states"):
        seq.build_problem([0.0, 1.0], states=storage)


@pytest.mark.parametrize("flag", ["store_states", "store_final_state"])
def test_public_storage_flags_have_one_replacement(flag):
    seq, _ = _sequence("qutip")
    with pytest.raises(ValueError, match="states"):
        seq.build_problem([0.0, 1.0], options={flag: False})


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_final_state_lookup_requires_a_retained_time(backend):
    seq, _ = _sequence(backend)
    result = seq.simulate(tlist=[0.0, 1.0, 2.0], states="final")
    np.testing.assert_array_equal(result._backend.to_array(result.state_at(2.0)),
                                  result._backend.to_array(result.final_state))
    with pytest.raises(RuntimeError, match='states="all"'):
        result.state_at(1.0)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("noisy", [False, True])
def test_batch_final_projections_do_not_require_histories(backend, noisy):
    seq, _ = _sequence(backend)
    batch = seq.simulate_batch(seq.vary("initial_state", [{"q": 1}, {"q": 0}]),
                               tlist=[0.0, 1.0], states="final", dissipation=noisy)
    target = seq._chip.backend.basis(2, 1)
    targets = [target, target]
    np.testing.assert_allclose(batch.final_overlap_magnitudes(targets),
                               [np.exp(-0.1) if noisy else 1.0, 0.0], atol=2e-6)
    if noisy:
        with pytest.raises(TypeError, match="ket"):
            batch.final_amplitudes(targets)
    else:
        np.testing.assert_allclose(np.abs(batch.final_amplitudes(targets)), [1.0, 0.0], atol=2e-6)


def test_state_lookup_is_exact_by_default_with_explicit_nearest():
    seq, _ = _sequence("qutip")
    result = seq.simulate(tlist=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="saved time"):
        result.state_at(0.9)
    assert result.state_at(0.9, method="nearest") is result.states[1]
    assert result.state_at(1.0) is result.states[1]
    for time in [-1.0, 3.0, np.nan]:
        with pytest.raises(ValueError, match="interval"):
            result.state_at(time, method="nearest")
    with pytest.raises(ValueError, match="method"):
        result.state_at(1.0, method="interpolate")
