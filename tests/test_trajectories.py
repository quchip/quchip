"""Native trajectory execution preserves weights, records, and logical point noise."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from quchip import Chip, QuantumSequence, Resonator, with_truncation
from quchip.engine import solve_many, solve_problem


def jump_problem(backend, *, storage=None, count=16):
    """Prepare a decaying single excitation with reproducible native noise."""
    r = Resonator(freq=1.0, levels=2, label="r", T1=1.0)
    chip = Chip([r], backend=backend, frame="rotating")
    if backend == "qutip":
        solver, args = "mcsolve", {"ntraj": count, "seeds": 42}
        options = {"progress_bar": "", "keep_runs_results": True}
    else:
        dq = pytest.importorskip("dynamiqs")
        import jax
        solver = "jssesolve"
        args = {"keys": jax.random.split(jax.random.key(42), count), "method": dq.method.Event(dtmax=0.02)}
        options = {}
    seq = QuantumSequence(chip)
    return seq, seq.build_problem([0., 0.5, 1.], solver=solver, run_args=args, options=options,
                                  states=storage, initial_state={"r": 1}, e_ops=chip.e_ops(r="n"))


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_jumps_match_native_events_and_weighted_averages(backend):
    """The adapter preserves native seeds, events, and ensemble averaging."""
    _, problem = jump_problem(backend, storage="all")
    result = solve_problem(problem)
    owner = problem.backend
    rhs = owner.prepare_hamiltonian(problem.engine_result, problem.tlist).rhs
    if backend == "qutip":
        import qutip
        native = qutip.mcsolve(rhs, problem.initial_state, problem.tlist,
                              owner._collapse_operators(problem.engine_result), e_ops=problem.e_ops,
                              options={**problem.options, "store_states": True, "store_final_state": False},
                              **problem.run_args)
        assert result.native.col_times == native.col_times
        expected = native.average_expect[0]
    else:
        import dynamiqs as dq
        native = dq.jssesolve(rhs, owner._collapse_operators(problem.engine_result), problem.initial_state,
                              problem.tlist, exp_ops=problem.e_ops, **problem.run_args)
        np.testing.assert_allclose(result.native.clicktimes, native.clicktimes)
        expected = native.mean_expects()[0]
    np.testing.assert_allclose(result.expect("r"), expected)
    assert len(result.run(0).states) == 3
    assert result.average().final_state.shape == (2, 2)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_jump_decay_and_explicit_per_run_truncation(backend):
    """Jump ensembles follow exponential decay and diagnostics check individual runs."""
    _, problem = jump_problem(backend, storage="none", count=128)
    with warnings.catch_warnings(record=True) as caught:
        result = solve_problem(with_truncation(problem))
    assert not any("boundary" in str(w.message) for w in caught)
    # Bernoulli excitation variance is bounded by 1/4: five standard errors.
    np.testing.assert_allclose(result.expect("r"), np.exp(-problem.tlist), atol=2.5 / np.sqrt(128))
    report = result.check_truncation(threshold=2.)
    assert len(report["runs"]) == 128
    assert report["maximum"]["r"] == pytest.approx(1.)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("diffusive", [False, True])
def test_parameter_batch_replays_assigned_point_noise(backend, diffusive):
    """Point keys/seeds are captured before grouping and each point replays alone."""
    seq, problem = jump_problem(backend)
    args = dict(problem.run_args)
    args.pop("seeds", None)
    if diffusive:
        if backend == "qutip":
            problem = replace(problem, solver="ssesolve", options={**problem.options, "dt": .001,
                                                                  "store_measurement": True})
        else:
            import dynamiqs as dq
            problem = replace(problem, solver="dssesolve")
            args["method"] = dq.method.EulerMaruyama(dt=.001)
    batch = seq.build_batch(seq.vary("r.T1", [1., 1.]), tlist=problem.tlist, solver=problem.solver,
                            run_args=args, options=problem.options, initial_state={"r": 1},
                            e_ops=problem.chip.e_ops(r="n"))
    results = solve_many(batch, progress=False)
    replay = solve_many([batch[1], batch[0]], progress=False)
    for i in range(2):
        np.testing.assert_allclose(results[i].expect("r"), solve_problem(batch[i]).expect("r"))
        np.testing.assert_allclose(results[i].expect("r"), replay[1-i].expect("r"))
    if backend == "qutip":
        if diffusive:
            assert not np.array_equal(results[0].native.measurement, results[1].native.measurement)
        else:
            assert results[0].native.col_times != results[1].native.col_times
    else:
        import jax
        assert not np.array_equal(jax.random.key_data(results[0].native.keys),
                                  jax.random.key_data(results[1].native.keys))


def test_native_arguments_cannot_replace_assembled_physics():
    """Forwarding cannot override model assembly or resolve conflicting storage silently."""
    _, problem = jump_problem("qutip")
    with pytest.raises(ValueError, match="assembled inputs"):
        replace(problem, run_args={"H": 0})
    with pytest.raises(ValueError, match="Conflicting"):
        solve_problem(replace(problem, states="all", options={"store_states": True}))
    assert problem.states is None
    result = solve_problem(problem)
    assert result.native.options["method"] != "diag"


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("pure", [True, False])
def test_diffusive_native_parity_and_state_invariants(backend, pure):
    """Diffusive adapters preserve native records, pure SSE states, and SME trace."""
    from quchip import with_monitoring
    _, base = jump_problem(backend, storage="all", count=8)
    if backend == "qutip":
        import qutip
        solver = "ssesolve" if pure else "smesolve"
        args = {"ntraj": 8, "seeds": 42}
        options = {"dt": 0.001, "store_measurement": True, "keep_runs_results": True, "progress_bar": ""}
    else:
        import dynamiqs as dq
        solver = "dssesolve" if pure else "dsmesolve"
        args = {**base.run_args, "method": dq.method.EulerMaruyama(dt=0.001)}
        options = {}
    eta = 1. if pure else .4
    problem = with_monitoring(replace(base, solver=solver, run_args=args, options=options),
                              {base.engine_result.slh.channels[0].key: eta})
    result = solve_problem(problem)
    owner = problem.backend
    rhs = owner.prepare_hamiltonian(problem.engine_result, problem.tlist).rhs
    operator = owner.from_canonical_operator(problem.engine_result.slh.channels[0].coupling)
    if backend == "qutip":
        kwargs = dict(sc_ops=[np.sqrt(eta)*operator], e_ops=problem.e_ops,
                      options={**options, "store_states": True, "store_final_state": False}, **args)
        if not pure:
            kwargs["c_ops"] = [np.sqrt(1-eta)*operator]
        native = getattr(qutip, solver)(rhs, problem.initial_state, problem.tlist, **kwargs)
        np.testing.assert_allclose(result.native.measurement, native.measurement)
        expected = native.average_expect[0]
    else:
        kwargs = dict(exp_ops=problem.e_ops, **args)
        if not pure:
            kwargs["etas"] = np.array([eta])
        state = {"psi0" if pure else "rho0": problem.initial_state}
        native = getattr(dq, solver)(rhs, [operator], tsave=problem.tlist, **state, **kwargs)
        np.testing.assert_allclose(result.native.measurements, native.measurements)
        expected = native.mean_expects()[0]
    np.testing.assert_allclose(result.expect("r"), expected)
    for state in result.run(0).states:
        array = np.asarray(owner.to_array(owner.state_to_dm(state)))
        if pure:
            # Native fixed-step SSE states need not be normalized at finite dt.
            np.testing.assert_allclose(np.trace(array @ array) / np.trace(array)**2, 1., atol=1e-12)
        else:
            np.testing.assert_allclose(np.trace(array), 1., atol=1e-12)
            # Compare positivity to the identical native discretization below.
            assert np.isfinite(np.linalg.eigvalsh(array)).all()
    batch = solve_many([problem, problem], progress=False)
    np.testing.assert_allclose(batch[0].expect("r"), result.expect("r"))


def test_monitored_generator_preserves_distinct_noncommuting_channels():
    """Inefficiency splitting retains identical channel identities and the Lindblad generator."""
    import qutip
    from quchip import CollapseChannel, with_monitoring
    from quchip.declarative import DeviceModel
    from quchip.engine.monitoring import monitored_operators

    class MeasuredQubit(DeviceModel):
        def local_hamiltonian(self, op, p):
            return 0. * op.n

        def dissipation(self, op, p):
            return (CollapseChannel(op.sigma_x, 0.2, "x"), CollapseChannel(op.sigma_z, 0.1, "z"),
                    CollapseChannel(op.sigma_x, 0.3, "x_again"))

    chip = Chip([MeasuredQubit(label="q", levels=2)], frame="lab")
    problem = QuantumSequence(chip).build_problem([0., .1], solver="smesolve")
    channels = problem.engine_result.slh.channels
    for eta in (0., .4, 1.):
        selected = with_monitoring(problem, {c.key: eta for c in channels},
                                   phases={c.key: .3 for c in channels})
        loss, monitored, etas = monitored_operators(selected)
        split = loss + [np.sqrt(e) * op for e, op in zip(etas, monitored)]
        split += [np.sqrt(1 - e) * op for e, op in zip(etas, monitored)]
        expected = qutip.liouvillian(
            0 * chip.backend.identity(2), problem.backend._collapse_operators(problem.engine_result)
        )
        actual = qutip.liouvillian(0 * chip.backend.identity(2), split)
        np.testing.assert_allclose(actual.full(), expected.full(), atol=1e-14)
        assert len(monitored) == 3
    selected = replace(selected, run_args={"ntraj": 4, "seeds": 3},
                       options={"method": "rouchon", "dt": .001, "keep_runs_results": True,
                                "store_measurement": "end", "progress_bar": ""})
    evolved = solve_problem(selected)
    assert np.asarray(evolved.native.measurement).shape[1] == 3
    for state in evolved.run(0).states:
        np.testing.assert_allclose(state.tr(), 1., atol=1e-12)
        assert np.linalg.eigvalsh(state.full()).min() >= -1e-12


def test_diffusive_fixed_noise_gradient_matches_finite_difference():
    """The Euler diffusive SSE pathwise gradient matches the same-key finite difference."""
    import jax
    import jax.numpy as jnp
    import dynamiqs as dq

    def value(lifetime):
        chip = Chip([Resonator(freq=1., levels=2, label="r", T1=lifetime)], backend="dynamiqs", frame="rotating")
        problem = QuantumSequence(chip).build_problem(
            (0., .1, .2), solver="dssesolve", initial_state={"r": 1}, e_ops=chip.e_ops(r="n"),
            run_args={"keys": jax.random.split(jax.random.key(6), 4), "method": dq.method.EulerMaruyama(dt=.001)})
        return jnp.real(solve_problem(problem).expect("r")[-1])

    derivative = jax.jit(jax.grad(value))(1.)
    step = 1e-4
    finite = (value(1.+step)-value(1.-step))/(2*step)
    np.testing.assert_allclose(derivative, finite, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_advanced_jump_sampling_retains_no_click_diagnostics(backend):
    """Native weighted averages and deterministic no-click cutoff samples remain available."""
    _, problem = jump_problem(backend, storage="none", count=8)
    if backend == "qutip":
        problem = replace(problem, options={**problem.options, "improved_sampling": True})
    else:
        import dynamiqs as dq

        problem = replace(
            problem, run_args={**problem.run_args, "method": dq.method.Event(dtmax=0.02, smart_sampling=True)}
        )
    result = solve_problem(with_truncation(problem))
    native_mean = (result.native.average_expect[0] if backend == "qutip" else result.native.mean_expects()[0])
    np.testing.assert_allclose(result.expect("r"), native_mean)
    diagnostics = result.check_truncation(threshold=2.)
    assert len(diagnostics["deterministic"]) == 1
    assert diagnostics["deterministic"][0]["r"] == pytest.approx(1.)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_native_current_identity_offset_and_variance(backend):
    """An authored coherent identity offset survives monitor phase rotation and native shot noise."""
    from quchip import CollapseChannel, with_monitoring
    from quchip.declarative import DeviceModel

    class Offset(DeviceModel):
        def local_hamiltonian(self, op, p):
            return 0. * op.n

        def dissipation(self, op, p):
            return (CollapseChannel((1.+.2j)*op.I, .5, "offset"),)

    chip = Chip([Offset(levels=2, label="q")], backend=backend, frame="lab")
    if backend == "qutip":
        solver = "smesolve"
        args = {"ntraj": 1024, "seeds": 71}
        options = {"dt": .001, "store_measurement": "end", "keep_runs_results": True, "progress_bar": ""}
    else:
        import dynamiqs as dq
        import jax
        solver = "dsmesolve"
        args = {"keys": jax.random.split(jax.random.key(71), 1024), "method": dq.method.EulerMaruyama(dt=.001)}
        options = {}
    problem = QuantumSequence(chip).build_problem([0., .1, .2], solver=solver, run_args=args, options=options)
    key = problem.engine_result.slh.channels[0].key
    problem = with_monitoring(problem, {key: .4}, phases={key: .3})
    result = solve_problem(problem)
    records = np.asarray(result.native.measurement if backend == "qutip" else result.native.measurements)
    expected_mean = 2*np.real(np.sqrt(.5*.4)*np.exp(-.3j)*(1.+.2j))
    # Native current is dW/dt, with interval variance 1/dt; five sampling standard errors.
    variance = 1/.1
    np.testing.assert_allclose(records.mean(), expected_mean, atol=5*np.sqrt(variance/records.size))
    np.testing.assert_allclose(records.var(), variance, atol=5*variance*np.sqrt(2/(records.size-1)))
    assert result.monitor_labels == (key,)
    np.testing.assert_allclose(result.run(0).population("q", 0), 1., atol=1e-12)



def test_native_event_buffer_exhaustion_cannot_look_like_completed_evolution():
    """Native raw data survives while analysis rejects Event's silently unfinished capped solve."""
    import dynamiqs as dq
    import jax

    chip = Chip([Resonator(freq=1., levels=3, label="r", T1=.1)], backend="dynamiqs", frame="rotating")
    problem = QuantumSequence(chip).build_problem(
        [0., .5, 1.], solver="jssesolve", initial_state={"r": 2}, e_ops=chip.e_ops(r="n"),
        run_args={"keys": jax.random.split(jax.random.key(11), 2), "method": dq.method.Event(dtmax=.01)},
        options={"nmaxclick": 1})
    incomplete = solve_problem(problem)
    assert incomplete.native.clicktimes.shape[-1] == 1
    with pytest.raises(Exception, match="exhausted nmaxclick"):
        incomplete.expect("r")
    complete = solve_problem(replace(problem, options={"nmaxclick": 8}))
    np.testing.assert_allclose(complete.expect("r"), [2., 0., 0.], atol=1e-12)
