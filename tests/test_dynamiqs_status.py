"""Native method and storage parity across status-bearing batch integration."""

from dataclasses import replace

import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence
from quchip.engine.ir import SolveBatch


@pytest.mark.parametrize("method_name", ["Euler", "Dopri5", "Dopri8", "Tsit5", "Kvaerno3", "Kvaerno5", "Expm",
                                         "Rouchon1", "Rouchon2", "Rouchon3"])
def test_native_method_batch_matches_separate_master_equations(method_name):
    import dynamiqs as dq

    method = getattr(dq.method, method_name)(**({"dt": 0.01} if method_name in ("Euler", "Rouchon1") else {}))
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0, label="q")
    chip = Chip([q], backend="dynamiqs", frame="rotating")
    problem = QuantumSequence(chip).build_problem([0.0, 0.1, 1.0], initial_state={"q": 1},
        e_ops={"q": q.number_operator()}, states="final", options={"method": method})
    second = replace(problem, initial_state=chip.backend.basis(2, 0))
    batch = chip.backend.solve_batch(SolveBatch(chip=chip, problems=(problem, second)), progress=False)
    for request, point in zip((problem, second), batch):
        reference = chip.backend.solve_problem(request)
        np.testing.assert_allclose(point.expect, reference.expect, atol=1e-12)
        np.testing.assert_allclose(chip.backend.to_array(point.final_state),
                                   chip.backend.to_array(reference.final_state), atol=1e-12)


@pytest.mark.parametrize("gradient_name", ["BackwardCheckpointed", "Direct", "Forward"])
def test_native_gradient_modes_keep_decay_derivative(gradient_name):
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp
    from quchip.engine import solve_batch

    gradient = getattr(dq.gradient, gradient_name)()

    def objective(t1):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=t1, label="q")
        chip = Chip([q], backend="dynamiqs", frame="rotating")
        problem = QuantumSequence(chip).build_problem([0.0, 0.1, 1.0], initial_state={"q": 1},
            e_ops={"q": q.number_operator()}, states="none", options={"gradient": gradient})
        results = solve_batch(SolveBatch(chip=chip, problems=(problem, problem)),
                              progress=False)
        return jnp.real(results.expect("q", reduce="last").sum())

    derivative = jax.jacfwd(objective) if gradient_name == "Forward" else jax.grad(objective)
    assert jax.jit(derivative)(jnp.asarray(10.0)) == pytest.approx(0.02 * np.exp(-0.1), abs=2e-7)


@pytest.mark.parametrize("multipart", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_expm_final_density_matrix_preserves_coherence_and_subsystem_dimensions(multipart, batched):
    import dynamiqs as dq
    from quchip.engine import solve_batch, solve_problem

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0, label="q")
    devices = [q] + ([DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="r")] if multipart else [])
    chip = Chip(devices, backend="dynamiqs", frame="rotating")
    state = ((chip.bare_state(q=0, r=1) + 1j * chip.bare_state(q=1, r=0)) if multipart
             else (chip.bare_state(q=0) + 1j * chip.bare_state(q=1))) / np.sqrt(2)
    problem = QuantumSequence(chip).build_problem([0.0, 0.1, 1.0], initial_state=state,
        states="final", options={"method": dq.method.Expm()})
    results = (solve_batch(SolveBatch(chip=chip, problems=(problem, problem)), progress=False)
               if batched else [solve_problem(problem)])
    decay = np.exp(-0.1)
    expected = np.zeros((4, 4) if multipart else (2, 2), dtype=complex)
    if multipart:
        expected[0, 0], expected[1, 1], expected[2, 2] = (1-decay)/2, 0.5, decay/2
        left, right = 1, 2
    else:
        expected[0, 0], expected[1, 1] = 1-decay/2, decay/2
        left, right = 0, 1
    expected[left, right] = -0.5j * np.sqrt(decay)
    expected[right, left] = 0.5j * np.sqrt(decay)
    for result in results:
        final = result.final_state
        assert final.dims == tuple(2 for _ in devices)
        assert not final.isket()
        np.testing.assert_allclose(chip.backend.to_array(final), expected, atol=1e-10)


@pytest.mark.parametrize("batched", [False, True])
def test_expm_rejects_nonfinite_native_output(batched):
    import dynamiqs as dq
    from quchip.engine import solve_batch, solve_problem

    problems = []
    for frequency in (5.0, 1e8):
        q = DuffingTransmon(freq=frequency, anharmonicity=-0.2, levels=2, label="q")
        chip = Chip([q], backend="dynamiqs", frame="lab")
        problems.append(QuantumSequence(chip).build_problem([0.0, 1.0], initial_state={"q": 1},
            states="none", options={"method": dq.method.Expm()}))
    with pytest.raises(RuntimeError, match="Nonfinite"):
        if batched:
            solve_batch(SolveBatch(chip=problems[0].chip, problems=tuple(problems)),
                         progress=False)
        else:
            solve_problem(problems[1])


@pytest.mark.parametrize("method_name", ["JumpMonteCarlo", "DiffusiveMonteCarlo"])
def test_stochastic_methods_have_an_explicit_support_boundary(method_name):
    import dynamiqs as dq
    import jax
    from quchip.engine import solve_problem

    trajectory_method = (dq.method.Event(dtmax=0.01) if method_name == "JumpMonteCarlo"
                         else dq.method.EulerMaruyama(dt=0.01))
    method = getattr(dq.method, method_name)(jax.random.split(jax.random.PRNGKey(0), 2), trajectory_method)
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0, label="q")
    problem = QuantumSequence(Chip([q], backend="dynamiqs")).build_problem([0.0, 1.0],
        initial_state={"q": 1}, options={"method": method})
    with pytest.raises(ValueError, match="deterministic.*Monte Carlo"):
        solve_problem(problem)


@pytest.mark.parametrize("batched", [False, True])
def test_expm_final_density_matrix_keeps_jit_gradient(batched):
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp
    from quchip.engine import solve_batch, solve_problem

    @jax.jit
    @jax.value_and_grad
    def objective(t1):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=t1, label="q")
        chip = Chip([q], backend="dynamiqs", frame="rotating")
        problem = QuantumSequence(chip).build_problem([0.0, 1.0], initial_state={"q": 1},
            states="final", options={"method": dq.method.Expm()})
        result = (solve_batch(SolveBatch(chip=chip, problems=(problem, problem)),
                             progress=False)[0] if batched
                  else solve_problem(problem))
        return jnp.real(chip.backend.to_array(result.final_state)[1, 1])

    value, derivative = objective(jnp.asarray(10.0))
    assert value == pytest.approx(np.exp(-0.1), abs=1e-10)
    assert derivative == pytest.approx(0.01 * np.exp(-0.1), abs=1e-10)


@pytest.mark.parametrize("batched", [False, True])
def test_expm_nonfinite_check_survives_gradient_only(batched):
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp
    from quchip.engine import solve_batch, solve_problem

    @jax.jit
    @jax.grad
    def objective(frequency):
        q = DuffingTransmon(freq=frequency, anharmonicity=-0.2, levels=2, label="q")
        chip = Chip([q], backend="dynamiqs", frame="lab")
        problem = QuantumSequence(chip).build_problem([0.0, 1.0], initial_state={"q": 1},
            states="final", options={"method": dq.method.Expm()})
        result = (solve_batch(SolveBatch(chip=chip, problems=(problem, problem)),
                             progress=False)[0] if batched
                  else solve_problem(problem))
        return jnp.real(result.final_state.to_jax()[1, 0])

    assert np.isfinite(objective(jnp.asarray(5.0)))
    with pytest.raises(Exception, match="Nonfinite"):
        objective(jnp.asarray(1e8)).block_until_ready()
