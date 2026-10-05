"""Stationary Lindblad solves through the public chip API."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quchip import Bath, Chip, Exact, Resonator, Sweep


def test_damped_mode_has_vacuum_steady_state() -> None:
    """A zero-temperature damped mode settles into vacuum with complete diagnostics."""
    mode = Resonator(freq=6.0, levels=4, label="r", T1=20.0)
    chip = Chip([mode], frame="rotating", backend="qutip")

    result = chip.steadystate(e_ops={mode: mode.number_operator()})

    np.testing.assert_allclose(chip.backend.to_array(result.state), np.diag([1.0, 0.0, 0.0, 0.0]), atol=1e-12)
    assert result.expect(mode) == pytest.approx(0.0, abs=1e-12)
    assert result.trace == pytest.approx(1.0, abs=1e-12)
    assert result.trace_error < 1e-12
    assert result.hermiticity_error < 1e-12
    assert result.positivity_error < 1e-12
    assert result.residual < 1e-12
    assert result.nullity == 1
    assert result.is_unique


@pytest.mark.validation
@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_state_positivity_is_computed_on_request_from_captured_state(backend, monkeypatch):
    mode = Resonator(freq=6.0, levels=2, T1=20.0, thermal_occupation=0.1, label="r")
    chip = Chip([mode], frame="rotating", backend=backend)
    xp = chip.backend.array_module
    eigvalsh = xp.linalg.eigvalsh
    evaluated = []

    def measured(state):
        evaluated.append(state.shape)
        return eigvalsh(state)

    monkeypatch.setattr(xp.linalg, "eigvalsh", measured)
    result = chip.steadystate(e_ops={"r": "n"})
    assert result.expect("r") == pytest.approx(1 / 12, abs=1e-10)
    assert evaluated == []
    mode.thermal_occupation = 0.5
    assert result.minimum_eigenvalue == pytest.approx(1 / 12, abs=1e-10)
    assert evaluated == [(2, 2)]
    assert result.positivity_error == pytest.approx(0.0, abs=1e-12)
    assert result.trace_error < 1e-12
    assert result.hermiticity_error < 1e-12


@pytest.mark.validation
def test_requested_stationary_diagnostic_keeps_native_gradient():
    import jax
    import jax.numpy as jnp

    @jax.jit
    @jax.value_and_grad
    def minimum(occupation):
        mode = Resonator(freq=6.0, levels=2, T1=20.0,
                         thermal_occupation=occupation, label="r")
        return Chip([mode], frame="rotating", backend="dynamiqs").steadystate().minimum_eigenvalue

    value, derivative = minimum(jnp.asarray(0.1))
    assert value == pytest.approx(1 / 12, abs=1e-10)
    assert derivative == pytest.approx(1 / 1.2**2, abs=1e-10)


def test_built_stationary_request_keeps_its_backend_and_local_context() -> None:
    """Source edits and a new default backend cannot reinterpret a built stationary solve."""
    from quchip.backend import reset_default_backend, set_default_backend
    from quchip.backend.qutip import QuTiPBackend
    from quchip.engine.steady_state import build_steadystate_problem, solve_steadystate_problem

    class LaterBackend(QuTiPBackend):
        def steadystate(self, problem):
            raise AssertionError("old stationary request used new backend")

    reset_default_backend()
    mode = Resonator(freq=6.0, levels=3, label="r", T1=20.0)
    chip = Chip([mode], frame="rotating")
    problem = build_steadystate_problem(chip, e_ops={mode: mode.number_operator()})
    mode.levels = 4
    mode.computational = True
    try:
        set_default_backend(LaterBackend())
        result = solve_steadystate_problem(problem)
        assert result.dims == (3,)
        assert result.device_info == (("r", False),)
        assert result.reduced_state("r").shape == (3, 3)
        assert result.expect("r") == pytest.approx(0.0, abs=1e-12)
    finally:
        reset_default_backend()


def test_qutip_skips_dense_rank_diagnostics_above_default_cap() -> None:
    """Large sparse QuTiP solves keep a sparse residual without forcing a dense SVD."""
    mode = Resonator(freq=6.0, levels=17, label="r", T1=20.0)

    result = Chip([mode], frame="rotating", backend="qutip").steadystate()

    assert result.residual < 1e-12
    assert result.nullity is None
    assert result.condition_number is None
    assert result.is_unique is None
    assert result.stats["uniqueness_checked"] is False


def test_qutip_returns_a_guess_only_when_it_is_stationary() -> None:
    """A stationary guess comes back without a solve; any other guess is solved away."""
    import qutip

    from quchip.engine.steady_state import build_steadystate_problem, solve_steadystate_problem

    mode = Resonator(freq=6.0, levels=4, label="r", T1=20.0, thermal_occupation=0.2)
    problem = build_steadystate_problem(Chip([mode], frame="rotating", backend="qutip"))
    solved = solve_steadystate_problem(problem)
    assert solved.stats["guess_reused"] is False

    reused = solve_steadystate_problem(problem, guess=solved.state)
    assert reused.stats["guess_reused"] is True
    assert reused.state is solved.state
    assert reused.residual == solved.residual

    resolved = solve_steadystate_problem(problem, guess=qutip.fock_dm(4, 0))
    assert resolved.stats["guess_reused"] is False
    np.testing.assert_allclose(resolved.state.full(), solved.state.full(), atol=1e-12)


def test_qutip_default_direct_solve_matches_qutip_steadystate() -> None:
    """The backend's own default direct solve reproduces qutip.steadystate's direct method."""
    import qutip

    from quchip import RWA, Capacitive
    from quchip.engine.steady_state import build_steadystate_problem, solve_steadystate_problem

    first = Resonator(freq=6.0, levels=4, label="a", T1=20.0, thermal_occupation=0.3)
    second = Resonator(freq=6.03, levels=3, label="b", T1=35.0, T2=30.0)
    chip = Chip([first, second], [Capacitive(first, second, g=0.05)], frame={"a": 6.0, "b": 6.0},
                approximation=RWA(), backend="qutip")
    problem = build_steadystate_problem(chip)

    ours = solve_steadystate_problem(problem)
    reference = qutip.steadystate(problem.backend.prepare_stationary(problem.engine_result).liouvillian)

    assert ours.state.dims == reference.dims
    assert ours.state.isherm
    np.testing.assert_allclose(ours.state.full(), reference.full(), atol=1e-13)
    assert ours.residual < 1e-12


@pytest.mark.validation
@pytest.mark.optional_backend
def test_dynamiqs_steady_state_is_jittable_and_differentiable() -> None:
    """The constrained dynamiqs solve preserves JIT and gradients through dissipative parameters."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    mode = Resonator(freq=6.0, levels=5, label="r", T1=20.0, thermal_occupation=0.1)
    chip = Chip([mode], frame="rotating", backend="dynamiqs")
    number = mode.number_operator()

    @jax.jit
    def occupation(thermal_occupation):
        shifted = chip.with_params({"r.thermal_occupation": thermal_occupation})
        result = shifted.steadystate(e_ops={"r": number})
        return jnp.real(result.expect("r"))

    value, gradient = jax.value_and_grad(occupation)(jnp.asarray(0.1))

    assert float(value) == pytest.approx(0.09993, rel=2e-3)
    assert float(gradient) == pytest.approx(0.996, rel=1e-2)


@pytest.mark.optional_backend
def test_dynamiqs_traced_nonunique_solve_returns_invalid_state() -> None:
    """A traced non-unique solve cannot silently expose an arbitrary density matrix."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    mode = Resonator(freq=6.0, levels=2, label="r", T2=10.0)
    chip = Chip([mode], frame="rotating", backend="dynamiqs")

    with pytest.raises(ValueError, match="unique stationary state"):
        chip.steadystate()

    @jax.jit
    def traced_trace_error(scale):
        shifted = chip.with_params({"r.T2": 10.0 * scale})
        return shifted.steadystate().trace_error

    assert jnp.isnan(traced_trace_error(jnp.asarray(1.0)))


def test_thermal_mode_matches_truncated_bose_distribution() -> None:
    """A thermal bath produces the Bose distribution after finite-level truncation."""
    frequency = 5.0
    temperature = 300.0
    levels = 12
    mode = Resonator(freq=frequency, levels=levels, label="r")
    chip = Chip(
        [mode],
        baths=[Bath("thermal", temperature=temperature, rate=0.05)],
        frame="rotating",
        backend="qutip",
    )

    result = chip.steadystate(e_ops={mode: mode.number_operator()})

    from quchip.utils.constants import k_B

    nbar = 1.0 / np.expm1(frequency / (k_B * temperature))
    ratio = nbar / (nbar + 1.0)
    probabilities = ratio ** np.arange(levels)
    probabilities /= probabilities.sum()
    expected = float(probabilities @ np.arange(levels))
    assert float(np.real(result.expect(mode))) == pytest.approx(expected, abs=1e-10)


def test_closed_system_rejects_non_unique_stationary_manifold() -> None:
    """A closed multi-level Hamiltonian is not assigned an arbitrary stationary state."""
    mode = Resonator(freq=6.0, levels=2, label="r")

    with pytest.raises(ValueError, match="unique stationary state"):
        Chip([mode], backend="qutip").steadystate()


def test_time_dependent_resolved_hamiltonian_is_rejected() -> None:
    """Stationary solving rejects component-owned time dependence and points to time evolution."""
    from quchip.extensions import FrequencyModulatedMode

    mode = FrequencyModulatedMode(
        frequency=5.0,
        modulation_amplitude=0.2,
        modulation_frequency=0.25,
        levels=3,
        label="m",
        T1=20.0,
    )

    with pytest.raises(ValueError, match="QuantumSequence"):
        Chip([mode], frame="lab", approximation=Exact(), backend="qutip").steadystate()


def test_steady_state_batch_preserves_sweep_shape_and_expectations() -> None:
    """Stationary parameter sweeps use the same named grid shape as other quchip batches."""
    mode = Resonator(freq=6.0, levels=6, label="r", T1=20.0, thermal_occupation=0.0)
    chip = Chip([mode], frame="rotating", backend="qutip")
    thermal = Sweep([0.0, 0.1, 0.2], name="r.thermal_occupation")

    result = chip.steadystate_batch(thermal, e_ops={mode: mode.number_operator()}, progress=False)

    assert result.shape == (3,)
    assert result.axes[0][0] == "r.thermal_occupation"
    np.testing.assert_allclose(np.real(result.expect(mode)), [0.0, 0.09998, 0.19936], atol=8e-4)
    with pytest.raises(FrozenInstanceError):
        result._shape = (99,)
