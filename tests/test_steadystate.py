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


def test_qutip_skips_dense_rank_diagnostics_above_default_cap() -> None:
    """Large sparse QuTiP solves keep a sparse residual without forcing a dense SVD."""
    mode = Resonator(freq=6.0, levels=17, label="r", T1=20.0)

    result = Chip([mode], frame="rotating", backend="qutip").steadystate()

    assert result.residual < 1e-12
    assert result.nullity is None
    assert result.condition_number is None
    assert result.is_unique is None
    assert result.stats["uniqueness_checked"] is False


@pytest.mark.optional_backend
def test_dynamiqs_steady_state_is_jittable_and_differentiable() -> None:
    """The constrained dynamiqs solve preserves JIT and gradients through dissipative parameters."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    mode = Resonator(freq=6.0, levels=5, label="r", T1=20.0, thermal_population=0.1)
    chip = Chip([mode], frame="rotating", backend="dynamiqs")
    number = mode.number_operator()

    @jax.jit
    def occupation(thermal_population):
        shifted = chip.with_params({"r.thermal_population": thermal_population})
        result = shifted.steadystate(e_ops={"r": number})
        return jnp.real(result.expect("r"))

    value = occupation(jnp.asarray(0.1))
    gradient = jax.grad(occupation)(jnp.asarray(0.1))

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
    mode = Resonator(freq=6.0, levels=6, label="r", T1=20.0, thermal_population=0.0)
    chip = Chip([mode], frame="rotating", backend="qutip")
    thermal = Sweep([0.0, 0.1, 0.2], name="r.thermal_population")

    result = chip.steadystate_batch(thermal, e_ops={mode: mode.number_operator()}, progress=False)

    assert result.shape == (3,)
    assert result.axes[0][0] == "r.thermal_population"
    np.testing.assert_allclose(np.real(result.expect(mode)), [0.0, 0.09998, 0.19936], atol=8e-4)
    with pytest.raises(FrozenInstanceError):
        result._shape = (99,)
