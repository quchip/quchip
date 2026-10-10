"""Forward- and reverse-mode derivatives through the local eigensolver."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from quchip import Chip, Resonator  # noqa: E402
from quchip.chip.couplings import Capacitive  # noqa: E402
from quchip.devices.transmon.duffing import DuffingTransmon  # noqa: E402
from quchip.engine.basis import _differentiable_eigenpairs, _differentiable_eigenvector  # noqa: E402


def _dispersive_chip() -> Chip:
    pytest.importorskip("dynamiqs")
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    resonator = Resonator(freq=7.0, levels=4, label="r")
    return Chip(
        [qubit, resonator], couplings=[Capacitive(qubit, resonator, g=0.1)], frame="rotating", backend="dynamiqs"
    )


def _resonator_frame(chip: Chip, freq: float) -> float:
    rebound = chip.with_params({"q.freq": freq})
    return rebound.resolve().resolved_frame.frequencies["r"]


def test_forward_and_reverse_derivatives_agree_with_finite_differences() -> None:
    """jacfwd and grad through a traced device frequency both match a central difference."""
    chip = _dispersive_chip()
    step = 1e-4
    reference = (_resonator_frame(chip, 5.0 + step) - _resonator_frame(chip, 5.0 - step)) / (2 * step)
    forward = float(jax.jacfwd(lambda f: _resonator_frame(chip, f))(5.0))
    reverse = float(jax.grad(lambda f: _resonator_frame(chip, f))(5.0))
    assert forward == pytest.approx(reference, rel=1e-4)
    assert reverse == pytest.approx(reference, rel=1e-4)
    assert abs(reference) > 1e-4


@pytest.mark.validation
def test_hessian_is_finite_away_from_degeneracies() -> None:
    """Second derivatives are finite and match a second difference at a nondegenerate operating point."""
    chip = _dispersive_chip()
    curvature = float(jax.jit(jax.hessian(lambda f: _resonator_frame(chip, f)))(5.0))
    step = 1e-3
    reference = (
        _resonator_frame(chip, 5.0 + step) - 2 * _resonator_frame(chip, 5.0) + _resonator_frame(chip, 5.0 - step)
    ) / step**2
    assert np.isfinite(curvature)
    assert curvature == pytest.approx(reference, rel=5e-2)


def test_eigenvector_tangent_is_orthogonal_to_the_eigenvector() -> None:
    """The parallel-transport gauge keeps v_i^dagger dv_i = 0 for every retained eigenvector."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    matrix = jnp.asarray(base + base.conj().T)
    direction = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    direction = jnp.asarray(direction + direction.conj().T)
    (_, vectors), (_, dvectors) = jax.jvp(lambda m: _differentiable_eigenpairs(m, 3), (matrix,), (direction,))
    connection = jnp.diagonal(jnp.conj(vectors).T @ dvectors)
    np.testing.assert_allclose(np.asarray(connection), 0.0, atol=1e-10)


def test_selected_eigenvector_derivative_is_exact_beside_a_tie() -> None:
    """One column's derivative matches the analytic 2x2 result while two other levels are exactly tied."""

    def ground_weight(coupling):
        # Levels 2 and 3 stay exactly degenerate, where differentiating eigh itself gives NaN.
        matrix = jnp.diag(jnp.array([0.0, 1.0, 2.0, 2.0], dtype=complex)).at[0, 1].set(coupling).at[1, 0].set(coupling)
        values, vectors = jnp.linalg.eigh(matrix)
        vector = _differentiable_eigenvector(matrix, values[0], values, vectors, vectors[:, 0])
        return jnp.abs(vector[0]) ** 2

    coupling = 0.1
    # The block [[0, g], [g, 1]] gives |<0|ground>|² = (1 + (1 + 4g²)^(-1/2)) / 2.
    exact = -2 * coupling / (1 + 4 * coupling**2) ** 1.5
    assert float(jax.grad(ground_weight)(coupling)) == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize("origin", [-1e8, 1e8])
def test_eigenprojector_derivative_is_independent_of_energy_origin(origin: float) -> None:
    """An identity energy shift cannot suppress a nondegenerate physical response."""
    matrix = jnp.array([[0.0, 0.02], [0.02, 0.04]])
    direction = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    def projector(m):
        _, vectors = _differentiable_eigenpairs(m, 1)
        return vectors @ vectors.conj().T

    step = 1e-6
    finite_difference = (projector(matrix + step * direction) - projector(matrix - step * direction)) / (2 * step)
    shifted = matrix + origin * jnp.eye(2)
    _, tangent = jax.jvp(projector, (shifted,), (direction,))
    np.testing.assert_allclose(tangent, finite_difference, rtol=1e-6, atol=1e-6)


@pytest.mark.validation
def test_canonical_energy_vector_phase_and_derivative_agree():
    """The largest authored component is positive and its local phase derivative is correct."""
    from quchip.engine.basis import resolve_local_basis

    def vectors(value):
        matrix = jnp.array([[0.0, value - 0.2j], [value + 0.2j, 1.0]])
        return resolve_local_basis(matrix).energy_vectors

    value = 0.3
    actual = vectors(value)
    pivots = actual[jnp.argmax(jnp.abs(actual), axis=0), jnp.arange(2)]
    np.testing.assert_allclose(pivots.imag, 0.0, atol=1e-12)
    assert np.all(np.asarray(pivots.real) > 0)
    step = 1e-5
    finite_difference = (vectors(value + step) - vectors(value - step)) / (2 * step)
    np.testing.assert_allclose(jax.jacfwd(vectors)(value), finite_difference, atol=1e-8)
    def loss(x):
        return jnp.real(vectors(x)[0, 1])
    reference = (loss(value + step) - loss(value - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(value), reference, atol=1e-8)


def test_traced_analysis_stays_inside_its_trace() -> None:
    """Analysis computed inside a trace is reused within that trace only, for traced and concrete chips alike."""
    chip = _dispersive_chip()
    single = Chip([DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")], backend="dynamiqs")

    def shifted(freq):
        rebound = chip.with_params({"q.freq": freq})

        def body(total, _):
            return total + rebound.freq("q"), None

        inner, _ = jax.lax.scan(body, 0.0, None, length=2)
        return inner + rebound.freq("q") + rebound.freq("r") - rebound.freq("q") + single.freq("q")

    value, slope = jax.jit(jax.value_and_grad(shifted))(5.0)
    again, _ = jax.jit(jax.value_and_grad(shifted))(5.1)
    eager = [chip.with_params({"q.freq": f}) for f in (5.0, 5.0 + 1e-5, 5.0 - 1e-5, 5.1)]
    expected = [2 * c.freq("q") + c.freq("r") + single.freq("q") for c in eager]
    np.testing.assert_allclose(float(value), expected[0], rtol=1e-12)
    np.testing.assert_allclose(float(slope), (expected[1] - expected[2]) / 2e-5, rtol=1e-6)
    np.testing.assert_allclose(float(again), expected[3], rtol=1e-12)
