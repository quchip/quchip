"""Exact reductions use one retained map for the complete Hamiltonian and jumps."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import RWA, Capacitive, Chip, DuffingTransmon, Exact, Resonator, eliminate
from quchip.chip.sw import bare_hamiltonian


def _chip(backend, g=.08, approximation=Exact()):
    q = DuffingTransmon(freq=5., anharmonicity=-.3, levels=4, label='q')
    r = Resonator(freq=7., levels=4, T1=100., thermal_occupation=.4, label='r')
    return Chip([q, r], [Capacitive(q, r, g=g, label='qr')], backend=backend, approximation=approximation)


def _oracle(chip):
    h, _, dims = bare_hamiltonian(chip, approximation=chip.approximation)
    h = np.asarray(h)
    energies, vectors = np.linalg.eigh(h)
    bare = np.arange(dims[0]) * dims[1]
    selected = np.argmax(np.abs(vectors[bare])**2, axis=1)
    assert len(set(selected)) == len(bare)
    w = vectors[np.ix_(bare, selected)]
    gram_values, gram_vectors = np.linalg.eigh(w @ w.conj().T)
    inverse_sqrt = (gram_vectors * gram_values**-.5) @ gram_vectors.conj().T
    unitary = inverse_sqrt @ w
    embedding = vectors[:, selected] @ unitary.conj().T
    return embedding.conj().T @ h @ embedding, embedding, energies, vectors, bare, selected


@pytest.mark.validation
@pytest.mark.parametrize('approximation', [Exact(), RWA()], ids=['exact', 'rwa'])
@pytest.mark.parametrize('backend', ['qutip', 'dynamiqs'])
def test_exact_model_retains_full_matrix_and_thermal_jump_coordinates(backend, approximation):
    # Under RWA() the reduction diagonalizes each excitation sector separately;
    # the oracle diagonalizes the full matrix.
    full = _chip(backend, approximation=approximation)
    expected_h, embedding, *_ = _oracle(full)
    expected_jumps = [embedding.conj().T @ np.asarray(full.backend.to_array(op)) @ embedding
                      for op in full.backend._collapse_operators(full.resolve(frame='lab'))]
    reduced = eliminate(full, 'r', method='exact').chip
    np.testing.assert_allclose(bare_hamiltonian(reduced)[0], expected_h, atol=1e-11)
    actual_jumps = [np.asarray(reduced.backend.to_array(op))
                    for op in reduced.backend._collapse_operators(reduced.resolve(frame='lab'))]
    assert len(actual_jumps) == len(expected_jumps) == 2
    for actual, expected in zip(actual_jumps, expected_jumps):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_exact_retained_map_is_isometric_and_independent_of_eigenvector_phases():
    from quchip.chip.sw import exact_subspace
    _, expected, energies, vectors, bare, selected = _oracle(_chip('qutip'))
    phases = np.exp(1j * np.arange(vectors.shape[1]) * .37)
    first = exact_subspace(energies, vectors, bare, selected)
    second = exact_subspace(energies, vectors * phases, bare, selected)
    np.testing.assert_allclose(first.embedding.conj().T @ first.embedding, np.eye(len(bare)), atol=1e-12)
    np.testing.assert_allclose(first.embedding, expected, atol=1e-12)
    np.testing.assert_allclose(second.embedding, first.embedding, atol=1e-12)
    np.testing.assert_allclose(second.hamiltonian, first.hamiltonian, atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(first.hamiltonian), sorted(energies[selected]), atol=1e-12)


@pytest.mark.validation
def test_exact_retained_matrix_jit_gradient_matches_independent_finite_difference():
    def value(g):
        reduced = eliminate(_chip('dynamiqs', g), 'r', method='exact').chip
        return jnp.real(reduced.resolve(frame='lab').hamiltonian().matrix(backend=reduced.backend)[3, 3])
    g, step = .08, 1e-5
    expected = _oracle(_chip('qutip', g))[0][3, 3].real
    derivative = (_oracle(_chip('qutip', g + step))[0][3, 3].real
                  - _oracle(_chip('qutip', g - step))[0][3, 3].real) / (2 * step)
    actual, gradient = jax.jit(jax.value_and_grad(value))(g)
    assert actual == pytest.approx(expected, abs=1e-11)
    assert gradient == pytest.approx(derivative, rel=1e-6, abs=1e-8)


@pytest.mark.validation
@pytest.mark.parametrize("execution", ["eager", "jit", "gradient"])
@pytest.mark.parametrize("invalidity", ["singular", "labels"])
def test_exact_projection_guards_survive_compiled_and_gradient_only_calls(execution, invalidity):
    from quchip.chip.sw import _inverse_sqrt_hermitian, exact_mode_subspace

    def value(parameter):
        if invalidity == "singular":
            return jnp.trace(_inverse_sqrt_hermitian(jnp.diag(jnp.array([1., parameter]))))
        h = jnp.diag(jnp.array([0., parameter, 5., parameter + 5.]))
        h = h.at[1, 2].set(.02).at[2, 1].set(.02)
        return jnp.real(exact_mode_subspace(h, ["q", "r"], (2, 2), "r", ["q"]).hamiltonian[1, 1])

    evaluate = {"eager": value, "jit": jax.jit(value), "gradient": jax.jit(jax.grad(value))}[execution]
    parameter = 0. if invalidity == "singular" else 5.
    message = "Gram matrix" if invalidity == "singular" else "Near-degenerate"
    with pytest.raises(Exception, match=message):
        jax.block_until_ready(evaluate(parameter))


def test_exact_retained_projection_batches_match_independent_models():
    from quchip.chip.sw import exact_mode_subspace

    def value(frequency):
        h = jnp.diag(jnp.array([0., frequency, 5., frequency + 5.]))
        h = h.at[1, 2].set(.02).at[2, 1].set(.02)
        return exact_mode_subspace(h, ["q", "r"], (2, 2), "r", ["q"]).hamiltonian

    frequencies = jnp.array([6., 7., 8.])
    actual = jax.jit(jax.vmap(value))(frequencies)
    expected = jnp.stack([value(frequency) for frequency in frequencies])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
