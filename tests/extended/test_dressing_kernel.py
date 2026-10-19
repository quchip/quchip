"""Unit tests for the ``quchip.engine.dressing`` JAX-traceable kernel.

These tests exercise the pure-array labeling primitives in isolation
from :class:`Chip`. Traceability through the full chip pipeline is
covered by :mod:`tests.extended.test_dressing_chip_traceable`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from quchip.engine.dressing import (
    BareProductReference,
    EigenstateReference,
    LabelingPath,
    assign_argmax,
    assign_rowwise_greedy,
    compute_overlaps,
    gauge_phases,
    label_eigensystem,
    track_path,
)


def _two_qubit_H(g: jnp.ndarray) -> jnp.ndarray:
    """Detuned two-level coupled to another two-level, coupling ``g``.

    Diagonal is ``diag(0, 1, 1 + delta, 2 + delta)`` with ``delta = 0.1``,
    so near-resonance mixing of ``|01⟩`` and ``|10⟩`` grows with ``g``.
    """
    delta = 0.1
    base = jnp.diag(jnp.array([0.0, 1.0, 1.0 + delta, 2.0 + delta], dtype=jnp.float32))
    sx = jnp.array(
        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        dtype=jnp.float32,
    )
    return base + g * sx


class TestAssignmentPolicies:
    def test_global_greedy_is_a_permutation_at_zero_coupling(self) -> None:
        """Global-greedy assignment yields a valid permutation even with degenerate bare labels."""
        H = jnp.diag(jnp.array([0.0, 1.0, 1.0, 2.0], dtype=jnp.float32))
        evals, evecs = jnp.linalg.eigh(H)
        labeling = label_eigensystem(evecs, BareProductReference(dims=(2, 2)))
        # Degenerate (0,1)/(1,0) pair still assigns a permutation with global greedy.
        assert sorted(labeling.indices.tolist()) == [0, 1, 2, 3]
        assert not bool(labeling.duplicates.any())

    def test_argmax_reports_duplicates_when_bare_labels_share_a_dressed_match(self) -> None:
        """Argmax assignment flags duplicate dressed-state matches while identity overlaps yield none."""
        # Synthetic overlap matrix where rows 1 and 2 both prefer column 1.
        # The kernel's ``duplicates`` flag is exactly designed for this case.
        overlaps = jnp.asarray(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.1, 0.7, 0.2, 0.0],  # shares column 1 with row 1 under argmax
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        indices, _, _ = assign_argmax(overlaps)
        assert indices.tolist() == [0, 1, 1, 3]

        repeated = label_eigensystem(jnp.sqrt(overlaps), BareProductReference(dims=(2, 2)),
                                      policy=assign_argmax)
        assert repeated.duplicates.tolist() == [False, True, True, False]

        # Identity overlaps give each row a unique dressed match.
        evecs = jnp.eye(4, dtype=jnp.float32)
        labeling = label_eigensystem(
            evecs, BareProductReference(dims=(2, 2)), policy=assign_argmax,
        )
        # Identity overlaps give each row a unique argmax — no duplicates here.
        assert not bool(labeling.duplicates.any())


    def test_rowwise_greedy_is_a_permutation(self) -> None:
        """Rowwise-greedy assignment returns a valid permutation for strongly hybridized overlaps."""
        # Strongly-hybridized overlaps (random unitary) — the rowwise policy must
        # still return a valid permutation (no duplicate dressed indices).
        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (8, 8)) + 1j * jax.random.normal(jax.random.fold_in(key, 1), (8, 8))
        q, _ = jnp.linalg.qr(a)
        overlaps = jnp.abs(q) ** 2
        indices, _, _ = assign_rowwise_greedy(overlaps)
        assert sorted(indices.tolist()) == list(range(8))


class TestReferences:
    def test_local_product_factors_match_explicit_complex_reference(self) -> None:
        dims = (2, 3, 2)
        def unitary(seed, dimension):
            key = jax.random.PRNGKey(seed)
            matrix = jax.random.normal(key, (dimension, dimension)) + 1j * jax.random.normal(
                jax.random.fold_in(key, 1), (dimension, dimension))
            return jnp.linalg.qr(matrix)[0]

        first, last = unitary(0, 2), unitary(1, 2)
        vectors = unitary(2, 12)[:, :7]
        reference = BareProductReference(dims=dims, local_vectors=(first, None, last))
        product = jnp.kron(jnp.kron(first, jnp.eye(3)), last)
        expected = jnp.abs(product.conj().T @ vectors) ** 2
        assert jnp.allclose(compute_overlaps(reference, vectors), expected, atol=1e-12)
        assert reference.keys[7] == (1, 0, 1)

    def test_local_product_reference_gradients_match_explicit_tensor_product(self) -> None:
        def objective(angle, factored):
            rotation = jnp.array([[jnp.cos(angle), -1j * jnp.sin(angle)],
                                  [-1j * jnp.sin(angle), jnp.cos(angle)]])
            vectors = jnp.eye(6, dtype=complex)
            product = jnp.kron(rotation, jnp.eye(3))
            reference = (BareProductReference((2, 3), local_vectors=(rotation, None)) if factored
                         else EigenstateReference(product.T, keys=tuple(range(6))))
            return jnp.sum(jnp.arange(36).reshape(6, 6) ** 2 * compute_overlaps(reference, vectors))

        actual = jax.jit(jax.value_and_grad(lambda angle: objective(angle, True)))(0.3)
        expected = jax.jit(jax.value_and_grad(lambda angle: objective(angle, False)))(0.3)
        assert abs(expected[1]) > 1.0
        assert jnp.allclose(jnp.asarray(actual), jnp.asarray(expected), atol=1e-10)


class TestTraceability:
    def test_label_eigensystem_is_jittable(self) -> None:
        """label_eigensystem is jittable and returns energies for all four labeled states."""
        ref = BareProductReference(dims=(2, 2))

        @jax.jit
        def labeled_energies(g):
            H = _two_qubit_H(g)
            evals, evecs = jnp.linalg.eigh(H)
            labeling = label_eigensystem(evecs, ref)
            return evals[labeling.indices]

        energies = labeled_energies(jnp.float32(0.05))
        assert energies.shape == (4,)

    def test_grad_through_labeled_energy_is_finite_away_from_crossings(self) -> None:
        """Gradient of a labeled energy through label_eigensystem is finite away from level crossings."""
        ref = BareProductReference(dims=(2, 2))

        def loss(g):
            H = _two_qubit_H(g)
            evals, evecs = jnp.linalg.eigh(H)
            labeling = label_eigensystem(evecs, ref)
            return evals[labeling.indices[3]]  # |11⟩ energy

        grad = jax.grad(loss)(jnp.float32(0.08))
        assert jnp.isfinite(grad)

    def test_vmap_over_parameter(self) -> None:
        """label_eigensystem vmaps over a batch of coupling parameters, returning finite ground energies."""
        ref = BareProductReference(dims=(2, 2))

        def labeled_ground(g):
            H = _two_qubit_H(g)
            evals, evecs = jnp.linalg.eigh(H)
            labeling = label_eigensystem(evecs, ref)
            return evals[labeling.indices[0]]

        gs = jnp.linspace(0.0, 0.1, 5, dtype=jnp.float32)
        ground_energies = jax.vmap(labeled_ground)(gs)
        assert ground_energies.shape == (5,)
        assert jnp.all(jnp.isfinite(ground_energies))


class TestGaugePhases:
    def test_zero_assigned_amplitude_takes_the_phase_of_the_largest_amplitude(self) -> None:
        """A column with no amplitude on its assigned label takes the phase of its largest amplitude."""
        root3 = np.sqrt(3.0)
        U = np.array([[root3, root3, root3, 0.0], [-2, 1, 1, root3], [1, -2, 1, root3], [1, 1, -2, root3]]) / 3
        _, evecs = np.linalg.eigh(U @ np.diag([1.0, 2.0, 3.0, 10.0]) @ U.T)
        labeling = label_eigensystem(evecs, BareProductReference(dims=(4,)))
        # Label 0 takes column 3, which has no amplitude on label 0.
        assert np.asarray(labeling.indices).tolist() == [3, 0, 1, 2]
        expected = U * np.array([-1.0, -1.0, -1.0, 1.0])
        for theta in (0.0, 0.7):
            # The reference is the identity basis, so the amplitudes are the vectors.
            vectors = evecs * np.exp(1j * theta * np.arange(1, 5))
            np.testing.assert_allclose(vectors * gauge_phases(vectors, labeling.indices), expected, atol=1e-12)
            traced = jax.jit(gauge_phases)(jnp.asarray(vectors), jnp.asarray(labeling.indices))
            np.testing.assert_allclose(vectors * np.asarray(traced), expected, atol=1e-12)

    def test_gauge_fixed_eigenvector_derivative_matches_central_differences(self) -> None:
        """A gauge-fixed eigenvector of a complex Hamiltonian has the derivative of its central differences."""
        reference = BareProductReference(dims=(2,))
        hop = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)

        def lower_state(phi):
            H = jnp.diag(jnp.array([0.0, 2.0], dtype=complex)) + jnp.exp(1j * phi) * hop + jnp.exp(-1j * phi) * hop.T
            _, evecs = jnp.linalg.eigh(H)
            labeling = label_eigensystem(evecs, reference)
            return (evecs * gauge_phases(evecs, labeling.indices))[:, labeling.indices[0]]

        phi, step = 0.3, 1e-6
        _, tangent = jax.jvp(lower_state, (phi,), (1.0,))
        central = (lower_state(phi + step) - lower_state(phi - step)) / (2 * step)
        np.testing.assert_allclose(tangent, central, atol=1e-8)


class TestTrackPath:
    def test_track_path_through_smooth_sweep(self) -> None:
        """track_path produces no swap events across a smooth sweep below any avoided crossing."""
        ref = BareProductReference(dims=(2, 2))
        gs = jnp.linspace(0.0, 0.1, 6, dtype=jnp.float32)

        def eig(g):
            return jnp.linalg.eigh(_two_qubit_H(g))

        evals_path, evecs_path = jax.vmap(eig)(gs)
        path = track_path(evals_path, evecs_path, ref)
        assert isinstance(path, LabelingPath)
        assert path.indices.shape == (6, 4)
        assert path.swap_events.shape == (5, 4)
        # Smooth sweep below any avoided crossing: no swaps.
        assert not bool(path.swap_events.any())

    def test_track_path_final_matches_direct_labeling(self) -> None:
        """track_path's final labeling matches a direct label_eigensystem call on the same eigensystem."""
        ref = BareProductReference(dims=(2, 2))
        gs = jnp.array([0.0, 0.05, 0.08], dtype=jnp.float32)

        def eig(g):
            return jnp.linalg.eigh(_two_qubit_H(g))

        evals_path, evecs_path = jax.vmap(eig)(gs)
        path = track_path(evals_path, evecs_path, ref)

        direct = label_eigensystem(evecs_path[-1], ref)
        # Path's final step and a direct labeling of the final eigensystem
        # should agree in the smooth (non-crossing) regime.
        assert path.final.indices.tolist() == direct.indices.tolist()
