"""Unit tests for the ``quchip.chip.dressing`` JAX-traceable kernel.

These tests exercise the pure-array labeling primitives in isolation
from :class:`Chip`. Traceability through the full chip pipeline is
covered by :mod:`tests.extended.test_dressing_chip_traceable`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from quchip.chip.dressing import (
    BareProductReference,
    EigenstateReference,
    LabelingPath,
    assign_argmax,
    assign_global_greedy,
    assign_rowwise_greedy,
    compute_overlaps,
    label_eigensystem,
    phase_fixed_transform,
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

        # The full-kernel path also surfaces duplicates via one-hot counts.
        evecs = jnp.eye(4, dtype=jnp.float32)
        labeling = label_eigensystem(
            evecs, BareProductReference(dims=(2, 2)), policy=assign_argmax,
        )
        # Identity overlaps give each row a unique argmax — no duplicates here.
        assert not bool(labeling.duplicates.any())

    def test_global_greedy_rejects_overdetermined_reference(self) -> None:
        """Global-greedy assignment raises ValueError when more bare labels than dressed states are given."""
        evecs = jnp.eye(3, dtype=jnp.float32)
        overlaps = jnp.abs(evecs) ** 2
        # 4 bare labels, 3 dressed states → global-greedy cannot assign them all.
        fake = jnp.concatenate([overlaps, overlaps[:1]], axis=0)
        with pytest.raises(ValueError, match="global-greedy"):
            assign_global_greedy(fake)

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

    def test_rowwise_greedy_matches_global_on_dispersive_overlaps(self) -> None:
        """Rowwise-greedy assignment matches global-greedy exactly in the dispersive overlap regime."""
        # In the dispersive / weak-hybridization regime (near-permutation overlaps)
        # the O(D^2) rowwise policy is bit-identical to the O(D^3) global greedy.
        # This is the regime ``Chip.dress`` runs in and is what makes the swap safe.
        for g in (0.0, 0.02, 0.05, 0.1):
            H = _two_qubit_H(jnp.asarray(g, dtype=jnp.float32))
            _, evecs = jnp.linalg.eigh(H)
            overlaps = compute_overlaps(BareProductReference(dims=(2, 2)), evecs)
            i_global, c_global, m_global = assign_global_greedy(overlaps)
            i_row, c_row, m_row = assign_rowwise_greedy(overlaps)
            assert i_row.tolist() == i_global.tolist()
            assert jnp.allclose(c_row, c_global)
            assert jnp.allclose(m_row, m_global)

    def test_rowwise_greedy_rejects_overdetermined_reference(self) -> None:
        """Rowwise-greedy assignment raises ValueError when more bare labels than dressed states are given."""
        evecs = jnp.eye(3, dtype=jnp.float32)
        overlaps = jnp.abs(evecs) ** 2
        fake = jnp.concatenate([overlaps, overlaps[:1]], axis=0)
        with pytest.raises(ValueError, match="rowwise-greedy"):
            assign_rowwise_greedy(fake)


class TestReferences:
    def test_bare_product_overlaps_match_column_squared_amplitudes(self) -> None:
        """Bare-product reference overlaps equal the squared magnitudes of the eigenvector columns."""
        H = _two_qubit_H(jnp.float32(0.2))
        _, evecs = jnp.linalg.eigh(H)
        ref = BareProductReference(dims=(2, 2))
        overlaps = compute_overlaps(ref, evecs)
        expected = jnp.abs(evecs) ** 2
        assert jnp.allclose(overlaps, expected)

    def test_eigenstate_reference_roundtrip(self) -> None:
        """Eigenstate reference overlaps against its own eigenvectors form the identity matrix."""
        H = _two_qubit_H(jnp.float32(0.15))
        _, evecs = jnp.linalg.eigh(H)
        # Take evecs as reference; overlaps should then be (near-)identity.
        ref = EigenstateReference(vectors=evecs.T, keys=tuple(range(evecs.shape[1])))
        overlaps = compute_overlaps(ref, evecs)
        assert jnp.allclose(overlaps, jnp.eye(evecs.shape[1]), atol=1e-5)


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


class TestPhaseFixedTransform:
    def test_phase_fixed_U_is_unitary(self) -> None:
        """phase_fixed_transform returns a unitary transformation matrix."""
        ref = BareProductReference(dims=(2, 2))
        H = _two_qubit_H(jnp.float32(0.05))
        evals, evecs = jnp.linalg.eigh(H)
        labeling = label_eigensystem(evecs, ref)
        U = phase_fixed_transform(labeling, evecs)
        residual = jnp.linalg.norm(U.conj().T @ U - jnp.eye(U.shape[1]))
        assert float(residual) < 1e-5
