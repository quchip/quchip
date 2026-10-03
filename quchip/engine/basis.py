"""Local energy-basis resolution and explicit transformation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from quchip.utils.jax_utils import contains_tracer


def _eigenpairs(matrix: Any, levels: int) -> tuple[Any, Any]:
    """Return the lowest ``levels`` eigenpairs from a plain Hermitian eigensolve."""
    values, vectors = jnp.linalg.eigh(matrix)
    return values[:levels], vectors[:, :levels]


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _differentiable_eigenpairs(matrix: Any, levels: int) -> tuple[Any, Any]:
    return _eigenpairs(matrix, levels)


@_differentiable_eigenpairs.defjvp
def _eigenpairs_jvp(
    levels: int, primals: tuple[Any], tangents: tuple[Any]
) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    """Return first-order tangents for the retained Hermitian eigenpairs.

    The rule Hermitianizes ``dH`` and evaluates
    ``dλ_i = <v_i|dH|v_i>`` and
    ``dv_i = Σ_{j≠i} v_j <v_j|dH|v_i> / (λ_i − λ_j)`` over the full
    eigensystem. Gaps at or below ``1e-9·(λ_max − λ_min)`` contribute zero;
    the tolerance is independent of the energy origin. The nested
    ``where`` also keeps the reciprocal off the masked branch. The diagonal
    connection is zero, fixing the parallel-transport gauge
    ``v_i† dv_i = 0``, and eigenvalue tangents are real.

    The rule is linear in ``dH``, so JAX obtains reverse mode by transposition;
    ``grad``, ``jacfwd``, and ``hessian`` work through traced device parameters.
    An exact degeneracy uses a zero eigenvector connection instead of a NaN.
    Second derivatives at a degeneracy remain undefined because the outer
    derivative passes through the rule's unmasked ``eigh``.
    """
    (matrix,) = primals
    (dmatrix,) = tangents
    values, vectors = jnp.linalg.eigh(matrix)
    retained = vectors[:, :levels]
    overlaps, dvectors = _masked_connection(values, vectors, dmatrix, values[:levels], retained)
    dvalues = jnp.real(jnp.diagonal(overlaps))
    return (values[:levels], retained), (dvalues, dvectors)


def _masked_connection(values: Any, vectors: Any, dmatrix: Any, energies: Any, columns: Any) -> tuple[Any, Any]:
    """Return ``<v_j|dH|columns>`` and the masked first-order change of ``columns``.

    ``(values, vectors)`` is the full eigensystem. Each column in ``columns``
    is its eigenvector at the corresponding entry in ``energies``.
    """
    hermitian = 0.5 * (dmatrix + dmatrix.conj().T)
    overlaps = vectors.conj().T @ hermitian @ columns
    gaps = energies[None, :] - values[:, None]
    tolerance = 1e-9 * (values[-1] - values[0])
    resolved = jnp.abs(gaps) > tolerance
    inverse_gaps = jnp.where(resolved, 1.0 / jnp.where(resolved, gaps, 1.0), 0.0)
    return overlaps, vectors @ (inverse_gaps * overlaps)


@jax.custom_jvp
def _differentiable_eigenvector(matrix: Any, energy: Any, values: Any, vectors: Any, vector: Any) -> Any:
    """Return ``vector``, the eigenvector of ``matrix`` at ``energy`` in ``(values, vectors)``.

    The derivative applies the masked rule of :func:`_differentiable_eigenpairs`
    to this column and depends on ``dH`` alone. It reuses the forward
    eigensystem, so unrelated eigenvalue ties leave the derivative finite
    instead of invoking JAX's full ``eigh`` derivative.
    """
    del matrix, energy, values, vectors
    return vector


@_differentiable_eigenvector.defjvp
def _eigenvector_jvp(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> tuple[Any, Any]:
    _, energy, values, vectors, vector = primals
    _, dvectors = _masked_connection(values, vectors, tangents[0], jnp.reshape(energy, (1,)), vector[:, None])
    return vector, dvectors[:, 0]


def _lowest_eigenpairs(matrix: Any, levels: int) -> tuple[Any, Any]:
    """Return the lowest ``levels`` eigenpairs without staging a constant matrix.

    A traced matrix uses :func:`_differentiable_eigenpairs`, which supports
    forward- and reverse-mode differentiation through device parameters. A
    constant matrix uses the plain eigensolve so
    ``jax.ensure_compile_time_eval`` can finish inside ``jit``; a
    custom-derivative call would always be staged.
    """
    if contains_tracer(matrix):
        values, vectors = _differentiable_eigenpairs(matrix, levels)
    else:
        values, vectors = _eigenpairs(matrix, levels)
    # Fix each phase by its largest authored-basis component (first on ties).
    # The pivot is locally constant; derivatives include the phase adjustment.
    pivots = vectors[jnp.argmax(jnp.abs(vectors), axis=0), jnp.arange(levels)]
    phases = jnp.conj(pivots) / jnp.abs(pivots)
    return values, vectors * phases


@dataclass(frozen=True)
class BasisRecord:
    """One device's fixed authored-to-solver transformation.

    Attributes
    ----------
    kind : {"native", "eigen"}
        Selected solver-basis policy.
    vectors : array_like
        Solver basis vectors as authored-basis columns.
    energies : array_like
        Retained isolated eigenenergies in GHz.
    energy_vectors : array_like
        Isolated energy eigenvectors as authored-basis columns.
    native_dim, resolved_dim : int
        Authored and retained local dimensions.
    authored_hamiltonian : array_like or None
        Captured authored-basis Hamiltonian in GHz, when retained.
    """

    kind: Literal["native", "eigen"]
    vectors: Any
    energies: Any
    energy_vectors: Any
    native_dim: int
    resolved_dim: int
    authored_hamiltonian: Any = field(default=None, repr=False, compare=False)

    def energy_state(self, level: int) -> Any:
        """Return one captured isolated energy ket in the solver basis.

        Parameters
        ----------
        level : int
            Zero-based isolated energy level.
        """
        if self.kind == "eigen":
            return jnp.eye(self.resolved_dim, dtype=jnp.complex128)[:, level]
        return self.energy_vectors[:, level]

    def energy_to_solver(self) -> Any | None:
        """Return the energy-to-solver map, or None when the bases coincide."""
        if self.kind == "eigen":
            return None
        if not contains_tracer(self.energy_vectors) and np.array_equal(
            self.energy_vectors, np.eye(self.resolved_dim)
        ):
            return None
        return self.energy_vectors

    @property
    def projector(self) -> Any:
        """Projector onto the retained authored subspace."""
        return self.vectors @ self.vectors.conj().T

    def transform_operator(self, operator: Any) -> Any:
        """Apply the recorded authored-to-solver transformation to an operator.

        Parameters
        ----------
        operator : array_like
            Square operator with shape ``(native_dim, native_dim)``.
        """
        if getattr(operator, "shape", None) != (self.native_dim, self.native_dim):
            raise ValueError(
                f"Operator must have native shape {(self.native_dim, self.native_dim)}, "
                f"got {getattr(operator, 'shape', None)}."
            )
        return self.vectors.conj().T @ operator @ self.vectors

    def level_operator(self) -> Any:
        """Return the energy-level index operator in the resolved solver basis."""
        if self.kind == "eigen":
            return jnp.asarray(np.diag(np.arange(self.resolved_dim, dtype=complex)))
        return self.authored_level_operator()

    def authored_level_operator(self) -> Any:
        """Energy-level index in the authored basis, on the retained subspace."""
        vectors = self.energy_vectors if contains_tracer(self.energy_vectors) else np.asarray(self.energy_vectors)
        indices = np.arange(vectors.shape[1], dtype=complex)
        return jnp.asarray((vectors * indices) @ vectors.conj().T)


def resolve_local_basis(
    hamiltonian: Any,
    *,
    basis: Literal["native", "eigen"] = "native",
    levels: int | None = None,
) -> BasisRecord:
    """Resolve one fixed local solver basis from a static authored Hamiltonian."""
    shape = getattr(hamiltonian, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Local Hamiltonian must be square, got shape {shape}.")
    native_dim = shape[0]
    if basis == "native":
        if levels is not None:
            raise ValueError("levels is only valid when basis='eigen'.")
        energies, energy_vectors = _lowest_eigenpairs(hamiltonian, native_dim)
        return BasisRecord(
            kind="native",
            vectors=jnp.asarray(np.eye(native_dim, dtype=hamiltonian.dtype)),
            energies=energies,
            energy_vectors=energy_vectors,
            native_dim=native_dim,
            resolved_dim=native_dim,
        )
    if basis != "eigen":
        raise ValueError(f"basis must be 'native' or 'eigen', got {basis!r}.")
    if levels is None:
        raise ValueError("levels is required when basis='eigen'.")
    if levels < 1 or levels > native_dim:
        raise ValueError(f"levels must be between 1 and {native_dim}, got {levels}.")
    energies, vectors = _lowest_eigenpairs(hamiltonian, levels)
    return BasisRecord(
        kind="eigen",
        vectors=vectors,
        energies=energies,
        energy_vectors=vectors,
        native_dim=native_dim,
        resolved_dim=levels,
    )


def resolve_device_basis(
    device: Any,
    *,
    basis: Literal["native", "eigen"],
    levels: int | None = None,
) -> BasisRecord:
    """Resolve a device from its exact authored static Hamiltonian."""
    from quchip.declarative.expr import materialize_array

    matrix = materialize_array(device.unresolved_hamiltonian())
    return resolve_local_basis(matrix, basis=basis, levels=levels)
