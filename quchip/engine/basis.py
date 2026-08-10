"""Local energy-basis resolution and explicit transformation records."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _lowest_eigenpairs(matrix: Any, levels: int) -> tuple[Any, Any]:
    values, vectors = jnp.linalg.eigh(matrix)
    return values[:levels], vectors[:, :levels]


def _lowest_eigenpairs_fwd(matrix: Any, levels: int) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    values, vectors = jnp.linalg.eigh(matrix)
    return (values[:levels], vectors[:, :levels]), (values, vectors)


def _lowest_eigenpairs_bwd(
    levels: int,
    residuals: tuple[Any, Any],
    cotangents: tuple[Any, Any],
) -> tuple[Any]:
    values, vectors = residuals
    values_bar, vectors_bar = cotangents
    retained = vectors[:, :levels]

    gradient = (retained * values_bar[None, :]) @ retained.conj().T
    gaps = values[None, :levels] - values[:, None]
    tolerance = 1e-9 * jnp.max(jnp.abs(values))
    resolved = jnp.abs(gaps) > tolerance
    inverse_gaps = jnp.where(
        resolved,
        1.0 / jnp.where(resolved, gaps, 1.0),
        0.0,
    )
    overlaps = vectors.conj().T @ vectors_bar
    gradient = gradient + vectors @ (inverse_gaps * overlaps) @ retained.conj().T
    gradient = 0.5 * (gradient + gradient.conj().T)
    return (gradient.astype(vectors.dtype),)


_lowest_eigenpairs.defvjp(_lowest_eigenpairs_fwd, _lowest_eigenpairs_bwd)


@dataclass(frozen=True)
class BasisRecord:
    """One device's fixed authored-to-solver transformation."""

    kind: Literal["native", "eigen"]
    vectors: Any
    energies: Any
    energy_vectors: Any
    native_dim: int
    resolved_dim: int

    @property
    def projector(self) -> Any:
        """Projector onto the retained authored subspace."""
        return self.vectors @ self.vectors.conj().T

    def transform_operator(self, operator: Any) -> Any:
        """Apply the recorded authored-to-solver transformation to an operator."""
        if getattr(operator, "shape", None) != (self.native_dim, self.native_dim):
            raise ValueError(
                f"Operator must have native shape {(self.native_dim, self.native_dim)}, "
                f"got {getattr(operator, 'shape', None)}."
            )
        return self.vectors.conj().T @ operator @ self.vectors

    def level_operator(self) -> Any:
        """Return the energy-level index operator in the resolved solver basis."""
        indices = jnp.diag(jnp.arange(self.energy_vectors.shape[1], dtype=jnp.complex128))
        if self.kind == "eigen":
            return indices
        return self.energy_vectors @ indices @ self.energy_vectors.conj().T


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
            vectors=jnp.eye(native_dim, dtype=hamiltonian.dtype),
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

    matrix = materialize_array(device.hamiltonian())
    return resolve_local_basis(matrix, basis=basis, levels=levels)


def semantic_to_solver_transform(device: Any, record: BasisRecord) -> Any | None:
    """Map semantic local levels into the resolved solver basis when needed.

    Fock devices label authored occupation states. Other local spaces label
    energy-ordered states. ``None`` means those labels already coincide with
    solver indices, allowing sparse band decomposition to stay sparse.
    """
    from quchip.devices.spaces import FockSpace

    if isinstance(device.local_space(), FockSpace):
        if record.kind == "native":
            return None
        local_vectors = jnp.eye(record.native_dim, dtype=jnp.complex128)[
            :, : record.resolved_dim
        ]
    else:
        if record.kind == "eigen":
            return None
        local_vectors = record.energy_vectors[:, : record.resolved_dim]
    return record.vectors.conj().T @ local_vectors
