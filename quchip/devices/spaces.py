"""Backend-neutral local Hilbert spaces and their named operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from operator import index
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np

from quchip.utils.jax_utils import contains_tracer


@dataclass(frozen=True)
class TruncationBoundary:
    """Authored basis indices near a cutoff and the corresponding convergence check.

    Parameters
    ----------
    indices : tuple[int, ...]
        Distinct non-negative boundary indices.
    description : str
        Human-readable boundary name.
    convergence_hint : str
        Suggested convergence comparison.
    """

    indices: tuple[int, ...]
    description: str
    convergence_hint: str

    def __post_init__(self) -> None:
        indices = tuple(index(value) for value in self.indices)
        if len(set(indices)) != len(indices) or any(value < 0 for value in indices):
            raise ValueError("Truncation boundary indices must be distinct and nonnegative")
        object.__setattr__(self, "indices", indices)


class LocalSpace(ABC):
    """Numerical realization of the operators used by one device model."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the authored local-space dimension."""

    @abstractmethod
    def matrix(self, name: str) -> Any:
        """Return one named operator as a JAX-compatible dense array.

        Parameters
        ----------
        name : str
            Operator name supported by the concrete space.
        """

    def truncation_boundary(self) -> TruncationBoundary | None:
        """Describe a cutoff, or return None for an intrinsically finite space."""
        raise NotImplementedError(f"{type(self).__name__} does not declare a truncation boundary")

    def operator(self, name: str, backend: Any) -> Any:
        """Lower one named local operator through ``backend``.

        Parameters
        ----------
        name : str
            Supported local operator name.
        backend : backend protocol
            Object that supplies native operator constructors.
        """
        return backend.from_array(
            self.matrix(name),
            dims=[[self.dimension], [self.dimension]],
        )


@dataclass(frozen=True)
class FockSpace(LocalSpace):
    """Finite Fock ladder with standard bosonic and qubit operators.

    Parameters
    ----------
    levels : int
        Hilbert-space dimension, at least 2. The ``a`` ladder is truncated at
        ``levels - 1``.
    """

    levels: int

    def __post_init__(self) -> None:
        if index(self.levels) < 2:
            raise ValueError(f"levels must be >= 2, got {self.levels}")

    def truncation_boundary(self) -> TruncationBoundary:
        return TruncationBoundary((self.levels - 1,), "Fock boundary", "Increase levels and compare observables.")

    @property
    def dimension(self) -> int:
        return self.levels

    def matrix(self, name: str) -> Any:
        """Return a named Fock-space matrix as a read-only NumPy array.

        Parameters
        ----------
        name : {"a", "adag", "n", "I", "sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus"}
            Operator to construct.
        """
        corners = {"sigma_x": [[0, 1], [1, 0]], "sigma_y": [[0, -1j], [1j, 0]], "sigma_z": [[1, 0], [0, -1]],
                   "sigma_plus": [[0, 0], [1, 0]], "sigma_minus": [[0, 1], [0, 0]]}
        if name in ("a", "adag"):
            value = np.diag(np.sqrt(np.arange(1, self.levels)), 1).astype(complex)
            value = value if name == "a" else value.conj().T
        elif name == "n":
            value = np.diag(np.arange(self.levels, dtype=complex))
        elif name == "I":
            value = np.eye(self.levels, dtype=complex)
        elif name in corners:
            value = np.pad(np.asarray(corners[name], dtype=complex), (0, self.levels - 2))
        else:
            raise ValueError(f"Unknown Fock-space operator {name!r}.")
        value.setflags(write=False)
        return value

    def operator(self, name: str, backend: Any) -> Any:
        if name == "a":
            return backend.destroy(self.levels)
        if name == "adag":
            return backend.create(self.levels)
        if name == "n":
            return backend.number(self.levels)
        if name == "I":
            return backend.identity(self.levels)
        if name in {"sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus"}:
            return backend.from_array(
                self.matrix(name),
                dims=[[self.levels], [self.levels]],
            )
        raise ValueError(f"Unknown Fock-space operator {name!r}.")


@dataclass(frozen=True)
class ChargeSpace(LocalSpace):
    """Finite integer-charge basis centered on zero charge.

    Parameters
    ----------
    num_basis : int
        Odd basis size, at least 3, with charges from ``-(num_basis-1)//2`` to
        ``+(num_basis-1)//2``.
    """

    num_basis: int

    def __post_init__(self) -> None:
        if index(self.num_basis) < 3 or self.num_basis % 2 == 0:
            raise ValueError(f"num_basis must be an odd integer >= 3, got {self.num_basis}")

    def truncation_boundary(self) -> TruncationBoundary:
        return TruncationBoundary((0, self.num_basis - 1), "charge-basis edges",
                                  "Increase num_basis and compare observables.")

    @property
    def dimension(self) -> int:
        return self.num_basis

    def matrix(self, name: str) -> Any:
        """Return a named charge-space matrix as a read-only NumPy array.

        Parameters
        ----------
        name : {"n", "cos_phi", "sin_phi", "I"}
            Operator to construct.
        """
        if name == "n":
            value = np.diag(np.arange(self.num_basis, dtype=complex) - (self.num_basis - 1) // 2)
        elif name == "cos_phi":
            value = 0.5 * (np.eye(self.num_basis, k=1, dtype=complex) + np.eye(self.num_basis, k=-1, dtype=complex))
        elif name == "sin_phi":
            value = (np.eye(self.num_basis, k=-1, dtype=complex) - np.eye(self.num_basis, k=1, dtype=complex)) / (2j)
        elif name == "I":
            value = np.eye(self.num_basis, dtype=complex)
        else:
            raise ValueError(f"Unknown charge-space operator {name!r}.")
        value.setflags(write=False)
        return value


@dataclass(frozen=True)
class PhaseGridSpace(LocalSpace):
    """Uniform endpoint-excluded phase grid with nonperiodic finite differences.

    The centered-difference stencil does not wrap across the grid boundary.
    Values beyond each endpoint are zero.
    Parameters
    ----------
    points : int
        Number of endpoint-excluded grid points, at least 3.
    extent : float
        Positive half-width of the grid, in dimensionless phase radians.
    """

    points: int
    extent: float

    def __post_init__(self) -> None:
        if index(self.points) < 3:
            raise ValueError(f"points must be >= 3, got {self.points}")
        if self.extent <= 0:
            raise ValueError(f"extent must be positive, got {self.extent}")

    def truncation_boundary(self) -> TruncationBoundary:
        return TruncationBoundary((0, self.points - 1), "phase-grid edges",
                                  "Increase the phase range at fixed grid spacing and compare observables.")

    @property
    def dimension(self) -> int:
        return self.points

    def matrix(self, name: str) -> Any:
        """Return a named phase-grid matrix, read-only NumPy unless ``extent`` is traced.

        Parameters
        ----------
        name : {"phi", "n", "n2", "cos_phi", "sin_phi", "I"}
            Operator to construct.
        """
        spacing = 2.0 * self.extent / self.points
        plus = np.eye(self.points, k=1, dtype=complex)
        minus = np.eye(self.points, k=-1, dtype=complex)
        if name in ("phi", "cos_phi", "sin_phi"):
            # The JAX kernels sample the grid, so values match their eager construction bit for bit.
            samples = {"phi": jnp.asarray, "cos_phi": jnp.cos, "sin_phi": jnp.sin}[name](
                jnp.linspace(-self.extent, self.extent, self.points, endpoint=False))
            if contains_tracer(samples):  # Staged inside an enclosing JAX trace.
                return jnp.diag(samples.astype(complex))
            value = np.diag(np.asarray(samples).astype(complex))
        elif name == "n":
            value = -1j * (plus - minus) / (2.0 * spacing)
        elif name == "n2":
            value = -(plus - 2.0 * np.eye(self.points) + minus) / spacing**2
        elif name == "I":
            value = np.eye(self.points, dtype=complex)
        else:
            raise ValueError(f"Unknown phase-grid operator {name!r}.")
        if isinstance(value, np.ndarray):  # A traced extent gives a JAX array.
            value.setflags(write=False)
        return value


class CustomSpace(LocalSpace):
    """Named local operators supplied as matrices or zero-argument JAX callables.

    Parameters
    ----------
    dimension : int
        Positive matrix dimension.
    operators : mapping[str, array or callable]
        Operator providers. Each resulting matrix needs shape
        ``(dimension, dimension)``.
    """

    def __init__(self, dimension: int, operators: Mapping[str, Any]) -> None:
        if index(dimension) < 1:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension
        self.operators = MappingProxyType(dict(operators))

    @property
    def dimension(self) -> int:
        return self._dimension

    def matrix(self, name: str) -> Any:
        try:
            provider = self.operators[name]
        except KeyError as exc:
            raise ValueError(f"Unknown custom-space operator {name!r}.") from exc
        value = provider() if callable(provider) else provider
        if getattr(value, "shape", None) != (self.dimension, self.dimension):
            raise ValueError(
                f"Custom operator {name!r} must have shape "
                f"{(self.dimension, self.dimension)}, got {getattr(value, 'shape', None)}."
            )
        return value
