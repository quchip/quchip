"""Backend-neutral local Hilbert spaces and their named operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp


class LocalSpace(ABC):
    """Numerical realization of the operators used by one device model."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the authored local-space dimension."""

    @abstractmethod
    def matrix(self, name: str) -> Any:
        """Return one named operator as a JAX-compatible dense array."""

    def operator(self, name: str, backend: Any) -> Any:
        """Lower one named local operator through ``backend``."""
        return backend.from_array(
            self.matrix(name),
            dims=[[self.dimension], [self.dimension]],
        )


@dataclass(frozen=True)
class FockSpace(LocalSpace):
    """Finite Fock ladder with the standard bosonic and qubit operators."""

    levels: int

    def __post_init__(self) -> None:
        if self.levels < 2:
            raise ValueError(f"levels must be >= 2, got {self.levels}")

    @property
    def dimension(self) -> int:
        return self.levels

    def matrix(self, name: str) -> Any:
        annihilation = jnp.diag(jnp.sqrt(jnp.arange(1, self.levels)), 1).astype(jnp.complex128)
        if name == "a":
            return annihilation
        if name == "adag":
            return annihilation.conj().T
        if name == "n":
            return jnp.diag(jnp.arange(self.levels, dtype=jnp.complex128))
        if name == "I":
            return jnp.eye(self.levels, dtype=jnp.complex128)
        zero = jnp.zeros((self.levels, self.levels), dtype=jnp.complex128)
        if name == "sigma_x":
            return zero.at[0, 1].set(1).at[1, 0].set(1)
        if name == "sigma_y":
            return zero.at[0, 1].set(-1j).at[1, 0].set(1j)
        if name == "sigma_z":
            return zero.at[0, 0].set(1).at[1, 1].set(-1)
        if name == "sigma_plus":
            return zero.at[1, 0].set(1)
        if name == "sigma_minus":
            return zero.at[0, 1].set(1)
        raise ValueError(f"Unknown Fock-space operator {name!r}.")

    def operator(self, name: str, backend: Any) -> Any:
        if name == "a":
            return backend.destroy(self.levels)
        if name == "adag":
            return backend.create(self.levels)
        if name == "n":
            return backend.number(self.levels)
        if name == "I":
            return backend.identity(self.levels)

        zero = backend.basis(self.levels, 0)
        one = backend.basis(self.levels, 1)
        p01 = backend.matmul(zero, backend.dag(one))
        p10 = backend.matmul(one, backend.dag(zero))
        if name == "sigma_x":
            return p01 + p10
        if name == "sigma_y":
            return -1j * p01 + 1j * p10
        if name == "sigma_z":
            return backend.matmul(zero, backend.dag(zero)) - backend.matmul(one, backend.dag(one))
        if name == "sigma_plus":
            return p10
        if name == "sigma_minus":
            return p01
        raise ValueError(f"Unknown Fock-space operator {name!r}.")


@dataclass(frozen=True)
class ChargeSpace(LocalSpace):
    """Finite integer-charge basis centered on zero charge."""

    num_basis: int

    def __post_init__(self) -> None:
        if self.num_basis < 3 or self.num_basis % 2 == 0:
            raise ValueError(f"num_basis must be an odd integer >= 3, got {self.num_basis}")

    @property
    def dimension(self) -> int:
        return self.num_basis

    def matrix(self, name: str) -> Any:
        plus = jnp.eye(self.num_basis, k=1, dtype=jnp.complex128)
        minus = jnp.eye(self.num_basis, k=-1, dtype=jnp.complex128)
        if name == "n":
            cutoff = (self.num_basis - 1) // 2
            value = jnp.diag(jnp.arange(-cutoff, cutoff + 1, dtype=jnp.complex128))
        elif name == "cos_phi":
            value = 0.5 * (plus + minus)
        elif name == "sin_phi":
            value = (plus - minus) / (2j)
        elif name == "I":
            value = jnp.eye(self.num_basis, dtype=jnp.complex128)
        else:
            raise ValueError(f"Unknown charge-space operator {name!r}.")
        return value


@dataclass(frozen=True)
class PhaseGridSpace(LocalSpace):
    """Uniform endpoint-excluded phase grid with nonperiodic finite differences.

    The centered-difference stencil does not wrap across the grid boundary;
    values beyond either endpoint are treated as zero.
    """

    points: int
    extent: float

    def __post_init__(self) -> None:
        if self.points < 3:
            raise ValueError(f"points must be >= 3, got {self.points}")
        if self.extent <= 0:
            raise ValueError(f"extent must be positive, got {self.extent}")

    @property
    def dimension(self) -> int:
        return self.points

    def matrix(self, name: str) -> Any:
        phase = jnp.linspace(-self.extent, self.extent, self.points, endpoint=False)
        spacing = 2.0 * self.extent / self.points
        plus = jnp.eye(self.points, k=1, dtype=jnp.complex128)
        minus = jnp.eye(self.points, k=-1, dtype=jnp.complex128)
        charge = -1j * (plus - minus) / (2.0 * spacing)
        if name == "phi":
            value = jnp.diag(phase.astype(jnp.complex128))
        elif name == "n":
            value = charge
        elif name == "n2":
            value = -(plus - 2.0 * jnp.eye(self.points) + minus) / spacing**2
        elif name == "cos_phi":
            value = jnp.diag(jnp.cos(phase).astype(jnp.complex128))
        elif name == "sin_phi":
            value = jnp.diag(jnp.sin(phase).astype(jnp.complex128))
        elif name == "I":
            value = jnp.eye(self.points, dtype=jnp.complex128)
        else:
            raise ValueError(f"Unknown phase-grid operator {name!r}.")
        return value


class CustomSpace(LocalSpace):
    """Named local operators supplied as matrices or zero-argument JAX callables."""

    def __init__(self, dimension: int, operators: Mapping[str, Any]) -> None:
        if dimension < 1:
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
