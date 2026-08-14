"""Explicit strategies for reducing an authored Hamiltonian."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterable


def _freeze_bands(value: Iterable[tuple[int, ...]] | None) -> frozenset[tuple[int, ...]] | None:
    """Validate and freeze an optional replacement band selection."""
    if value is None:
        return None
    try:
        bands = frozenset(value)
    except TypeError as error:
        raise TypeError("keep_bands must be an iterable of integer tuples.") from error
    if not bands:
        raise ValueError("keep_bands must contain at least one integer tuple.")
    valid = all(
        isinstance(band, tuple)
        and band
        and all(not isinstance(number, bool) and isinstance(number, int) for number in band)
        for band in bands
    )
    if not valid:
        raise TypeError("keep_bands must be an iterable of non-empty integer tuples.")
    return bands


class Approximation(ABC):
    """Immutable engine strategy applied after authored physics is assembled."""

    filters_terms: bool = False

    def keeps_operator_band(self, weights: tuple[int, ...]) -> bool:
        """Return whether a structural operator band survives reduction."""
        del weights
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return the stable serialized strategy tag."""
        if type(self) is Exact:
            return {"type": "Exact"}
        if type(self) is RWA:
            data: dict[str, Any] = {"type": "RWA"}
            if self.keep_bands is not None:
                data["keep_bands"] = [list(band) for band in sorted(self.keep_bands)]
            return data
        raise TypeError(f"{type(self).__name__} must define its own stable approximation serialization.")

    @staticmethod
    def from_dict(data: Any) -> "Approximation":
        """Restore one explicit strategy and reject retired schemas."""
        if not isinstance(data, dict):
            raise TypeError(
                "approximation must be a serialized Exact or RWA strategy, "
                f"got {type(data).__name__}."
            )
        strategy_type = data.get("type")
        if strategy_type == "Exact" and set(data) == {"type"}:
            return Exact()
        if strategy_type == "RWA" and set(data) <= {"type", "keep_bands"}:
            bands = data.get("keep_bands")
            return RWA(None if bands is None else {tuple(band) for band in bands})
        if strategy_type in {"Exact", "RWA"}:
            fields = sorted(set(data) - {"type", "keep_bands"})
            raise TypeError(f"Unsupported serialized approximation fields: {fields}")
        raise ValueError(f"Unknown approximation strategy {strategy_type!r}.")


@dataclass(frozen=True)
class Exact(Approximation):
    """Retain every term in the authored finite-dimensional Hamiltonian."""


@dataclass(frozen=True, init=False)
class RWA(Approximation):
    """First-order structural rotating-wave selection.

    ``keep_bands`` replaces, rather than extends, total-excitation conservation.
    """

    keep_bands: frozenset[tuple[int, ...]] | None
    filters_terms = True

    def __init__(self, keep_bands: Iterable[tuple[int, ...]] | None = None) -> None:
        object.__setattr__(self, "keep_bands", _freeze_bands(keep_bands))

    def keeps_operator_band(self, weights: tuple[int, ...]) -> bool:
        if self.keep_bands is not None:
            return weights in self.keep_bands
        return sum(weights) == 0


def require_approximation(value: Any) -> Approximation:
    """Validate a public approximation value without Boolean coercion."""
    if not isinstance(value, Approximation):
        raise TypeError(f"approximation must be an Exact or RWA strategy, got {type(value).__name__}.")
    return value


__all__ = ["Approximation", "Exact", "RWA"]
