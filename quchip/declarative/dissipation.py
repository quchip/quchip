"""Authored Lindblad dissipation values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quchip.declarative.expr import as_operator_expr, as_scalar_expr
from quchip.utils.jax_utils import maybe_concrete_scalar


def collapse_parameter_paths(operator: Any, rate: Any) -> tuple[str, ...]:
    """Return expression parameter paths in authored operator-then-rate order."""
    paths: list[str] = []
    for value in (operator, rate):
        parameter_paths = getattr(value, "parameter_paths", None)
        if parameter_paths is not None:
            paths.extend(parameter_paths())
    return tuple(dict.fromkeys(paths))


@dataclass(frozen=True)
class CollapseChannel:
    """One unscaled Lindblad operator and its rate in inverse nanoseconds."""

    operator: Any
    rate: Any
    name: str

    def __post_init__(self) -> None:
        concrete_rate = maybe_concrete_scalar(self.rate)
        if concrete_rate is not None and concrete_rate < 0:
            raise ValueError("CollapseChannel.rate must be non-negative.")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CollapseChannel.name must be a non-empty string.")


def normalize_dissipation(
    authored: Any,
    *,
    labels: tuple[str, ...],
    dims: tuple[int, ...],
    owner: Any,
    scope: str,
    allowed: dict[str, Any] | None = None,
    bindings: dict[str, Any] | None = None,
) -> tuple[tuple[CollapseChannel, tuple[str, ...]], ...]:
    """Normalize one owner's authored local dissipation and infer dependencies."""
    if not isinstance(authored, tuple):
        raise TypeError(
            f"{type(owner).__name__}.dissipation() must return a tuple of "
            f"CollapseChannel values; got {type(authored).__name__}."
        )
    normalized: list[tuple[CollapseChannel, tuple[str, ...]]] = []
    for channel in authored:
        if not isinstance(channel, CollapseChannel):
            raise TypeError(
                f"{type(owner).__name__}.dissipation() must return "
                f"CollapseChannel values; got {type(channel).__name__}."
            )
        operator = as_operator_expr(
            channel.operator,
            labels=labels,
            dims=dims,
            name=rf"\hat L_{{{scope},{channel.name}}}",
            owner=owner,
            scope=scope,
            allowed=allowed,
        )
        rate = as_scalar_expr(
            channel.rate,
            name=rf"\gamma_{{{scope},{channel.name}}}",
            owner=owner,
            scope=scope,
            allowed=allowed,
        )
        if bindings:
            operator = operator.with_bindings(bindings)
            rate = rate.with_bindings(bindings)
        normalized.append(
            (
                CollapseChannel(operator, rate, channel.name),
                collapse_parameter_paths(operator, rate),
            )
        )
    return tuple(normalized)
