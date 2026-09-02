"""Public solve-time observables for accessible SLH output fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class OutputAmplitude:
    r"""Mean output field ``<b_out>`` at one exposure reference plane."""

    exposure: str

    def __init__(self, exposure: str | Any) -> None:
        object.__setattr__(self, "exposure", resolve_label(exposure))


@dataclass(frozen=True)
class OutputQuadrature:
    r"""Mean field quadrature ``Re[exp(-i phase) <b_out>]``."""

    exposure: str
    phase: Any = 0.0

    def __init__(self, exposure: str | Any, *, phase: Any = 0.0) -> None:
        object.__setattr__(self, "exposure", resolve_label(exposure))
        object.__setattr__(self, "phase", phase)


@dataclass(frozen=True)
class OutputPhotonFlux:
    r"""Normally ordered output photon flux ``<b_out dagger b_out>``."""

    exposure: str

    def __init__(self, exposure: str | Any) -> None:
        object.__setattr__(self, "exposure", resolve_label(exposure))


OutputObservable: TypeAlias = OutputAmplitude | OutputQuadrature | OutputPhotonFlux


def is_output_observable(value: Any) -> bool:
    """Return whether *value* is a public output-field observable spec."""
    return isinstance(value, (OutputAmplitude, OutputQuadrature, OutputPhotonFlux))


__all__ = ["OutputAmplitude", "OutputPhotonFlux", "OutputQuadrature"]
