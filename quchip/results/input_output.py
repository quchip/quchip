"""Results for continuous-wave port scattering calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class SParameterResult:
    """Complex small-signal scattering over a declared VNA sweep grid."""

    frequencies: Any
    input_port: str
    output_ports: tuple[str, ...]
    axes: tuple[tuple[str, Any], ...]
    shape: tuple[int, ...]
    steady_states: tuple[Any, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    _response: Mapping[tuple[str, str], Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_response", MappingProxyType(dict(self._response)))
        object.__setattr__(
            self,
            "diagnostics",
            tuple(MappingProxyType(dict(item)) for item in self.diagnostics),
        )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Names of the result axes, in array order."""
        return tuple(name for name, _ in self.axes)

    def s(self, output: Any, input: Any | None = None) -> Any:
        """Return one complex ``S(output, input)`` array."""
        output_label = resolve_label(output)
        input_label = self.input_port if input is None else resolve_label(input)
        try:
            return self._response[(output_label, input_label)]
        except KeyError:
            available = [key for key in self._response]
            raise KeyError(f"S({output_label!r}, {input_label!r}) is unavailable. Available: {available}") from None

    @property
    def s11(self) -> Any:
        """Reflection at the swept input port."""
        return self.s(self.input_port, self.input_port)

    @property
    def s21(self) -> Any:
        """Transmission to the first requested output distinct from the input."""
        output = next((label for label in self.output_ports if label != self.input_port), None)
        if output is None:
            raise AttributeError("s21 requires an output port distinct from the swept input port.")
        return self.s(output, self.input_port)

    def __array__(self) -> np.ndarray:
        return np.asarray(self.s11)


@dataclass(frozen=True)
class OutputSpectrumResult:
    """Normally ordered stationary output-field fluctuation spectrum."""

    port: str
    frequencies: Any
    fluctuation_spectrum: Any
    output_photon_flux: Any
    coherent_flux: Any
    incoherent_flux: Any
    steady_state: Any
    fourier_convention: str = "2 Re integral_0^inf d tau exp(+i 2 pi f tau) C(tau)"

    @property
    def total_flux(self) -> Any:
        """Mean normally ordered output photon flux."""
        return self.output_photon_flux


@dataclass(frozen=True)
class OutputCorrelationResult:
    """Normalized stationary output-field correlation versus delay."""

    order: int
    input_port: str
    output_port: str
    delays: Any
    values: Any
    unnormalized: Any
    input_intensity: Any
    output_intensity: Any
    steady_state: Any
    normalization: str

    @property
    def port(self) -> str:
        """Delayed output port, retained for single-port result code."""
        return self.output_port

    @property
    def intensity(self) -> Any:
        """Delayed output intensity, retained for single-port result code."""
        return self.output_intensity
