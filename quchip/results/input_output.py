"""Results for continuous-wave port scattering calculations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quchip.utils.jax_utils import is_jax_array, select_array_module

from quchip.utils.labeling import resolve_label


class _LazyDiagnostics(Mapping[str, Any]):
    """Read-only diagnostic values with explicitly deferred calculations."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> Any:
        value = self._values[key]
        return value() if callable(value) else value

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        return key in self._values


@dataclass(frozen=True)
class SParameterResult:
    """Complete selected-port small-signal scattering over a sweep grid.

    Around a phase-sensitive operating point, the response is
    ``delta <b_out> = S delta beta + T conj(delta beta)``. ``matrix`` stores ``S``
    and ``conjugate_matrix`` stores ``T``. Both have shape
    ``(*shape, n_ports, n_ports)`` and use ``[..., output, input]`` indexing in
    ``ports`` order. ``s(output, input)`` and ``t(output, input)`` select
    individual entries.

    The passive-linear route reports zero for ``T``. ``numpy.asarray(result)``
    returns ``matrix``.

    Attributes
    ----------
    frequencies : scalar or array_like
        Probe frequencies in GHz.
    ports : tuple of str
        Port labels in output/input matrix order.
    axes : tuple
        Sweep-axis ``(name, values)`` pairs. ``shape`` is the sweep shape.
    diagnostics : tuple of mapping
        Per-point solver diagnostics.
    matrix, conjugate_matrix : array_like
        ``S`` and phase-conjugating ``T`` arrays with shape
        ``(*shape, n_ports, n_ports)``.
    shape : tuple of int
        Sweep-grid shape preceding the matrix axes.
    """
    # The stationary route computes both matrices from one shifted-Liouvillian
    # factorization.

    frequencies: Any
    ports: tuple[str, ...]
    axes: tuple[tuple[str, Any], ...]
    shape: tuple[int, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    matrix: Any
    conjugate_matrix: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            tuple(item if isinstance(item, _LazyDiagnostics) else _LazyDiagnostics(item) for item in self.diagnostics),
        )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Names of the result axes, in array order."""
        return tuple(name for name, _ in self.axes)

    def s(self, output: Any, input: Any) -> Any:
        """Return ``S(output, input)`` over the sweep grid.

        Parameters
        ----------
        output, input : port object or str
            Output row and input column.
        """
        return self.matrix[..., self._index(output), self._index(input)]

    def t(self, output: Any, input: Any) -> Any:
        """Return phase-conjugating ``T(output, input)`` over the sweep grid.

        Parameters
        ----------
        output, input : port object or str
            Output row and input column.
        """
        return self.conjugate_matrix[..., self._index(output), self._index(input)]

    @property
    def s11(self) -> Any:
        """Return reflection from the first selected plane back onto itself."""
        return self.matrix[..., 0, 0]

    @property
    def s21(self) -> Any:
        """Return transmission from the first selected plane to the second."""
        if len(self.ports) < 2:
            raise AttributeError("s21 requires at least two ports.")
        return self.matrix[..., 1, 0]

    def _index(self, plane: Any) -> int:
        label = resolve_label(plane)
        try:
            return self.ports.index(label)
        except ValueError:
            raise KeyError(f"Port {label!r} is not in this result. Available: {list(self.ports)}") from None

    def __array__(self) -> np.ndarray:
        return np.asarray(self.matrix)


@dataclass(frozen=True)
class MeanFieldResponseResult:
    """Stationary mean output fields from a finite coherent probe.

    ``values`` stores ``<b_out>`` at every selected plane with shape
    ``(*shape, n_ports)`` in ``ports`` order. ``incident`` stores the input
    amplitude ``beta`` broadcast to ``shape``. ``axes`` lists chip and pump sweep
    axes first, followed by ``"amplitude"`` and ``"frequency"`` when those
    arguments are arrays.

    If no fixed pump leaves a coherent mean at that plane and carrier,
    ``ratio(plane)`` approaches the corresponding small-signal S-parameter as
    ``beta`` goes to zero. ``ratio(plane)`` is ``NaN`` where ``beta`` is zero.
    The result contains one stationary mean-field branch. It does not encode
    sweep-rate hysteresis or metastable branches.

    Attributes
    ----------
    ports : tuple of str
        Selected output labels in the final array axis.
    input : str
        Probe input label.
    frequencies, amplitudes : scalar or array_like
        Probe values in GHz and ``1/sqrt(ns)``.
    axes : tuple
        Sweep-axis ``(name, values)`` pairs. ``shape`` is their array shape.
    diagnostics : tuple of mapping
        Per-point stationary-solver diagnostics.
    values : array_like
        Complex output means with shape ``(*shape, n_ports)``.
    incident : array_like
        Incident amplitude broadcast over ``shape``.
    shape : tuple of int
        Sweep-grid shape preceding the port axis.
    """

    ports: tuple[str, ...]
    input: str
    frequencies: Any
    amplitudes: Any
    axes: tuple[tuple[str, Any], ...]
    shape: tuple[int, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    values: Any
    incident: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            tuple(item if isinstance(item, _LazyDiagnostics) else _LazyDiagnostics(item) for item in self.diagnostics),
        )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Names of the result axes, in array order."""
        return tuple(name for name, _ in self.axes)

    def mean(self, plane: Any) -> Any:
        """Return stationary ``<b_out>`` at ``plane`` with shape ``shape``.

        Parameters
        ----------
        plane : port object or str
            Output reference plane.
        """
        label = resolve_label(plane)
        try:
            return self.values[..., self.ports.index(label)]
        except ValueError:
            raise KeyError(f"Port {label!r} is not in this result. Available: {list(self.ports)}") from None

    def ratio(self, plane: Any) -> Any:
        """Return the stationary ``<b_out>/beta`` at ``plane`` with shape ``shape``.

        The result is complex ``NaN`` where the incident ``beta`` is zero. Its
        zero-amplitude limit is the corresponding small-signal S-parameter when
        no fixed pump leaves a coherent mean at that plane and carrier.

        Parameters
        ----------
        plane : port object or str
            Output reference plane.
        """
        mean = self.mean(plane)
        xp = select_array_module(is_jax_array(mean))
        incident = xp.asarray(self.incident)
        zero = incident == 0
        return xp.where(zero, xp.nan + 0j, mean / xp.where(zero, 1.0, incident))


@dataclass(frozen=True)
class OutputSpectrumResult:
    """Stationary output-field fluctuation spectra and signal photon fluxes.

    ``signal_fluctuation_spectrum`` is device-generated spectral excess, with
    input-system interference for thermal fields. It can be negative.
    ``added_noise_spectrum`` is directly propagated thermal and amplifier noise,
    and ``total_fluctuation_spectrum`` is their sum. ``signal_photon_flux`` is
    the propagated device-field flux, split into ``signal_coherent_flux`` and
    ``signal_incoherent_flux``. These fluxes exclude added noise, because
    converting a spectral density to flux requires a detection bandwidth.

    Attributes
    ----------
    port : str
        Output port label.
    frequencies : array_like
        Offset frequencies in GHz.
    total_fluctuation_spectrum, signal_fluctuation_spectrum, added_noise_spectrum : array_like
        Normal-ordered spectral densities in frequency order.
    signal_photon_flux, signal_coherent_flux, signal_incoherent_flux : scalar
        Propagated flux terms in photons/ns.
    steady_state : SteadyStateResult
        Captured stationary state used for the spectra.
    fourier_convention : str
        Definition of the reported two-sided spectrum.
    """

    port: str
    frequencies: Any
    total_fluctuation_spectrum: Any
    signal_fluctuation_spectrum: Any
    added_noise_spectrum: Any
    signal_photon_flux: Any
    signal_coherent_flux: Any
    signal_incoherent_flux: Any
    steady_state: Any
    fourier_convention: str = "2 Re integral_0^inf d tau exp(-j 2 pi f tau) C(tau); engineering j = -i"


@dataclass(frozen=True)
class OutputCorrelationResult:
    """Normalized stationary output-field correlation versus delay.

    Attributes
    ----------
    order : {1, 2}
        Correlation order.
    input_port, output_port : str
        Correlation input and delayed output labels.
    delays : array_like
        Non-negative delays in ns.
    values, unnormalized : array_like
        Normalized and raw correlation values in delay order.
    input_intensity, output_intensity : scalar
        Mean photon fluxes used for normalization.
    steady_state : SteadyStateResult
        Captured stationary state.
    normalization : str
        Human-readable normalization convention.
    """

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
        """Delayed output port, kept for single-port result code."""
        return self.output_port

    @property
    def intensity(self) -> Any:
        """Delayed output intensity, kept for single-port result code."""
        return self.output_intensity
