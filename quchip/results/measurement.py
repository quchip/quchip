"""Captured physical field statistics and inexpensive IQ receiver processing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quchip.analysis.field_statistics import block_diagonal, normal_spectrum, quadrature_transfer
from quchip.utils.constants import TWO_PI, hbar
from quchip.results.input_output import MeanFieldResponseResult
from quchip.utils.jax_utils import array_namespace, contains_tracer, is_jax_array, select_array_module
from quchip.results.receiver import IQReceiver, gaussian_samples, integrate_noise
from quchip.utils.labeling import resolve_label


def _capture(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.setflags(write=False)
    return value


def _index(ports: tuple[str, ...], output: Any) -> int:
    label = resolve_label(output)
    if label not in ports:
        raise KeyError(f"Unknown output {label!r}; available: {list(ports)}")
    return ports.index(label)


@dataclass(frozen=True)
class VNAMeasurementSamples:
    """Synthetic Gaussian IQ draws with sample, sweep, then output axes.

    Attributes
    ----------
    ports : tuple of str
        Output labels in the final array axis.
    input : str
        Driven input label.
    axes : tuple
        Named sweep-axis metadata.
    incident : array_like
        Incident complex field in ``1/sqrt(ns)``.
    values : array_like
        Sampled complex fields with leading shot and final output axes.
    receiver : IQReceiver
        Receiver used to integrate the physical spectra.
    distribution : str
        Sampling approximation.
    """

    ports: tuple[str, ...]
    input: str
    axes: tuple[tuple[str, Any], ...]
    incident: Any
    values: Any
    receiver: IQReceiver
    distribution: str = "Gaussian field-moment approximation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _capture(self.values))
        object.__setattr__(self, "incident", _capture(self.incident))

    def field(self, output: Any) -> Any:
        """Return sampled complex fields in ``1/sqrt(ns)``.

        Parameters
        ----------
        output : port object or str
            Output plane to select.
        """
        return self.values[..., _index(self.ports, output)]

    def ratio(self, output: Any) -> Any:
        """Return sampled output/input ratios, undefined for zero input.

        Parameters
        ----------
        output : port object or str
            Output plane to select.
        """
        xp = array_namespace(self.values)
        zero = self.incident == 0
        return xp.where(zero, xp.nan + 0j, self.field(output) / xp.where(zero, 1.0, self.incident))


@dataclass(frozen=True)
class VNAMeasurementStatistics:
    """Integrated output means and full covariance of ``(I0,Q0,I1,Q1,...)``.

    Attributes
    ----------
    ports : tuple of str
        Output labels in covariance order.
    input : str
        Driven input label.
    axes : tuple
        Named sweep-axis metadata.
    incident : array_like
        Incident complex field in ``1/sqrt(ns)``.
    values : array_like
        Integrated complex output means.
    iq_covariance : array_like
        Full real IQ covariance with two axes per output.
    receiver : IQReceiver
        Receiver used for integration.
    contributions : mapping
        Named integrated covariance contributions.
    """

    ports: tuple[str, ...]
    input: str
    axes: tuple[tuple[str, Any], ...]
    incident: Any
    values: Any
    iq_covariance: Any
    receiver: IQReceiver
    contributions: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("incident", "values", "iq_covariance"):
            object.__setattr__(self, name, _capture(getattr(self, name)))
        object.__setattr__(self, "contributions", MappingProxyType(
            {key: _capture(value) for key, value in self.contributions.items()}))

    def mean(self, output: Any) -> Any:
        """Return the integrated complex mean at an output.

        Parameters
        ----------
        output : port object or str
            Output plane to select.
        """
        return self.values[..., _index(self.ports, output)]

    def covariance(self, output: Any, other: Any = None) -> Any:
        """Return a 2x2 IQ covariance or cross-output covariance block.

        Parameters
        ----------
        output : port object or str
            First output plane.
        other : port object, str, or None, optional
            Second output plane. ``None`` selects ``output``.
        """
        i = 2 * _index(self.ports, output)
        j = i if other is None else 2 * _index(self.ports, other)
        return self.iq_covariance[..., i:i+2, j:j+2]

    def noise_contributions(self, output: Any) -> Mapping[str, Any]:
        """Return integrated source covariance blocks, with detector vacuum.

        Parameters
        ----------
        output : port object or str
            Output plane to select.
        """
        i = 2 * _index(self.ports, output)
        return MappingProxyType({key: value[..., i:i+2, i:i+2] for key, value in self.contributions.items()})

    def sample(self, count: int, *, seed: int | None = None, key: Any = None) -> VNAMeasurementSamples:
        """Draw joint Gaussian IQ samples without a physical solver run.

        Use seed for NumPy draws or an explicit JAX random key. The covariance
        includes anomalous/cross-output second moments but does not specify
        higher-order non-Gaussian photon statistics.

        Parameters
        ----------
        count : int
            Positive number of samples.
        seed : int or None, optional
            NumPy random seed.
        key : jax.Array or None, optional
            JAX random key. Do not use it together with ``seed``.
        """
        xp = array_namespace(self.values)
        mean = xp.stack((xp.real(self.values), xp.imag(self.values)), axis=-1)
        draws = gaussian_samples(mean.reshape((*self.values.shape[:-1], -1)), self.iq_covariance,
                                 count, seed=seed, key=key)
        values = draws[..., 0::2] + 1j * draws[..., 1::2]
        return VNAMeasurementSamples(self.ports, self.input, self.axes, self.incident, values, self.receiver)

    def calibrate(self, factors: Mapping[Any, Any]) -> VNAMeasurementStatistics:
        """Multiply output fields and covariance axes by calibration factors.

        Parameters
        ----------
        factors : mapping
            Complex amplitude factor keyed by output plane. Outputs that are not in the mapping use one.
        """
        xp = select_array_module(is_jax_array(self.values) or contains_tracer(tuple(factors.values())))
        normalized = {resolve_label(port): value for port, value in factors.items()}
        for port in normalized:
            _index(self.ports, port)
        gains = xp.asarray([normalized.get(port, 1.0) for port in self.ports])
        blocks = quadrature_transfer(gains, gains, xp)
        transform = block_diagonal(blocks, xp)
        transform = xp.real(transform)
        covariance = transform @ self.iq_covariance @ transform.T
        contributions = {name: transform @ value @ transform.T for name, value in self.contributions.items()}
        return VNAMeasurementStatistics(self.ports, self.input, self.axes, self.incident,
                                     self.values * gains, covariance, self.receiver, contributions)


@dataclass(frozen=True)
class VNAMeasurement(MeanFieldResponseResult):
    """Physical means and spectra captured before you choose a receiver.

    noise_components maps physical source labels to (white covariance, excess
    spectral covariance). The excess spectral covariance has an
    offset-frequency axis before its two IQ axes. Device-generated excess
    includes input-system correlations and can be negative. It is not an
    independent random source.

    Attributes
    ----------
    ports : tuple of str
        Captured output labels.
    input : str
        Driven input label.
    values : array_like
        Complex output means with the output axis last.
    incident : array_like
        Incident complex field in ``1/sqrt(ns)``.
    frequencies, amplitudes : array_like
        Probe frequencies in GHz and incident amplitudes.
    shape : tuple of int
        Broadcast sweep shape.
    axes : tuple
        Named sweep-axis metadata.
    diagnostics : mapping
        Captured solve diagnostics.
    noise_frequencies : array_like
        Two-sided offsets in GHz.
    noise_components : mapping
        Named white and excess IQ covariance spectra.
    output_delays : array_like
        Output reference-plane delays in ns.
    parameters : tuple of mapping
        Bound parameters for each sweep point.
    modes : tuple of str
        Captured internal Fock-mode labels.
    mode_amplitudes, photon_numbers : array_like
        Internal stationary means and occupations.
    mode_frequencies : array_like
        Internal stationary-frame frequencies in GHz.
    conventions : tuple of str
        Units and statistical conventions.
    """

    noise_frequencies: Any
    noise_components: Mapping[str, tuple[Any, Any]]
    output_delays: Any
    parameters: tuple[Mapping[str, Any], ...]
    modes: tuple[str, ...]
    mode_amplitudes: Any
    photon_numbers: Any
    mode_frequencies: Any
    conventions: tuple[str, ...] = (
        "b=I+jQ; engineering e^{+jωt}, j = −i; frequencies in GHz; time in ns",
        "Physical normal-order spectra; detector vacuum added by the receiver",
        "Independent stationary experiments across sweep points",
        "Gaussian sampling uses second moments, not full photon statistics",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("values", "incident", "noise_frequencies", "frequencies", "amplitudes", "output_delays",
                     "mode_amplitudes", "photon_numbers", "mode_frequencies"):
            object.__setattr__(self, name, _capture(getattr(self, name)))
        object.__setattr__(self, "axes", tuple((name, _capture(value)) for name, value in self.axes))
        object.__setattr__(self, "parameters", tuple(MappingProxyType(
            {name: _capture(value) for name, value in point.items()}) for point in self.parameters))
        object.__setattr__(self, "noise_components", MappingProxyType({
            name: (_capture(white), _capture(excess)) for name, (white, excess) in self.noise_components.items()}))

    def _mode_index(self, mode: Any) -> int:
        label = resolve_label(mode)
        if label not in self.modes:
            raise KeyError(f"Unknown Fock mode {label!r}; available: {list(self.modes)}")
        return self.modes.index(label)

    def mode_amplitude(self, mode: Any) -> Any:
        """Return captured <a> in the mode's stationary frame, in sqrt(photons).

        Use ``mode_frequency(mode)`` for that frame's frequency in GHz. The
        value is the internal field with the full declared wiring included. The
        returned array follows the measurement's sweep axes.

        Parameters
        ----------
        mode : device object or str
            Captured Fock mode.
        """
        return self.mode_amplitudes[..., self._mode_index(mode)]

    def photon_number(self, mode: Any) -> Any:
        """Return captured <a†a>, with coherent and incoherent occupation.

        Available for authored Fock modes and nonlinear modes. The general
        solver keeps the declared basis projection and truncation. Receiver
        integration and calibration do not change this occupation.

        Parameters
        ----------
        mode : device object or str
            Captured Fock mode.
        """
        return self.photon_numbers[..., self._mode_index(mode)]

    def mode_frequency(self, mode: Any) -> Any:
        """Return a mode's captured stationary frame frequency in GHz.

        Parameters
        ----------
        mode : device object or str
            Captured Fock mode.
        """
        return self.mode_frequencies[..., self._mode_index(mode)]

    def noise_contributions(self, output: Any) -> Mapping[str, Any]:
        """Return physical IQ spectral contributions at the captured offsets.

        Parameters
        ----------
        output : port object or str
            Output plane to select.
        """
        i = 2 * _index(self.ports, output)
        return MappingProxyType({name: white[..., None, i:i+2, i:i+2] + excess[..., i:i+2, i:i+2]
                                 for name, (white, excess) in self.noise_components.items()})

    def noise_spectrum(self, output: Any, *, unit: str = "quanta") -> Any:
        """Return physical output noise in quanta, W/Hz, or dBm/Hz.

        The final axis is ``noise_frequencies`` relative to each probe carrier.
        This normally ordered spectrum does not include coherent signal and
        receiver vacuum. Power units use hf times occupation at the absolute
        sideband frequency and require positive physical frequencies. No solve
        is run.

        Parameters
        ----------
        output : port object or str
            Output plane to select.
        unit : {"quanta", "W/Hz", "dBm/Hz"}, default="quanta"
            Spectral-density unit.
        """
        xp = array_namespace(self.values)
        spectrum = normal_spectrum(sum(self.noise_contributions(output).values()), xp)
        if unit == "quanta":
            return spectrum
        if unit not in {"W/Hz", "dBm/Hz"}:
            raise ValueError("unit must be 'quanta', 'W/Hz', or 'dBm/Hz'.")
        frequency = xp.broadcast_to(xp.asarray(self.frequencies), self.shape)[..., None] + self.noise_frequencies
        if not contains_tracer(frequency) and np.any(np.asarray(frequency) <= 0):
            raise ValueError("Power noise spectra require positive absolute sideband frequencies.")
        power = TWO_PI*hbar*frequency*1e9*spectrum
        if unit == "W/Hz":
            return power
        return xp.where(power > 0, 10*xp.log10(xp.where(power > 0, power, 1.0)/1e-3), -xp.inf)

    def statistics(self, *, receiver: IQReceiver) -> VNAMeasurementStatistics:
        """Integrate captured spectra and detector vacuum without a solver call.

        Parameters
        ----------
        receiver : IQReceiver
            Boxcar receiver and optional digital transfer.
        """
        covariance, contributions, mean_gain = integrate_noise(
            self.values, self.noise_frequencies, self.noise_components, self.output_delays, receiver)
        return VNAMeasurementStatistics(self.ports, self.input, self.axes, self.incident,
                                     self.values * mean_gain, covariance, receiver, contributions)

    def sample(
        self, count: int, *, receiver: IQReceiver, seed: int | None = None, key: Any = None,
    ) -> VNAMeasurementSamples:
        """Integrate for a receiver and draw samples from the captured moments.

        Parameters
        ----------
        count : int
            Positive number of samples.
        receiver : IQReceiver
            Boxcar receiver and optional digital transfer.
        seed : int or None, optional
            NumPy random seed.
        key : jax.Array or None, optional
            JAX random key. Do not use it together with ``seed``.
        """
        return self.statistics(receiver=receiver).sample(count, seed=seed, key=key)
