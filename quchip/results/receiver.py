"""Shared heterodyne receiver integration and random sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from quchip.analysis.field_statistics import quadrature_transfer
from quchip.utils.jax_utils import contains_tracer, is_jax_array, select_array_module


@dataclass(frozen=True)
class IQReceiver:
    """Configure ideal heterodyne detection after a physical measurement.

    Parameters
    ----------
    integration_time : scalar
        Positive boxcar duration in ns. White detector and field noise scale
        as ``1 / integration_time``.
    transfer : callable or None, optional
        Digital complex amplitude transfer function. It receives offset
        frequencies in GHz, must be supported on the captured spectral grid,
        and its value at zero also scales the mean field. It does not alter the
        simulated chip or its physical noise sources.
    tolerance : float, default=0.02
        Relative convergence and spectral-support tolerance. Must satisfy
        ``0 < tolerance < 1``.

    Notes
    -----
    The receiver integrates the captured two-sided normally ordered spectrum
    with the sinc-squared boxcar response and adds detector vacuum with
    covariance ``1 / (2 * integration_time)`` per IQ quadrature. See Caves,
    *Phys. Rev. D* 26, 1817 (1982), doi:10.1103/PhysRevD.26.1817, for the
    phase-preserving amplifier noise convention.
    """

    integration_time: Any
    transfer: Callable[[Any], Any] | None = field(default=None, repr=False)
    tolerance: float = 0.02

    def __post_init__(self) -> None:
        if not contains_tracer(self.integration_time):
            value = np.asarray(self.integration_time)
            if value.ndim or not np.isfinite(value) or value <= 0:
                raise ValueError("integration_time must be a finite positive scalar in ns.")
        if self.transfer is not None and not callable(self.transfer):
            raise TypeError("Receiver transfer must be callable.")
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            raise ValueError("Receiver tolerance must lie between zero and one.")


def validate_samples(count: int, seed: Any, key: Any, values: Any) -> Any:
    """Validate Gaussian-shot arguments and choose NumPy or JAX arithmetic.

    Parameters
    ----------
    count : int
        Number of draws; must be positive.
    seed : int or None
        NumPy generator seed. Do not pass together with ``key``.
    key : jax.Array or None
        Explicit JAX PRNG key required when sampling traced values.
    values : object
        Mean and covariance values used to infer the array namespace.

    Returns
    -------
    module
        NumPy-like namespace selected for the draw.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Sample count must be a positive integer.")
    if seed is not None and key is not None:
        raise ValueError("Specify seed or key, not both.")
    if key is None and contains_tracer(values):
        raise ValueError("Sampling traced statistics requires an explicit JAX random key.")
    if key is not None:
        import jax.numpy as jnp
        return jnp
    from jax.tree_util import tree_leaves
    return select_array_module(any(is_jax_array(v) for v in tree_leaves(values)))


def gaussian_samples(mean: Any, covariance: Any, count: int, *, seed: Any = None, key: Any = None) -> Any:
    """Draw correlated real Gaussian vectors.

    Parameters
    ----------
    mean : array_like
        Mean with shape ``(..., n)``.
    covariance : array_like
        Positive-definite covariance with shape ``(..., n, n)``.
    count : int
        Number of draws.
    seed, key : optional
        Use ``seed`` for NumPy or ``key`` for JAX; passing both is invalid.

    Returns
    -------
    array
        Samples with shape ``(count, ..., n)`` in the input array namespace.
    """
    xp = validate_samples(count, seed, key, (mean, covariance))
    mean, covariance = xp.asarray(mean), xp.asarray(covariance)
    shape = (count, *mean.shape)
    if key is None:
        standard = xp.asarray(np.random.default_rng(seed).normal(size=shape))
    else:
        import jax.random
        standard = jax.random.normal(key, shape, dtype=covariance.dtype)
    root = xp.linalg.cholesky(covariance)
    return mean + xp.einsum("...ij,n...j->n...i", root, standard)


def noise_grid(frequencies: Any = None) -> np.ndarray:
    """Validate or construct the two-sided spectral offset grid in GHz.

    Parameters
    ----------
    frequencies : 1-D array_like or None, optional
        Strictly increasing, finite, symmetric offsets containing zero. With
        ``None``, return the default grid spanning ``-0.1`` to ``+0.1`` GHz.

    Returns
    -------
    numpy.ndarray
        Validated offsets in GHz.
    """
    if frequencies is None:
        positive = np.geomspace(1e-9, 0.1, 161)
        return np.concatenate((-positive[::-1], [0.0], positive))
    values = np.asarray(frequencies, dtype=float)
    if (values.ndim != 1 or len(values) < 5 or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0) or not np.any(values == 0)
            or not np.allclose(values, -values[::-1], atol=0, rtol=1e-12)):
        raise ValueError("noise_frequencies must be finite, increasing, symmetric offsets including zero (GHz).")
    return values


def _engineering_iq(spectrum: Any, xp: Any) -> Any:
    """Reverse both Q axes and conjugate spectra, preserving physical sideband labels."""
    signs = xp.where(xp.arange(spectrum.shape[-1]) % 2, -1, 1)
    return signs[:, None] * xp.conj(spectrum) * signs[None, :]


def integrate_noise(values: Any, noise_frequencies: Any, noise_components: Any,
                    output_delays: Any, receiver: IQReceiver) -> tuple[Any, Any, Any]:
    """Integrate normal spectra plus detector vacuum; return covariance, budget, DC gain.

    Captured IQ spectra and the receiver transfer use the engineering convention.
    """
    count = values.shape[-1]
    receiver_upper = None if receiver.transfer is None else receiver.transfer(noise_frequencies)
    xp = select_array_module(is_jax_array(values)
                             or contains_tracer((receiver.integration_time, receiver_upper)))
    trapezoid = getattr(xp, "trapezoid", None) or xp.trapz
    frequencies = xp.asarray(noise_frequencies)
    delays = xp.repeat(output_delays, 2, axis=-1)
    overlap = xp.maximum(1 - xp.abs(delays[..., :, None] - delays[..., None, :]) / receiver.integration_time, 0)
    window = xp.sinc(frequencies * receiver.integration_time) ** 2
    mean_gain = 1.0
    transform = None
    if receiver.transfer is not None:
        upper = xp.asarray(receiver_upper) + xp.zeros_like(frequencies)
        lower = xp.asarray(receiver.transfer(-frequencies)) + xp.zeros_like(frequencies)
        mean_gain = receiver.transfer(xp.asarray(0.0))
        local = quadrature_transfer(upper, lower, xp)
        transform = xp.kron(xp.eye(count), local)
    components = dict(noise_components)
    components["receiver.vacuum"] = (xp.broadcast_to(xp.eye(2 * count) / 2,
                                                    (*values.shape[:-1], 2 * count, 2 * count)),
                                     xp.zeros((*values.shape[:-1], len(frequencies), 2 * count, 2 * count)))
    contributions, coarse, edge_bounds = {}, [], []
    for name, (white, excess) in components.items():
        white, excess = xp.asarray(white), xp.asarray(excess)
        if transform is None:
            spectrum = excess
            analytic = white * overlap / receiver.integration_time
        else:
            relative = delays[..., :, None] - delays[..., None, :]
            phase = xp.exp(-2j * xp.pi * frequencies[:, None, None] * relative[..., None, :, :])
            physical = excess + white[..., None, :, :] * phase
            spectrum = transform @ physical @ xp.conj(xp.swapaxes(transform, -1, -2))
            analytic = 0.0
        edge_bounds.append(xp.max(xp.abs(spectrum[..., xp.asarray([0, -1]), :, :]))
                           / (xp.pi**2 * frequencies[-1] * receiver.integration_time**2))
        weighted = xp.real(spectrum) * window[:, None, None]
        contributions[name] = analytic + trapezoid(weighted, x=frequencies, axis=-3)
        coarse.append(analytic + trapezoid(weighted[..., ::2, :, :], x=frequencies[::2], axis=-3))
    covariance = sum(contributions.values())
    covariance = (covariance + xp.swapaxes(covariance, -1, -2)) / 2
    if not contains_tracer((covariance, receiver.integration_time)):
        scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
        error = float(np.max(np.abs(np.asarray(covariance - sum(coarse))))) / scale
        if error > receiver.tolerance:
            raise ValueError("Receiver integration is unresolved on the captured noise_frequencies; "
                             "capture a finer spectral grid.")
        if sum(float(value) for value in edge_bounds) / scale > receiver.tolerance:
            raise ValueError("Receiver integration extends beyond captured spectral support; "
                             "capture a wider noise_frequencies grid.")
        if receiver.transfer is not None:
            edge = np.max(np.abs(np.asarray(receiver.transfer(frequencies[xp.asarray([0, -1])]))))
            if edge > receiver.tolerance:
                raise ValueError("Receiver filter extends beyond captured spectral support; "
                                 "capture a wider noise_frequencies grid.")
        if np.min(np.linalg.eigvalsh(np.asarray(covariance))) <= 0:
            raise ValueError("Integrated IQ covariance is not positive; "
                             "check spectral resolution and model validity.")
    return covariance, contributions, mean_gain
