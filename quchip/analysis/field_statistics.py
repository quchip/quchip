"""Stationary output statistics shared by spectra and captured measurements."""

from typing import Any

from quchip.engine.field_noise import input_noise_matrix
from quchip.engine.ir import CanonicalOperator


def canonical(values: Any, template: CanonicalOperator, *, tag: str) -> CanonicalOperator:
    """Preserve the resolved operator's physical basis for a regression source."""
    return CanonicalOperator.from_dense(values, dims=template.dims, basis=template.basis,
                                       subsystem_labels=template.subsystem_labels, tag=tag)


def field_sources(engine: Any, rho: Any, xp: Any) -> tuple[Any, Any, Any]:
    """Return centered L, output regression sources and direct thermal noise.

    With K=S†L, source i is (L_i-<L_i>)rho + sum_j S_ij n_j [K_j,rho].
    The input-system correlations are essential: at thermal equilibrium a
    matched cavity's output stays thermal, rather than adding fluorescence
    on top of the same incoming bath a second time.
    """
    operators = xp.stack([xp.asarray(operator.to_dense()) for operator in engine.slh.L])
    means = xp.einsum("ijk,kj->i", operators, rho)
    centered = operators - means[:, None, None] * xp.eye(rho.shape[0])[None, :, :]
    white = input_noise_matrix(engine.slh, xp)
    commutators = operators @ rho - rho @ operators
    sources = centered @ rho + xp.einsum("ij,jkl->ikl", white, commutators)
    return centered, sources, white


def quadrature_spectrum(
    engine: Any, rho: Any, backend: Any, prepared: Any, labels: tuple[str, ...], frequencies: Any,
) -> Any:
    """Return the normally ordered IQ cross-spectrum.

    Coordinates are (I_0,Q_0,I_1,Q_1,...), with b=I+iQ. Detector vacuum is
    deliberately absent. The excess spectrum includes anomalous correlations
    and can have negative eigenvalues for squeezed fields.
    """
    xp = backend.array_module
    centered, emission, _ = field_sources(engine, rho, xp)
    indices_by_key = {channel.key: i for i, channel in enumerate(engine.slh.channels)}
    indices = [indices_by_key[label] for label in labels]
    sources, observables = [], []
    for index in indices:
        field, source = centered[index], emission[index]
        template = engine.slh.channels[index].coupling
        for quadrature, factor in (("I", 1.0), ("Q", -1j)):
            label = f"{index}:{quadrature}"
            observable = (factor * field + xp.conj(factor * field.T)) / 2
            conditioned = (factor * source + xp.conj(factor * source.T)) / 2
            sources.append((label, canonical(-conditioned, template, tag=f"iq-source:{label}")))
            observables.append((label, canonical(observable, template, tag=f"iq-observable:{label}")))
    response = backend.stationary_resolvent(engine, tuple(sources), tuple(observables), frequencies, prepared=prepared)
    causal = xp.stack([xp.stack([response[(b, a)] for b, _ in sources], axis=-1)
                       for a, _ in observables], axis=-2)
    excess = causal + xp.conj(xp.swapaxes(causal, -1, -2))
    return excess


def complex_covariance(values: Any, xp: Any) -> Any:
    """Convert E[z_i z_j*] to the covariance of (Re z_i, Im z_i)."""
    return proper_spectrum(values, values, xp)


def normal_spectrum(iq: Any, xp: Any) -> Any:
    """Recover the upper-sideband normal spectrum from a single-output IQ block."""
    return xp.real(iq[..., 0, 0] + iq[..., 1, 1] + 1j*(iq[..., 1, 0]-iq[..., 0, 1]))


def quadrature_transfer(upper: Any, lower: Any, xp: Any) -> Any:
    """Convert complex upper/lower sideband transmission to an IQ transfer."""
    even = (upper + xp.conj(lower)) / 2
    odd = (upper - xp.conj(lower)) / 2
    return xp.stack((xp.stack((even, 1j * odd), axis=-1),
                     xp.stack((-1j * odd, even), axis=-1)), axis=-2)


def block_diagonal(blocks: Any, xp: Any) -> Any:
    """Assemble per-output IQ blocks, preserving leading batch axes."""
    count, rows, columns = blocks.shape[-3:]
    return xp.einsum("ij,...iab->...iajb", xp.eye(count), blocks).reshape(
        (*blocks.shape[:-3], count * rows, count * columns))


def proper_spectrum(upper: Any, lower: Any, xp: Any) -> Any:
    """Convert normal upper/lower cross-spectra to normally ordered IQ spectra."""
    return field_transfer(upper, lower, xp) / 2


def field_transfer(upper: Any, lower: Any, xp: Any) -> Any:
    """Convert a matrix of field transfers to interleaved IQ coordinates."""
    blocks = quadrature_transfer(upper, lower, xp)
    rows, columns = upper.shape[-2:]
    return xp.swapaxes(blocks, -3, -2).reshape((*upper.shape[:-2], 2*rows, 2*columns))
