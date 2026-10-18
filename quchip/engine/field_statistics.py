"""Stationary output statistics and their propagation through resolved wiring."""

from typing import Any

from quchip.engine.field_noise import input_noise_matrix
from quchip.engine.ir import CanonicalOperator
from quchip.engine.output_network import pad_mixing
from quchip.engine.reference import (
    FieldChannel, ReferenceFilter, cw_transfer, noise_colors, noise_contributions, source_occupation, time_shift,
)


def canonical(values: Any, template: CanonicalOperator, *, tag: str) -> CanonicalOperator:
    """Keep the resolved operator's physical basis for a regression source."""
    return CanonicalOperator.from_dense(values, dims=template.dims, basis=template.basis,
                                       subsystem_labels=template.subsystem_labels, tag=tag)


def field_sources(engine: Any, rho: Any, xp: Any) -> tuple[Any, Any, Any]:
    """Return centered L, output regression sources, and direct thermal noise.

    With K=S†L, source i is (L_i-<L_i>)rho + sum_j S_ij n_j [K_j,rho]. The
    input-system correlations are necessary because, at thermal equilibrium, a
    matched cavity's output stays thermal. Without them, the fluorescence would
    add the same incoming bath a second time.
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

    The coordinates are (I_0,Q_0,I_1,Q_1,...), with b=I+iQ. The detector vacuum
    is intentionally absent. The excess spectrum includes anomalous
    correlations, so it can have negative eigenvalues for squeezed fields.
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
    """Assemble per-output IQ blocks, keeping the leading batch axes."""
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


def propagate_noise(
    fields: tuple[FieldChannel, ...], scattering: Any, graph: Any, excess: Any,
    labels: tuple[str, ...], frequency: Any, offsets: Any, xp: Any, *, include_inputs: bool = True,
) -> tuple[dict[str, tuple[Any, Any]], Any]:
    """Apply one physical source/transfer owner to the spectrum of either stationary solver."""
    indices = {channel.key: i for i, channel in enumerate(fields)}
    selected = [indices[label] for label in labels]
    channels = [fields[i] for i in selected]
    mixed = graph is not None
    upper = xp.stack([cw_transfer(c.reference.outbound, frequency + offsets, xp) + xp.zeros_like(offsets)
                      for c in channels], axis=-1)
    lower = xp.stack([cw_transfer(c.reference.outbound, frequency - offsets, xp) + xp.zeros_like(offsets)
                      for c in channels], axis=-1)
    if mixed:
        graph_positive = [graph.evaluate(frequency + offset, xp) for offset in offsets]
        graph_negative = [graph.evaluate(frequency - offset, xp) for offset in offsets]
        graph_upper = xp.stack([pad_mixing(point[0], len(fields), xp)[xp.asarray(selected)]
                                for point in graph_positive])
        graph_lower = xp.stack([pad_mixing(point[0], len(fields), xp)[xp.asarray(selected)]
                                for point in graph_negative])
        transform = field_transfer(upper[..., None] * graph_upper, lower[..., None] * graph_lower, xp)
    else:
        transform = block_diagonal(quadrature_transfer(upper, lower, xp), xp)
    transformed = transform @ excess @ xp.conj(xp.swapaxes(transform, -1, -2))
    size = 2 * len(labels)
    zero = xp.zeros((size, size))
    empty = xp.zeros((len(offsets), size, size), dtype=complex)
    components: dict[str, tuple[Any, Any]] = {"device.correlations": (zero, transformed)}
    gains = xp.stack([cw_transfer(c.reference.outbound, frequency, xp) for c in channels])
    if mixed:
        dc = gains[:, None] * pad_mixing(graph.evaluate(frequency, xp)[0], len(fields), xp)[xp.asarray(selected)]
        dc_transform = field_transfer(dc, dc, xp)
    else:
        dc_transform = block_diagonal(quadrature_transfer(gains, gains, xp), xp)
    colored_output = any(isinstance(element, ReferenceFilter) for c in channels for element in c.reference.outbound)
    scattering = xp.asarray(scattering)
    if not mixed:
        scattering = scattering[xp.asarray(selected)]
    colored_output = colored_output or (graph is not None and graph.colored)
    for index, channel in enumerate(fields if include_inputs else ()):
        column = scattering[:, index]
        source_matrix = column[:, None] * xp.conj(column[None, :])
        inbound = channel.reference.inbound
        reference_sources = any(source_occupation(element) is not None for element in inbound)
        if reference_sources:
            positive = noise_contributions(inbound, frequency + offsets, xp)
            negative = noise_contributions(inbound, frequency - offsets, xp)
            colors = noise_colors(inbound)
            for name, up in positive.items():
                direct = proper_spectrum(up[..., None, None] * source_matrix,
                                         negative[name][..., None, None] * source_matrix, xp)
                spectrum = transform @ direct @ xp.conj(xp.swapaxes(transform, -1, -2))
                components[f"input.{channel.key}.{name}"] = ((zero, spectrum) if colored_output or colors[name] else
                    (xp.real(dc_transform @ direct[len(offsets)//2] @ xp.conj(dc_transform.T)), empty))
        elif channel.input_occupation is not None:
            direct = complex_covariance(source_matrix * channel.input_occupation, xp)
            if colored_output:
                components[f"input.{channel.key}"] = (
                    zero, transform @ direct @ xp.conj(xp.swapaxes(transform, -1, -2)))
            else:
                components[f"input.{channel.key}"] = (xp.real(dc_transform @ direct @ xp.conj(dc_transform.T)), empty)
    if mixed:
        for name in graph_positive[0][1]:
            up = xp.stack([point[1][name][xp.asarray(selected)[:, None], xp.asarray(selected)[None, :]]
                           for point in graph_positive]) * upper[..., :, None] * xp.conj(upper[..., None, :])
            down = xp.stack([point[1][name][xp.asarray(selected)[:, None], xp.asarray(selected)[None, :]]
                             for point in graph_negative]) * lower[..., :, None] * xp.conj(lower[..., None, :])
            spectrum = proper_spectrum(up, down, xp)
            components[f"network.{name}"] = ((zero, spectrum) if colored_output else
                                              (xp.real(spectrum[len(offsets)//2]), empty))
    for index, channel in enumerate(channels):
        elements = channel.reference.outbound
        positive = noise_contributions(elements, frequency + offsets, xp)
        negative = noise_contributions(elements, frequency - offsets, xp)
        colors = noise_colors(elements)
        for name, up in positive.items():
            block = quadrature_transfer(up, negative[name], xp) / 2
            selector = xp.eye(len(labels))[index:index+1].T @ xp.eye(len(labels))[index:index+1]
            spectrum = xp.kron(selector, block)
            if colors[name]:
                components[f"output.{channel.key}.{name}"] = (zero, spectrum)
            else:
                components[f"output.{channel.key}.{name}"] = (xp.real(spectrum[0]), empty)
    return components, xp.asarray([time_shift(c.reference.outbound) for c in channels])
