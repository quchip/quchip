"""Propagation of physical field noise through resolved wiring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from quchip.analysis.field_statistics import (
    block_diagonal, complex_covariance, field_transfer, proper_spectrum, quadrature_transfer,
)
from quchip.engine.output_network import pad_mixing
from quchip.engine.reference import (
    FieldChannel, ReferenceFilter, cw_transfer, noise_colors, noise_contributions, source_occupation, time_shift,
)


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


@dataclass(frozen=True)
class ReadoutWiring:
    """Capture only field references and downstream mixing, without quantum operators."""

    fields: tuple[FieldChannel, ...]
    exposed: tuple[str, ...]
    graph: Any
    array_module: Any

    @classmethod
    def capture(cls, slh: Any, array_module: Any) -> ReadoutWiring:
        """Keep the solved wiring independently of later component edits."""
        return cls(tuple(FieldChannel(c.key, c.reference, c.input_occupation) for c in slh.channels),
                   tuple(c.key for c in slh.external_channels), slh.output_network, array_module)

    def iq_readout(self, output: Any, *, means: Any, frequency: Any, receiver: Any,
                   noise_frequencies: Any = None) -> Any:
        """Propagate conditional coherent boundary templates and downstream noise."""
        from quchip.results.receiver import _engineering_iq, integrate_noise, noise_grid
        from quchip.results.terminal import IQReadout
        from quchip.utils.jax_utils import contains_tracer, is_jax_array, is_jax_namespace, select_array_module
        from quchip.utils.labeling import resolve_label

        label = resolve_label(output)
        if label not in self.exposed:
            raise ValueError(f"Unknown exposed output {label!r}; available: {self.exposed}.")
        templates = {resolve_label(k): v for k, v in means.items()} if isinstance(means, Mapping) else {label: means}
        if not templates or any(k not in self.exposed for k in templates):
            raise ValueError("Conditional fields must name captured exposed boundary channels.")
        xp = select_array_module(is_jax_namespace(self.array_module)
                                 or any(is_jax_array(v) for v in templates.values())
                                 or contains_tracer((frequency, receiver.integration_time, tuple(templates.values()))))
        if not contains_tracer(frequency) and (np.ndim(frequency) or not np.isfinite(frequency)):
            raise ValueError("Readout frequency must be a finite scalar in GHz.")
        arrays = {k: xp.conj(xp.asarray(v, dtype=complex)) for k, v in templates.items()}
        first = next(iter(arrays.values()))
        if first.ndim != 1 or len(first) < 1 or any(a.shape != first.shape for a in arrays.values()):
            raise ValueError("Conditional fields must be equal-length vectors ordered by physical outcome.")
        selected = next(i for i, c in enumerate(self.fields) if c.key == label)
        fields = xp.stack([arrays.get(c.key, xp.zeros_like(first)) for c in self.fields], axis=-1)
        mixing = xp.eye(len(self.fields)) if self.graph is None else pad_mixing(
            self.graph.evaluate(frequency, xp)[0], len(self.fields), xp)
        gain = cw_transfer(self.fields[selected].reference.outbound, frequency, xp)
        centers = xp.conj((fields @ mixing.T)[..., selected] * gain)
        offsets = xp.asarray(noise_grid(noise_frequencies))
        size = 2 * (len(self.fields) if self.graph is not None else 1)
        components, delays = propagate_noise(
            self.fields, xp.eye(len(self.fields)), self.graph,
            xp.zeros((len(offsets), size, size), dtype=complex), (label,), frequency, offsets, xp,
            include_inputs=False)
        components = {name: (_engineering_iq(white, xp), _engineering_iq(excess, xp))
                      for name, (white, excess) in components.items()}
        covariance, budget, dc_gain = integrate_noise(xp.zeros(1), offsets, components, delays, receiver)
        return IQReadout(centers * dc_gain, covariance, budget)
