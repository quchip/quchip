"""Automatic output sampling over captured physics, separate from solver steps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from quchip.engine.ir import (
    CoefficientRef, EnvelopeRef, Shift, Window, decompose_carrier_bands,
    evaluate_signal_program, signal_children,
)
from quchip.engine.solver_hints import _static_spectral_span, operator_norm_bound
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer

_POINTS_PER_PERIOD = 32
_POINTS_PER_DECAY = 16
_MAX_AUTOMATIC_POINTS = 1_000_000


@dataclass(frozen=True)
class AutomaticTimeGrid:
    """Internal interval intent, replaced by an array before a request escapes."""

    stop: float

    @property
    def bounds(self) -> tuple[float, float]:
        return (0.0, self.stop)


def interval_bounds(tlist: Any) -> tuple[Any, Any]:
    return tlist.bounds if isinstance(tlist, AutomaticTimeGrid) else (tlist[0], tlist[-1])


def _concrete(value: Any) -> np.ndarray:
    if value is None or contains_tracer(value):
        raise ValueError(
            "Automatic sampling requires concrete physics and timing. Build a grid outside "
            "JAX tracing and reuse it as an explicit tlist for differentiated solves."
        )
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("Automatic sampling requires finite physics and timing.")
    return array


def signal_feature_times(signal: Any, shift: float = 0.0) -> list[np.ndarray]:
    if isinstance(signal, Shift):
        return signal_feature_times(signal.child, shift + float(_concrete(signal.delta_t)))
    if isinstance(signal, CoefficientRef):
        raise NotImplementedError("Pass an explicit tlist for a custom TimeCoefficient with opaque time dependence.")
    features = []
    if isinstance(signal, Window):
        features.append(_concrete([signal.start, signal.stop]) + shift)
    if isinstance(signal, EnvelopeRef):
        times = _concrete(signal.envelope.sampling_times())
        duration = float(_concrete(signal.envelope.duration))
        if times.ndim != 1 or np.any((times < 0) | (times > duration)):
            raise ValueError("Envelope sampling_times() must return local times between zero and duration.")
        features.append(times + shift)
    for child in signal_children(signal):
        features.extend(signal_feature_times(child, shift))
    return features


def _problem_scales(problem: Any) -> tuple[float, list[tuple[float, Any]], list[np.ndarray]]:
    """Return a persistent sample rate and carrier-free dynamic band bounds."""
    from quchip.engine.observables import BandMeta, OutputMeta
    from quchip.engine.reference import time_shift
    from quchip.engine.truncation import boundary_sampling_frequency

    engine = problem.engine_result
    static = float(_concrete(_static_spectral_span(engine.static_terms))) / TWO_PI
    decay = sum(
        2 * float(_concrete(term.rate)) * float(_concrete(operator_norm_bound(term.operator))) ** 2
        for term in engine.collapse_terms
    )
    demodulation = (float(_concrete(boundary_sampling_frequency(problem.truncation)))
                    if problem.truncation is not None and problem.truncation.sampled else 0.0)
    output_shifts = set()
    channels = {channel.key: channel for channel in engine.slh.external_channels}
    for meta in problem.e_ops_meta or ():
        if isinstance(meta, BandMeta):
            labels = (meta.device_labels,) if isinstance(meta.device_labels, str) else meta.device_labels
            weights = (meta.weight,) if isinstance(meta.weight, int) else meta.weight
            frequency = sum(problem.resolved_frame.demod_freqs[label] * weight
                            for label, weight in zip(labels, weights))
            demodulation = max(demodulation, abs(float(_concrete(frequency))))
        elif isinstance(meta, OutputMeta):
            channel = channels[meta.observable.exposure]
            demodulation = max(demodulation, abs(float(_concrete(channel.carrier))))
            output_shifts.add(float(_concrete(time_shift(channel.reference.outbound))))
    rate = max(_POINTS_PER_PERIOD * (static + demodulation), _POINTS_PER_DECAY * decay)
    bands = []
    for term in engine.dynamic_terms:
        norm = float(_concrete(operator_norm_bound(term.operator)))
        for band in decompose_carrier_bands(term.time_dependence.signal):
            bands.append((norm, band))
    features = []
    if output_shifts:
        from quchip.engine.ir import CarrierBand

        for bound in engine.coherent_inputs:
            bands.extend((0.0, band) for band in decompose_carrier_bands(bound.beta))
        source_bands = tuple(bands)
        bounds = _concrete(interval_bounds(problem.tlist))
        for shift in output_shifts:
            if shift:
                # The reference plane delays the entire field, including emission.
                bands.extend((norm, CarrierBand(Shift(band.envelope, shift), band.freq))
                             for norm, band in source_bands)
                shifted = bounds + shift
                features.extend([shifted, np.nextafter(shifted, -np.inf),
                                 np.nextafter(shifted, np.inf)])
    return rate, bands, features


def automatic_tlist(problems: list[Any], *, combine: bool = False) -> np.ndarray:
    """Sample the union of features at the fastest required rate across points.

    Matrix norm bounds cover coherent and dissipative evolution; local envelope
    samples estimate amplitudes. These are sampling heuristics, not error bounds.
    """
    start, stop = map(float, _concrete(interval_bounds(problems[0].tlist)))
    features = [np.array([start, stop])]
    scales = []
    for problem in problems:
        rate, bands, output_features = _problem_scales(problem)
        scales.append((rate, bands))
        features.extend(output_features)
        for _, band in bands:
            features.extend(signal_feature_times(band.envelope))
    knots = np.unique(np.concatenate(features))
    knots = knots[(knots >= start) & (knots <= stop)]
    left, right = knots[:-1], knots[1:]
    # Probe inside each cell so a boundary jump cannot refine an entire idle.
    probe = np.stack([np.nextafter(left, right), (left + right) / 2, np.nextafter(right, left)])
    rates = np.zeros_like(left)
    for persistent, bands in scales:
        coherent = np.zeros_like(left)
        carrier = np.zeros_like(left)
        for norm, band in bands:
            values = _concrete(evaluate_signal_program(band.envelope, probe))
            peak = np.max(np.abs(np.broadcast_to(values, probe.shape)), axis=0)
            coherent += 2 * norm * peak / TWO_PI
            frequency = abs(float(_concrete(band.freq))) / TWO_PI
            carrier = np.maximum(carrier, np.where(peak > 0, frequency, 0.0))
        point_rates = persistent + _POINTS_PER_PERIOD * (coherent + carrier)
        rates = rates + point_rates if combine else np.maximum(rates, point_rates)
    counts = np.maximum(1, np.ceil((right - left) * rates))
    total = 1 + float(np.sum(counts))
    if not np.isfinite(total) or total > _MAX_AUTOMATIC_POINTS:
        raise ValueError(
            f"Automatic sampling would require {total:g} points. Choose an appropriate "
            "rotating frame or pass an explicit tlist."
        )
    return np.concatenate([np.linspace(a, b, int(count), endpoint=False)
                           for a, b, count in zip(left, right, counts)] + [np.array([stop])])


def sample_problems(problems: list[Any]) -> list[Any]:
    """Give equal-duration points one adequate grid; preserve distinct intervals."""
    groups: dict[tuple[float, float], list[int]] = {}
    for index, problem in enumerate(problems):
        start, stop = interval_bounds(problem.tlist)
        groups.setdefault((float(_concrete(start)), float(_concrete(stop))), []).append(index)
    sampled = list(problems)
    for indices in groups.values():
        times = automatic_tlist([problems[index] for index in indices])
        for index in indices:
            sampled[index] = replace(problems[index], tlist=times)
    return sampled
