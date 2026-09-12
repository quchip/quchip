"""Physical stationary acquisition, independent of receiver configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from quchip.analysis.field_statistics import proper_spectrum, quadrature_spectrum
from quchip.analysis.field_noise import propagate_noise
from quchip.results.receiver import noise_grid
from quchip.engine.ir import LinearResponseProblem
from quchip.devices.spaces import FockSpace
from quchip.engine.linear_response import try_build_linear_response_problem
from quchip.engine.reference import (
    FieldChannel, cw_transfer,
)
from quchip.results.measurement import VNAMeasurement
from quchip.sweep import Sweep, ZippedSweep, _iter_axis_points
from quchip.utils.jax_utils import contains_tracer
from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class _Acquisition:
    mean: Any
    components: dict[str, tuple[Any, Any]]
    delays: Any
    diagnostics: Any
    mode_amplitudes: Any
    photon_numbers: Any
    mode_frequencies: Any


def capture_modes(chip: Any, operating: Any, modes: tuple[str, ...]) -> tuple[Any, Any, Any]:
    """Read authored Fock-mode observables in the solved stationary frames."""
    from quchip.chip.observables import prepare_local_op

    backend = chip.backend
    xp = backend.array_module
    amplitudes, numbers, frequencies = [], [], []
    for label in modes:
        device = chip.device_map[label]
        basis = operating.engine.bases[label]
        state = operating.state.reduced_state(label)
        amplitudes.append(backend.expect(prepare_local_op(device, "a", basis, backend), state))
        numbers.append(xp.real(backend.expect(prepare_local_op(device, "n", basis, backend), state)))
        frequencies.append(operating.engine.resolved_frame.frequencies[label])
    return xp.asarray(amplitudes), xp.asarray(numbers), xp.asarray(frequencies)


def capture_noise(operating: Any, backend: Any, labels: tuple[str, ...], frequency: Any, offsets: Any) -> Any:
    """Compute joint physical IQ spectra and source budgets from one operating point."""
    xp = backend.array_module
    engine = operating.engine
    rho = xp.asarray(backend.to_array(operating.state.state))
    regression_labels = tuple(channel.key for channel in engine.slh.channels) if engine.slh.output_network else labels
    excess = quadrature_spectrum(engine, rho, backend, operating.prepared, regression_labels, offsets)
    fields = tuple(FieldChannel(c.key, c.reference, c.input_occupation) for c in engine.slh.channels)
    return propagate_noise(fields, engine.slh.S, engine.slh.output_network, excess, labels, frequency, offsets, xp)


def capture_linear(
    problem: LinearResponseProblem, backend: Any, labels: tuple[str, ...], input_label: str,
    frequency: Any, amplitude: Any, offsets: Any,
) -> _Acquisition:
    """Acquire harmonic means and normal spectra without a Fock-space truncation."""
    xp = backend.array_module
    count = problem.scattering.shape[0]
    indices = {channel.key: i for i, channel in enumerate(problem.field_channels)}
    selected = xp.asarray([indices[label] for label in labels])
    probe = indices[input_label]
    # The validated symmetric grid supplies both sidebands and the carrier.
    frequencies = frequency + offsets
    solved = backend.linear_response(replace(problem, frequencies=frequencies, plane_indices=tuple(range(count))))
    response = solved.responses[:, selected, :]
    gains = xp.stack([cw_transfer(problem.field_channels[indices[label]].reference.outbound, frequency, xp)
                      for label in labels])
    incoming = cw_transfer(problem.field_channels[probe].reference.inbound, frequency, xp)
    mean = gains * response[len(offsets)//2, :, probe] * incoming * amplitude
    occupations = xp.asarray([0.0 if c.input_occupation is None else c.input_occupation
                              for c in problem.field_channels] + [0.0]*(count-len(problem.field_channels)))
    normal = (response * occupations) @ xp.conj(xp.swapaxes(response, -1, -2))
    direct = xp.asarray(problem.scattering)[selected]
    background = (direct * occupations) @ xp.conj(direct.T)
    excess = proper_spectrum(normal-background, normal[::-1]-background, xp)
    components, delays = propagate_noise(problem.field_channels, problem.scattering, None, excess,
                                         labels, frequency, offsets, xp)
    mode_amplitudes = solved.mode_amplitudes[len(offsets)//2, :, probe] * incoming * amplitude
    numbers = xp.abs(mode_amplitudes)**2 + xp.real(xp.diag(solved.mode_covariance))
    return _Acquisition(mean, components, delays,
                        {"solver": "linear_response", "mode_count": len(problem.mode_labels),
                         "residual": xp.max(solved.residuals)},
                        mode_amplitudes, numbers, xp.full((len(problem.mode_labels),), frequency))


def measure(
    vna: Any, frequencies: Any, amplitudes: Any, variations: tuple[Sweep | ZippedSweep, ...], *,
    input: Any, outputs: Any, noise_frequencies: Any, options: Any, progress: bool,
) -> VNAMeasurement:
    """Acquire physical fields and correlations once per probe/sweep point."""
    from quchip.analysis.vna import (
        _axis_values, _operating_point, _plane_means, _public_axes, _resolve_exposure,
    )
    if isinstance(outputs, str):
        raise TypeError("outputs must be a sequence of plane objects or labels, not a string.")
    labels = tuple(vna.ports) if outputs is None else tuple(_resolve_exposure(vna.chip, plane) for plane in outputs)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("outputs must contain distinct exposed planes.")
    if input is None:
        if len(vna.ports) != 1:
            raise ValueError("measure() requires input= when more than one port is selected.")
        input_label = vna.ports[0]
    else:
        input_label = _resolve_exposure(vna.chip, input)
    if any(tone.port == input_label for tone in vna._tones):
        raise ValueError("The measurement probe must be the only tone on its input.")
    frequency_values, frequency_axis = _axis_values(frequencies)
    amplitude_values, amplitude_axis = _axis_values(amplitudes)
    vna._validate_variations(variations, reserved=(
        *(("frequency",) if frequency_axis else ()), *(("amplitude",) if amplitude_axis else ())))
    shape, variation_points = _iter_axis_points(variations)
    axes = _public_axes(variations)
    if amplitude_axis:
        shape += (len(amplitude_values),)
        axes += (("amplitude", amplitude_values),)
    if frequency_axis:
        shape += (len(frequency_values),)
        axes += (("frequency", frequency_values),)
    offsets = noise_grid(noise_frequencies)
    xp = vna.chip.backend.array_module
    modes = tuple(device.label for device in vna.chip.devices if isinstance(device.local_space(), FockSpace))
    captured: list[_Acquisition] = []
    incident, parameters = [], []
    points: list[Any] = []
    for _, params in variation_points:
        chip = vna._chip_at(params)
        linear = (try_build_linear_response_problem(chip, frequency_values, plane_labels=labels)
                  if not vna._tones and options is None else None)
        if linear is not None and not contains_tracer((linear.hamiltonian, linear.couplings)):
            couplings = np.asarray(linear.couplings)
            drift = -1j*np.asarray(linear.hamiltonian)-couplings.conj().T@couplings/2
            if np.any(np.linalg.eigvals(drift).real >= 0):
                raise ValueError("A stationary harmonic measurement requires all modes to decay.")
        points.extend((chip, linear, params, amplitude, frequency)
                      for amplitude in amplitude_values for frequency in frequency_values)
    if progress:
        from tqdm import tqdm
        points = tqdm(points, desc="VNA measurement")
    for chip, linear, params, amplitude, frequency in points:
        parameters.append(dict(chip.parameters))
        tones = vna._tone_values(params) + ((input_label, frequency, amplitude),)
        if linear is None:
            operating = _operating_point(chip, tones, frequency, (), options)
            mean = _plane_means(operating.engine, operating.state.state, chip.backend, labels, tones, frequency)
            components, output_delays = capture_noise(operating, chip.backend, labels, frequency, xp.asarray(offsets))
            amplitudes_at_point, numbers_at_point, frames_at_point = capture_modes(chip, operating, modes)
            acquired = _Acquisition(mean, components, output_delays, operating.diagnostics,
                                    amplitudes_at_point, numbers_at_point, frames_at_point)
        else:
            acquired = capture_linear(
                linear, chip.backend, labels, input_label, frequency, amplitude, xp.asarray(offsets))
        captured.append(acquired)
        incident.append(xp.asarray(amplitude))
    size = 2 * len(labels)
    names = tuple(captured[0].components)
    if any(tuple(point.components) != names for point in captured):
        raise ValueError("Measurement variations must preserve noise source structure.")
    components = {
        name: (xp.stack([point.components[name][0] for point in captured]).reshape((*shape, size, size)),
               xp.stack([point.components[name][1] for point in captured]).reshape((*shape, len(offsets), size, size)))
        for name in names
    }
    return VNAMeasurement(
        ports=labels, input=resolve_label(input_label),
        frequencies=frequency_values if frequency_axis else frequency_values[0],
        amplitudes=amplitude_values if amplitude_axis else amplitude_values[0], axes=axes, shape=shape,
        diagnostics=tuple(point.diagnostics for point in captured),
        values=xp.stack([point.mean for point in captured]).reshape((*shape, len(labels))),
        incident=xp.stack(incident).reshape(shape), noise_frequencies=offsets, noise_components=components,
        output_delays=xp.stack([point.delays for point in captured]).reshape((*shape, len(labels))),
        parameters=tuple(parameters), modes=modes,
        mode_amplitudes=xp.stack([point.mode_amplitudes for point in captured]).reshape((*shape, len(modes))),
        photon_numbers=xp.stack([point.photon_numbers for point in captured]).reshape((*shape, len(modes))),
        mode_frequencies=xp.stack([point.mode_frequencies for point in captured]).reshape((*shape, len(modes))),
    )
