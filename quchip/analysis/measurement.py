"""Physical stationary acquisition, independent of receiver configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from quchip.engine.field_statistics import propagate_noise, proper_spectrum, quadrature_spectrum
from quchip.engine.routes import (
    WORKFLOWS, SolveRoute, array_name, automatic_cutoff, largest_change, plan_sectors,
    sector_diagnostics, warn_cutoff,
)
from quchip.utils.memory import fits_memory
from quchip.cutoff import active_cutoff
from quchip.results.receiver import _engineering_iq, noise_grid
from quchip.engine.ir import LinearResponseProblem
from quchip.devices.spaces import FockSpace
from quchip.engine.linear_response import try_build_linear_response_problem
from quchip.backend._sector_stationary import padded
from quchip.engine.reference import (
    FieldChannel, cw_transfer,
)
from quchip.results.measurement import VNAMeasurement
from quchip.results.input_output import MeanFieldResponseResult, _LazyDiagnostics
from quchip.sweep import Sweep, ZippedSweep, _iter_axis_points
from quchip.utils.jax_utils import contains_tracer
from quchip.utils.labeling import resolve_label


# The noise comparison runs when the larger cutoff of a pair has at most this many states.
_NOISE_CHECK_STATES = 1024
_HARMONIC = ("The chip is passive and harmonic, so its mode equations give the response and the noise without a "
             "Fock truncation.")


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
    from quchip.engine.observables import prepare_local_op

    backend = chip.backend
    xp = backend.array_module
    space = operating.engine.excitation_space
    labels = tuple(device.label for device in chip.devices)
    amplitudes, numbers, frequencies = [], [], []
    for label in modes:
        device = chip.device_map[label]
        basis = operating.engine.bases[label]
        # Under an excitation cutoff, the state holds only the retained labels.
        state = (operating.state.reduced_state(label) if space is None
                 else space.reduced(operating.state.state, backend, (labels.index(label),)))
        amplitudes.append(backend.expect(prepare_local_op(device, "a", basis, backend), state))
        numbers.append(xp.real(backend.expect(prepare_local_op(device, "n", basis, backend), state)))
        frequencies.append(operating.engine.resolved_frame.frequencies[label])
    return xp.asarray(amplitudes), xp.asarray(numbers), xp.asarray(frequencies)


def capture_noise(operating: Any, backend: Any, labels: tuple[str, ...], frequency: Any, offsets: Any) -> Any:
    """Calculate joint physical IQ spectra and source budgets from one operating point."""
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
    route = SolveRoute("linear_response", _HARMONIC, len(problem.mode_labels), array_name(backend))
    return _Acquisition(mean, components, delays,
                        {"solver": route.name, "route": route, "mode_count": len(problem.mode_labels),
                         "residual": xp.max(solved.residuals)},
                        mode_amplitudes, numbers, xp.full((len(problem.mode_labels),), frequency))


def measure(
    vna: Any, frequencies: Any, amplitudes: Any, variations: tuple[Sweep | ZippedSweep, ...], *,
    input: Any, outputs: Any, noise_frequencies: Any, options: Any, progress: bool,
) -> VNAMeasurement:
    """Acquire physical fields and correlations once for each probe/sweep point."""
    from quchip.analysis.vna import _resolve_exposure
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
    return _acquire(vna, frequencies, amplitudes, variations, labels=labels, input_label=input_label,
                    options=options, progress=progress, noise=True, noise_frequencies=noise_frequencies)


def _acquire(
    vna: Any, frequencies: Any, amplitudes: Any, variations: tuple[Sweep | ZippedSweep, ...], *,
    labels: tuple[str, ...], input_label: str, options: Any, progress: bool,
    noise: bool = False, noise_frequencies: Any = None,
) -> Any:
    """Acquire one finite-probe grid, with optional noise and mode observables."""
    from quchip.analysis.vna import _axis_values, _operating_point, _plane_means, _public_axes

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
    offsets = noise_grid(noise_frequencies) if noise else None
    xp = vna.chip.backend.array_module
    modes = tuple(device.label for device in vna.chip.devices
                  if isinstance(device.local_space(), FockSpace)) if noise else ()
    incident, parameters = [], []
    points: list[Any] = []
    concrete_grid = not contains_tracer((frequency_values, amplitude_values))
    method = "VNA.measure()" if noise else "VNA.finite_power()"
    workflow = WORKFLOWS[method]
    for _, params in variation_points:
        chip, pumps = vna._chip_at(params), vna._tone_values(params)
        cutoff = active_cutoff(chip)
        # Under a cutoff, the projected model is not the harmonic chip, so the harmonic route does not apply.
        linear = (try_build_linear_response_problem(chip, frequency_values, plane_labels=labels)
                  if noise and not vna._tones and options is None and cutoff is None else None)
        if linear is not None and not contains_tracer((linear.hamiltonian, linear.couplings)):
            couplings = np.asarray(linear.couplings)
            drift = -1j*np.asarray(linear.hamiltonian)-couplings.conj().T@couplings/2
            if np.any(np.linalg.eigvals(drift).real >= 0):
                raise ValueError("A stationary harmonic measurement requires all modes to decay.")
        sector, reason = None, _HARMONIC
        if linear is None:
            # Pumps share the frame of the probe only at their own frequency.
            sector, reason = plan_sectors(
                chip, workflow, labels=labels, inputs=(input_label, *(port for port, _, _ in pumps)), modes=modes,
                frequencies=(*frequency_values, *(frequency for _, frequency, _ in pumps)) if pumps else (),
                options=options, concrete=concrete_grid and not contains_tracer(pumps))
        points.extend((chip, linear, sector, pumps, amplitude, frequency, reason)
                      for amplitude in amplitude_values for frequency in frequency_values)
    captured: list[Any] = [None] * len(points)
    sector_points = [index for index, point in enumerate(points) if point[2] is not None]
    if sector_points:
        solved = _sector_acquire([points[index] for index in sector_points], labels=labels, offsets=offsets,
                                 noise=noise, progress=progress, method=method, xp=xp)
        for index, acquired in zip(sector_points, solved, strict=True):
            captured[index] = acquired
    remaining = [index for index, point in enumerate(points) if point[2] is None]
    iterator: Any = remaining
    if progress and remaining:
        from tqdm import tqdm
        iterator = tqdm(remaining, desc="VNA measurement" if noise else "VNA power")
    for index in iterator:
        chip, linear, _, pumps, amplitude, frequency, reason = points[index]
        tones = pumps + ((input_label, frequency, xp.conj(amplitude)),)
        if linear is None:
            operating = _operating_point(chip, tones, frequency, (), options, reason=reason)
            mean = _plane_means(operating.engine, operating.state.state, chip.backend, labels, tones, frequency)
            components, output_delays = (capture_noise(operating, chip.backend, labels, frequency, xp.asarray(offsets))
                                         if noise else ({}, None))
            amplitudes_at_point, numbers_at_point, frames_at_point = (
                capture_modes(chip, operating, modes) if noise else (None, None, None))
            acquired = _Acquisition(mean, components, output_delays, operating.diagnostics,
                                    amplitudes_at_point, numbers_at_point, frames_at_point)
        else:
            acquired = capture_linear(
                linear, chip.backend, labels, input_label, frequency, xp.conj(amplitude), xp.asarray(offsets))
        captured[index] = acquired
    for chip, _, _, _, amplitude, _, _ in points:
        if noise:
            parameters.append(dict(chip.parameters))
        incident.append(xp.asarray(amplitude))
    response = dict(
        ports=labels, input=resolve_label(input_label),
        frequencies=frequency_values if frequency_axis else frequency_values[0],
        amplitudes=amplitude_values if amplitude_axis else amplitude_values[0], axes=axes, shape=shape,
        diagnostics=tuple(point.diagnostics for point in captured),
        values=xp.conj(xp.stack([point.mean for point in captured]).reshape((*shape, len(labels)))),
        incident=xp.stack(incident).reshape(shape),
    )
    if not noise:
        return MeanFieldResponseResult(**response)
    assert offsets is not None
    size = 2 * len(labels)
    names = tuple(captured[0].components)
    if any(tuple(point.components) != names for point in captured):
        raise ValueError("Measurement variations must preserve noise source structure.")
    components = {
        name: (_engineering_iq(xp.stack([point.components[name][0] for point in captured])
                              .reshape((*shape, size, size)), xp),
               _engineering_iq(xp.stack([point.components[name][1] for point in captured])
                              .reshape((*shape, len(offsets), size, size)), xp))
        for name in names
    }
    return VNAMeasurement(
        **response, noise_frequencies=offsets, noise_components=components,
        output_delays=xp.stack([point.delays for point in captured]).reshape((*shape, len(labels))),
        parameters=tuple(parameters), modes=modes,
        mode_amplitudes=xp.stack([point.mode_amplitudes for point in captured]).reshape((*shape, len(modes))),
        photon_numbers=xp.stack([point.photon_numbers for point in captured]).reshape((*shape, len(modes))),
        mode_frequencies=xp.stack([point.mode_frequencies for point in captured]).reshape((*shape, len(modes))),
    )


@dataclass(frozen=True)
class _SectorPoint:
    """One sector solve at one grid point, in the physics convention."""

    solve: Any
    mean: Any
    mode_amplitudes: Any
    photon_numbers: Any
    excess: Any
    noise: dict[str, Any]
    # Noise solutions that start the noise solves of the next cutoff; None when they did not fit in memory.
    responses: Any = None


def _sector_pass(entries: list[Any], cutoff: int, *, offsets: Any, noise: bool, guesses: list[_SectorPoint] | None,
                 progress: bool, method: str) -> list[_SectorPoint]:
    """Solve every grid point on ``N <= cutoff``, starting from nearby solutions."""
    iterator: Any = range(len(entries))
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"{method} N<={cutoff}")
    points: list[_SectorPoint] = []
    previous: tuple[Any, Any] | None = None
    for index in iterator:
        chip, _, model, pumps, amplitude, frequency, _ = entries[index]
        frequency = float(frequency)
        internal = (complex(np.conj(amplitude)), *(complex(value) for _, _, value in pumps))
        guess = None
        if guesses is not None:
            guess = padded(guesses[index].solve.state, model.size(cutoff))
        elif previous is not None and previous[0] is chip:
            guess = previous[1]
        solved = model.stationary(cutoff, frequency, internal, guess)
        amplitudes, numbers = model.mode_values(cutoff, solved.state)
        excess, info, responses = None, {}, None
        if noise:
            starts = None if guesses is None else guesses[index].responses
            excess, info, responses = model.quadrature_spectrum(cutoff, frequency, internal, solved.state, offsets,
                                                                starts)
        points.append(_SectorPoint(solved, model.means(cutoff, frequency, internal, solved.state),
                                   amplitudes, numbers, excess, info, responses))
        previous = (chip, solved.state)
    return points


def _sector_change(lower: list[_SectorPoint], upper: list[_SectorPoint], noise: bool) -> float:
    """Compare the means, mode observables and, optionally, the noise spectra of two cutoffs."""
    changes = [largest_change([p.mean for p in lower], [p.mean for p in upper]),
               largest_change([p.mode_amplitudes for p in lower], [p.mode_amplitudes for p in upper]),
               largest_change([p.photon_numbers for p in lower], [p.photon_numbers for p in upper])]
    if noise:
        changes.append(largest_change([spectrum for p in lower for spectrum in p.excess],
                                      [spectrum for p in upper for spectrum in p.excess]))
    return max(changes)


def _sector_acquire(entries: list[Any], *, labels: tuple[str, ...], offsets: Any, noise: bool,
                    progress: bool, method: str, xp: Any) -> list[_Acquisition]:
    """Acquire grid points on excitation sectors at the chip's cutoff, or at an automatic one."""
    models = [entry[2] for entry in entries]
    first = models[0]
    fixed = first.sectors.cutoff

    def solve(cutoff: int, lower: list[_SectorPoint] | None) -> list[_SectorPoint]:
        # A comparison above the noise limit compares the means and mode observables only.
        spectra = noise and (lower is None or first.size(cutoff) <= _NOISE_CHECK_STATES)
        return _sector_pass(entries, cutoff, offsets=offsets, noise=spectra, guesses=lower, progress=progress,
                            method=method)

    def complete(cutoff: int, points: list[_SectorPoint]) -> list[_SectorPoint]:
        if not noise or points[0].excess is not None:
            return points
        return _sector_pass(entries, cutoff, offsets=offsets, noise=True, guesses=points, progress=progress,
                            method=method)

    if fixed is not None:
        cutoff, choice, check = fixed, "explicit", None
        points = solve(cutoff, None)
    else:
        unique = {id(model): model for model in models}.values()
        choice = "automatic"

        def fits(total: int) -> bool:
            return all(fits_memory(model.required_memory(total)) for model in unique)

        start = None
        if noise:
            # The means settle no later than the noise spectra, so the full ladder starts at the lower
            # cutoff of the first pair whose means agree.
            kept, _, means = automatic_cutoff(
                first.max_total, lambda cutoff, lower: _sector_pass(
                    entries, cutoff, offsets=offsets, noise=False, guesses=lower, progress=progress, method=method),
                lambda lower, upper: _sector_change(lower, upper, False), states=first.size, fits=fits)
            start = kept - 1 if means is not None and means.status == "passed" else kept
        cutoff, points, check = automatic_cutoff(
            first.max_total, solve, lambda lower, upper: _sector_change(lower, upper, upper[0].excess is not None),
            complete=complete, partial=lambda points: noise and points[0].excess is None, states=first.size,
            fits=fits, start=start)
        if check is not None:
            quantities = "means" + (", mode amplitudes and photon numbers" if first.modes else "")
            note = ""
            if noise and check.status == "passed":
                if check.states <= _NOISE_CHECK_STATES:
                    quantities = quantities.replace(" and ", ", ") + " and noise spectra"
                else:
                    note = f" (noise spectra not compared above {_NOISE_CHECK_STATES:,} states)"
            warn_cutoff(method, cutoff, check, quantities, note=note)
    acquired = []
    for entry, point in zip(entries, points, strict=True):
        _, _, model, _, _, frequency, reason = entry
        components: dict[str, tuple[Any, Any]] = {}
        delays = None
        frames: np.ndarray = np.zeros(0)
        if noise:
            components, delays = propagate_noise(model.fields, model.scattering, None, point.excess, labels,
                                                 float(frequency), np.asarray(offsets), np)
            components = {name: (xp.asarray(white), xp.asarray(excess)) for name, (white, excess) in components.items()}
            # Every mode is solved in the probe frame.
            frames = np.full(len(model.modes), float(frequency))
        diagnostics = {**sector_diagnostics(model, cutoff, point.solve, reason, choice, check), **point.noise}
        if check is not None:
            diagnostics["cutoff_check"]["noise_compared"] = (noise and check.status == "passed"
                                                             and check.states <= _NOISE_CHECK_STATES)
        acquired.append(_Acquisition(xp.asarray(point.mean), components, None if delays is None else xp.asarray(delays),
                                     _LazyDiagnostics(diagnostics), xp.asarray(point.mode_amplitudes),
                                     xp.asarray(point.photon_numbers), xp.asarray(frames)))
    return acquired
