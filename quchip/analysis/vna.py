"""Continuous-wave scattering through declared Markovian ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quchip.engine.input_output import (
    add_port_inputs,
    dense_liouvillian,
    port_operators,
    same_frequency,
    small_signal_response,
)
from quchip.engine.ir import CanonicalOperator, EngineResult, SteadyStateProblem
from quchip.engine.steady_state import solve_steadystate_problem
from quchip.results.input_output import (
    OutputCorrelationResult,
    OutputSpectrumResult,
    SParameterResult,
)
from quchip.sweep import Sweep, ZippedSweep, _iter_axis_points
from quchip.utils.constants import TWO_PI, hbar
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar


@dataclass(frozen=True)
class PortTone:
    """A fixed coherent input field entering through one declared port."""

    port: str
    freq: Any
    amplitude: Any
    _index: int


class VNA:
    """Sweep one input port and report complex output-field response."""

    def __init__(self, chip: Any, *, input: Any, outputs: list[Any] | tuple[Any, ...]) -> None:
        self.chip = chip
        self.input = chip.port(input)
        self.outputs = tuple(chip.port(port) for port in outputs)
        if not self.outputs:
            raise ValueError("VNA requires at least one output port.")
        labels = [port.label for port in self.outputs]
        if len(set(labels)) != len(labels):
            raise ValueError(f"VNA output ports must be unique, got {labels}.")
        self._tones: list[PortTone] = []
        self._variations: set[Sweep] = set()

    def pump(self, port: Any, *, freq: Any, amplitude: Any) -> PortTone:
        """Add a fixed background tone and return its variation handle."""
        resolved = self.chip.port(port)
        tone = PortTone(resolved.label, freq, amplitude, len(self._tones))
        self._tones.append(tone)
        return tone

    def vary(self, tone: PortTone, parameter: str, values: Any) -> Sweep:
        """Create a sweep axis for one fixed tone's frequency or amplitude."""
        if tone._index >= len(self._tones) or self._tones[tone._index] is not tone:
            raise ValueError("The tone does not belong to this VNA.")
        if parameter not in {"freq", "amplitude"}:
            raise ValueError("A port tone can vary only 'freq' or 'amplitude'.")
        variation = Sweep(values, name=f"__vna_tone_{tone._index}_{parameter}")
        self._variations.add(variation)
        return variation

    @staticmethod
    def zip(*variations: Sweep) -> ZippedSweep:
        """Pair tone variations element by element."""
        return Sweep.zip(*variations)

    def sweep(
        self,
        frequencies: Any,
        *variations: Sweep | ZippedSweep,
        amplitude: Any | None = None,
        options: dict | None = None,
        progress: bool = False,
    ) -> SParameterResult:
        """Compute differential or finite-amplitude scattering response."""
        self._validate_variations(variations)
        freq_values, freq_is_axis = _axis_values(frequencies)
        amplitude_values, amplitude_is_axis = _amplitude_values(amplitude)
        variation_shape, variation_points = _iter_axis_points(variations)
        shape = variation_shape
        if amplitude_is_axis:
            shape += (len(amplitude_values),)
        if freq_is_axis:
            shape += (len(freq_values),)

        points = [
            (coord, params, amplitude_index, probe_amplitude, frequency_index, frequency)
            for coord, params in variation_points
            for amplitude_index, probe_amplitude in enumerate(amplitude_values)
            for frequency_index, frequency in enumerate(freq_values)
        ]
        iterator: Any = points
        if progress:
            from tqdm import tqdm

            iterator = tqdm(points, desc="VNA")

        responses: dict[str, list[Any]] = {port.label: [] for port in self.outputs}
        steady_states: list[Any] = []
        diagnostics: list[dict[str, Any]] = []
        operating_cache: dict[tuple[tuple[int, ...], int], tuple[Any, Any, Any]] = {}
        for coord, params, _amplitude_index, probe_amplitude, frequency_index, frequency in iterator:
            cache_key = (coord, frequency_index)
            if cache_key not in operating_cache:
                tones = self._tone_values(params)
                engine = self._resolved_engine(frequency, tones)
                operating_engine = add_port_inputs(engine, self.chip.backend, tones)
                operating = _solve_engine(self.chip, operating_engine, options)
                resolved_operators = port_operators(operating_engine, self.chip.backend)
                operating_cache[cache_key] = (operating_engine, operating, resolved_operators)
            operating_engine, operating, resolved_operators = operating_cache[cache_key]

            if amplitude is None:
                values = small_signal_response(
                    operating_engine,
                    operating.state,
                    self.chip.backend,
                    resolved_operators,
                    self.input.label,
                    tuple(port.label for port in self.outputs),
                )
                state = operating
            else:
                driven_engine = add_port_inputs(
                    operating_engine,
                    self.chip.backend,
                    ((self.input.label, frequency, probe_amplitude),),
                )
                driven = _solve_engine(self.chip, driven_engine, options)
                values = _finite_response(
                    operating.state,
                    driven.state,
                    self.chip.backend,
                    resolved_operators,
                    self.input.label,
                    tuple(port.label for port in self.outputs),
                    probe_amplitude,
                )
                state = driven

            for label, value in values.items():
                responses[label].append(value)
            steady_states.append(state)
            diagnostics.append(
                {
                    "residual": state.residual,
                    "trace_error": state.trace_error,
                    "positivity_error": state.positivity_error,
                    "condition_number": state.condition_number,
                }
            )

        xp = self.chip.backend.array_module
        response_arrays = {
            (label, self.input.label): xp.reshape(xp.asarray(values), shape)
            for label, values in responses.items()
        }
        axes = _public_axes(variations, self._tones)
        if amplitude_is_axis:
            axes += (("amplitude", amplitude_values),)
        if freq_is_axis:
            axes += (("frequency", freq_values),)
        photon_fluxes = (
            None
            if amplitude is None
            else xp.abs(xp.asarray(amplitude_values if amplitude_is_axis else amplitude)) ** 2
        )
        return SParameterResult(
            frequencies=freq_values if freq_is_axis else freq_values[0],
            input_port=self.input.label,
            output_ports=tuple(port.label for port in self.outputs),
            input_amplitudes=(amplitude_values if amplitude_is_axis else amplitude),
            input_photon_fluxes=photon_fluxes,
            input_powers=_input_powers(
                photon_fluxes,
                freq_values if freq_is_axis else freq_values[0],
                amplitude_is_axis=amplitude_is_axis,
                frequency_is_axis=freq_is_axis,
                variation_rank=len(variation_shape),
                result_shape=shape,
                xp=xp,
            ),
            axes=axes,
            shape=shape,
            steady_states=tuple(steady_states),
            diagnostics=tuple(diagnostics),
            _response=response_arrays,
        )

    def _validate_variations(self, variations: tuple[Sweep | ZippedSweep, ...]) -> None:
        for variation in variations:
            members = variation.sweeps if isinstance(variation, ZippedSweep) else (variation,)
            if any(member not in self._variations for member in members):
                raise ValueError("VNA variations must be created by this VNA's vary() method.")

    def output_spectrum(
        self,
        output: Any,
        *,
        frequencies: Any,
        options: dict | None = None,
    ) -> OutputSpectrumResult:
        """Return the normally ordered output fluctuation spectrum.

        Frequencies are offsets in GHz from the stationary tone frame. The
        coherent carrier is reported separately because it is a delta peak,
        not a finite sampled spectral density.
        """
        output_port = self.chip.port(output)
        frequency_values, _ = _axis_values(frequencies)
        engine, state, operators, incoming = self._stationary_output(options)
        xp = self.chip.backend.array_module
        rho = xp.asarray(self.chip.backend.to_array(state.state), dtype=complex)
        field = _output_field_matrix(
            operators[output_port.label], incoming.get(output_port.label, 0.0), xp
        )
        mean = xp.trace(field @ rho)
        identity = xp.eye(rho.shape[0], dtype=complex)
        fluctuation = field - mean * identity
        intensity = xp.real(xp.trace(xp.conj(xp.swapaxes(field, -1, -2)) @ field @ rho))
        coherent_flux = xp.abs(mean) ** 2
        source = (fluctuation @ rho).T.reshape(-1)
        observable = xp.conj(xp.swapaxes(fluctuation, -1, -2))
        liouvillian = dense_liouvillian(engine, self.chip.backend, operation="Output spectrum")
        spectra = []
        for frequency in frequency_values:
            shifted = liouvillian + 1j * (2.0 * np.pi) * xp.asarray(frequency) * xp.eye(
                liouvillian.shape[0], dtype=complex
            )
            propagated = -xp.linalg.pinv(shifted) @ source
            matrix = propagated.reshape(rho.shape).T
            spectra.append(2.0 * xp.real(xp.trace(observable @ matrix)))
        return OutputSpectrumResult(
            port=output_port.label,
            frequencies=frequency_values,
            fluctuation_spectrum=xp.asarray(spectra),
            output_photon_flux=intensity,
            coherent_flux=coherent_flux,
            incoherent_flux=intensity - coherent_flux,
            steady_state=state,
        )

    def g1(
        self,
        output: Any,
        delays: Any,
        *,
        input: Any | None = None,
        options: dict | None = None,
    ) -> OutputCorrelationResult:
        """Return normalized first-order output coherence."""
        return self._correlation(output, delays, input=input, order=1, options=options)

    def g2(
        self,
        output: Any,
        delays: Any,
        *,
        input: Any | None = None,
        options: dict | None = None,
    ) -> OutputCorrelationResult:
        """Return normalized second-order output intensity correlation."""
        return self._correlation(output, delays, input=input, order=2, options=options)

    def _correlation(
        self,
        output: Any,
        delays: Any,
        *,
        input: Any | None,
        order: int,
        options: dict | None,
    ) -> OutputCorrelationResult:
        output_port = self.chip.port(output)
        input_port = output_port if input is None else self.chip.port(input)
        delay_values, _ = _axis_values(delays)
        if not contains_tracer(delay_values) and np.any(np.asarray(delay_values) < 0):
            raise ValueError("Stationary output correlations require non-negative delays.")
        engine, state, operators, incoming = self._stationary_output(options)
        backend = self.chip.backend
        xp = backend.array_module
        rho = xp.asarray(backend.to_array(state.state), dtype=complex)
        output_field = _output_field_matrix(
            operators[output_port.label], incoming.get(output_port.label, 0.0), xp
        )
        input_field = _output_field_matrix(
            operators[input_port.label], incoming.get(input_port.label, 0.0), xp
        )
        output_field_dag = xp.conj(xp.swapaxes(output_field, -1, -2))
        input_field_dag = xp.conj(xp.swapaxes(input_field, -1, -2))
        output_number = output_field_dag @ output_field
        input_number = input_field_dag @ input_field
        output_intensity = xp.real(xp.trace(output_number @ rho))
        input_intensity = xp.real(xp.trace(input_number @ rho))
        concrete_output = maybe_concrete_scalar(output_intensity)
        concrete_input = maybe_concrete_scalar(input_intensity)
        if (
            concrete_output is not None
            and concrete_output <= 0
            or concrete_input is not None
            and concrete_input <= 0
        ):
            raise ValueError("Normalized output correlations require nonzero output intensity.")

        initial = input_field @ rho if order == 1 else input_field @ rho @ input_field_dag
        observable = output_field_dag if order == 1 else output_number
        liouvillian = dense_liouvillian(engine, backend, operation=f"g{order}")
        initial_vector = initial.T.reshape(-1)
        unnormalized = []
        for delay in delay_values:
            evolved = _matrix_exponential(liouvillian * xp.asarray(delay), xp) @ initial_vector
            evolved_state = evolved.reshape(rho.shape).T
            unnormalized.append(xp.trace(observable @ evolved_state))
        raw = xp.asarray(unnormalized)
        denominator = (
            xp.sqrt(input_intensity * output_intensity)
            if order == 1
            else input_intensity * output_intensity
        )
        same_port = input_port.label == output_port.label
        return OutputCorrelationResult(
            order=order,
            input_port=input_port.label,
            output_port=output_port.label,
            delays=delay_values,
            values=raw / denominator,
            unnormalized=raw,
            input_intensity=input_intensity,
            output_intensity=output_intensity,
            steady_state=state,
            normalization=(
                "G1(tau) / G1(0)"
                if order == 1 and same_port
                else "G1(output, input; tau) / sqrt(I_output I_input)"
                if order == 1
                else "G2(tau) / G1(0)^2"
                if same_port
                else "G2(output, input; tau) / (I_output I_input)"
            ),
        )

    def _stationary_output(
        self,
        options: dict | None,
    ) -> tuple[EngineResult, Any, dict[str, CanonicalOperator], dict[str, Any]]:
        tones = self._tone_values({})
        incoming: dict[str, Any] = {}
        for port_label, _frequency, amplitude in tones:
            incoming[port_label] = incoming.get(port_label, 0.0) + amplitude
        target = self.input.resolve_targets(self.chip)[0]
        reference_frequency = getattr(self.chip[target], "freq")
        for port_label, frequency, _ in tones:
            if port_label == self.input.label:
                reference_frequency = frequency
                break
        engine = self._resolved_engine(reference_frequency, tones)
        driven_engine = add_port_inputs(engine, self.chip.backend, tones)
        state = _solve_engine(self.chip, driven_engine, options)
        return driven_engine, state, port_operators(driven_engine, self.chip.backend), incoming

    def _tone_values(self, params: dict[str, Any]) -> tuple[tuple[str, Any, Any], ...]:
        tones: list[tuple[str, Any, Any]] = []
        for tone in self._tones:
            freq = params.get(f"__vna_tone_{tone._index}_freq", tone.freq)
            amplitude = params.get(f"__vna_tone_{tone._index}_amplitude", tone.amplitude)
            tones.append((tone.port, freq, amplitude))
        return tuple(tones)

    def _resolved_engine(self, probe_frequency: Any, tones: tuple[tuple[str, Any, Any], ...]) -> EngineResult:
        device_frequencies: dict[str, Any] = {}
        stationary_frequencies: list[Any] = []
        for port_label, frequency, _ in (*tones, (self.input.label, probe_frequency, 0.0)):
            stationary_frequencies.append(frequency)
            port = self.chip.port(port_label)
            for target in port.resolve_targets(self.chip):
                if target in device_frequencies and not same_frequency(device_frequencies[target], frequency):
                    raise ValueError(
                        f"Ports address {target!r} with distinct stationary tones. "
                        "Use QuantumSequence for time evolution; periodic/Floquet steady states are not supported."
                    )
                device_frequencies[target] = frequency
        # Exchange terms are stationary only when both endpoints share a
        # frame. Carry the frame of an addressed filter or resonator through
        # its passive exchange network; diagonal couplings such as CrossKerr
        # deliberately do not join the components, so pump and probe may keep
        # independent carriers.
        changed = True
        while changed:
            changed = False
            for coupling in self.chip.couplings:
                if not getattr(coupling, "folds_exchange", False):
                    continue
                left = coupling.device_a_label
                right = coupling.device_b_label
                left_frequency = device_frequencies.get(left)
                right_frequency = device_frequencies.get(right)
                if left_frequency is None and right_frequency is not None:
                    device_frequencies[left] = right_frequency
                    changed = True
                elif right_frequency is None and left_frequency is not None:
                    device_frequencies[right] = left_frequency
                    changed = True
                elif (
                    left_frequency is not None
                    and right_frequency is not None
                    and not same_frequency(left_frequency, right_frequency)
                ):
                    raise ValueError(
                        f"Exchange-coupled devices {left!r} and {right!r} are addressed by "
                        "distinct stationary tones. Use QuantumSequence for time evolution; "
                        "periodic/Floquet steady states are not supported."
                    )
        # A single carrier defines one global rotating frame. Applying it to
        # every device keeps passive number-conserving paths stationary, such
        # as feedline -> Purcell filter -> readout, without inventing a port on
        # each unaddressed internal mode. Multiple carriers still use their
        # explicitly addressed frames and are rejected below if any terms
        # remain time dependent.
        one_carrier = all(
            same_frequency(stationary_frequencies[0], frequency)
            for frequency in stationary_frequencies[1:]
        )
        frame = stationary_frequencies[0] if one_carrier else device_frequencies
        engine = self.chip.resolve(frame=frame)
        if engine.dynamic_terms:
            raise ValueError(
                "The selected tones leave dynamic Hamiltonian terms after frame and approximation resolution. "
                "Use QuantumSequence for time evolution; periodic/Floquet steady states are not supported."
            )
        return engine


def _axis_values(values: Any) -> tuple[Any, bool]:
    if np.ndim(values) == 0:
        return (values,), False
    array = values if hasattr(values, "shape") else np.asarray(values)
    if len(array) == 0:
        raise ValueError("VNA frequency sweep cannot be empty.")
    return array, True


def _amplitude_values(amplitude: Any | None) -> tuple[Any, bool]:
    if amplitude is None or np.ndim(amplitude) == 0:
        return (amplitude,), False
    array = amplitude if hasattr(amplitude, "shape") else np.asarray(amplitude)
    if len(array) == 0:
        raise ValueError("VNA amplitude sweep cannot be empty.")
    return array, True


def _public_axis_name(name: str, tones: list[PortTone]) -> str:
    prefix = "__vna_tone_"
    if not name.startswith(prefix):
        return name
    body = name[len(prefix):]
    index_text, parameter = body.split("_", 1)
    return f"{tones[int(index_text)].port}.{parameter}"


def _public_axes(
    variations: tuple[Sweep | ZippedSweep, ...],
    tones: list[PortTone] | None = None,
) -> tuple[tuple[str, Any], ...]:
    tone_list = tones or []
    axes: list[tuple[str, Any]] = []
    for variation in variations:
        if isinstance(variation, ZippedSweep):
            names = tuple(_public_axis_name(item.name, tone_list) for item in variation.sweeps)
            values = tuple(
                {names[j]: item.values[index] for j, item in enumerate(variation.sweeps)}
                for index in range(variation.size)
            )
            axes.append(("/".join(names), values))
        else:
            axes.append((_public_axis_name(variation.name, tone_list), variation.values))
    return tuple(axes)


def _solve_engine(chip: Any, engine: EngineResult, options: dict | None) -> Any:
    problem = SteadyStateProblem(
        chip=chip,
        engine_result=engine,
        e_ops=None,
        e_ops_meta=None,
        resolved_frame=engine.resolved_frame,
        options={} if options is None else options,
    )
    return solve_steadystate_problem(problem)


def _finite_response(
    operating_state: Any,
    driven_state: Any,
    backend: Any,
    port_operators: dict[str, CanonicalOperator],
    input_label: str,
    output_labels: tuple[str, ...],
    amplitude: Any,
) -> dict[str, Any]:
    xp = backend.array_module
    if not contains_tracer(amplitude) and np.ndim(amplitude) == 0 and complex(amplitude) == 0:
        raise ValueError("Finite-amplitude VNA response requires a nonzero input amplitude.")
    operating = xp.asarray(backend.to_array(operating_state), dtype=complex)
    driven = xp.asarray(backend.to_array(driven_state), dtype=complex)
    response: dict[str, Any] = {}
    for label in output_labels:
        operator = xp.asarray(port_operators[label].to_dense())
        change = -xp.trace(operator @ (driven - operating))
        if label == input_label:
            change = change + xp.asarray(amplitude)
        response[label] = change / xp.asarray(amplitude)
    return response


def _output_field_matrix(operator: CanonicalOperator, incoming: Any, xp: Any) -> Any:
    coupling = xp.asarray(operator.to_dense(), dtype=complex)
    return xp.asarray(incoming) * xp.eye(coupling.shape[0], dtype=complex) - coupling


def _input_powers(
    photon_fluxes: Any | None,
    frequencies: Any,
    *,
    amplitude_is_axis: bool,
    frequency_is_axis: bool,
    variation_rank: int,
    result_shape: tuple[int, ...],
    xp: Any,
) -> Any | None:
    """Convert GHz and photons/ns to coherent input power in watts."""
    if photon_fluxes is None:
        return None
    flux = xp.asarray(photon_fluxes)
    frequency = xp.asarray(frequencies)
    if amplitude_is_axis and frequency_is_axis:
        flux = flux[..., None]
    power = hbar * TWO_PI * 1e18 * flux * frequency
    if not result_shape:
        return power
    power = xp.reshape(power, (1,) * variation_rank + tuple(power.shape))
    return xp.broadcast_to(power, result_shape)


def _matrix_exponential(matrix: Any, xp: Any) -> Any:
    if xp.__name__.startswith("jax"):
        import jax.scipy.linalg as jsp_linalg

        return jsp_linalg.expm(matrix)
    from scipy.linalg import expm

    return expm(np.asarray(matrix, dtype=complex))
