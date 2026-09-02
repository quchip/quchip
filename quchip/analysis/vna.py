"""Continuous-wave scattering through declared Markovian ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quchip.engine.input_output import (
    add_port_inputs,
    port_operators,
    resolve_stationary_engine,
)
from quchip.engine.ir import CanonicalOperator, EngineResult, SteadyStateProblem
from quchip.engine.steady_state import solve_steadystate_problem
from quchip.results.input_output import (
    OutputCorrelationResult,
    OutputSpectrumResult,
    SParameterResult,
)
from quchip.sweep import Sweep, ZippedSweep, _axis_metadata, _iter_axis_points
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar
from quchip.utils.labeling import resolve_label


@dataclass(frozen=True)
class PortTone:
    """A fixed coherent input field entering through one declared port."""

    port: str
    freq: Any
    amplitude: Any
    _index: int


@dataclass(frozen=True)
class _ExposureRef:
    """Validated external SLH exposure retained by label."""

    label: str


class VNA:
    """Sweep one incident exposure and report differential output scattering."""

    def __init__(self, chip: Any, *, input: Any, outputs: list[Any] | tuple[Any, ...]) -> None:
        self.chip = chip
        self.input = _resolve_exposure(chip, input)
        self.outputs = tuple(_resolve_exposure(chip, port) for port in outputs)
        if not self.outputs:
            raise ValueError("VNA requires at least one output port.")
        labels = [port.label for port in self.outputs]
        if len(set(labels)) != len(labels):
            raise ValueError(f"VNA output ports must be unique, got {labels}.")
        self._tones: list[PortTone] = []
        self._variations: set[Sweep] = set()

    def pump(self, port: Any, *, freq: Any, amplitude: Any) -> PortTone:
        """Add a fixed background tone and return its variation handle."""
        resolved = _resolve_exposure(self.chip, port)
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
        options: dict | None = None,
        progress: bool = False,
    ) -> SParameterResult:
        """Compute the differential scattering response around fixed pumps."""
        self._validate_variations(variations)
        freq_values, freq_is_axis = _axis_values(frequencies)
        variation_shape, variation_points = _iter_axis_points(variations)
        shape = variation_shape
        if freq_is_axis:
            shape += (len(freq_values),)

        points = [
            (coord, params, frequency_index, frequency)
            for coord, params in variation_points
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
        for coord, params, frequency_index, frequency in iterator:
            cache_key = (coord, frequency_index)
            if cache_key not in operating_cache:
                tones = self._tone_values(params)
                engine = resolve_stationary_engine(
                    self.chip,
                    tuple((label, tone_frequency) for label, tone_frequency, _ in tones)
                    + ((self.input.label, frequency),),
                )
                operating_engine = add_port_inputs(engine, self.chip.backend, tones)
                operating = _solve_engine(self.chip, operating_engine, options)
                resolved_operators = port_operators(operating_engine, self.chip.backend)
                operating_cache[cache_key] = (operating_engine, operating, resolved_operators)
            operating_engine, operating, resolved_operators = operating_cache[cache_key]

            values = _small_signal_response(
                operating_engine,
                operating.state,
                self.chip.backend,
                resolved_operators,
                self.input.label,
                tuple(port.label for port in self.outputs),
                frequency,
            )
            state = operating

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
        if freq_is_axis:
            axes += (("frequency", freq_values),)
        return SParameterResult(
            frequencies=freq_values if freq_is_axis else freq_values[0],
            input_port=self.input.label,
            output_ports=tuple(port.label for port in self.outputs),
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
        output_port = _resolve_exposure(self.chip, output)
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
        source = _canonical_matrix(
            -(fluctuation @ rho),
            operators[output_port.label],
            tag=f"spectrum-source:{output_port.label}",
        )
        observable = _canonical_matrix(
            xp.conj(xp.swapaxes(fluctuation, -1, -2)),
            operators[output_port.label],
            tag=f"spectrum-observable:{output_port.label}",
        )
        response = self.chip.backend.stationary_resolvent(
            engine,
            source,
            ((output_port.label, observable),),
            frequency_values,
        )[output_port.label]
        spectra = 2.0 * xp.real(response)
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
        output_port = _resolve_exposure(self.chip, output)
        input_port = output_port if input is None else _resolve_exposure(self.chip, input)
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

        initial = _canonical_matrix(
            input_field @ rho if order == 1 else input_field @ rho @ input_field_dag,
            operators[input_port.label],
            tag=f"g{order}-initial:{input_port.label}",
        )
        observable = _canonical_matrix(
            output_field_dag if order == 1 else output_number,
            operators[output_port.label],
            tag=f"g{order}-observable:{output_port.label}",
        )
        raw = backend.stationary_propagate(
            engine,
            initial,
            ((output_port.label, observable),),
            delay_values,
        )[output_port.label]
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
        reference_frequency = next(
            (frequency for port_label, frequency, _ in tones if port_label == self.input.label),
            _exposure_reference_frequency(self.chip, self.input.label),
        )
        engine = resolve_stationary_engine(
            self.chip,
            tuple((label, frequency) for label, frequency, _ in tones)
            + ((self.input.label, reference_frequency),),
        )
        driven_engine = add_port_inputs(engine, self.chip.backend, tones)
        state = _solve_engine(self.chip, driven_engine, options)
        incoming = _stationary_output_backgrounds(driven_engine, tones, self.chip.backend)
        operators = port_operators(driven_engine, self.chip.backend)
        xp = self.chip.backend.array_module
        for channel in driven_engine.slh.external_channels:
            frequency = (
                0.0
                if channel.collapse.frame_frequency is None
                else channel.collapse.frame_frequency
            )
            phase = xp.exp(
                1j
                * TWO_PI
                * xp.asarray(frequency)
                * xp.asarray(channel.reference_delay)
            )
            operators[channel.key] = operators[channel.key].scaled(
                phase,
                tag=f"reference_plane:{channel.key}",
            )
            incoming[channel.key] = phase * incoming[channel.key]
        return driven_engine, state, operators, incoming

    def _tone_values(self, params: dict[str, Any]) -> tuple[tuple[str, Any, Any], ...]:
        tones: list[tuple[str, Any, Any]] = []
        for tone in self._tones:
            freq = params.get(f"__vna_tone_{tone._index}_freq", tone.freq)
            amplitude = params.get(f"__vna_tone_{tone._index}_amplitude", tone.amplitude)
            tones.append((tone.port, freq, amplitude))
        return tuple(tones)


def _axis_values(values: Any) -> tuple[Any, bool]:
    if np.ndim(values) == 0:
        return (values,), False
    array = values if hasattr(values, "shape") else np.asarray(values)
    if len(array) == 0:
        raise ValueError("VNA frequency sweep cannot be empty.")
    return array, True


def _resolve_exposure(chip: Any, value: Any) -> _ExposureRef:
    """Resolve a Port object or label against the external SLH boundary."""
    label = resolve_label(value)
    available = [channel.key for channel in chip.resolve().slh.external_channels]
    if label not in available:
        raise ValueError(f"Unknown VNA exposure {label!r}. Available exposures: {available}.")
    return _ExposureRef(label)


def _exposure_reference_frequency(chip: Any, label: str) -> Any:
    """Return the exposure carrier in the chip's natural rotating frame."""
    resolved = chip.resolve(frame="rotating")
    for channel in resolved.slh.external_channels:
        if channel.key == label:
            frequency = channel.collapse.frame_frequency
            return 0.0 if frequency is None else frequency
    available = [channel.key for channel in resolved.slh.external_channels]
    raise ValueError(f"Unknown VNA exposure {label!r}. Available exposures: {available}.")


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
    return _axis_metadata(
        variations,
        rename=lambda name: _public_axis_name(name, tone_list),
    )


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


def _stationary_output_backgrounds(
    engine: EngineResult,
    tones: tuple[tuple[str, Any, Any], ...],
    backend: Any,
) -> dict[str, Any]:
    """Return ``S beta`` at each Markov boundary for fixed reference-plane tones."""
    xp = backend.array_module
    external = engine.slh.external_channels
    exposure_index = {channel.key: index for index, channel in enumerate(external)}
    incident = [xp.asarray(0.0 + 0.0j) for _ in external]
    for label, frequency, amplitude in tones:
        index = exposure_index[label]
        incident[index] = incident[index] + xp.asarray(amplitude) * xp.exp(
            1j * TWO_PI * xp.asarray(frequency) * xp.asarray(external[index].reference_delay)
        )
    return {
        channel.key: sum(
            (
                xp.asarray(engine.slh.S[output_index, input_index]) * beta
                for input_index, beta in enumerate(incident)
            ),
            start=xp.asarray(0.0 + 0.0j),
        )
        for output_index, channel in enumerate(external)
    }


def _small_signal_response(
    engine: EngineResult,
    state: Any,
    backend: Any,
    port_operators: dict[str, CanonicalOperator],
    input_label: str,
    output_labels: tuple[str, ...],
    frequency: Any,
) -> dict[str, Any]:
    """Evaluate the port response through the backend stationary resolvent."""
    xp = backend.array_module
    rho = xp.asarray(backend.to_array(state), dtype=complex)
    external = engine.slh.external_channels
    exposure_index = {channel.key: index for index, channel in enumerate(external)}
    input_index = exposure_index[input_label]
    template = port_operators[external[0].key]
    input_operator = xp.zeros(template.shape, dtype=complex)
    for output_index, channel in enumerate(external):
        input_operator = input_operator + xp.conj(
            xp.asarray(engine.slh.S[output_index, input_index])
        ) * xp.asarray(port_operators[channel.key].to_dense(), dtype=complex)
    input_dag = xp.conj(xp.swapaxes(input_operator, -1, -2))
    source = _canonical_matrix(
        input_dag @ rho - rho @ input_dag,
        template,
        tag=f"linear-response-source:{input_label}",
    )
    response = backend.stationary_resolvent(
        engine,
        source,
        tuple((label, port_operators[label]) for label in output_labels),
        (0.0,),
    )
    input_phase = xp.exp(
        1j * TWO_PI * xp.asarray(frequency) * xp.asarray(external[input_index].reference_delay)
    )
    result: dict[str, Any] = {}
    for label in output_labels:
        output_index = exposure_index[label]
        output_phase = xp.exp(
            1j
            * TWO_PI
            * xp.asarray(frequency)
            * xp.asarray(external[output_index].reference_delay)
        )
        boundary = xp.asarray(engine.slh.S[output_index, input_index]) + response[label][0]
        result[label] = input_phase * output_phase * boundary
    return result


def _output_field_matrix(operator: CanonicalOperator, incoming: Any, xp: Any) -> Any:
    coupling = xp.asarray(operator.to_dense(), dtype=complex)
    return xp.asarray(incoming) * xp.eye(coupling.shape[0], dtype=complex) + coupling


def _canonical_matrix(
    values: Any,
    template: CanonicalOperator,
    *,
    tag: str,
) -> CanonicalOperator:
    """Attach resolved subsystem metadata to a stationary query matrix."""
    return CanonicalOperator.from_dense(
        values,
        dims=template.dims,
        basis=template.basis,
        subsystem_labels=template.subsystem_labels,
        tag=tag,
    )
