"""Continuous-wave scattering through declared Markovian ports."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np

from quchip.utils.values import DeferredValue
from quchip.backend.containers import PreparedStationary
from quchip.engine.input_output import (
    add_port_inputs,
    port_operators,
    resolve_stationary_engine,
    same_frequency,
)
from quchip.engine.linear_response import try_build_linear_response_problem
from quchip.results.measurement import VNAMeasurement
from quchip.engine.output_network import output_mixing
from quchip.engine.reference import cw_transfer, has_amplifier, ReferenceFilter, ReferenceLoss
from quchip.engine.ir import CanonicalOperator, EngineResult, SteadyStateProblem
from quchip.engine.steady_state import solve_steadystate_problem
from quchip.results.input_output import (
    MeanFieldResponseResult,
    OutputCorrelationResult,
    OutputSpectrumResult,
    SParameterResult,
    _LazyDiagnostics,
)
from quchip.results.steady_state import SteadyStateResult
from quchip.sweep import Sweep, ZippedSweep, _axis_metadata, _iter_axis_points
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar
from quchip.utils.labeling import resolve_label
from quchip.analysis.field_statistics import canonical as _canonical_matrix

_TONE_PREFIX = "__vna_tone_"


@dataclass(frozen=True)
class _OperatingPoint:
    prepared: PreparedStationary
    state: SteadyStateResult
    tones: tuple[tuple[str, Any, Any], ...]

    @property
    def engine(self) -> EngineResult:
        return self.prepared.engine_result

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return _LazyDiagnostics({
            "solver": "stationary_resolvent",
            "residual": self.state.residual,
            "trace_error": self.state.trace_error,
            "positivity_error": DeferredValue(partial(getattr, self.state, "positivity_error")),
            "condition_number": partial(getattr, self.state, "condition_number"),
        })


@dataclass(frozen=True)
class PortTone:
    """A fixed coherent input field entering through one declared port.

    Attributes
    ----------
    port : str
        Resolved external network-port label.
    freq : scalar
        Carrier frequency in GHz.
    amplitude : scalar
        Complex incident field amplitude in ``1/sqrt(ns)``.
    """

    port: str
    freq: Any
    amplitude: Any
    _index: int
    _owner: "VNA"

    def vary(self, field: str, values: Any, *, name: str | None = None) -> Sweep:
        """Create a sweep axis for this tone's frequency or amplitude.

        Parameters
        ----------
        field : {"freq", "amplitude"}
            Tone attribute to vary. Frequency is in GHz and amplitude is in
            ``1/sqrt(ns)``.
        values : array_like
            Values in sweep order.
        name : str or None, optional
            Public result-axis name; defaults to ``"<port>.<field>"``.

        Returns
        -------
        Sweep
            Axis for :meth:`VNA.sweep`, :meth:`VNA.finite_power`, or
            :meth:`VNA.measure`.
        """
        if field not in {"freq", "amplitude"}:
            raise ValueError(f"A port tone can vary only 'freq' or 'amplitude', got {field!r}.")
        return _ToneAxis(
            values,
            name=f"{_TONE_PREFIX}{self._index}_{field}",
            owner=self._owner,
            public_name=name or f"{self.port}.{field}",
        )


class _ToneAxis(Sweep):
    """Sweep axis owned by one VNA tone; ``public_name`` labels the result axis."""

    def __init__(self, values: Any, *, name: str, owner: "VNA", public_name: str) -> None:
        super().__init__(values, name=name)
        self.owner = owner
        self.public_name = public_name


class VNA:
    """Small-signal scattering between selected instrument ports of one chip.

    ``VNA(chip)`` selects every exposed instrument port in the chip's ``PortNetwork``.
    Pass exposed port objects or their labels as ``ports`` to select a subset;
    a bare label string is rejected.

    S-parameters follow the engineering ``e^{+jωt}`` convention, where ``j = −i``.
    Probe and pump amplitudes, returned fields, correlations, and IQ statistics use this
    convention. Internal mode observables retain the physics convention.

    Parameters
    ----------
    chip : Chip
        Chip with a ``PortNetwork``.
    ports : sequence of port objects or str, optional
        Selected external ports; ``None`` selects all external ports.
    """

    def __init__(self, chip: Any, *, ports: Sequence[Any] | None = None) -> None:
        self.chip = chip
        if ports is None:
            if chip.port_network is None:
                raise ValueError("VNA requires a chip with a PortNetwork.")
            ports = chip.port_network.external_ports
        elif isinstance(ports, str) or not isinstance(ports, Sequence):
            raise TypeError("VNA ports must be a sequence of network ports or labels.")
        self.ports = tuple(_resolve_exposure(chip, plane) for plane in ports)
        if not self.ports:
            raise ValueError("VNA requires at least one port.")
        if len(set(self.ports)) != len(self.ports):
            raise ValueError(f"VNA ports must be unique, got {list(self.ports)}.")
        self._tones: list[PortTone] = []

    def pump(self, port: Any, *, freq: Any, amplitude: Any) -> PortTone:
        """Add a fixed coherent pump and return its sweep handle.

        Parameters
        ----------
        port : port object or str
            External port receiving the pump.
        freq : scalar
            Pump frequency in GHz.
        amplitude : scalar
            Complex incident field in ``1/sqrt(ns)``.

        Returns
        -------
        PortTone
            Handle for varying pump frequency or amplitude.
        """
        tone = PortTone(_resolve_exposure(self.chip, port), freq, amplitude, len(self._tones), self)
        self._tones.append(tone)
        return tone

    @staticmethod
    def zip(*variations: Sweep) -> ZippedSweep:
        """Pair sweep axes element by element instead of taking a product.

        Parameters
        ----------
        *variations : Sweep
            Axes with equal lengths.

        Returns
        -------
        ZippedSweep
            Composite sweep evaluated at corresponding points.
        """
        return Sweep.zip(*variations)

    def sweep(
        self,
        frequencies: Any,
        *variations: Sweep | ZippedSweep,
        options: dict | None = None,
        progress: bool = False,
    ) -> SParameterResult:
        """Sweep the complete selected-port small-signal matrix around fixed pumps.

        The result contains every ``S(output, input)`` between the selected ports.
        Its ``matrix`` has shape ``(*sweep_axes, n_ports, n_ports)`` and is
        indexed ``[..., output, input]``. Ordinary ``Sweep`` axes name paths in
        ``chip.parameters`` and rebind the chip at each point; they may be mixed
        with pump-tone axes. At each frequency, the passive-linear route
        uses one multi-right-hand-side mode-space solve. The stationary route solves
        one pumped operating point, then uses one shifted-Liouvillian factorization
        for every input port.

        Parameters
        ----------
        frequencies : scalar or array_like
            Probe frequencies in GHz.
        *variations : Sweep or ZippedSweep
            Chip and pump axes; Cartesian axes combine and zipped axes vary together.
        options : dict or None, optional
            Stationary solver options; ``None`` permits the passive-linear path.
        progress : bool, default=False
            Show a progress bar for stationary solves.

        Returns
        -------
        SParameterResult
            Matrix with shape ``(*shape, n_ports, n_ports)`` and indexing
            ``[..., output, input]``.
        """
        freq_values, freq_is_axis = _axis_values(frequencies)
        self._validate_variations(variations, reserved=("frequency",) if freq_is_axis else ())
        variation_shape, variation_points = _iter_axis_points(variations)
        shape = variation_shape
        if freq_is_axis:
            shape += (len(freq_values),)

        labels = self.ports
        xp = self.chip.backend.array_module
        axes = _public_axes(variations)
        if freq_is_axis:
            axes += (("frequency", freq_values),)

        def result(diagnostics: Any, matrix: Any, conjugate: Any) -> SParameterResult:
            block = (*shape, len(labels), len(labels))
            return SParameterResult(
                frequencies=freq_values if freq_is_axis else freq_values[0],
                ports=labels,
                axes=axes,
                shape=shape,
                diagnostics=tuple(diagnostics),
                matrix=xp.conj(xp.reshape(matrix, block)),
                conjugate_matrix=xp.conj(xp.reshape(conjugate, block)),
            )

        chip_points = [(self._tone_values(p), self._chip_at(p)) for _, p in variation_points]
        if not self._tones and options is None:
            linear = self._linear_sweep([chip for _, chip in chip_points], freq_values, labels)
            if linear is not None:
                matrix = xp.concatenate(linear[0])
                return result(linear[1], matrix, xp.zeros_like(matrix))

        points = [(tones, chip, frequency) for tones, chip in chip_points for frequency in freq_values]
        iterator: Any = points
        if progress:
            from tqdm import tqdm

            iterator = tqdm(points, desc="VNA")

        blocks: list[tuple[Any, Any]] = []
        diagnostics: list[Mapping[str, Any]] = []
        for tones, chip, frequency in iterator:
            operating = _operating_point(
                chip, tones, frequency, labels, options
            )
            blocks.append(_small_signal_matrix(operating, labels, frequency))
            diagnostics.append(operating.diagnostics)

        return result(diagnostics, *(xp.stack(part) for part in zip(*blocks, strict=True)))

    def finite_power(
        self,
        frequencies: Any,
        amplitudes: Any,
        *variations: Sweep | ZippedSweep,
        input: Any | None = None,
        options: dict | None = None,
        progress: bool = False,
    ) -> MeanFieldResponseResult:
        """Solve the stationary mean output fields for a finite coherent probe.

        ``frequencies`` are in GHz. ``amplitudes`` are incident field amplitudes
        ``beta`` in ``1/sqrt(ns)``; complex values encode phase. The probe enters
        ``input`` beside any fixed pumps. When one port is selected, ``input``
        defaults to that port; otherwise it is required. A fixed pump on the probe
        input is rejected.

        At each grid point, the method solves the stationary Liouvillian in the probe
        frame. Chip and pump sweep axes precede ``"amplitude"`` and ``"frequency"``.
        Every selected port must resolve at the probe frequency. The returned
        ``MeanFieldResponseResult`` reports ``<b_out>`` using the same reference-plane
        and hidden-channel bookkeeping as ``sweep()``, and stores the incident
        ``beta`` broadcast over the grid.

        This method always uses the stationary Liouvillian route. It does not use the
        passive-linear mode-space shortcut or follow sweep-rate hysteresis and
        metastable branches.

        Parameters
        ----------
        frequencies, amplitudes : scalar or array_like
            Probe frequencies in GHz and incident fields in ``1/sqrt(ns)``.
        *variations : Sweep or ZippedSweep
            Chip and pump axes.
        input : port object or str, optional
            Probe input; required when multiple ports are selected.
        options : dict or None, optional
            Stationary solver options.
        progress : bool, default=False
            Show a progress bar.

        Returns
        -------
        MeanFieldResponseResult
            Complex means with shape ``(*shape, n_ports)`` and incident fields
            with shape ``shape``.
        """
        if input is None:
            if len(self.ports) != 1:
                raise ValueError("finite_power() requires input= when more than one port is selected.")
            input_label = self.ports[0]
        else:
            input_label = _resolve_exposure(self.chip, input)
            if input_label not in self.ports:
                raise ValueError(f"finite_power() input {input_label!r} is not a selected port {list(self.ports)}.")
        if any(tone.port == input_label for tone in self._tones):
            raise ValueError(
                f"Plane {input_label!r} already carries a fixed pump. The finite_power() probe must be "
                "the only tone entering the selected input so that <b_out>/beta is unambiguous."
            )
        from quchip.analysis.measurement import _acquire

        return _acquire(self, frequencies, amplitudes, variations, labels=self.ports,
                        input_label=input_label, options=options, progress=progress)

    def _validate_variations(
        self, variations: tuple[Sweep | ZippedSweep, ...], *, reserved: Sequence[str] = ()
    ) -> None:
        names = list(reserved)
        chip_paths = set(self.chip.parameters)
        for variation in variations:
            members = variation.sweeps if isinstance(variation, ZippedSweep) else (variation,)
            for member in members:
                if isinstance(member, _ToneAxis):
                    if member.owner is not self:
                        raise ValueError(
                            "VNA tone variations must come from tone.vary(...) on a tone created by this VNA."
                        )
                elif member.name not in chip_paths:
                    raise KeyError(
                        f"Unknown Chip parameter path {member.name!r} in VNA sweep. "
                        f"Available paths: {sorted(chip_paths)}"
                    )
                if isinstance(variation, ZippedSweep):
                    names.append(_public_name(member))
        names.extend(name for name, _ in _public_axes(variations))
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"VNA axis names must be unique, got duplicates {duplicates}.")

    def measure(
        self, frequencies: Any, amplitudes: Any, *variations: Sweep | ZippedSweep,
        input: Any = None, outputs: Sequence[Any] | None = None, noise_frequencies: Any = None,
        options: dict | None = None, progress: bool = False,
    ) -> VNAMeasurement:
        """Capture stationary means and joint physical noise before choosing a receiver.

        Frequencies are probe GHz, amplitudes are in 1/sqrt(ns). Optional
        noise_frequencies are an increasing symmetric grid of offsets in GHz,
        including zero. The default spans ±0.1 GHz with logarithmic spacing
        down to 1 Hz. Choose a grid covering the model's noise features.
        Returned data supports receiver integration and Gaussian sampling
        without another solve; it is not a quantum-trajectory distribution.
        Internal Fock-mode amplitudes and occupations are captured as well;
        query them with mode_amplitude(device) and photon_number(device).

        Parameters
        ----------
        frequencies, amplitudes : scalar or array_like
            Probe frequencies in GHz and incident fields in ``1/sqrt(ns)``.
        *variations : Sweep or ZippedSweep
            Chip and pump axes.
        input : port object or str, optional
            Probe input; required when multiple ports are selected.
        outputs : sequence of port objects or str, optional
            Captured outputs; ``None`` selects all selected ports.
        noise_frequencies : array_like or None, optional
            Symmetric increasing offset grid in GHz, including zero.
        options : dict or None, optional
            Stationary solver options.
        progress : bool, default=False
            Show a progress bar.

        Returns
        -------
        VNAMeasurement
            Means and physical noise spectra reusable by receiver processing.
        """
        from quchip.analysis.measurement import measure
        return measure(self, frequencies, amplitudes, variations, input=input, outputs=outputs,
                       noise_frequencies=noise_frequencies, options=options, progress=progress)

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

        Parameters
        ----------
        output : port object or str
            Output plane.
        frequencies : scalar or array_like
            Offset frequencies in GHz.
        options : dict or None, optional
            Stationary solver options.
        """
        output_port = _resolve_exposure(self.chip, output)
        frequency_values, _ = _axis_values(frequencies)
        operating = self._stationary_output(output_port, options)
        engine, state, tones = operating.engine, operating.state, operating.tones
        operators = port_operators(engine, self.chip.backend)
        incoming = _stationary_output_backgrounds(engine, tones, self.chip.backend)
        xp = self.chip.backend.array_module
        rho = xp.asarray(self.chip.backend.to_array(state.state), dtype=complex)
        channel = next(item for item in engine.slh.external_channels if item.key == output_port)
        carrier = _output_carrier(engine, output_port, tones)
        all_fields = xp.stack([_output_field_matrix(operators[item.key], incoming.get(item.key, 0.0), xp)
                               for item in engine.slh.channels])
        index = next(i for i, item in enumerate(engine.slh.channels) if item.key == output_port)
        boundary_field = xp.einsum("i,ijk->jk", output_mixing(engine.slh, carrier, xp)[index], all_fields)
        field = cw_transfer(channel.reference.outbound, carrier, xp) * boundary_field
        field_rho = field @ rho
        mean = xp.trace(field_rho)
        intensity = xp.real(xp.einsum("ij,ij->", field.conj(), field_rho))
        coherent_flux = xp.abs(mean) ** 2
        from quchip.analysis.measurement import capture_noise
        components, _ = capture_noise(operating, self.chip.backend, (output_port,), carrier,
                                      xp.asarray(frequency_values))

        from quchip.analysis.field_statistics import normal_spectrum

        def source_spectrum(pair: tuple[Any, Any]) -> Any:
            white, excess = pair
            return normal_spectrum(white + excess, xp)

        signal = source_spectrum(components["device.correlations"])
        added_noise = sum((source_spectrum(pair) for name, pair in components.items()
                           if name != "device.correlations"), xp.zeros_like(frequency_values))
        return OutputSpectrumResult(
            port=output_port,
            frequencies=frequency_values,
            total_fluctuation_spectrum=xp.asarray(signal + added_noise),
            signal_fluctuation_spectrum=xp.asarray(signal),
            added_noise_spectrum=added_noise,
            signal_photon_flux=intensity,
            signal_coherent_flux=coherent_flux,
            signal_incoherent_flux=intensity - coherent_flux,
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
        """Return normalized first-order output coherence ``g1``.

        Parameters
        ----------
        output : port object or str
            Delayed output plane.
        delays : scalar or array_like
            Non-negative delays in ns.
        input : port object or str, optional
            Cross-correlation input; defaults to ``output``.
        options : dict or None, optional
            Stationary solver options.

        Returns
        -------
        OutputCorrelationResult
            Normalized and raw correlations on the delay grid.
        """
        return self._correlation(output, delays, input=input, order=1, options=options)

    def g2(
        self,
        output: Any,
        delays: Any,
        *,
        input: Any | None = None,
        options: dict | None = None,
    ) -> OutputCorrelationResult:
        """Return normalized second-order output intensity correlation ``g2``.

        Parameters
        ----------
        output : port object or str
            Delayed output plane.
        delays : scalar or array_like
            Non-negative delays in ns.
        input : port object or str, optional
            Cross-correlation input; defaults to ``output``.
        options : dict or None, optional
            Stationary solver options.

        Returns
        -------
        OutputCorrelationResult
            Normalized and raw second-order correlations on the delay grid.
        """
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
        operating = self._stationary_output(output_port, options)
        engine, state, tones = operating.engine, operating.state, operating.tones
        backend = self.chip.backend
        operators = port_operators(engine, backend)
        incoming = _stationary_output_backgrounds(engine, tones, backend)
        xp = backend.array_module
        rho = xp.asarray(backend.to_array(state.state), dtype=complex)
        runs = {item.key: item.reference.outbound for item in engine.slh.external_channels}
        if engine.slh.output_network is not None or has_amplifier(runs[output_port]) or has_amplifier(runs[input_port]):
            raise NotImplementedError(
                "Normalized g1 and g2 through an amplifier require a detection bandwidth for its "
                "broadband added noise. Request the correlation at a reference plane before the amplifier."
            )

        thermal = any(channel.input_occupation is not None or any(
            (isinstance(element, ReferenceLoss) and element.occupation is not None)
            or (isinstance(element, ReferenceFilter) and element.loss_occupation is not None)
            for element in (*channel.reference.inbound, *channel.reference.outbound))
            for channel in engine.slh.channels)
        if thermal:
            raise NotImplementedError("Normalized g1 and g2 with thermal network fields require a detection "
                                      "bandwidth. Use measure().statistics() for integrated second moments.")

        def plane_field(key: str) -> Any:
            factor = cw_transfer(runs[key], _output_carrier(engine, key, tones), xp)
            return factor * _output_field_matrix(operators[key], incoming.get(key, 0.0), xp)

        output_field = plane_field(output_port)
        input_field = plane_field(input_port)
        output_field_dag = xp.conj(xp.swapaxes(output_field, -1, -2))
        input_field_dag = xp.conj(xp.swapaxes(input_field, -1, -2))
        output_number = output_field_dag @ output_field
        input_number = input_field_dag @ input_field
        output_intensity = xp.real(xp.einsum("ij,ji->", output_number, rho))
        input_intensity = xp.real(xp.einsum("ij,ji->", input_number, rho))
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
            operators[input_port],
            tag=f"g{order}-initial:{input_port}",
        )
        observable = _canonical_matrix(
            output_field_dag if order == 1 else output_number,
            operators[output_port],
            tag=f"g{order}-observable:{output_port}",
        )
        raw = backend.stationary_propagate(
            engine,
            initial,
            ((output_port, observable),),
            delay_values,
            prepared=operating.prepared,
        )[output_port]
        denominator = (
            xp.sqrt(input_intensity * output_intensity)
            if order == 1
            else input_intensity * output_intensity
        )
        same_port = input_port == output_port
        raw = xp.conj(raw)
        return OutputCorrelationResult(
            order=order,
            input_port=input_port,
            output_port=output_port,
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
        output_label: str,
        options: dict | None,
    ) -> _OperatingPoint:
        """Solve stationary output statistics in the applicable reference frames.

        Pump tones define the stationary frames when present; otherwise the requested
        output plane's own reference frequency does.
        """
        tones = self._tone_values({})
        frames = tuple((label, frequency) for label, frequency, _ in tones) or (
            (output_label, _exposure_reference_frequency(self.chip, output_label)),
        )
        engine = resolve_stationary_engine(self.chip, frames)
        driven_engine = add_port_inputs(engine, self.chip.backend, tones)
        return _solve_engine(self.chip, driven_engine, tones, options)


    def _linear_sweep(
        self, chips: list[Any], frequencies: Any, labels: tuple[str, ...]
    ) -> tuple[list[Any], list[Mapping[str, Any]]] | None:
        """Solve every chip point in mode space, or return ``None`` if any point is ineligible."""
        matrices: list[Any] = []
        diagnostics: list[Mapping[str, Any]] = []
        for chip in chips:
            problem = try_build_linear_response_problem(chip, frequencies, plane_labels=labels)
            if problem is None:
                return None
            solved = chip.backend.linear_response(problem)
            indices = list(problem.plane_indices)
            transfer = (
                problem.outbound_transfer[:, indices][:, :, None]
                * problem.inbound_transfer[:, indices][:, None, :]
            )
            matrices.append(chip.backend.array_module.asarray(transfer * solved.responses))
            diagnostics.extend(
                _LazyDiagnostics({
                    "solver": "linear_response",
                    "mode_count": len(problem.mode_labels),
                    "residual": solved.residuals[index],
                    "condition_number": lambda index=index, condition=solved._condition_numbers: condition()[index],
                })
                for index in range(len(frequencies))
            )
        return matrices, diagnostics

    def _chip_at(self, params: dict[str, Any]) -> Any:
        """Return the chip after rebinding the ordinary sweep values in ``params``."""
        bindings = {name: value for name, value in params.items() if name not in self._tone_keys()}
        return self.chip.with_params(bindings) if bindings else self.chip

    def _tone_keys(self) -> set[str]:
        return {
            f"{_TONE_PREFIX}{tone._index}_{field}"
            for tone in self._tones
            for field in ("freq", "amplitude")
        }

    def _tone_values(self, params: dict[str, Any]) -> tuple[tuple[str, Any, Any], ...]:
        tones: list[tuple[str, Any, Any]] = []
        for tone in self._tones:
            freq = params.get(f"{_TONE_PREFIX}{tone._index}_freq", tone.freq)
            amplitude = params.get(f"{_TONE_PREFIX}{tone._index}_amplitude", tone.amplitude)
            tones.append((tone.port, freq, self.chip.backend.array_module.conj(amplitude)))
        return tuple(tones)


def _axis_values(values: Any) -> tuple[Any, bool]:
    if np.ndim(values) == 0:
        return (values,), False
    array = values if hasattr(values, "shape") else np.asarray(values)
    if len(array) == 0:
        raise ValueError("VNA frequency sweep cannot be empty.")
    return array, True


def _resolve_exposure(chip: Any, value: Any) -> str:
    """Find an external network port by object or label."""
    label = resolve_label(value)
    network = chip.port_network
    if network is None:
        raise ValueError("VNA requires a chip with a PortNetwork.")
    available = [exposure.label for exposure in network.external_ports]
    if label not in available:
        raise ValueError(f"Unknown VNA port {label!r}. Available network ports: {available}.")
    return label


def _exposure_reference_frequency(chip: Any, label: str) -> Any:
    """Return the exposure carrier in the chip's natural rotating frame."""
    resolved = chip.resolve(frame="rotating")
    for channel in resolved.slh.external_channels:
        if channel.key == label:
            frequency = channel.collapse.frame_frequency
            if frequency is None and resolved.slh.output_network is not None:
                carriers = [item.collapse.frame_frequency for item in resolved.slh.channels
                            if item.collapse.frame_frequency is not None]
                if carriers and all(same_frequency(carriers[0], other) for other in carriers[1:]):
                    return carriers[0]
            return 0.0 if frequency is None else frequency
    available = [channel.key for channel in resolved.slh.external_channels]
    raise ValueError(f"Unknown VNA port {label!r}. Available network ports: {available}.")


def _public_name(sweep: Sweep) -> str:
    return sweep.public_name if isinstance(sweep, _ToneAxis) else sweep.name


def _public_axes(variations: tuple[Sweep | ZippedSweep, ...]) -> tuple[tuple[str, Any], ...]:
    public = {
        sweep.name: sweep.public_name
        for axis in variations
        for sweep in (axis.sweeps if isinstance(axis, ZippedSweep) else (axis,))
        if isinstance(sweep, _ToneAxis)
    }
    return _axis_metadata(variations, rename=lambda name: public.get(name, name))


def _operating_point(
    chip: Any,
    tones: tuple[tuple[str, Any, Any], ...],
    frequency: Any,
    labels: tuple[str, ...],
    options: dict | None,
) -> _OperatingPoint:
    """Solve the pumped stationary state with every selected plane framed at ``frequency``."""
    engine = resolve_stationary_engine(
        chip,
        tuple((label, tone_frequency) for label, tone_frequency, _ in tones)
        + tuple((label, frequency) for label in labels),
    )
    operating_engine = add_port_inputs(engine, chip.backend, tones)
    return _solve_engine(chip, operating_engine, tones, options)


def _solve_engine(
    chip: Any, engine: EngineResult, tones: tuple[tuple[str, Any, Any], ...], options: dict | None,
) -> _OperatingPoint:
    problem = SteadyStateProblem(
        chip=chip,
        engine_result=engine,
        e_ops=None,
        e_ops_meta=None,
        resolved_frame=engine.resolved_frame,
        options={} if options is None else options,
    )
    prepared = problem.backend.prepare_stationary(engine)
    return _OperatingPoint(prepared, solve_steadystate_problem(problem, prepared=prepared), tones)


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
        incident[index] = incident[index] + xp.asarray(amplitude) * cw_transfer(
            external[index].reference.inbound, frequency, xp
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


def _small_signal_matrix(
    operating: _OperatingPoint,
    labels: tuple[str, ...],
    frequency: Any,
) -> tuple[Any, Any]:
    """Return the selected-port ``S_ji(f)`` and phase-conjugating ``T_ji(f)`` matrices.

    Rows are output ports and columns are input ports. Around a phase-sensitive
    operating point the linear response is ``delta <b_out> = S delta beta + T
    conj(delta beta)``; one shifted-Liouvillian factorization serves both source
    sets for all input columns.
    """
    engine = operating.engine
    backend = operating.prepared.backend
    channels = port_operators(engine, backend)
    xp = backend.array_module
    rho = xp.asarray(backend.to_array(operating.state.state), dtype=complex)
    external = engine.slh.external_channels
    exposure_index = {channel.key: index for index, channel in enumerate(external)}
    indices = [exposure_index[label] for label in labels]
    template = channels[external[0].key]
    operators = xp.stack(
        [xp.asarray(channels[channel.key].to_dense(), dtype=complex) for channel in engine.slh.channels]
    )
    scattering = xp.asarray(engine.slh.S)
    sources = []
    for column, index in enumerate(indices):
        input_operator = xp.tensordot(xp.conj(scattering[:, index]), operators, axes=1)
        input_dag = xp.conj(xp.swapaxes(input_operator, -1, -2))
        normal_source = _canonical_matrix(input_dag @ rho - rho @ input_dag, template, tag=f"normal-source:{column}")
        conjugate_source = _canonical_matrix(
            rho @ input_operator - input_operator @ rho, template, tag=f"conjugate-source:{column}"
        )
        sources.append((f"normal:{column}", normal_source))
        sources.append((f"conjugate:{column}", conjugate_source))
    response = backend.stationary_resolvent(
        engine,
        tuple(sources),
        tuple((channel.key, channels[channel.key]) for channel in engine.slh.channels),
        (0.0,),
        prepared=operating.prepared,
    )
    inbound = xp.stack([cw_transfer(external[i].reference.inbound, frequency, xp) for i in indices])
    outbound = xp.stack([cw_transfer(external[i].reference.outbound, frequency, xp) for i in indices])

    def gather(kind: str) -> Any:
        columns = range(len(labels))
        return xp.stack([xp.stack([response[(f"{kind}:{c}", channel.key)][0] for c in columns])
                         for channel in engine.slh.channels])

    direct = scattering[:, xp.asarray(indices)]
    mixing = output_mixing(engine.slh, frequency, xp)[xp.asarray(indices)]
    gain = outbound[:, None]
    return (
        gain * inbound[None, :] * (mixing @ (direct + gather("normal"))),
        gain * xp.conj(inbound)[None, :] * (mixing @ gather("conjugate")),
    )


def _output_carrier(engine: EngineResult, key: str, tones: Any) -> Any:
    """Return the carrier a plane's outbound run is evaluated at.

    A coupling-free composed output inherits the unique tone that structurally
    feeds it; without one, an outbound reference run has no defined carrier.
    """
    external = {item.key: index for index, item in enumerate(engine.slh.external_channels)}
    row = external[key]
    channel = engine.slh.external_channels[row]
    if channel.collapse.frame_frequency is not None or (not channel.reference.outbound
                                                       and engine.slh.output_network is None):
        return channel.carrier
    feeding = [frequency for label, frequency, _ in tones if engine.slh.feeds(row, external[label])]
    if not feeding or any(not same_frequency(feeding[0], other) for other in feeding[1:]):
        raise ValueError(
            f"Output plane {key!r} carries no coupling and is fed by {len(feeding)} tones; "
            "its outbound reference sections need exactly one carrier."
        )
    return feeding[0]


def _plane_means(
    engine: EngineResult,
    state: Any,
    backend: Any,
    labels: tuple[str, ...],
    tones: tuple[tuple[str, Any, Any], ...],
    frequency: Any,
) -> Any:
    """Return stationary output means at the selected ports.

    For each plane, apply its outbound continuous-wave transfer to
    ``sum_i S_ji beta_boundary,i + <L_j>`` at ``frequency``. Every selected
    plane must resolve at that frequency.
    """
    xp = backend.array_module
    rho = xp.asarray(backend.to_array(state), dtype=complex)
    operators = port_operators(engine, backend)
    backgrounds = _stationary_output_backgrounds(engine, tones, backend)
    channels = {channel.key: channel for channel in engine.slh.external_channels}
    boundaries = xp.stack([backgrounds.get(channel.key, 0.0)
                           + xp.einsum("ij,ji->", xp.asarray(operators[channel.key].to_dense()), rho)
                           for channel in engine.slh.channels])
    boundaries = output_mixing(engine.slh, frequency, xp) @ boundaries
    means = []
    indices = {channel.key: i for i, channel in enumerate(engine.slh.channels)}
    for label in labels:
        channel = channels[label]
        carrier = _output_carrier(engine, label, tones)
        # A coupling-free, unilluminated output is zero at any requested carrier.
        is_zero = not any(engine.slh.feeds(
            indices[label], indices[tone[0]],
        ) for tone in tones) and channel.collapse.frame_frequency is None
        if not is_zero and not same_frequency(carrier, frequency):
            raise ValueError(
                f"Plane {label!r} resolves at {carrier!r} GHz, not at the probe frequency "
                f"{frequency!r} GHz. Narrow VNA(..., ports=...) to ports at the probe carrier."
            )
        boundary = boundaries[indices[label]]
        means.append(cw_transfer(channel.reference.outbound, carrier, xp) * boundary)
    return xp.stack(means)


def _output_field_matrix(operator: CanonicalOperator, incoming: Any, xp: Any) -> Any:
    coupling = xp.asarray(operator.to_dense(), dtype=complex)
    return xp.asarray(incoming) * xp.eye(coupling.shape[0], dtype=complex) + coupling
