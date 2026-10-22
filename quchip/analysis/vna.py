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
    reject_retained_mode_responses,
    resolve_stationary_engine,
    same_frequency,
    stationary_frame_plan,
)
from quchip.engine.linear_response import response_poles, try_build_weak_probe_problem
from quchip.engine.conditional_response import build_conditional_response
from quchip.results.measurement import VNAMeasurement
from quchip.engine.output_network import output_mixing
from quchip.engine.reference import cw_transfer, has_amplifier, ReferenceFilter, ReferenceLoss
from quchip.engine.ir import CanonicalOperator, EngineResult, LinearResponseProblem, SteadyStateProblem
from quchip.engine.routes import (
    WEAK_PROBE, SolveRoute, array_name, automatic_cutoff, largest_change, plan_sectors, require_workflow,
    sector_diagnostics, warn_cutoff, weak_probe_reason,
)
from quchip.engine.steady_state import solve_steadystate_problem
from quchip.results.input_output import (
    MeanFieldResponseResult,
    OutputCorrelationResult,
    OutputSpectrumResult,
    ResonanceResult,
    SParameterResult,
    _LazyDiagnostics,
)
from quchip.results.steady_state import SteadyStateResult
from quchip.sweep import Sweep, ZippedSweep, _axis_metadata, _iter_axis_points
from quchip.utils.jax_utils import concrete_array_module, contains_tracer, maybe_concrete_scalar
from quchip.utils.labeling import bare_label_from_mapping, resolve_label
from quchip.utils.memory import fits_memory
from quchip.engine.field_statistics import canonical as _canonical_matrix, normal_spectrum, propagate_noise

_TONE_PREFIX = "__vna_tone_"
_VACUUM_REFERENCE = ("The when= reference is the vacuum, which is stationary, so a weak probe keeps the vacuum and "
                     "the one-excitation states.")
_FROZEN_REFERENCE = ("The when= reference freezes one dressed state, so the response solves the coherences between "
                     "its sector and the adjacent sectors.")


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
        values = {
            "solver": "stationary_resolvent",
            "route": self.state.diagnostics["route"],
            "residual": self.state.residual,
            "trace_error": self.state.trace_error,
            "positivity_error": DeferredValue(partial(getattr, self.state, "positivity_error")),
            "condition_number": partial(getattr, self.state, "condition_number"),
        }
        space = self.engine.excitation_space
        if space is not None:
            values.update(excitation_cutoff=max(space.totals), retained_dimension=space.size)
        return _LazyDiagnostics(values)


@dataclass(frozen=True)
class PortTone:
    """A fixed coherent input field that enters through one declared port.

    Attributes
    ----------
    port : str
        Resolved label of the external network port.
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
            Public name of the result axis. The default is ``"<port>.<field>"``.

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


@dataclass(frozen=True)
class _PortTransfers:
    """Reference-plane transfers of the selected ports at given probe frequencies."""

    chip: Any
    labels: tuple[str, ...]

    def __call__(self, frequencies: Any) -> tuple[Any, Any]:
        # The route does not depend on the probe frequencies, so the rebuilt problem exists.
        rebuilt_problem = try_build_weak_probe_problem(self.chip, frequencies, plane_labels=self.labels)
        assert rebuilt_problem is not None
        rebuilt = rebuilt_problem[0]
        indices = list(rebuilt.plane_indices)
        return rebuilt.outbound_transfer[:, indices], rebuilt.inbound_transfer[:, indices]


class VNA:
    """Small-signal scattering between selected instrument ports of one chip.

    ``VNA(chip)`` selects every exposed instrument port in the chip's ``PortNetwork``.
    To select a subset, pass exposed port objects or their labels as ``ports``. A bare
    label string is rejected.

    S-parameters follow the engineering ``e^{+jωt}`` convention, where ``j = i``. Probe
    and pump amplitudes, returned fields, correlations, and IQ statistics use this
    convention, while internal mode observables keep the physics convention.

    Parameters
    ----------
    chip : Chip
        Chip with a ``PortNetwork``.
    ports : sequence of port objects or str, optional
        Selected external ports. ``None`` selects all external ports.
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
            Handle to vary the pump frequency or amplitude.
        """
        tone = PortTone(_resolve_exposure(self.chip, port), freq, amplitude, len(self._tones), self)
        self._tones.append(tone)
        return tone

    @staticmethod
    def zip(*variations: Sweep) -> ZippedSweep:
        """Pair sweep axes element by element instead of a product.

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
        when: Mapping[Any, int] | None = None,
        options: dict | None = None,
        progress: bool = False,
    ) -> SParameterResult:
        """Sweep the selected ports' complete small-signal matrix around fixed pumps.

        The result contains every ``S(output, input)`` between the selected ports.
        Its ``matrix`` has shape ``(*sweep_axes, n_ports, n_ports)`` and is indexed
        ``[..., output, input]``. Ordinary ``Sweep`` axes name paths in
        ``chip.parameters``, rebind the chip at each point, and can mix with
        pump-tone axes. At each frequency, the passive-linear route solves one
        multi-right-hand-side mode-space system. The stationary route solves one
        pumped operating point, then uses one shifted-Liouvillian factorization for
        every input port. A pump-free model that is static in the lab frame uses
        one stationary state per chip point and a resolvent at each probe frequency.

        Parameters
        ----------
        frequencies : scalar or array_like
            Probe frequencies in GHz.
        *variations : Sweep or ZippedSweep
            Chip and pump axes. Cartesian axes combine and zipped axes vary together.
        when : mapping of device objects or labels to int, optional
            Frozen dressed-reference label. Unspecified devices have level zero.
            Supports zero or one total excitation, static conserving models and
            vacuum inputs. ``None`` uses the ordinary stationary response.
            An excited reference is fixed in the probe source, while its response
            includes all loss channels. It does not model an evolving preparation.
        options : dict or None, optional
            Stationary solver options. ``None`` permits the passive-linear path.
            Explicit ``when`` requires ``None``.
        progress : bool, default=False
            Show a progress bar for stationary solves.

        Returns
        -------
        SParameterResult
            Matrix with shape ``(*shape, n_ports, n_ports)`` and indexing
            ``[..., output, input]``.
        """
        require_workflow(self.chip, "VNA.sweep()")
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

        reference = None
        if when is not None:
            if not isinstance(when, Mapping):
                raise TypeError("VNA when must be a mapping of device objects or labels to integer levels.")
            reference = bare_label_from_mapping(tuple(d.label for d in self.chip.devices), when, {})
            if any(isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, np.integer)) or level < 0
                   for level in reference) or sum(reference) > 1:
                raise ValueError("Conditional VNA requires integer levels with zero or one total excitation.")
            if self._tones:
                raise ValueError("Conditional VNA does not support fixed pumps.")
            if options is not None:
                raise ValueError("Conditional VNA does not accept stationary solver options.")
        chip_points = [(self._tone_values(p), self._chip_at(p)) for _, p in variation_points]
        if reference is not None:
            conditional = self._conditional_sweep([chip for _, chip in chip_points], freq_values, labels, reference)
            matrix = xp.concatenate(conditional[0])
            return result(conditional[1], matrix, xp.zeros_like(matrix))
        space = "The backend's stationary Liouvillian solves the product space."
        if self._tones:
            reason = f"Pump tones need the stationary state of the pumped chip. {space}"
        elif options is not None:
            reason = "Solver options select the backend's stationary Liouvillian on the product space."
        else:
            linear = self._linear_sweep([chip for _, chip in chip_points], freq_values, labels)
            if not isinstance(linear, str):
                matrix = xp.concatenate(linear[0])
                return result(linear[1], matrix, xp.zeros_like(matrix))
            reason = f"{linear} {space}"

        points = [(tones, chip, frequency) for tones, chip in chip_points for frequency in freq_values]
        iterator: Any = points
        if progress:
            from tqdm import tqdm

            iterator = tqdm(points, desc="VNA")

        blocks: list[tuple[Any, Any]] = []
        diagnostics: list[Mapping[str, Any]] = []
        previous: tuple[Any, Any, _OperatingPoint] | None = None
        lab_operating: _OperatingPoint | None = None
        for tones, chip, frequency in iterator:
            same_point = previous is not None and previous[0] is tones and previous[1] is chip
            if not same_point and not tones:
                reject_retained_mode_responses(chip)
                engine = chip._resolve(frame="lab")
                lab_operating = None if engine.dynamic_terms else _solve_engine(chip, engine, tones, options,
                                                                                reason=reason)
            if lab_operating is not None:
                operating = lab_operating
            else:
                guess = previous[2].state.state if previous is not None and same_point else None
                operating = _operating_point(chip, tones, frequency, labels, options, guess=guess, reason=reason)
            previous = (tones, chip, operating)
            blocks.append(_small_signal_matrix(
                operating, labels, frequency, offset=frequency if lab_operating is not None else 0.0,
            ))
            diagnostics.append(operating.diagnostics)

        return result(diagnostics, *(xp.stack(part) for part in zip(*blocks, strict=True)))

    def resonances(self) -> ResonanceResult:
        """Return the poles of the weak-probe response between the selected ports.

        The poles come from the model that :meth:`sweep` solves on its
        ``"linear_response"`` and ``"vacuum_response"`` routes, without a fit.
        Each eigenvalue ``λ_p`` of the drift ``A = -iH - C†C/2`` gives a pole
        frequency ``f_p = -Im λ_p / 2π`` in GHz and a response linewidth
        ``κ_p = -2 Re λ_p`` in 1/ns, which includes port, internal, Purcell and
        dephasing broadening. The frequencies and linewidths stay
        differentiable under JAX on a fixed structure away from degenerate
        poles. Compositions and residues are computed when they are read.

        Returns
        -------
        ResonanceResult
            Poles in increasing frequency, with their composition, dominant
            mode and residues.

        Raises
        ------
        ValueError
            If the VNA has pump tones, if no weak-probe route applies, with the
            reason that :meth:`sweep` gives, or if two ill-conditioned poles
            are too close to resolve, as near an exceptional point.
        """
        require_workflow(self.chip, "VNA.resonances()")
        if self._tones:
            raise ValueError("resonances() describes the weak-probe response without pumps. Remove the pump tones.")
        labels = self.ports
        built = try_build_weak_probe_problem(self.chip, np.zeros(1), plane_labels=labels)
        if built is None:
            raise ValueError(f"resonances() needs a weak-probe route of VNA.sweep(). "
                             f"{weak_probe_reason(self.chip, labels)}")
        problem, name = built
        route = SolveRoute(name, WEAK_PROBE[name], len(problem.mode_labels), array_name(self.chip.backend))
        system = response_poles(problem)
        xp = concrete_array_module(system.values)
        return ResonanceResult(
            frequencies=-xp.imag(system.values) / (2.0 * np.pi),
            linewidths=-2.0 * xp.real(system.values),
            # The public convention conjugates the pole i λ / 2π.
            poles=xp.conj(1j * system.values / (2.0 * np.pi)),
            mode_labels=problem.mode_labels,
            ports=labels,
            route=route,
            _eigensystem=system,
            _problem=problem,
            _transfers=_PortTransfers(self.chip, labels),
        )

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
        ``beta`` in ``1/sqrt(ns)``. Complex values encode phase. The probe enters
        ``input`` beside any fixed pumps. ``input`` defaults to the selected port
        when only one is selected, and is required otherwise. A fixed pump on the
        probe input is rejected.

        At each grid point, the method solves the stationary Liouvillian in the probe
        frame. Chip and pump sweep axes precede ``"amplitude"`` and ``"frequency"``.
        Every selected port must resolve at the probe frequency. The returned
        ``MeanFieldResponseResult`` reports ``<b_out>`` with the same reference-plane
        and hidden-channel bookkeeping as ``sweep()``. It also stores the incident
        ``beta``, broadcast over the grid.

        This method does not use the passive-linear mode-space shortcut or follow
        sweep-rate hysteresis and metastable branches. A chip whose static model
        conserves the total excitation ``N``, with ports that lower ``N``, other
        channels that lower or conserve it, vacuum inputs, no thermal occupation,
        pumps only at the probe frequency and no output network, solves on
        excitation sectors. A finite probe couples every ``N`` to ``N ± 1``, so
        this route keeps the product states with ``N <= K``. With
        ``chip.with_space(ExcitationCutoff(max_total=K))``, ``K`` is the chip's.
        Without a cutoff, a chip with more than 64 product states starts at
        ``K = 2``, compares the result with ``K + 1`` at every grid point, raises
        ``K`` until the change is below ``1e-6`` relative, and warns which ``K``
        it kept. This is a convergence check, not an error bound. ``options``
        other than ``None`` solve the backend's stationary Liouvillian. Each
        entry of ``diagnostics`` records the route and its reason under
        ``"route"``.

        Parameters
        ----------
        frequencies, amplitudes : scalar or array_like
            Probe frequencies in GHz and incident fields in ``1/sqrt(ns)``.
        *variations : Sweep or ZippedSweep
            Chip and pump axes.
        input : port object or str, optional
            Probe input, required when multiple ports are selected.
        options : dict or None, optional
            Stationary solver options. Without a cutoff, a dictionary selects the
            full product space.
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
        """Capture stationary means and joint physical noise before you choose a receiver.

        Frequencies are probe frequencies in GHz and amplitudes are in
        1/sqrt(ns). Optional noise_frequencies are an increasing symmetric grid
        of offsets in GHz, including zero. The default spans ±0.1 GHz with
        logarithmic spacing down to 1 Hz. Choose a grid that covers the model's
        noise features. The returned data supports receiver integration and
        Gaussian sampling without another solve, but it is not a
        quantum-trajectory distribution. The method also captures internal
        Fock-mode amplitudes and occupations, which you query with
        mode_amplitude(device) and photon_number(device).

        An excitation-conserving chip solves on the product states with
        ``N <= K``, as :meth:`finite_power` describes. Without a cutoff, the
        automatic check also compares the noise spectra on the requested offset
        grid when the ``K + 1`` space has at most 1024 states.

        Parameters
        ----------
        frequencies, amplitudes : scalar or array_like
            Probe frequencies in GHz and incident fields in ``1/sqrt(ns)``.
        *variations : Sweep or ZippedSweep
            Chip and pump axes.
        input : port object or str, optional
            Probe input, required when multiple ports are selected.
        outputs : sequence of port objects or str, optional
            Captured outputs. ``None`` selects all selected ports.
        noise_frequencies : array_like or None, optional
            Symmetric increasing offset grid in GHz, including zero.
        options : dict or None, optional
            Stationary solver options. ``None`` permits the passive-linear
            shortcut and the excitation sectors. A dictionary selects the
            backend's stationary Liouvillian.
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

        A chip with pumps at one frequency solves on excitation sectors under
        the conditions of :meth:`finite_power`, and chooses ``K`` as it does.
        ``result.diagnostics["route"]`` records the route and its reason.

        Parameters
        ----------
        output : port object or str
            Output plane.
        frequencies : scalar or array_like
            Offset frequencies in GHz.
        options : dict or None, optional
            Stationary solver options. A dictionary selects the backend's
            stationary Liouvillian.
        """
        workflow = require_workflow(self.chip, "VNA.output_spectrum()")
        output_port = _resolve_exposure(self.chip, output)
        frequency_values, _ = _axis_values(frequencies)
        sector, reason = self._sector_output(workflow, (output_port,), options, frequency_values)
        if sector is not None:
            return _sector_spectrum(sector, output_port, frequency_values)
        operating = self._stationary_output(output_port, options, reason=reason)
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
            diagnostics=operating.diagnostics,
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

        The excitation sectors apply as in :meth:`output_spectrum`.

        Parameters
        ----------
        output : port object or str
            Delayed output plane.
        delays : scalar or array_like
            Non-negative delays in ns.
        input : port object or str, optional
            Cross-correlation input. The default is ``output``.
        options : dict or None, optional
            Stationary solver options. A dictionary selects the backend's
            stationary Liouvillian.

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

        The excitation sectors apply as in :meth:`output_spectrum`.

        Parameters
        ----------
        output : port object or str
            Delayed output plane.
        delays : scalar or array_like
            Non-negative delays in ns.
        input : port object or str, optional
            Cross-correlation input. The default is ``output``.
        options : dict or None, optional
            Stationary solver options. A dictionary selects the backend's
            stationary Liouvillian.

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
        workflow = require_workflow(self.chip, f"VNA.g{order}()")
        output_port = _resolve_exposure(self.chip, output)
        input_port = output_port if input is None else _resolve_exposure(self.chip, input)
        delay_values, _ = _axis_values(delays)
        if not contains_tracer(delay_values) and np.any(np.asarray(delay_values) < 0):
            raise ValueError("Stationary output correlations require non-negative delays.")
        sector, reason = self._sector_output(workflow, tuple(dict.fromkeys((output_port, input_port))), options,
                                             delay_values)
        if sector is not None:
            return _sector_correlation(sector, order, output_port, input_port, delay_values)
        operating = self._stationary_output(output_port, options, reason=reason)
        engine, state, tones = operating.engine, operating.state, operating.tones
        backend = self.chip.backend
        operators = port_operators(engine, backend)
        incoming = _stationary_output_backgrounds(engine, tones, backend)
        xp = backend.array_module
        rho = xp.asarray(backend.to_array(state.state), dtype=complex)
        runs = {item.key: item.reference.outbound for item in engine.slh.external_channels}
        _require_detectable(runs, [(channel.input_occupation, channel.reference) for channel in engine.slh.channels],
                            output_port, input_port, mixed=engine.slh.output_network is not None)

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
            normalization=_normalization(order, input_port == output_port),
            diagnostics=operating.diagnostics,
        )

    def _stationary_output(
        self,
        output_label: str,
        options: dict | None,
        *,
        reason: str = "The backend's stationary Liouvillian solves the output statistics.",
    ) -> _OperatingPoint:
        """Solve stationary output statistics in the applicable reference frames.

        Pump tones define the stationary frames when present; otherwise the requested
        output plane's own reference frequency does.
        """
        tones = self._tone_values({})
        engine = resolve_stationary_engine(self.chip, self._output_frames(output_label, tones))
        driven_engine = add_port_inputs(engine, self.chip.backend, tones)
        return _solve_engine(self.chip, driven_engine, tones, options, reason=reason)

    def _output_frames(self, output_label: str, tones: tuple[tuple[str, Any, Any], ...]) -> tuple[tuple[str, Any], ...]:
        """Return the tone frequencies, or the output's reference frequency without tones."""
        return tuple((label, frequency) for label, frequency, _ in tones) or (
            (output_label, _exposure_reference_frequency(self.chip, output_label)),
        )

    def _sector_output(
        self, workflow: Any, labels: tuple[str, ...], options: dict | None, grid: Any,
    ) -> tuple["_SectorOutput | None", str]:
        """Return the sector plan of a stationary output workflow, or ``None``, and the reason.

        ``labels`` lists the output first. ``grid`` holds the offsets or delays,
        which the host solve needs as concrete values.
        """
        tones = self._tone_values({})
        model, reason = plan_sectors(
            self.chip, workflow, labels=labels, inputs=tuple(port for port, _, _ in tones),
            frequencies=tuple(frequency for _, frequency, _ in tones), options=options,
            concrete=not contains_tracer((tones, grid)))
        if model is None:
            return None, reason
        from quchip.engine.assembly import _prepare_engine_assembly

        frames = self._output_frames(labels[0], tones)
        plan, resolution = stationary_frame_plan(self.chip, frames)
        _, frame = _prepare_engine_assembly(self.chip, plan, self.chip.approximation, resolution=resolution)
        return _SectorOutput(model, reason, workflow.name, float(frames[0][1]),
                             tuple(complex(amplitude) for _, _, amplitude in tones), frame), reason


    def _linear_sweep(
        self, chips: list[Any], frequencies: Any, labels: tuple[str, ...]
    ) -> tuple[list[Any], list[Mapping[str, Any]]] | str:
        """Solve every chip point in mode space, or return why a point needs the stationary route."""
        matrices: list[Any] = []
        diagnostics: list[Mapping[str, Any]] = []
        for chip in chips:
            built = try_build_weak_probe_problem(chip, frequencies, plane_labels=labels)
            if built is None:
                return weak_probe_reason(chip, labels)
            problem, name = built
            route = SolveRoute(name, WEAK_PROBE[name], len(problem.mode_labels), array_name(chip.backend))
            solved = chip.backend.linear_response(problem)
            indices = list(problem.plane_indices)
            transfer = (
                problem.outbound_transfer[:, indices][:, :, None]
                * problem.inbound_transfer[:, indices][:, None, :]
            )
            matrices.append(chip.backend.array_module.asarray(transfer * solved.responses))
            diagnostics.extend(
                _LazyDiagnostics({
                    "solver": name,
                    "route": route,
                    "mode_count": len(problem.mode_labels),
                    "residual": solved.residuals[index],
                    "condition_number": lambda index=index, condition=solved._condition_numbers: condition()[index],
                })
                for index in range(len(frequencies))
            )
        return matrices, diagnostics

    def _conditional_sweep(
        self, chips: list[Any], frequencies: Any, labels: tuple[str, ...], reference: tuple[int, ...],
    ) -> tuple[list[Any], list[Mapping[str, Any]]]:
        """Solve a fixed-reference response at each parameter point."""
        matrices: list[Any] = []
        diagnostics: list[Mapping[str, Any]] = []
        for chip in chips:
            problem, metadata = build_conditional_response(chip, frequencies, labels, reference)
            vacuum = isinstance(problem, LinearResponseProblem)
            metadata["route"] = SolveRoute(
                metadata["solver"], _VACUUM_REFERENCE if vacuum else _FROZEN_REFERENCE,
                metadata["mode_count"] if vacuum else metadata["coherence_count"], array_name(chip.backend))
            solved = chip.backend.linear_response(problem) if vacuum else chip.backend.coherence_response(problem)
            indices = list(problem.plane_indices)
            transfer = problem.outbound_transfer[:, indices, None] * problem.inbound_transfer[:, None, indices]
            matrices.append(transfer * solved.responses)

            def condition(index: int, value: Any = solved._condition_numbers) -> Any:
                values = value()
                return None if values is None else values[index]

            diagnostics.extend(_LazyDiagnostics({
                **metadata, "residual": solved.residuals[index],
                "condition_number" if vacuum else "block_condition_number": partial(condition, index),
            }) for index in range(len(frequencies)))
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
    *,
    guess: Any | None = None,
    reason: str,
) -> _OperatingPoint:
    """Solve the pumped stationary state with every selected plane framed at ``frequency``.

    ``guess`` is the operating state at another probe frequency; the backend
    may reuse it when it remains stationary in these frames. ``reason``
    explains the route in the diagnostics.
    """
    engine = resolve_stationary_engine(
        chip,
        tuple((label, tone_frequency) for label, tone_frequency, _ in tones)
        + tuple((label, frequency) for label in labels),
    )
    operating_engine = add_port_inputs(engine, chip.backend, tones)
    return _solve_engine(chip, operating_engine, tones, options, guess=guess, reason=reason)


def _solve_engine(
    chip: Any, engine: EngineResult, tones: tuple[tuple[str, Any, Any], ...], options: dict | None,
    *, guess: Any | None = None, reason: str,
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
    return _OperatingPoint(prepared, solve_steadystate_problem(problem, prepared=prepared, guess=guess, reason=reason),
                           tones)


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
    *,
    offset: Any = 0.0,
) -> tuple[Any, Any]:
    """Return the selected-port ``S_ji(f)`` and phase-conjugating ``T_ji(f)`` matrices.

    Rows are output ports and columns are input ports. Around a phase-sensitive
    operating point the linear response is ``delta <b_out> = S delta beta + T
    conj(delta beta)``; one shifted-Liouvillian factorization serves both source
    sets for all input columns. ``offset`` is the probe frequency relative to
    the stationary frame, in GHz. At nonzero offset the conjugate input lies
    at the opposite Fourier frequency and contributes no same-carrier ``T``.
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
        (offset,),
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
        xp.where(xp.asarray(offset) == 0.0,
                 gain * xp.conj(inbound)[None, :] * (mixing @ gather("conjugate")), 0.0),
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


@dataclass(frozen=True)
class _SectorOutput:
    """The sector model of a stationary output workflow, with its coherent inputs at one carrier frequency."""

    model: Any
    reason: str
    method: str
    frequency: float
    amplitudes: tuple[complex, ...]
    frame: Any


def _sector_solve(sector: _SectorOutput, measure: Any, change: Any, quantities: str,
                  stacklevel: int) -> tuple[int, Any, str, Any]:
    """Solve on the chip's cutoff, or on an automatic one, and measure the stationary state.

    Returns the kept cutoff, the solve with its measured values, the choice
    of the cutoff and the check of an automatic one.
    """
    from quchip.analysis.measurement import _embed

    model = sector.model

    def solve(cutoff: int, lower: Any) -> Any:
        guess = None if lower is None else _embed(lower[0].state, model.size(cutoff))
        solved = model.stationary(cutoff, sector.frequency, sector.amplitudes, guess)
        return solved, measure(cutoff, solved.state)

    if model.sectors.cutoff is not None:
        return model.sectors.cutoff, solve(model.sectors.cutoff, None), "explicit", None
    cutoff, results, check = automatic_cutoff(
        model.max_total, solve, lambda lower, upper: change(lower[1], upper[1]), states=model.size,
        fits=lambda total: fits_memory(model.required_memory(total)))
    if check is not None:
        warn_cutoff(sector.method, cutoff, model.size(cutoff), check, quantities, stacklevel=stacklevel)
    return cutoff, results, "automatic", check


def _sector_spectrum(sector: _SectorOutput, output_port: str, offsets: Any) -> OutputSpectrumResult:
    """Return the output spectrum of a sector solve, in the frame of the coherent inputs."""
    model, frequency, amplitudes = sector.model, sector.frequency, sector.amplitudes
    grid = np.asarray(offsets, dtype=float)

    def source(pair: tuple[Any, Any]) -> Any:
        white, excess = pair
        return normal_spectrum(white + excess, np)

    def measure(cutoff: int, state: Any) -> tuple[Any, ...]:
        excess, info = model.quadrature_spectrum(cutoff, frequency, amplitudes, state, grid)
        components, _ = propagate_noise(model.fields, model.scattering, None, excess, (output_port,), frequency,
                                        grid, np)
        added = sum((source(pair) for name, pair in components.items() if name != "device.correlations"),
                    np.zeros(len(grid)))
        field = model.field(cutoff, frequency, amplitudes, output_port)
        intensity = float(np.real((field.conj().T @ field).multiply(state.T).sum()))
        coherent = abs(complex(field.multiply(state.T).sum())) ** 2
        return source(components["device.correlations"]), added, intensity, coherent, info

    # The warning names the caller of VNA.output_spectrum().
    cutoff, (solved, (signal, added, intensity, coherent, info)), choice, check = _sector_solve(
        sector, measure, _spectrum_change, "spectra and photon fluxes", stacklevel=4)
    diagnostics = {**sector_diagnostics(model, cutoff, solved, sector.reason, choice, check), **info}
    xp = model.chip.backend.array_module
    return OutputSpectrumResult(
        port=output_port,
        frequencies=offsets,
        total_fluctuation_spectrum=xp.asarray(signal + added),
        signal_fluctuation_spectrum=xp.asarray(signal),
        added_noise_spectrum=xp.asarray(added),
        signal_photon_flux=xp.asarray(intensity),
        signal_coherent_flux=xp.asarray(coherent),
        signal_incoherent_flux=xp.asarray(intensity - coherent),
        steady_state=model.steady_state(cutoff, solved, sector.frame, diagnostics),
        diagnostics=_LazyDiagnostics(diagnostics),
    )


def _spectrum_change(lower: tuple[Any, ...], upper: tuple[Any, ...]) -> float:
    """Compare two spectrum solves at each offset, and their total and coherent photon fluxes.

    The fluxes share the floor of the total flux, which bounds the coherent one.
    """
    return max(largest_change(list(lower[0]), list(upper[0])), largest_change(list(lower[1]), list(upper[1])),
               largest_change(lower[2:4], upper[2:4]))


def _correlation_change(lower: tuple[Any, ...], upper: tuple[Any, ...]) -> float:
    """Compare two correlation solves at each delay, and each of their two intensities."""
    return max(largest_change(list(lower[0]), list(upper[0])), largest_change(lower[1:2], upper[1:2]),
               largest_change(lower[2:3], upper[2:3]))


def _sector_correlation(sector: _SectorOutput, order: int, output_port: str, input_port: str,
                        delays: Any) -> OutputCorrelationResult:
    """Return a normalized output correlation of a sector solve, in the frame of the coherent inputs."""
    model, frequency, amplitudes = sector.model, sector.frequency, sector.amplitudes
    references = model.response.references
    _require_detectable({key: reference.outbound for key, reference in zip(model.keys, references, strict=True)},
                        [(None, reference) for reference in references], output_port, input_port, mixed=False)
    grid = np.asarray(delays, dtype=float)

    def measure(cutoff: int, state: Any) -> tuple[Any, float, float]:
        values = model.correlation(cutoff, frequency, amplitudes, state, order, output_port, input_port, grid)
        if values[1] <= 0 or values[2] <= 0:
            raise ValueError("Normalized output correlations require nonzero output intensity.")
        return values

    # The warning names the caller of VNA.g1() or VNA.g2().
    cutoff, (solved, (raw, input_intensity, output_intensity)), choice, check = _sector_solve(
        sector, measure, _correlation_change, "correlations and intensities", stacklevel=5)
    raw = np.conj(raw)
    denominator = np.sqrt(input_intensity * output_intensity) if order == 1 else input_intensity * output_intensity
    diagnostics = sector_diagnostics(model, cutoff, solved, sector.reason, choice, check)
    xp = model.chip.backend.array_module
    return OutputCorrelationResult(
        order=order,
        input_port=input_port,
        output_port=output_port,
        delays=delays,
        values=xp.asarray(raw / denominator),
        unnormalized=xp.asarray(raw),
        input_intensity=xp.asarray(input_intensity),
        output_intensity=xp.asarray(output_intensity),
        steady_state=model.steady_state(cutoff, solved, sector.frame, diagnostics),
        normalization=_normalization(order, input_port == output_port),
        diagnostics=_LazyDiagnostics(diagnostics),
    )


def _require_detectable(runs: Mapping[str, Any], channels: Sequence[tuple[Any, Any]], output_port: str,
                        input_port: str, *, mixed: bool) -> None:
    """Raise when normalized g1 and g2 need a detection bandwidth.

    ``runs`` maps each external channel to its outbound reference run, and
    ``channels`` holds the input occupation and the reference of every channel.
    """
    if mixed or has_amplifier(runs[output_port]) or has_amplifier(runs[input_port]):
        raise NotImplementedError(
            "Normalized g1 and g2 through an amplifier require a detection bandwidth for its "
            "broadband added noise. Request the correlation at a reference plane before the amplifier."
        )
    thermal = any(occupation is not None or any(
        (isinstance(element, ReferenceLoss) and element.occupation is not None)
        or (isinstance(element, ReferenceFilter) and element.loss_occupation is not None)
        for element in (*reference.inbound, *reference.outbound))
        for occupation, reference in channels)
    if thermal:
        raise NotImplementedError("Normalized g1 and g2 with thermal network fields require a detection "
                                  "bandwidth. Use measure().statistics() for integrated second moments.")


def _normalization(order: int, same_port: bool) -> str:
    """Return the normalization convention of an output correlation."""
    if order == 1:
        return "G1(tau) / G1(0)" if same_port else "G1(output, input; tau) / sqrt(I_output I_input)"
    return "G2(tau) / G1(0)^2" if same_port else "G2(output, input; tau) / (I_output I_input)"
