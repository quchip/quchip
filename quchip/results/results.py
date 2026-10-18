"""Backend-agnostic wrapper around solver output.

The engine wraps each backend's solver output here, so callers use the same
state, expectation, partial-trace, population, and batch interfaces.

Numerical population and overlap accessors return NumPy arrays for QuTiP and
JAX arrays for Dynamiqs, even for eager calls. Host conversion is explicit.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quchip.backend import Backend, SolverResult
from quchip.devices.base import BaseDevice
from quchip.results._batch import BatchResult
from quchip.utils.constants import TWO_PI
from quchip.utils.labeling import resolve_label

# Mirrors quchip.viz.results.StateMode so the plot_state shim can re-expose the
# same literal without a hard import of the lazily loaded viz module.
StateMode = Literal["population", "dm"]
Frame = Literal["integration", "lab"]

if TYPE_CHECKING:
    from quchip.engine.ir import SLHChannel, SolveBatch, SolveProblem


@dataclass(frozen=True)
class ObservableTrace:
    """One named expectation-value trace, with pre- and post-processing values.

    Attributes
    ----------
    values
        Post-processed expectation values over time (e.g. demodulated,
        phase-corrected, band-summed). User code usually reads these.
    raw
        The same quantity before post-processing. Use it to debug frame
        conventions, band decomposition, and demodulation.
    """

    values: Any
    raw: Any


@dataclass(frozen=True)
class OutputFieldTrace:
    """Complete transient field reported at one external reference plane.

    ``amplitude`` is the complex mean field ``<b_out>``. ``photon_flux`` is the
    normally ordered ``<b_out dagger b_out>`` in photons/ns. ``raw_*`` keep the
    same moments at the Markov boundary, before propagation through the outbound
    reference run. Complex amplitudes use the physics ``e^{-iωt}`` convention.
    Conjugate ``raw_amplitude`` to give a boundary template to ``iq_readout()``.

    Attributes
    ----------
    exposure : str
        External network-port label.
    times : array_like
        Saved times in ns.
    amplitude, raw_amplitude : array_like
        Propagated and Markov-boundary complex means in ``1/sqrt(ns)``.
    photon_flux, raw_photon_flux : array_like
        Propagated and Markov-boundary normally ordered fluxes in photons/ns.
    """

    exposure: str
    times: Any
    amplitude: Any
    photon_flux: Any
    raw_amplitude: Any
    raw_photon_flux: Any

    def quadrature(self, phase: Any = 0.0) -> Any:
        r"""Return ``Re[exp(-i phase) <b_out>]`` without another solve.

        Parameters
        ----------
        phase : scalar, default=0.0
            Quadrature angle in radians.
        """
        from quchip.utils.jax_utils import array_namespace

        xp = array_namespace(self.amplitude)
        return xp.real(xp.exp(-1j * xp.asarray(phase)) * self.amplitude)

    @property
    def final_amplitude(self) -> Any:
        """Return the final complex output amplitude."""
        return self.amplitude[..., -1]

    @property
    def final_photon_flux(self) -> Any:
        """Return the final normally ordered photon flux."""
        return self.photon_flux[..., -1]


_NO_STATES_MSG = 'Full state history is unavailable; run with states="all" to retain it.'


class SimulationResult:
    """Backend-agnostic container for the output of one solve.

    Stored states follow the requested storage policy. ``times`` uses ns, and
    ``dims`` follows chip order. Dict-form ``e_ops`` become
    :class:`ObservableTrace` entries in ``observable_traces``.

    Parameters
    ----------
    solver_result : SolverResult
        Backend output to wrap.
    backend : Backend
        Backend that owns the native states and arrays.
    dims : sequence of int
        Hilbert dimensions in chip order.
    device_info, observable_traces, output_traces, channels, bases : optional
        Captured device, observable, output, channel, and basis metadata.
    dissipation : bool, default=True
        Whether collapse channels were included.
    excitation_space : optional
        Captured exact excitation coordinate map, or None for product coordinates.
    readout_wiring : optional
        Captured downstream readout wiring.
    frame : ResolvedFrame or None, optional
        Integration frame of the saved states.
    label_targets : LabelTargets or None, optional
        Snapshot of the solve's chip that prepares label targets.

    Attributes
    ----------
    frame : ResolvedFrame or None
        Integration frame of the saved states: the frame frequency of each device and the frame mode.
    """

    def __init__(
        self,
        solver_result: SolverResult,
        backend: Backend,
        dims: list[int] | tuple[int, ...],
        *,
        device_info: list[tuple[str, bool]] | tuple[tuple[str, bool], ...] | None = None,
        observable_traces: dict[Any, ObservableTrace | list[ObservableTrace]] | None = None,
        output_traces: dict[Any, OutputFieldTrace] | None = None,
        channels: tuple[SLHChannel, ...] = (),
        bases: dict[str, Any] | None = None,
        dissipation: bool = True,
        readout_wiring: Any = None,
        excitation_space: Any = None,
        frame: Any = None,
        label_targets: Any = None,
    ) -> None:
        self.frame = frame
        self._label_targets = label_targets
        self._frame_map_cache: Any = None
        self._expanded_states: dict[int, Any] = {}
        self._excitation_space = excitation_space
        self._readout_wiring = readout_wiring
        self._backend = backend
        self._truncation: Any = None
        self._boundary_traces: tuple[Any, ...] | None = None
        self.dissipation = dissipation
        self._bases = {} if bases is None else dict(bases)
        self._channels: dict[str, SLHChannel] = {channel.key: channel for channel in channels}
        self._jump_rate_cache: dict[str, Any] = {}
        self.times = backend.array_module.asarray(solver_result.times, dtype=float)
        self._states = solver_result.states
        self._expect_data: dict[Any, ObservableTrace | list[ObservableTrace]] | None = observable_traces
        self._output_data = {} if output_traces is None else dict(output_traces)
        self.solver = solver_result.solver
        self.stats = dict(solver_result.stats) if solver_result.stats else {}
        self.dims = list(dims)
        self.device_info = device_info
        self._final_state = solver_result.final_state
        # Cache native trajectories and the labels used by the full population dictionary.
        self._stacked_cache: Any = None
        self._basis_labels_cache: list[tuple[int, ...]] | None = None

    # ------------------------------------------------------------------
    # Observables — dict-form expectation traces
    # ------------------------------------------------------------------

    @property
    def observable_traces(self) -> dict[Any, ObservableTrace | list[ObservableTrace]] | None:
        """Return the dict of named :class:`ObservableTrace` entries, or ``None``.

        The value is ``None`` when ``e_ops`` was not a dict.
        """
        # Visualization and analysis code that needs the full dict should read
        # this property, not the private `_expect_data` attribute.
        return self._expect_data

    def _resolve_trace(self, key: Any, index: int | None = None) -> ObservableTrace:
        if not isinstance(self._expect_data, dict):
            raise TypeError("Expectation-value access requires dict-form e_ops (pass e_ops as a dict)")
        resolved_key: Any = (
            tuple(resolve_label(k) for k in key) if isinstance(key, tuple) else resolve_label(key)
        )
        trace = self._expect_data[resolved_key]
        if isinstance(trace, list):
            if index is None:
                raise ValueError(
                    f"expect[{key!r}] contains {len(trace)} traces; specify index=0..{len(trace) - 1}"
                )
            return trace[index]
        if index is not None:
            raise ValueError(f"expect[{key!r}] is a single trace, not a list; drop the index argument")
        return trace

    def expect(self, key: Any, index: int | None = None) -> Any:
        """Return an expectation trace over ``self.times``.

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, optional
            Entry of a list-valued observable.
        """
        return self._resolve_trace(key, index).values

    def observable_at(self, t: Any, values: Any, *, method: str = "exact") -> Any:
        """Select saved observable values at scalar or array times.

        Values have time on their last axis, and the result has their leading
        axes followed by the query shape. ``nearest`` selects the earlier
        sample on a tie. ``interpolate`` is linear, including for complex
        values. Out-of-interval queries raise an error. No solve or state
        interpolation occurs.

        Parameters
        ----------
        t : scalar or array_like
            Query times in ns.
        values : array_like
            Saved values with time on the final axis.
        method : {"exact", "nearest", "interpolate"}, default="exact"
            Time-selection rule.
        """
        from quchip.results._time import observable_at

        return observable_at(self.times, t, values, method, self._backend.array_module)

    def expect_final(self, key: Any, index: int | None = None) -> Any:
        """Return the final expectation value for an observable.

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, optional
            Entry of a list-valued observable.
        """
        return self._resolve_trace(key, index).values[-1]

    # Alias: reads nicely at call sites that say "give me the values array".
    expect_values = expect

    # ------------------------------------------------------------------
    # External fields
    # ------------------------------------------------------------------

    @property
    def outputs(self) -> dict[Any, OutputFieldTrace]:
        """Return complete output-field traces keyed by network port label."""
        return dict(self._output_data)

    def output(self, exposure: Any) -> OutputFieldTrace:
        """Return one field trace by network port or label.

        Parameters
        ----------
        exposure : port object or str
            External output reference plane.
        """
        label = resolve_label(exposure)
        try:
            return self._output_data[label]
        except KeyError as exc:
            raise KeyError(
                f"No output for {label!r}. Available network ports: {sorted(self._output_data)}"
            ) from exc

    # ------------------------------------------------------------------
    # Collapse channels
    # ------------------------------------------------------------------

    @property
    def collapse_channels(self) -> tuple[str, ...]:
        """Return the resolved SLH channel keys accepted by :meth:`jump_rate`.

        Hidden channels use keys such as ``"hidden.q.thermal_emission"`` and
        ``"hidden.r.internal_photon_loss"``. Channels exposed at external
        ports use their network labels.
        """
        return tuple(self._channels)

    def _channel_key(self, key: Any) -> str:
        label = resolve_label(key)
        if label in self._channels:
            return label
        matches = [name for name in self._channels if name.split("#")[0].partition(".")[2] == label]
        if len(matches) == 1:
            return matches[0]
        detail = "matches several channels" if matches else "matches no channel"
        raise KeyError(f"{label!r} {detail}. Available channels: {list(self._channels)}")

    def jump_rate(self, key: Any) -> Any:
        """Return the cached jump rate ``<L†L>(t)`` for one resolved channel.

        The rate, in ``1/ns``, is evaluated after the solve from stored states.
        ``key`` can be a resolved key from :attr:`collapse_channels`, a network
        port, or a ``"<device>.<channel>"`` label that identifies one channel.
        Address a port composed into a network through its external network
        port.

        At an external network port, this rate equals ``raw_photon_flux`` only with
        vacuum input. Dephasing and thermal-absorption channels can also have
        nonzero jump rates, so this quantity alone is not an excitation-loss rate.

        Parameters
        ----------
        key : channel object or str
            Resolved channel key, external port, or unambiguous local label.
        """
        name = self._channel_key(key)
        if name not in self._jump_rate_cache:
            from quchip.engine.observables import collapse_number_operator

            operator = collapse_number_operator(self._channels[name].coupling, self._backend, tag=f"flux:{name}")
            flux = self._backend.array_module.real(self._backend.expect_over_time(operator, self._stacked_states()))
            from quchip.utils.jax_utils import contains_tracer

            if not contains_tracer(flux):
                self._jump_rate_cache[name] = flux
            return flux
        return self._jump_rate_cache[name]

    def collapse_flux(self, key: Any) -> Any:
        """Deprecated alias for :meth:`jump_rate`.

        Parameters
        ----------
        key : channel object or str
            Resolved channel key.
        """
        from quchip.utils.deprecation import warn_renamed

        warn_renamed("collapse_flux()", "jump_rate()")
        return self.jump_rate(key)

    def collapse_integral(self, key: Any) -> Any:
        """Return the cumulative expected jump count for one channel.

        The cumulative trapezoid integral of :meth:`jump_rate` on the result
        grid. Its last entry is the expected number of jumps over the whole
        solve.

        Parameters
        ----------
        key : channel object or str
            Resolved channel key.
        """
        xp = self._backend.array_module
        flux = self.jump_rate(key)
        increments = 0.5 * (flux[1:] + flux[:-1]) * (self.times[1:] - self.times[:-1])
        return xp.concatenate([xp.zeros((1,), dtype=increments.dtype), xp.cumsum(increments)])

    # ------------------------------------------------------------------
    # States, partial traces, overlaps
    # ------------------------------------------------------------------

    @property
    def states(self) -> Any:
        """Return the kept native state history, or tell how to save it."""
        if self._states is None or len(self._states) == 0:
            raise RuntimeError(_NO_STATES_MSG)
        if self._excitation_space is not None:
            from math import prod
            from quchip.utils.memory import dense_bytes, require_memory
            ket = self._backend.is_ket(self._states[0])
            # A stacked history keeps the cached states and a new stack.
            copies = 1 if isinstance(self._states, list) else 2
            dimension = prod(self.dims)
            require_memory(dense_bytes(len(self._states), *((dimension,) if ket else (dimension, dimension)),
                                       copies=copies),
                           task="Full pulse history export", remedy="Use compact_state() or local observables.")
            lifted = [self._export_state(state, index) for index, state in enumerate(self._states)]
            return lifted if isinstance(self._states, list) else self._backend.stack_states(lifted)
        return self._states

    def _stacked_states(self) -> Any:
        """Return the trajectory as one ``(T, …)`` native stacked array (cached).

        This is the batched form the over-time extractors consume — built
        once per result. On dynamiqs it is the solver's own stacked
        ``QArray`` (no copy); on QuTiP it is a single ``np.stack`` of the
        stored ``Qobj``s. Keeping it as one array (rather than ``T`` separate
        objects) is what collapses the per-point extractor loop.
        """
        cached = getattr(self, "_stacked_cache", None)
        if cached is None:
            if self._states is None or len(self._states) == 0:
                raise RuntimeError(_NO_STATES_MSG)
            cached = self._backend.stack_states(self._states)
            from quchip.utils.jax_utils import contains_tracer

            if not contains_tracer(cached):
                self._stacked_cache = cached
        return cached

    def _is_ket_trajectory(self) -> bool:
        """Return whether the stored trajectory is kets (vs density matrices)."""
        return self._backend.is_ket(self._stacked_states()[0])

    @property
    def _basis_labels(self) -> list[tuple[int, ...]]:
        """Cache solver-product labels for the full population dictionary."""
        cached = self._basis_labels_cache
        if cached is None:
            cached = list(itertools.product(*[range(d) for d in self.dims]))
            self._basis_labels_cache = cached
        return cached

    def _export_state(self, state, index):
        """Lift one saved compact state, caching it by its time index."""
        if self._excitation_space is None:
            return state
        from quchip.utils.jax_utils import contains_tracer
        key = None if contains_tracer(index) else int(index)
        if key in self._expanded_states:
            return self._expanded_states[key]
        lifted = self._excitation_space.lift(state, self._backend)
        if key is not None and not contains_tracer(lifted):
            self._expanded_states[key] = lifted
        return lifted

    def _target_state(self, target):
        from quchip.engine.states import CompactState, compatible_values
        backend = self._backend
        space = self._excitation_space
        if isinstance(target, CompactState):
            if target.values.shape[-1] != 1:
                raise TypeError("Overlap targets must be kets.")
            values = compatible_values(
                target, tuple(label for label, _ in self.device_info), self.dims, self._bases, backend,
            )
            if space is not None:
                xp = backend.array_module
                # The saved states have no amplitude outside this space.
                columns = np.asarray([target.space.rows.get(tuple(level), target.space.size) for level in space.levels])
                padded = xp.concatenate([values, xp.zeros((1, 1), dtype=values.dtype)])
                return backend.from_array(padded[columns], dims=[[space.size], [1]])
            native = backend.from_array(values, dims=[[target.space.size], [1]])
            return target.space.lift(native, backend)
        native = backend.coerce_state(target, dims=tuple(self.dims))
        return native if space is None else space.project(native, backend)

    def _frame_map(self):
        """Return the map between the lab frame and the frame of the saved states."""
        from quchip.engine.frames import FrameMap
        from quchip.utils.jax_utils import contains_tracer

        if self._frame_map_cache is not None:
            return self._frame_map_cache
        if self.frame is None or self.device_info is None:
            raise RuntimeError("This result has no captured frame; it was not created by a solve.")
        frame_map = FrameMap.create(self.frame, tuple(label for label, _ in self.device_info), self._bases,
                                    self._backend.array_module, space=self._excitation_space)
        if not contains_tracer(frame_map.energies):
            self._frame_map_cache = frame_map
        return frame_map

    def _saved_dims(self) -> list[int]:
        return list(self.dims if self._excitation_space is None else self._excitation_space.solver_dims)

    def _in_frame(self, state, index, frame):
        """Return a saved native state in the requested frame, in saved coordinates."""
        if _frame_choice(frame) == "integration":
            return state
        backend = self._backend
        values = backend.to_array(state)
        time = self.times[index]
        frame_map = self._frame_map()
        ket = backend.is_ket(state)
        mapped = (frame_map.kets(values.reshape(-1), time, to_lab=True).reshape(values.shape) if ket
                  else frame_map.operators(values, time, to_lab=True))
        dims = self._saved_dims()
        return backend.from_array(mapped, dims=[dims, [1] * len(dims) if ket else dims])

    def _exported(self, state, index, frame):
        """Return a saved state in captured product coordinates and the requested frame."""
        if _frame_choice(frame) == "integration":
            return self._export_state(state, index)
        state = self._in_frame(state, index, frame)
        return state if self._excitation_space is None else self._excitation_space.lift(state, self._backend)

    def _resolve_target(self, target, frame, dressed, times):
        """Return a fixed native target, or an array with one target ket per time.

        A ket keeps its integration-frame meaning unless ``frame`` is ``"lab"``.
        A bare label is fixed in the integration frame. A dressed label k
        turns with its label: ``U(t)†|d_k⟩ exp(−i 2π t Σᵢ fᵢ kᵢ)``.
        """
        frame = _frame_choice(frame)
        backend = self._backend
        if isinstance(target, (Mapping, str)):
            if frame == "lab":
                raise ValueError("A label target turns with its label in every frame; frame= applies to ket targets.")
            if self._label_targets is None:
                raise RuntimeError("This result has no captured chip for label targets; it was not created by a solve.")
            levels = self._label_targets.levels(target, self.dims)
            space = self._excitation_space
            if not dressed:
                vector = self._label_targets.bare(levels, self._bases, space)
                dims = self._saved_dims()
                return backend.from_array(vector.reshape(-1, 1), dims=[dims, [1] * len(dims)]), None
            frame_map = self._frame_map()
            vector = self._label_targets.dressed(levels, self._bases, space)
            xp = backend.array_module
            phase = xp.exp(-1j * TWO_PI * xp.asarray(times) * frame_map.label_energy(levels))
            return None, phase[:, None] * frame_map.kets(vector, times)
        if dressed:
            raise ValueError("dressed=True applies to label targets, such as {'q0': 0, 'q1': 1}.")
        fixed = self._target_state(target)
        if frame == "integration":
            return fixed, None
        return None, self._frame_map().kets(backend.to_array(fixed).reshape(-1), times)

    def compact_state(self, t: Any = None) -> Any:
        """Return a reusable saved state without product-space expansion.

        Parameters
        ----------
        t : scalar or None, optional
            Saved time in ns. None selects the final state.

        Returns
        -------
        CompactState
            Captured excitation state accepted as initial_state by a later solve.
        """
        from quchip.engine.states import CompactState
        if self._excitation_space is None:
            raise ValueError("This result stores full product states. Use state() to reuse them.")
        state = self._raw_final_state() if t is None else self._raw_state_at(t)
        return CompactState(self._backend.to_array(state), self._excitation_space)

    def overlap(self, target: Any, *, frame: Frame = "integration", dressed: bool = False) -> Any:
        """Return the overlap with *target* at every stored time.

        For ket trajectories, the method returns ``|<target|psi(t)>|**2``. For
        density matrices, it returns ``<target|rho(t)|target>``. The result
        stays in the backend's array module, so it is differentiable. One
        batched op covers the leading time axis, with no per-point loop.

        A ket target is a fixed vector of the integration frame, so a target
        such as |+⟩ is written in that frame. With ``frame="lab"``, the target
        is a fixed lab-frame vector, such as a dressed state from
        ``chip.state()``. A label target, such as ``{"q0": 0, "q1": 1}``, is
        the bare product state of local energy levels. With ``dressed=True``,
        it is the dressed eigenstate ``|d_k⟩`` of label k, which turns with
        its label: ``U(t)†|d_k⟩ exp(−i 2π t Σᵢ fᵢ kᵢ)``, where ``U(t)†`` maps
        the lab frame into the integration frame and ``fᵢ`` are the frame
        frequencies. An idle dressed state then reads 1 in every frame. A
        snapshot of the solve's chip supplies ``|d_k⟩``, with the solve's
        approximation.

        Parameters
        ----------
        target : state_like, mapping, or str
            Target ket in the captured full Hilbert space, a compatible ``CompactState``,
            or a label of local energy levels.
        frame : {"integration", "lab"}, default="integration"
            Frame in which a ket target is fixed.
        dressed : bool, default=False
            Use the dressed eigenstate of a label target.
        """
        backend = self._backend
        target, moving = self._resolve_target(target, frame, dressed, self.times)
        if moving is not None:
            return _moving_projection(backend, moving, self._stacked_states(), amplitude=False)
        vector = backend.to_array(target)
        dims = self.dims if self._excitation_space is None else self._excitation_space.solver_dims
        projector = backend.from_array(vector @ backend.array_module.conj(vector.T),
                                       dims=[list(dims), list(dims)])
        series = backend.expect_over_time(projector, self._stacked_states())
        return backend.array_module.abs(series)

    def amplitude_array(self, target: Any, *, frame: Frame = "integration", dressed: bool = False) -> Any:
        """Return the phase-sensitive complex projection ``<target|psi(t)>`` for kets.

        Density-matrix trajectories raise :class:`TypeError`, because a mixed
        state has no single phase-sensitive amplitude. For a mixed state, use
        :meth:`overlap`. The method uses one batched op, with no per-point
        loop. Targets, ``frame`` and ``dressed`` are as in :meth:`overlap`.
        With ``frame="lab"``, the phase is the lab-frame phase. A dressed label
        target gives a phase that turns at ``Σᵢ fᵢ kᵢ − E_k``, the detuning of
        the label's frame energy from its dressed energy ``E_k``.

        Parameters
        ----------
        target : state_like, mapping, or str
            Target ket in the captured full Hilbert space, a compatible ``CompactState``,
            or a label of local energy levels.
        frame : {"integration", "lab"}, default="integration"
            Frame in which a ket target is fixed.
        dressed : bool, default=False
            Use the dressed eigenstate of a label target.
        """
        backend = self._backend
        if not self._is_ket_trajectory():
            raise TypeError(
                "amplitude_array() requires ket trajectories; use overlap() for density matrices."
            )
        target, moving = self._resolve_target(target, frame, dressed, self.times)
        if moving is not None:
            return _moving_projection(backend, moving, self._stacked_states(), amplitude=True)
        return backend.array_module.asarray(
            backend.overlap_over_time(target, self._stacked_states())
        )

    def _resolve_device_idx(self, device: str | BaseDevice) -> tuple[int, str]:
        label = resolve_label(device)
        if self.device_info is None:
            raise RuntimeError("device_info not available — result was not created by simulate()")
        for i, (lbl, _) in enumerate(self.device_info):
            if lbl == label:
                return i, label
        available = [lbl for lbl, _ in self.device_info]
        raise ValueError(f"Device '{label}' not found in device_info. Available: {available}")

    def state(self, t: float | None = None, *, dm: bool = False, frame: Frame = "integration") -> Any:
        """Return a kept state, optionally promoted to a density matrix.

        Parameters
        ----------
        t : float or None, optional
            Saved time in ns. ``None`` selects the final state.
        dm : bool, default=False
            Promote a ket to a density matrix.
        frame : {"integration", "lab"}, default="integration"
            Frame of the returned state.
        """
        s = (self._exported(self._raw_final_state(), len(self.times) - 1, frame) if t is None
             else self.state_at(t, frame=frame))
        if dm and self._backend.is_ket(s):
            return self._backend.state_to_dm(s)
        return s

    def state_at(self, t: Any, *, method: str = "exact", frame: Frame = "integration") -> Any:
        """Return a saved native state in captured product coordinates.

        Saved states are in the integration frame of :attr:`frame`. With
        ``frame="lab"``, the method maps the state into the lab frame at its
        saved time.

        Parameters
        ----------
        t : scalar
            Saved time in ns.
        method : {"exact", "nearest"}, default="exact"
            Time-selection rule. The lookup never interpolates states.
        frame : {"integration", "lab"}, default="integration"
            Frame of the returned state.
        """
        return self._exported(*self._saved_state(t, method), frame)

    def _raw_state_at(self, t: Any, *, method: str = "exact") -> Any:
        """Return a kept state at a scalar time without state interpolation.

        Parameters
        ----------
        t : scalar
            Query time in ns.
        method : {"exact", "nearest"}, default="exact"
            Time-selection rule.
        """
        return self._saved_state(t, method)[0]

    def _saved_state(self, t: Any, method: str) -> tuple[Any, Any]:
        """Return a kept native state and its time index."""
        from quchip.results._time import require_valid, time_selection
        from quchip.utils.jax_utils import contains_tracer

        if method not in ("exact", "nearest"):
            raise ValueError('State lookup method must be "exact" or "nearest".')
        xp = self._backend.array_module
        if xp.asarray(t).ndim != 0:
            raise ValueError("state_at() requires a scalar time.")
        index, _, _ = time_selection(self.times, t, method, xp)
        if self._states is None or len(self._states) == 0:
            if not contains_tracer(index) and int(index) != len(self.times) - 1:
                raise RuntimeError(_NO_STATES_MSG)
            return require_valid(self._raw_final_state(), index != len(self.times) - 1, _NO_STATES_MSG), index
        if contains_tracer(index):
            return self._stacked_states()[index], index
        return self._states[int(index)], index

    def dm_at(self, t: float, *, method: str = "exact", frame: Frame = "integration") -> Any:
        """Return a density matrix at a kept time, and promote kets on demand.

        Parameters
        ----------
        t : float
            Query time in ns.
        method : {"exact", "nearest"}, default="exact"
            Time-selection rule.
        frame : {"integration", "lab"}, default="integration"
            Frame of the returned state.
        """
        state = self.state_at(t, method=method, frame=frame)
        return self._backend.state_to_dm(state) if self._backend.is_ket(state) else state

    @property
    def final_state(self) -> Any:
        """Return the last kept state in captured product coordinates."""
        return self._export_state(self._raw_final_state(), len(self.times) - 1)

    def _raw_final_state(self):
        if self._states is not None and len(self._states) > 0:
            return self._states[-1]
        if self._final_state is not None:
            return self._final_state
        raise RuntimeError('No final state available; run with states="all" or states="final" to retain it.')

    def _final_projection(self, target: Any, *, amplitude: bool, frame: Frame, dressed: bool) -> Any:
        """Project the final saved state on a target, without its history."""
        backend = self._backend
        state = self._raw_final_state()
        is_ket = backend.is_ket(state)
        if amplitude and not is_ket:
            raise TypeError("Final amplitudes require ket states; use final_overlap_magnitudes for density matrices.")
        stacked = backend.stack_states([state])
        target, moving = self._resolve_target(target, frame, dressed, self.times[-1:])
        if moving is not None:
            return _moving_projection(backend, moving, stacked, amplitude=amplitude)[0]
        value = backend.overlap_over_time(target, stacked)[0]
        return value if amplitude else backend.array_module.abs(value) ** (2 if is_ket else 1)

    def iq_readout(self, output: Any, *, means: Any, frequency: Any, receiver: Any,
                   noise_frequencies: Any = None) -> Any:
        """Propagate conditional coherent boundary fields through captured output wiring.

        means gives one noiseless complex field per outcome, in 1/sqrt(ns), at
        the selected Markov boundary channel. A mapping gives fields at several
        boundary channels. Unspecified fields are vacuum. The output line adds
        its resolved gain, filter loss noise, and amplifier noise. The given
        means, returned IQ, and receiver transfers follow the VNA engineering
        convention. Conjugate ``output(...).raw_amplitude`` before you use its
        stationary value as a boundary template. This stationary coherent-state
        readout model excludes quantum-device correlations and occupied boundary
        inputs. Use calibrated IQReadout distributions when those effects are
        included in a detector calibration.

        Parameters
        ----------
        output : port object or str
            External output reference plane.
        means : array_like or mapping
            Engineering-convention Markov-boundary means in ``1/sqrt(ns)``.
        frequency : scalar
            Carrier frequency in GHz.
        receiver : IQReceiver
            Boxcar receiver and optional digital transfer.
        noise_frequencies : array_like or None, optional
            Two-sided offsets in GHz. ``None`` uses the standard grid.
        """
        if self._readout_wiring is None:
            raise RuntimeError("No captured output wiring is available for this result.")
        return self._readout_wiring.iq_readout(output, means=means, frequency=frequency,
                                             receiver=receiver, noise_frequencies=noise_frequencies)

    def measure(self, *devices: Any, t: Any = None, basis: Any = "energy") -> Any:
        """Measure kept states in local energy bases, without further evolution.

        Pass multiple devices for joint outcomes, t for an exact saved time, or
        basis='solver'. Custom local unitary columns are expressed in the captured
        energy basis of the stored integration frame. Give one matrix for one
        device, or a device mapping. No phase-frame conversion is applied. Samples
        at different times represent independently terminated experiments.

        Parameters
        ----------
        *devices : device object or str
            Measured devices. An empty selection measures all devices.
        t : scalar, array_like, or None, optional
            Saved time or times in ns. ``None`` selects the final state.
        basis : {"energy", "solver"}, array_like, or mapping, default="energy"
            Captured local measurement basis.
        """
        from quchip.results.terminal import measure_result
        return measure_result(self, devices, t=t, basis=basis)

    def reduced_state(self, t: float, device: str | BaseDevice, *, frame: Frame = "integration") -> Any:
        """Partial-trace a saved state down to one device.

        Parameters
        ----------
        t : float
            Saved time in ns.
        device : device object or str
            Device to keep.
        frame : {"integration", "lab"}, default="integration"
            Frame of the returned state.
        """
        dev_idx, _ = self._resolve_device_idx(device)
        state = self._in_frame(*self._saved_state(t, "exact"), frame)
        if self._excitation_space is not None:
            return self._excitation_space.reduced(state, self._backend, (dev_idx,))
        return self._backend.ptrace(state, dev_idx, self.dims)

    # ------------------------------------------------------------------
    # Populations
    # ------------------------------------------------------------------

    @property
    def populations(self) -> dict[tuple[int, ...], Any]:
        """Return per-basis-state populations ``|<n1, n2, ...|psi(t)>|**2`` over time.

        Keys index the solver's product basis, which is not necessarily the
        local energy basis. Each value is a native real array over
        ``self.times``. Use :meth:`population` for isolated energy-level
        occupations.

        This method requires ``states="all"``. For density-matrix trajectories,
        it reads the DM diagonal at each timestep.
        """
        backend = self._backend
        if self._excitation_space is not None:
            from math import prod
            from quchip.utils.memory import require_memory
            require_memory((256 + 8 * len(self.dims) + 16 * len(self.times)) * prod(self.dims),
                           task="Full product population dictionary",
                           remedy="Use population(device, level) instead.")
            states = self._backend.stack_states(self.states)
            values = backend.array_module.real(backend.populations_over_time(states))
            return {label: values[:, i] for i, label in enumerate(self._basis_labels)}
        basis_labels = self._basis_labels

        # One batched diagonal read over the leading time axis -> (T, ∏dims),
        # never building a per-timestep density matrix for ket trajectories.
        all_diags = backend.array_module.real(backend.populations_over_time(self._stacked_states()))
        return {label: all_diags[:, i] for i, label in enumerate(basis_labels)}

    def population(self, device: str | BaseDevice, level: int = 0) -> Any:
        """Return occupation of a captured isolated energy level in native arrays.

        Parameters
        ----------
        device : device object or str
            Captured device.
        level : int, default=0
            Zero-based isolated energy level.
        """
        backend = self._backend
        xp = backend.array_module
        dev_idx, label = self._resolve_device_idx(device)
        dev_dim = self.dims[dev_idx]
        if not (0 <= level < dev_dim):
            raise ValueError(
                f"Level {level} out of range for device '{label}' with {dev_dim} levels (0..{dev_dim - 1})."
            )

        if self._excitation_space is not None:
            diags = backend.populations_over_time(self._stacked_states())
            selected = self._excitation_space.levels[:, dev_idx] == level
            return xp.real(diags @ xp.asarray(selected, dtype=float))
        basis = self._bases.get(label)
        if basis is not None and basis.energy_to_solver() is not None:
            vector = xp.asarray(basis.energy_state(level))
            projector = backend.from_array(xp.outer(vector, vector.conj()), dims=[[dev_dim], [dev_dim]])
            embedded = backend.embed(projector, dev_idx, self.dims)
            return xp.real(backend.expect_over_time(embedded, self._stacked_states()))

        # Marginalize the other devices without enumerating product-basis labels.
        diags = backend.populations_over_time(self._stacked_states())
        axes = tuple(index + 1 for index in range(len(self.dims)) if index != dev_idx)
        marginal = xp.sum(diags.reshape((len(diags), *self.dims)), axis=axes)
        return xp.real(marginal @ xp.asarray(np.arange(dev_dim) == level, dtype=float))

    # ------------------------------------------------------------------
    # Plot shims — delegate to the (lazy) viz module
    # ------------------------------------------------------------------

    def plot_populations(
        self,
        *,
        trace_out: str | BaseDevice | list[str | BaseDevice] | None = None,
        computational: bool = False,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Plot per-basis-state populations over time.

        Parameters
        ----------
        trace_out : device object, str, sequence, or None, optional
            Devices to trace out.
        computational : bool, default=False
            Restrict labels to computational levels.
        ax : matplotlib.axes.Axes or None, optional
            Destination axes.
        **kwargs
            Forwarded to :func:`quchip.viz.plot_populations`.
        """
        from quchip.viz.results import plot_populations

        return plot_populations(self, trace_out=trace_out, computational=computational, ax=ax, **kwargs)

    def plot_state(
        self,
        index: int,
        *,
        trace_out: str | BaseDevice | list[str | BaseDevice] | None = None,
        computational: bool = False,
        mode: StateMode = "population",
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Plot one stored state as populations or a density matrix.

        Parameters
        ----------
        index : int
            Saved-state index.
        trace_out : device object, str, sequence, or None, optional
            Devices to trace out.
        computational : bool, default=False
            Restrict labels to computational levels.
        mode : {"population", "density_matrix"}, default="population"
            Plot representation.
        ax : matplotlib.axes.Axes or None, optional
            Destination axes.
        **kwargs
            Forwarded to :func:`quchip.viz.plot_state`.
        """
        from quchip.viz.results import plot_state

        return plot_state(
            self, index, trace_out=trace_out, computational=computational, mode=mode, ax=ax, **kwargs
        )

    def plot_expectation(self, *, keys: list[Any] | None = None, ax: Any = None, **kwargs: Any) -> Any:
        """Plot expectation-value traces over time.

        Parameters
        ----------
        keys : list or None, optional
            Observable keys. ``None`` plots every captured trace.
        ax : matplotlib.axes.Axes or None, optional
            Destination axes.
        **kwargs
            Forwarded to :func:`quchip.viz.plot_expectation`.
        """
        from quchip.viz.results import plot_expectation

        return plot_expectation(self, keys=keys, ax=ax, **kwargs)

    def plot_wigner(
        self,
        index: int = -1,
        *,
        trace_out: str | BaseDevice | list[str | BaseDevice] | None = None,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the Wigner function of one stored state.

        Parameters
        ----------
        index : int, default=-1
            Saved-state index.
        trace_out : device object, str, sequence, or None, optional
            Devices to trace out before plotting.
        ax : matplotlib.axes.Axes or None, optional
            Destination axes.
        **kwargs
            Forwarded to :func:`quchip.viz.plot_wigner`.
        """
        from quchip.viz.results import plot_wigner

        return plot_wigner(self, index, trace_out=trace_out, ax=ax, **kwargs)

    def check_truncation(self, *, threshold: float = 1e-3) -> dict[Any, Any]:
        """Deprecated method form of :func:`quchip.check_truncation`.

        quchip 0.6 removes this method. Call ``quchip.check_truncation(result)``.

        Parameters
        ----------
        threshold : float, default=1e-3
            Maximum accepted boundary population.
        """
        from quchip.engine.truncation import check_truncation
        from quchip.utils.deprecation import warn_renamed

        warn_renamed(f"{type(self).__name__}.check_truncation()", "quchip.check_truncation(result)", removal="0.6")
        return check_truncation(self, threshold=threshold)

    def __repr__(self) -> str:
        t_min = float(self.times[0]) if len(self.times) > 0 else 0.0
        t_max = float(self.times[-1]) if len(self.times) > 0 else 0.0
        parts = [
            f"SimulationResult(solver={self.solver!r}",
            f"t=[{t_min:.1f}, {t_max:.1f}] ns",
            f"steps={len(self.times)}",
            f"dims={self.dims}",
        ]
        if self._expect_data is not None:
            parts.append(f"expect=dict({len(self._expect_data)} keys)")
        if self._output_data:
            parts.append(f"outputs=dict({len(self._output_data)} keys)")
        return ", ".join(parts) + ")"


class SimulationBatchResult(BatchResult[SimulationResult]):
    """Ordered, immutable batch of :class:`SimulationResult` with stacked helpers.

    Returned by :func:`~quchip.engine.solve_many` and by any sweep that solves
    many problems in one call. The batch keeps the iteration order, so each
    per-element result maps one-to-one onto the input that produced it.

    The ``final_*`` helpers stack along a new leading batch axis in the
    backend's array module. The grid-aware :meth:`expect` / :meth:`population`
    accept ``reduce='last'`` for a final-value slice, so a loss function that
    sums over the batch stays JAX-traceable end-to-end.

    Attributes
    ----------
    results : tuple of SimulationResult
        Per-point results in sweep order.
    shape : tuple of int
        Sweep-grid shape.
    axes : tuple
        Named sweep-axis metadata.
    """

    def measure(self, *devices: Any, t: Any = None, basis: Any = "energy") -> Any:
        """Measure kept states in local energy bases, without further evolution.

        Pass multiple devices for joint outcomes, t for an exact saved time, or
        basis='solver'. Custom local unitary columns are expressed in the captured
        energy basis of the stored integration frame. Give one matrix for one
        device, or a device mapping. No phase-frame conversion is applied. Samples
        at different times represent independently terminated experiments.

        Parameters
        ----------
        *devices : device object or str
            Measured devices. An empty selection measures all devices.
        t : scalar, array_like, or None, optional
            Saved time or times in ns. ``None`` selects final states.
        basis : {"energy", "solver"}, array_like, or mapping, default="energy"
            Captured local measurement basis.
        """
        from quchip.results.terminal import measure_result
        return measure_result(self, devices, t=t, basis=basis)

    def _require_shared_times(self, values: Any) -> Any:
        from quchip.results._time import require_valid

        if not self._results:
            raise RuntimeError("Empty batch has no time coordinates.")
        xp = self.backend.array_module
        first = self._results[0].times
        mismatch = xp.asarray(False)
        message = "Batch points have different time grids; inspect individual results or reduce each trace."
        for result in self._results[1:]:
            if result.times.shape != first.shape:
                raise ValueError(message)
            mismatch = mismatch | xp.any(result.times != first)
        return require_valid(values, mismatch, message)

    @property
    def times(self) -> Any:
        """Return shared native coordinates, or tell why time grids are incompatible."""
        return self._require_shared_times(self._results[0].times if self._results else None)

    def _trace_values(self, values: list[Any], reduce: str | None) -> Any:
        if reduce is None:
            values = self._require_shared_times(values)
        return self._reshape([self._reduce_time_axis(value, reduce) for value in values])

    def _check_targets_len(self, targets: list[Any] | tuple[Any, ...]) -> None:
        if len(targets) != len(self._results):
            raise ValueError(f"Expected {len(self._results)} targets, got {len(targets)}.")

    def _reduce_time_axis(self, values: Any, reduce: str | None) -> Any:
        if reduce is None:
            return values
        xp = self.backend.array_module
        if reduce == "last":
            return values[..., -1]
        if reduce == "max":
            return xp.max(values, axis=-1)
        if reduce == "mean":
            return xp.mean(values, axis=-1)
        raise ValueError("reduce must be one of None, 'last', 'max', or 'mean'.")

    def expect(self, key: Any, index: int | None = None, *, reduce: str | None = None) -> Any:
        """Return expectation traces on the natural sweep grid.

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, optional
            Entry of a list-valued observable.
        reduce : {None, "last", "max", "mean"}, optional
            Reduction along the time axis.
        """
        return self._trace_values([r.expect(key, index=index) for r in self._results], reduce)

    def output(self, exposure: Any) -> OutputFieldTrace:
        """Return one complete field trace on the natural sweep grid.

        Parameters
        ----------
        exposure : port object or str
            External output reference plane.
        """
        traces = [result.output(exposure) for result in self._results]
        if not traces:
            raise RuntimeError("Empty batch has no output fields.")
        times, amplitude, flux, raw_amplitude, raw_flux = self._require_shared_times((
            traces[0].times, [trace.amplitude for trace in traces], [trace.photon_flux for trace in traces],
            [trace.raw_amplitude for trace in traces], [trace.raw_photon_flux for trace in traces],
        ))
        return OutputFieldTrace(
            exposure=traces[0].exposure,
            times=times,
            amplitude=self._reshape(amplitude),
            photon_flux=self._reshape(flux),
            raw_amplitude=self._reshape(raw_amplitude),
            raw_photon_flux=self._reshape(raw_flux),
        )

    def population(self, device: str | BaseDevice, level: int = 0, *, reduce: str | None = None) -> Any:
        """Return population traces on the natural sweep grid.

        Parameters
        ----------
        device : device object or str
            Captured device.
        level : int, default=0
            Zero-based isolated energy level.
        reduce : {None, "last", "max", "mean"}, optional
            Reduction along the time axis.
        """
        return self._trace_values([r.population(device, level) for r in self._results], reduce)

    def jump_rate(self, key: Any, *, reduce: str | None = None) -> Any:
        """Return the jump-rate traces of one channel on the natural sweep grid.

        ``reduce`` accepts ``None``, ``"last"``, ``"max"``, or ``"mean"``
        and acts on the time axis.

        Parameters
        ----------
        key : channel object or str
            Resolved channel key.
        reduce : {None, "last", "max", "mean"}, optional
            Reduction along the time axis.
        """
        return self._trace_values([r.jump_rate(key) for r in self._results], reduce)

    def collapse_flux(self, key: Any, *, reduce: str | None = None) -> Any:
        """Deprecated alias for :meth:`jump_rate`.

        Parameters
        ----------
        key : channel object or str
            Resolved channel key.
        reduce : {None, "last", "max", "mean"}, optional
            Reduction along the time axis.
        """
        from quchip.utils.deprecation import warn_renamed

        warn_renamed("collapse_flux()", "jump_rate()")
        return self.jump_rate(key, reduce=reduce)

    def collapse_integral(self, key: Any, *, reduce: str | None = None) -> Any:
        """Return the cumulative jump counts of one channel on the natural sweep grid.

        ``reduce`` accepts ``None``, ``"last"``, ``"max"``, or ``"mean"``
        and acts on the time axis. ``reduce="last"`` returns the expected
        number of jumps in each solve.

        Parameters
        ----------
        key : channel object or str
            Resolved channel key.
        reduce : {None, "last", "max", "mean"}, optional
            Reduction along the time axis.
        """
        return self._trace_values([r.collapse_integral(key) for r in self._results], reduce)

    def _final_projections(self, targets: list[Any] | tuple[Any, ...], *, amplitude: bool, frame: Frame,
                           dressed: bool) -> Any:
        self._check_targets_len(targets)
        return self._stack([
            result._final_projection(target, amplitude=amplitude, frame=frame, dressed=dressed)
            for result, target in zip(self._results, targets)
        ])

    def final_overlap_magnitudes(self, targets: list[Any] | tuple[Any, ...], *, frame: Frame = "integration",
                                 dressed: bool = False) -> Any:
        """Return final target-state populations without histories.

        Targets, ``frame`` and ``dressed`` are as in :meth:`SimulationResult.overlap`.

        Parameters
        ----------
        targets : sequence of state_like, mapping, or str
            One target ket or label per batch result.
        frame : {"integration", "lab"}, default="integration"
            Frame in which a ket target is fixed.
        dressed : bool, default=False
            Use the dressed eigenstate of a label target.
        """
        return self._final_projections(targets, amplitude=False, frame=frame, dressed=dressed)

    def final_amplitudes(self, targets: list[Any] | tuple[Any, ...], *, frame: Frame = "integration",
                         dressed: bool = False) -> Any:
        """Return final complex ket amplitudes without histories.

        Targets, ``frame`` and ``dressed`` are as in :meth:`SimulationResult.amplitude_array`.

        Parameters
        ----------
        targets : sequence of state_like, mapping, or str
            One target ket or label per batch result.
        frame : {"integration", "lab"}, default="integration"
            Frame in which a ket target is fixed.
        dressed : bool, default=False
            Use the dressed eigenstate of a label target.
        """
        return self._final_projections(targets, amplitude=True, frame=frame, dressed=dressed)

# ---------------------------------------------------------------------------
# Result wrapping helpers
# ---------------------------------------------------------------------------


def wrap_solver_result(solver_result: SolverResult, problem: SolveProblem, backend: Backend, *,
                       label_targets: Any = None) -> SimulationResult:
    """Wrap a raw backend :class:`SolverResult` into a user-facing :class:`SimulationResult`.

    The engine-side :class:`~quchip.engine.ir.SolveProblem` carries the
    metadata needed to rebuild dict-form observables
    (:func:`~quchip.engine.observables.build_observable_traces`) and to label
    devices for partial-trace helpers.

    Parameters
    ----------
    solver_result : SolverResult
        Raw backend solve output.
    problem : SolveProblem
        Frozen request carrying result metadata.
    backend : Backend
        Backend that owns the native states and arrays.
    label_targets : LabelTargets or None, optional
        Record captured when the solve finished. None captures it now.
    """
    if label_targets is None:
        label_targets = capture_label_targets(problem, {})
    if solver_result.native is not None:
        from quchip.results.trajectories import TrajectoryResult

        return TrajectoryResult(solver_result.native, problem, label_targets=label_targets)
    from quchip.engine.truncation import boundary_traces

    plan = problem.truncation
    samples = None
    if plan is not None and plan.sampled:
        raw = solver_result.expect
        flat = list(raw.values()) if isinstance(raw, dict) else list(raw or ())
        count = len(plan.operators)
        diagnostic = flat[-count:] if count else ()
        user_values = flat[:-count] if count else flat
        samples = boundary_traces(plan, diagnostic, problem.tlist, backend)
        solver_result = replace(solver_result, expect=user_values or None)
    engine_result = problem.engine_result
    observable_traces = None
    output_traces = None
    if problem.e_ops_meta is not None:
        from quchip.engine.observables import build_observable_traces

        observable_traces, output_traces = build_observable_traces(
            solver_result,
            problem.tlist,
            dict_meta=problem.e_ops_meta,
            resolved_frame=problem.resolved_frame,
            engine_result=engine_result,
        )
    from quchip.results.terminal import ReadoutWiring

    result = SimulationResult(
        solver_result=solver_result,
        backend=backend,
        dims=engine_result.dims if engine_result.excitation_space is None else engine_result.excitation_space.dims,
        excitation_space=engine_result.excitation_space,
        device_info=problem.device_info,
        observable_traces=observable_traces,
        output_traces=output_traces,
        channels=engine_result.slh.channels if engine_result.dissipation else (),
        bases=engine_result.bases,
        dissipation=engine_result.dissipation,
        readout_wiring=ReadoutWiring.capture(engine_result.slh, backend.array_module),
        frame=problem.resolved_frame,
        label_targets=label_targets,
    )
    result._truncation = plan
    result._boundary_traces = samples
    result.stats["states"] = problem.states
    result.stats["excitation_reduction"] = engine_result.metadata.get("excitation_reduction")
    return result


def wrap_solver_results_from_batch(
    solver_results: list[SolverResult],
    batch: SolveBatch,
    backend: Backend,
) -> list[SimulationResult]:
    """Wrap backend results with each batch point's resolved solve context."""
    if len(solver_results) != batch.batch_size:
        raise RuntimeError(f"Backend returned {len(solver_results)} results for {batch.batch_size} batch points.")
    captured: dict[int, list[tuple[Any, Any, Any]]] = {}
    return [
        wrap_solver_result(solver_result, problem, backend, label_targets=capture_label_targets(problem, captured))
        for solver_result, problem in zip(solver_results, batch.problems)
    ]


def capture_label_targets(problem: SolveProblem, captured: dict[int, list[tuple[Any, Any, Any]]]) -> Any:
    """Capture the label-target record of a solve, once for each chip and approximation of a batch.

    Parameters
    ----------
    problem : SolveProblem
        Finished solve request.
    captured : dict
        ``(chip, approximation, record)`` entries already captured for this batch, by chip identity.
        The function adds new ones.
    """
    from quchip.engine.states import LabelTargets

    if problem.chip is None:
        return None
    approximation = problem.engine_result.approximation
    entries = captured.setdefault(id(problem.chip), [])
    for chip, known, record in entries:
        if chip is problem.chip and known == approximation:
            return record
    record = LabelTargets.capture(problem.chip, approximation)
    entries.append((problem.chip, approximation, record))
    return record


def _frame_choice(frame: Any) -> str:
    if not isinstance(frame, str) or frame not in ("integration", "lab"):
        raise ValueError(f'frame must be "integration" or "lab"; got {frame!r}.')
    return frame


def _moving_projection(backend: Backend, targets: Any, stacked: Any, *, amplitude: bool) -> Any:
    """Project stacked states on one target ket per time.

    Kets give ``⟨v(t)|ψ(t)⟩``, or its squared magnitude when ``amplitude`` is
    False. Density matrices give ``⟨v(t)|ρ(t)|v(t)⟩``.
    """
    xp = backend.array_module
    states = xp.asarray(backend.to_array(stacked), dtype=complex)
    targets = xp.asarray(targets, dtype=complex)
    if backend._is_ket_stack(states):
        values = xp.einsum("ti,ti->t", xp.conj(targets), states[..., 0])
        return values if amplitude else xp.abs(values) ** 2
    return xp.abs(xp.einsum("ti,tij,tj->t", xp.conj(targets), states, targets))
