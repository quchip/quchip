"""Backend-agnostic wrapper around solver output.

The engine wraps each backend's solver output here so callers use the same
state, expectation, partial-trace, population, and batch interfaces.

Numerical population and overlap accessors return NumPy arrays for QuTiP and
JAX arrays for Dynamiqs, including eager calls. Host conversion is explicit.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quchip.backend import Backend, SolverResult
from quchip.devices.base import BaseDevice
from quchip.results._batch import BatchResult
from quchip.utils.labeling import resolve_label

DEFAULT_TRUNCATION_THRESHOLD = 1e-3

# Mirrors quchip.viz.results.StateMode so the plot_state shim can re-expose the
# same literal without a hard import of the lazily loaded viz module.
StateMode = Literal["population", "dm"]

if TYPE_CHECKING:
    from quchip.engine.ir import SLHChannel, SolveBatch, SolveProblem


@dataclass(frozen=True)
class ObservableTrace:
    """One named expectation-value trace, with pre- and post-processing values.

    Attributes
    ----------
    values
        Post-processed expectation values over time (e.g. demodulated,
        phase-corrected, band-summed). This is what user code normally
        wants.
    raw
        The same quantity before post-processing — useful for debugging
        frame conventions, band decomposition, and demodulation.
    """

    values: Any
    raw: Any


@dataclass(frozen=True)
class OutputFieldTrace:
    """Complete transient field reported at one external reference plane.

    ``amplitude`` is the complex mean field ``<b_out>`` and ``photon_flux``
    is the normally ordered ``<b_out dagger b_out>`` in photons/ns.
    ``raw_*`` retain the same moments at the Markov boundary, before propagation
    through the outbound reference run.

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

    Stored states follow the requested storage policy; ``times`` uses ns and
    ``dims`` follows chip order. Dict-form ``e_ops`` become
    :class:`ObservableTrace` entries in ``observable_traces``.

    Parameters
    ----------
    solver_result : SolverResult
        Backend output to wrap.
    backend : Backend
        Backend owning native states and arrays.
    dims : sequence of int
        Hilbert dimensions in chip order.
    device_info, observable_traces, output_traces, channels, bases : optional
        Captured device, observable, output, channel, and basis metadata.
    dissipation : bool, default=True
        Whether collapse channels were included.
    readout_wiring : optional
        Captured downstream readout wiring.
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
    ) -> None:
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
        # Lazily populated caches, built once and reused across accessors; declared
        # here so the cached attributes are explicit rather than first appearing
        # mid-method. ``_stacked_cache`` holds the (T, …) stacked trajectory used by
        # the batched over-time extractors; ``_basis_labels_cache`` holds the
        # full-chip Fock-tuple basis read by the population / truncation helpers.
        self._stacked_cache: Any = None
        self._basis_labels_cache: list[tuple[int, ...]] | None = None

    # ------------------------------------------------------------------
    # Observables — dict-form expectation traces
    # ------------------------------------------------------------------

    @property
    def observable_traces(self) -> dict[Any, ObservableTrace | list[ObservableTrace]] | None:
        """Return the dict of named :class:`ObservableTrace` entries, or ``None``.

        ``None`` when ``e_ops`` was not passed as a dict. Visualization and
        analysis code that wants the full dict should read this property
        rather than the private ``_expect_data`` attribute.
        """
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

        Values have time on their last axis. The result has their leading
        axes followed by the query shape. ``nearest`` selects the earlier
        sample on a tie; ``interpolate`` is linear, including complex values.
        Out-of-interval queries raise. No solve or state interpolation occurs.

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

        The rate is evaluated post-solve from stored states and has units of
        ``1/ns``. ``key`` may be a resolved key from
        :attr:`collapse_channels`, a network port, or a
        ``"<device>.<channel>"`` label that identifies one channel. Address a
        port composed into a network through its external network port.

        At an external network port, this rate equals ``raw_photon_flux`` only with
        vacuum input. Dephasing and thermal-absorption channels can also have
        nonzero jump rates, so this quantity alone is not an excitation-loss
        rate.

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

        This is the cumulative trapezoid integral of :meth:`jump_rate` on
        the result grid. Its last entry is the expected number of jumps over
        the whole solve.

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
        """Return the retained native state history, or explain how to save it."""
        if self._states is None or len(self._states) == 0:
            raise RuntimeError(_NO_STATES_MSG)
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
            cached = self._backend.stack_states(self.states)
            from quchip.utils.jax_utils import contains_tracer

            if not contains_tracer(cached):
                self._stacked_cache = cached
        return cached

    def _is_ket_trajectory(self) -> bool:
        """Return whether the stored trajectory is kets (vs density matrices)."""
        return self._backend.is_ket(self.states[0])

    @property
    def _basis_labels(self) -> list[tuple[int, ...]]:
        """Return the full-chip computational basis as Fock tuples ``(n1, …, nK)`` (cached).

        One entry per basis vector of the whole chip, in ``dims`` order. Built
        once and shared by :attr:`populations`, :meth:`population`, and
        :meth:`check_truncation` rather than each recomputing the product.
        """
        cached = self._basis_labels_cache
        if cached is None:
            cached = list(itertools.product(*[range(d) for d in self.dims]))
            self._basis_labels_cache = cached
        return cached

    def overlap(self, target: Any) -> Any:
        """Return the overlap with *target* at every stored time.

        For ket trajectories returns ``|<target|psi(t)>|**2``; for density
        matrices returns ``<target|rho(t)|target>``. Stays in the backend's
        array module so the result is differentiable. One batched op over the
        leading time axis — no per-point loop.

        Parameters
        ----------
        target : state_like
            Target ket in the captured full Hilbert space.
        """
        backend = self._backend
        target = backend.coerce_state(target, dims=tuple(self.dims))
        projector = backend.matmul(target, backend.dag(target))
        series = backend.expect_over_time(projector, self._stacked_states())
        return backend.array_module.abs(series)

    def amplitude_array(self, target: Any) -> Any:
        """Return the phase-sensitive complex projection ``<target|psi(t)>`` for kets.

        Density-matrix trajectories raise :class:`TypeError` — there is no
        single phase-sensitive amplitude for a mixed state; use
        :meth:`overlap` instead. One batched op, no per-point loop.

        Parameters
        ----------
        target : state_like
            Target ket in the captured full Hilbert space.
        """
        backend = self._backend
        if not self._is_ket_trajectory():
            raise TypeError(
                "amplitude_array() requires ket trajectories; use overlap() for density matrices."
            )
        target = backend.coerce_state(target, dims=tuple(self.dims))
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

    def state(self, t: float | None = None, *, dm: bool = False) -> Any:
        """Return a retained state, optionally promoted to a density matrix.

        Parameters
        ----------
        t : float or None, optional
            Saved time in ns; ``None`` selects the final state.
        dm : bool, default=False
            Promote a ket to a density matrix.
        """
        s = self.final_state if t is None else self.state_at(t)
        if dm and self._backend.is_ket(s):
            return self._backend.state_to_dm(s)
        return s

    def state_at(self, t: Any, *, method: str = "exact") -> Any:
        """Return a retained state at a scalar time; states are never interpolated.

        Parameters
        ----------
        t : scalar
            Query time in ns.
        method : {"exact", "nearest"}, default="exact"
            Time-selection rule.
        """
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
            return require_valid(self.final_state, index != len(self.times) - 1, _NO_STATES_MSG)
        if contains_tracer(index):
            return self._stacked_states()[index]
        return self._states[int(index)]

    def dm_at(self, t: float, *, method: str = "exact") -> Any:
        """Return a density matrix at a retained time, promoting kets on demand.

        Parameters
        ----------
        t : float
            Query time in ns.
        method : {"exact", "nearest"}, default="exact"
            Time-selection rule.
        """
        state = self.state_at(t, method=method)
        return self._backend.state_to_dm(state) if self._backend.is_ket(state) else state

    @property
    def final_state(self) -> Any:
        """Return the last history entry or the separately retained final state."""
        if self._states is not None and len(self._states) > 0:
            return self._states[-1]
        if self._final_state is not None:
            return self._final_state
        raise RuntimeError('No final state available; run with states="all" or states="final" to retain it.')

    def iq_readout(self, output: Any, *, means: Any, frequency: Any, receiver: Any,
                   noise_frequencies: Any = None) -> Any:
        """Propagate conditional coherent boundary fields through captured output wiring.

        means gives one noiseless complex field per outcome, in 1/sqrt(ns),
        at the selected Markov boundary channel. A mapping supplies fields at
        several boundary channels; unspecified fields are vacuum. The output
        line adds its resolved gain, filter loss noise, and amplifier noise.
        This stationary coherent-state readout model excludes quantum-device
        correlations and occupied boundary inputs. Use calibrated IQReadout
        distributions when those effects are included in a detector calibration.

        Parameters
        ----------
        output : port object or str
            External output reference plane.
        means : array_like or mapping
            Conditional Markov-boundary means in ``1/sqrt(ns)``.
        frequency : scalar
            Carrier frequency in GHz.
        receiver : IQReceiver
            Boxcar receiver and optional digital transfer.
        noise_frequencies : array_like or None, optional
            Two-sided offsets in GHz; ``None`` uses the standard grid.
        """
        if self._readout_wiring is None:
            raise RuntimeError("No captured output wiring is available for this result.")
        return self._readout_wiring.iq_readout(output, means=means, frequency=frequency,
                                             receiver=receiver, noise_frequencies=noise_frequencies)

    def measure(self, *devices: Any, t: Any = None, basis: Any = "energy") -> Any:
        """Measure retained states in local energy bases, without further evolution.

        Pass multiple devices for joint outcomes, t for an exact saved time,
        or basis='solver'. Custom local unitary columns are expressed in the
        captured energy basis of the stored integration frame: one matrix for
        one device, or a device mapping. No phase-frame conversion is applied.
        Samples at different times represent independently terminated experiments.

        Parameters
        ----------
        *devices : device object or str
            Measured devices; an empty selection measures all devices.
        t : scalar, array_like, or None, optional
            Saved time or times in ns; ``None`` selects the final state.
        basis : {"energy", "solver"}, array_like, or mapping, default="energy"
            Captured local measurement basis.
        """
        from quchip.results.terminal import measure_result
        return measure_result(self, devices, t=t, basis=basis)

    def reduced_state(self, t: float, device: str | BaseDevice) -> Any:
        """Partial-trace a saved state down to one device.

        Parameters
        ----------
        t : float
            Saved time in ns.
        device : device object or str
            Device to retain.
        """
        dev_idx, _ = self._resolve_device_idx(device)
        return self._backend.ptrace(self.state_at(t), dev_idx, self.dims)

    # ------------------------------------------------------------------
    # Populations
    # ------------------------------------------------------------------

    @property
    def populations(self) -> dict[tuple[int, ...], Any]:
        """Return per-basis-state populations ``|<n1, n2, ...|psi(t)>|**2`` over time.

        Keys index the solver's product basis, which need not be the local
        energy basis. Each value is a native real array over ``self.times``.
        Use :meth:`population` for isolated energy-level occupations.

        Requires ``states="all"``; density-matrix trajectories are handled
        transparently by reading the diagonal of each timestep's DM.
        """
        backend = self._backend
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

        basis = self._bases.get(label)
        if basis is not None and basis.energy_to_solver() is not None:
            vector = xp.asarray(basis.energy_state(level))
            projector = backend.from_array(xp.outer(vector, vector.conj()), dims=[[dev_dim], [dev_dim]])
            embedded = backend.embed(projector, dev_idx, self.dims)
            return xp.real(backend.expect_over_time(embedded, self._stacked_states()))

        # Full-chip diagonal populations (T, ∏dims) in one batched op, then sum
        # the basis states whose Fock index on *device* equals *level* — the
        # marginal P(level) without a per-point ptrace/expect loop.
        diags = backend.populations_over_time(self._stacked_states())
        basis_labels = self._basis_labels
        select = xp.asarray(
            np.array([1.0 if tup[dev_idx] == level else 0.0 for tup in basis_labels], dtype=float)
        )
        return xp.real(diags @ select)

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
            Observable keys; ``None`` plots every captured trace.
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

    def check_truncation(self, *, threshold: float = DEFAULT_TRUNCATION_THRESHOLD) -> dict[str, Any]:
        """Report each device's maximum boundary population over available samples.

        This warning heuristic can miss excursions between samples. Increase the
        relevant cutoff and compare observables to establish convergence. Native
        arrays remain differentiable; warning thresholds are evaluated only for
        concrete results. Boundaries are declared by the captured component model.

        Parameters
        ----------
        threshold : float, default=1e-3
            Maximum accepted boundary population.
        """
        from quchip.engine.truncation import evaluate_boundaries
        from quchip.utils.jax_utils import contains_tracer

        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("truncation threshold must be finite and nonnegative")
        plan, traces = evaluate_boundaries(self)
        for reason in plan.unavailable:
            warnings.warn(f"Truncation diagnostic unavailable: {reason}", UserWarning, stacklevel=2)
        xp = self._backend.array_module
        observed: dict[str, Any] = {}
        maxima = tuple(xp.maximum(0.0, xp.max(trace)) for trace in traces)
        traced = contains_tracer(maxima)
        if traced:
            warnings.warn(
                "Truncation boundary populations are traced; warning thresholds cannot be evaluated here. "
                "Check the concrete result afterward.",
                UserWarning, stacklevel=2,
            )
        for check, maximum in zip(plan.checks, maxima, strict=True):
            observed[check.label] = xp.maximum(observed.get(check.label, 0.0), maximum)
            if not traced and float(maximum) > threshold:
                warnings.warn(
                    f"Device {check.label!r}: maximum sampled {check.boundary.description} population "
                    f"({self.stats.get('truncation_sampling', 'saved time grid')}) "
                    f"{float(maximum):.3g} > threshold {threshold:.3g}. "
                    f"{check.boundary.convergence_hint} This is a sampling heuristic, not an error bound.",
                    UserWarning, stacklevel=2,
                )
        return observed

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

    Returned by :func:`~quchip.engine.solve_many` and by any sweep that
    solves many problems in one call. The batch preserves iteration order
    so that per-element results map one-to-one onto the inputs that
    produced them.

    The ``final_*`` helpers stack along a new leading batch axis in the
    backend's array module, and the grid-aware :meth:`expect` /
    :meth:`population` accept ``reduce='last'`` for a final-value slice, so
    a loss function that sums over the batch stays JAX-traceable end-to-end.

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
        """Measure retained states in local energy bases, without further evolution.

        Pass multiple devices for joint outcomes, t for an exact saved time,
        or basis='solver'. Custom local unitary columns are expressed in the
        captured energy basis of the stored integration frame: one matrix for
        one device, or a device mapping. No phase-frame conversion is applied.
        Samples at different times represent independently terminated experiments.

        Parameters
        ----------
        *devices : device object or str
            Measured devices; an empty selection measures all devices.
        t : scalar, array_like, or None, optional
            Saved time or times in ns; ``None`` selects final states.
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
        """Return shared native coordinates, or explain incompatible time grids."""
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
        """Return one channel's jump-rate traces on the natural sweep grid.

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
        """Return one channel's cumulative jump counts on the natural sweep grid.

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

    def _final_projections(self, targets: list[Any] | tuple[Any, ...], *, amplitude: bool) -> Any:
        self._check_targets_len(targets)
        values = []
        for result, target in zip(self._results, targets):
            backend = result._backend
            state = result.final_state
            is_ket = backend.is_ket(state)
            if amplitude and not is_ket:
                raise TypeError(
                    "Final amplitudes require ket states; use final_overlap_magnitudes for density matrices."
                )
            target = backend.coerce_state(target, dims=tuple(result.dims))
            value = backend.overlap_over_time(target, backend.stack_states([state]))[0]
            if not amplitude:
                value = backend.array_module.abs(value) ** (2 if is_ket else 1)
            values.append(value)
        return self._stack(values)

    def final_overlap_magnitudes(self, targets: list[Any] | tuple[Any, ...]) -> Any:
        """Return final target-state populations without requiring histories.

        Parameters
        ----------
        targets : sequence of state_like
            One target ket per batch result.
        """
        return self._final_projections(targets, amplitude=False)

    def final_amplitudes(self, targets: list[Any] | tuple[Any, ...]) -> Any:
        """Return final complex ket amplitudes without requiring histories.

        Parameters
        ----------
        targets : sequence of state_like
            One target ket per batch result.
        """
        return self._final_projections(targets, amplitude=True)

# ---------------------------------------------------------------------------
# Result wrapping helpers
# ---------------------------------------------------------------------------


def _wrap(
    solver_result: SolverResult,
    backend: Backend,
    *,
    device_info: tuple[tuple[str, bool], ...],
    tlist: Any,
    e_ops_meta: Any,
    resolved_frame: Any,
    engine_result: Any,
) -> SimulationResult:
    observable_traces = None
    output_traces = None
    if e_ops_meta is not None:
        from quchip.engine.observables import build_observable_traces

        observable_traces, output_traces = build_observable_traces(
            solver_result,
            tlist,
            dict_meta=e_ops_meta,
            resolved_frame=resolved_frame,
            engine_result=engine_result,
        )
    from quchip.analysis.field_noise import ReadoutWiring

    return SimulationResult(
        solver_result=solver_result,
        backend=backend,
        dims=engine_result.dims,
        device_info=device_info,
        observable_traces=observable_traces,
        output_traces=output_traces,
        channels=engine_result.slh.channels if engine_result.dissipation else (),
        bases=engine_result.bases,
        dissipation=engine_result.dissipation,
        readout_wiring=ReadoutWiring.capture(engine_result.slh, backend.array_module),
    )


def wrap_solver_result(solver_result: SolverResult, problem: SolveProblem, backend: Backend) -> SimulationResult:
    """Wrap a raw backend :class:`SolverResult` into a user-facing :class:`SimulationResult`.

    The engine-side :class:`~quchip.engine.ir.SolveProblem` carries the
    metadata needed to rebuild dict-form observables
    (:func:`~quchip.engine.observables.build_observable_traces`)
    and to label devices for partial-trace helpers.

    Parameters
    ----------
    solver_result : SolverResult
        Raw backend solve output.
    problem : SolveProblem
        Frozen request carrying result metadata.
    backend : Backend
        Backend that owns the native states and arrays.
    """
    if solver_result.native is not None:
        from quchip.results.trajectories import TrajectoryResult

        return TrajectoryResult(solver_result.native, problem)
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
    result = _wrap(
        solver_result,
        backend,
        device_info=problem.device_info,
        tlist=problem.tlist,
        e_ops_meta=problem.e_ops_meta,
        resolved_frame=problem.resolved_frame,
        engine_result=problem.engine_result,
    )
    result._truncation = plan
    result._boundary_traces = samples
    result.stats["states"] = problem.states
    return result


def wrap_solver_results_from_batch(
    solver_results: list[SolverResult],
    batch: SolveBatch,
    backend: Backend,
) -> list[SimulationResult]:
    """Wrap backend results using each batch point's resolved solve context."""
    if len(solver_results) != batch.batch_size:
        raise RuntimeError(f"Backend returned {len(solver_results)} results for {batch.batch_size} batch points.")
    return [
        wrap_solver_result(solver_result, problem, backend)
        for solver_result, problem in zip(solver_results, batch.problems)
    ]
