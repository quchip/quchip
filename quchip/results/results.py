"""Backend-agnostic wrapper around solver output.

This module is the contract between backends and users. The engine emits
backend-agnostic physics descriptions, and each backend converts them into
its own optimal solver-ready form; the *result* of solving is then
re-wrapped here so that user code never has to know whether QuTiP or
dynamiqs/JAX produced the numbers.

Key classes:

- :class:`SimulationResult` — one solve's states, times, expectations,
  and partial-trace / population accessors.
- :class:`SimulationBatchResult` — ordered, immutable batch of results
  with helpers that stack across the batch for e.g. sweeps.
- :class:`ObservableTrace` — one named expectation-value trace, keeping
  the pre-processing and post-processing values side by side (band
  reconstruction, demodulation, etc.).

All helpers that return arrays stay inside the backend's array module
(JAX / NumPy) to preserve differentiability. The convenience wrappers
:meth:`SimulationResult.overlap` and :meth:`SimulationResult.population`
additionally materialize to a concrete :class:`numpy.ndarray` when the
underlying values are concrete (e.g. the QuTiP backend, or an eager
dynamiqs call) — but return the backend-native array unchanged when
concretization would break differentiability (a traced value under
``jax.jit``/``grad`` on the dynamiqs backend).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quchip.backend import Backend, SolverResult
from quchip.devices.base import BaseDevice
from quchip.utils.labeling import resolve_label

DEFAULT_TRUNCATION_THRESHOLD = 1e-3

# Mirrors quchip.viz.results.StateMode so the plot_state shim can re-expose the
# same literal without a hard import of the lazily loaded viz module.
StateMode = Literal["population", "dm"]

if TYPE_CHECKING:
    from quchip.engine.ir import SolveBatch, SolveProblem


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


_NO_STATES_MSG = "No states stored — pass options={'store_states': True} to the solver."


class SimulationResult:
    """Backend-agnostic container for the output of one solve.

    A :class:`SimulationResult` bundles everything a user needs after a
    solve finishes:

    - ``times`` — the time grid the solver stored (ns).
    - ``states`` — the list of stored states (kets or density matrices),
      or ``None`` if ``store_states`` was off.
    - ``solver`` — the name reported by the backend.
    - ``stats`` — a plain ``dict`` of solver statistics.
    - ``dims`` — the per-device Hilbert-space dimensions, in chip order.
    - ``device_info`` — ``[(label, computational), …]`` for partial-trace
      helpers to resolve device indices by label or by object.

    Expectation values live in ``observable_traces`` when ``e_ops`` was
    passed as a dict. Each entry is an :class:`ObservableTrace` or a list
    of them (list-valued observables, one per band, etc.).
    """

    def __init__(
        self,
        solver_result: SolverResult,
        backend: Backend,
        dims: list[int],
        *,
        device_info: list[tuple[str, bool]] | None = None,
        observable_traces: dict[Any, ObservableTrace | list[ObservableTrace]] | None = None,
    ) -> None:
        self._backend = backend
        self.times = backend.array_module.asarray(solver_result.times, dtype=float)
        self.states = solver_result.states
        self._expect_data: dict[Any, ObservableTrace | list[ObservableTrace]] | None = observable_traces
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
        """Return the full expectation-value array for observable *key* over ``self.times``."""
        return self._resolve_trace(key, index).values

    def expect_final(self, key: Any, index: int | None = None) -> Any:
        """Return the final expectation value for observable *key* (``self.expect(key)[-1]``)."""
        return self._resolve_trace(key, index).values[-1]

    # Alias: reads nicely at call sites that say "give me the values array".
    expect_values = expect

    # ------------------------------------------------------------------
    # States, partial traces, overlaps
    # ------------------------------------------------------------------

    def _require_states(self) -> list[Any]:
        if self.states is None:
            raise RuntimeError(_NO_STATES_MSG)
        return list(self.states)

    def _stacked_states(self) -> Any:
        """Return the trajectory as one ``(T, …)`` native stacked array (cached).

        This is the batched form the over-time extractors consume — built
        once per result. On dynamiqs it is the solver's own stacked
        ``QArray`` (no copy); on QuTiP it is a single ``np.stack`` of the
        stored ``Qobj``s. Keeping it as one array (rather than ``T`` separate
        objects) is what collapses the per-point extractor loop.
        """
        if self.states is None:
            raise RuntimeError(_NO_STATES_MSG)
        cached = getattr(self, "_stacked_cache", None)
        if cached is None:
            cached = self._backend.stack_states(self.states)
            self._stacked_cache = cached
        return cached

    def _is_ket_trajectory(self) -> bool:
        """Return whether the stored trajectory is kets (vs density matrices)."""
        return self._backend.is_ket(self._require_states()[0])

    @property
    def _basis_labels(self) -> list[tuple[int, ...]]:
        """Return the full-chip computational basis as Fock tuples ``(n1, …, nK)`` (cached).

        One entry per basis vector of the whole chip, in ``dims`` order. Built
        once and shared by :attr:`populations`, :meth:`population_array`, and
        :meth:`check_truncation` rather than each recomputing the product.
        """
        cached = self._basis_labels_cache
        if cached is None:
            cached = list(itertools.product(*[range(d) for d in self.dims]))
            self._basis_labels_cache = cached
        return cached

    @staticmethod
    def _as_numpy_or_native(arr: Any) -> Any:
        """Coerce *arr* to a real NumPy array, else return it unchanged.

        Under the QuTiP backend the values are concrete and become a
        ``numpy.ndarray``; under JAX (dynamiqs) a traced array cannot be cast
        to ``float`` during ``jit``/``grad``, so the backend-native array is
        returned verbatim to keep the call differentiable.
        """
        try:
            return np.asarray(arr, dtype=float)
        except Exception:
            return arr

    def overlap_array(self, target: Any) -> Any:
        """Return the overlap with *target* at every stored time.

        For ket trajectories returns ``|<target|psi(t)>|**2``; for density
        matrices returns ``<target|rho(t)|target>``. Stays in the backend's
        array module so the result is differentiable. One batched op over the
        leading time axis — no per-point loop.
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
        :meth:`overlap_array` instead. One batched op, no per-point loop.
        """
        backend = self._backend
        if not self._is_ket_trajectory():
            raise TypeError(
                "amplitude_array() requires ket trajectories; use overlap_array() for density matrices."
            )
        target = backend.coerce_state(target, dims=tuple(self.dims))
        return backend.array_module.asarray(
            backend.overlap_over_time(target, self._stacked_states())
        )

    def overlap(self, target: Any) -> Any:
        """Wrap :meth:`overlap_array` for convenience.

        Returns a NumPy array under the QuTiP backend; under JAX (dynamiqs)
        returns the backend-native array so the call stays JIT/grad-friendly.
        """
        return self._as_numpy_or_native(self.overlap_array(target))

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
        """Return the state at time *t* (or final state if *t* is ``None``); optionally coerced to a DM."""
        s = self.final_state if t is None else self.state_at(t)
        if dm and self._backend.is_ket(s):
            return self._backend.state_to_dm(s)
        return s

    def state_at(self, t: float) -> Any:
        """Return the state at the stored time nearest to *t* (ns)."""
        states = self._require_states()
        idx = int(np.argmin(np.abs(self.times - t)))
        return states[idx]

    def dm_at(self, t: float) -> Any:
        """Return the density matrix at the stored time nearest to *t* (ns) — promotes kets on demand."""
        return self.state(t, dm=True)

    @property
    def final_state(self) -> Any:
        """Return the final state — explicit ``final_state`` if stored, else the last stored trajectory entry."""
        if self._final_state is not None:
            return self._final_state
        if self.states is not None and len(self.states) > 0:
            return self.states[-1]
        raise RuntimeError(
            "No final state available — pass options={'store_final_state': True} or {'store_states': True}"
        )

    def reduced_state(self, t: float, device: str | BaseDevice) -> Any:
        """Partial-trace the state at time *t* down to *device*'s subspace."""
        dev_idx, _ = self._resolve_device_idx(device)
        return self._backend.ptrace(self.state_at(t), dev_idx, self.dims)

    # ------------------------------------------------------------------
    # Populations
    # ------------------------------------------------------------------

    @property
    def populations(self) -> dict[tuple[int, ...], np.ndarray]:
        """Return per-basis-state populations ``|<n1, n2, ...|psi(t)>|**2`` over time.

        Returns a dict keyed by Fock tuple ``(n1, n2, ..., nK)`` — one per
        computational basis vector of the full chip — mapping to a real
        ``numpy.ndarray`` of length ``len(self.times)``.

        Requires ``store_states``; density-matrix trajectories are handled
        transparently by reading the diagonal of each timestep's DM.
        """
        backend = self._backend
        self._require_states()
        basis_labels = self._basis_labels

        # One batched diagonal read over the leading time axis -> (T, ∏dims),
        # never building a per-timestep density matrix for ket trajectories.
        all_diags = np.asarray(
            backend.populations_over_time(self._stacked_states()), dtype=float
        )
        return {label: all_diags[:, i] for i, label in enumerate(basis_labels)}

    def population_array(self, device: str | BaseDevice, level: int = 0) -> Any:
        """Return the population of Fock *level* on *device* over time, in the backend's array module."""
        backend = self._backend
        xp = backend.array_module
        dev_idx, label = self._resolve_device_idx(device)
        dev_dim = self.dims[dev_idx]
        if not (0 <= level < dev_dim):
            raise ValueError(
                f"Level {level} out of range for device '{label}' with {dev_dim} levels (0..{dev_dim - 1})."
            )

        # Full-chip diagonal populations (T, ∏dims) in one batched op, then sum
        # the basis states whose Fock index on *device* equals *level* — the
        # marginal P(level) without a per-point ptrace/expect loop.
        diags = backend.populations_over_time(self._stacked_states())
        basis_labels = self._basis_labels
        select = xp.asarray(
            np.array([1.0 if tup[dev_idx] == level else 0.0 for tup in basis_labels], dtype=float)
        )
        return xp.real(diags @ select)

    def population(self, device: str | BaseDevice, level: int = 0) -> Any:
        """Wrap :meth:`population_array` for convenience.

        Returns a NumPy array under the QuTiP backend; under JAX (dynamiqs)
        returns the backend-native array so the call stays JIT/grad-friendly.
        Use :meth:`population_array` directly when you want to keep gradient
        flow regardless of context.
        """
        return self._as_numpy_or_native(self.population_array(device, level))

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
        """Plot per-basis-state populations over time (delegates to :func:`quchip.viz.plot_populations`)."""
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
        """Plot one stored state as populations or a density matrix (delegates to :func:`quchip.viz.plot_state`)."""
        from quchip.viz.results import plot_state

        return plot_state(
            self, index, trace_out=trace_out, computational=computational, mode=mode, ax=ax, **kwargs
        )

    def plot_expectation(self, *, keys: list[Any] | None = None, ax: Any = None, **kwargs: Any) -> Any:
        """Plot expectation-value traces over time (delegates to :func:`quchip.viz.plot_expectation`)."""
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
        """Plot the Wigner function of one stored state (delegates to :func:`quchip.viz.plot_wigner`)."""
        from quchip.viz.results import plot_wigner

        return plot_wigner(self, index, trace_out=trace_out, ax=ax, **kwargs)

    def check_truncation(
        self,
        *,
        threshold: float = DEFAULT_TRUNCATION_THRESHOLD,
        top_levels: int = 1,
    ) -> dict[str, float]:
        """Emit a ``UserWarning`` per device whose top-``top_levels`` population exceeds *threshold*.

        Uses the final stored state only — cheap, one full-chip diagonal
        read, no per-timestep loop, no partial traces. Silently no-ops
        when no final state is available (e.g. ``store_states`` and
        ``store_final_state`` both disabled).

        Returns the per-device top-level population actually observed,
        keyed by device label, so callers can surface the numbers without
        re-parsing the warning text.
        """
        if self.device_info is None:
            return {}
        try:
            final = self.final_state
        except RuntimeError:
            return {}

        # The check reads the final state into NumPy to inspect Fock-level
        # populations, which concretizes a traced value. As the default-on
        # safety net now fires on every solve path, no-op when the state is
        # traced so JIT/grad through a solve stays intact.
        from quchip.utils.jax_utils import contains_tracer

        if contains_tracer(final):
            return {}

        backend = self._backend
        dm = backend.as_density_matrix(final)
        diag = np.real(np.diag(np.asarray(backend.to_array(dm), dtype=complex)))

        basis_labels = self._basis_labels
        observed: dict[str, float] = {}
        for dev_idx, (label, _computational) in enumerate(self.device_info):
            dev_dim = self.dims[dev_idx]
            k = min(top_levels, dev_dim)
            top_range = range(dev_dim - k, dev_dim)
            pop = float(sum(diag[i] for i, tup in enumerate(basis_labels) if tup[dev_idx] in top_range))
            observed[label] = pop
            if pop > threshold:
                warnings.warn(
                    f"Device '{label}': top-{k} Fock-level population {pop:.3g} > threshold {threshold:.3g}. "
                    f"Consider increasing `levels` to avoid truncation error.",
                    UserWarning,
                    stacklevel=2,
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
        return ", ".join(parts) + ")"


class SimulationBatchResult:
    """Ordered, immutable batch of :class:`SimulationResult` with stacked helpers.

    Returned by :func:`~quchip.engine.solve_many` and by any sweep that
    solves many problems in one call. The batch preserves iteration order
    so that per-element results map one-to-one onto the inputs that
    produced them.

    The ``final_*`` helpers stack along a new leading batch axis in the
    backend's array module, and the grid-aware :meth:`expect` /
    :meth:`population` accept ``reduce='last'`` for a final-value slice, so
    a loss function that sums over the batch stays JAX-traceable end-to-end.
    """

    def __init__(
        self,
        results: list[SimulationResult],
        *,
        shape: tuple[int, ...] | None = None,
        axes: tuple[tuple[str, Any], ...] | None = None,
    ) -> None:
        self._results = tuple(results)
        if shape is None:
            self._shape: tuple[int, ...] = (len(self._results),)
            self._axes = (("batch", tuple(range(len(self._results)))),) if axes is None else tuple(axes)
        else:
            self._shape = tuple(shape)
            self._axes = () if axes is None else tuple(axes)
        if int(np.prod(self._shape, dtype=int)) != len(self._results):
            raise ValueError(
                f"Batch shape {self._shape} has {int(np.prod(self._shape, dtype=int))} points, "
                f"but results contains {len(self._results)} elements."
            )
        if self._axes and len(self._axes) != len(self._shape):
            raise ValueError(f"Expected {len(self._shape)} axis descriptors, got {len(self._axes)}.")

    @property
    def results(self) -> tuple[SimulationResult, ...]:
        """Return the per-element :class:`SimulationResult` objects in input order."""
        return self._results

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the natural sweep-grid shape for this batch."""
        return self._shape

    @property
    def axes(self) -> tuple[tuple[str, Any], ...]:
        """Return sweep-axis metadata as ``(name, values)`` pairs."""
        return self._axes

    @property
    def backend(self) -> Backend:
        """Return the backend shared by every element (raises on an empty batch)."""
        if not self._results:
            raise RuntimeError("Empty batch has no backend.")
        return self._results[0]._backend

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    @staticmethod
    def _axis_member_names(axis_name: Any) -> tuple[str, ...]:
        if isinstance(axis_name, tuple):
            return tuple(str(part) for part in axis_name)
        text = str(axis_name)
        return tuple(part for part in text.split("/") if part)

    def _coordinate_from_dict(self, item: dict[str, int]) -> tuple[int, ...]:
        if not self._axes:
            raise TypeError("Dictionary indexing requires named sweep axes.")

        coord: list[int] = []
        consumed: set[str] = set()
        missing: list[str] = []
        for axis_name, _ in self._axes:
            direct_name = str(axis_name)
            member_names = self._axis_member_names(axis_name)
            provided_names = [name for name in member_names if name in item]

            if direct_name in item:
                provided_names.append(direct_name)

            if not provided_names:
                missing.append("/".join(member_names))
                continue

            provided_indices = {int(item[name]) for name in provided_names}
            if len(provided_indices) != 1:
                raise ValueError(f"Zipped axis {direct_name!r} constituent names must use the same index.")
            coord.append(provided_indices.pop())
            consumed.update(provided_names)

        unknown = sorted(set(item) - consumed)
        if unknown:
            axis_names = [name for axis_name, _ in self._axes for name in self._axis_member_names(axis_name)]
            raise KeyError(f"Unknown sweep axis names {unknown}. Available: {axis_names}")
        if missing:
            raise KeyError(f"Missing sweep axis indices for {missing}.")
        return tuple(coord)

    def __getitem__(self, item: int | slice | dict[str, int]) -> SimulationResult | SimulationBatchResult:
        if isinstance(item, dict):
            coord = self._coordinate_from_dict(item)
            return self._results[int(np.ravel_multi_index(coord, self._shape))]
        if isinstance(item, slice):
            selected = self._results[item]
            indices = tuple(range(*item.indices(len(self._results))))
            return SimulationBatchResult(
                list(selected),
                shape=(len(indices),),
                axes=(("batch", indices),),
            )
        return self._results[item]

    def _check_targets_len(self, targets: list[Any] | tuple[Any, ...]) -> None:
        if len(targets) != len(self._results):
            raise ValueError(f"Expected {len(self._results)} targets, got {len(targets)}.")

    def _stack(self, values: list[Any]) -> Any:
        return self.backend.array_module.asarray(values)

    def _reshape(self, values: list[Any]) -> Any:
        array = self._stack(values)
        return self.backend.array_module.reshape(array, self._shape + tuple(array.shape[1:]))

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

    def with_sweep_metadata(
        self,
        *,
        shape: tuple[int, ...],
        axes: tuple[tuple[str, Any], ...],
    ) -> "SimulationBatchResult":
        """Return an equivalent batch annotated with sweep-axis metadata."""
        return SimulationBatchResult(list(self._results), shape=shape, axes=axes)

    def expect(self, key: Any, index: int | None = None, *, reduce: str | None = None) -> Any:
        """Return expectation traces reshaped to the natural sweep grid."""
        values = self._reshape([r.expect(key, index=index) for r in self._results])
        return self._reduce_time_axis(values, reduce)

    def population(self, device: str | BaseDevice, level: int = 0, *, reduce: str | None = None) -> Any:
        """Return population traces reshaped to the natural sweep grid."""
        values = self._reshape([r.population_array(device, level) for r in self._results])
        return self._reduce_time_axis(values, reduce)

    def final_overlap_magnitudes(self, targets: list[Any] | tuple[Any, ...]) -> Any:
        """Return stacked final overlap magnitudes, one per ``(result, target)`` pair."""
        self._check_targets_len(targets)
        return self._stack([r.overlap_array(t)[-1] for r, t in zip(self._results, targets)])

    def final_amplitudes(self, targets: list[Any] | tuple[Any, ...]) -> Any:
        """Return stacked final complex amplitudes (phase-sensitive), one per ``(result, target)`` pair."""
        self._check_targets_len(targets)
        return self._stack([r.amplitude_array(t)[-1] for r, t in zip(self._results, targets)])

    def __repr__(self) -> str:
        axis_names = [str(name) for name, _ in self._axes]
        return f"SimulationBatchResult(n={len(self._results)}, shape={self._shape}, axes={axis_names})"


# ---------------------------------------------------------------------------
# Result wrapping helpers
# ---------------------------------------------------------------------------


def _wrap(
    solver_result: SolverResult,
    backend: Backend,
    *,
    chip: Any,
    tlist: Any,
    e_ops_meta: Any,
    resolved_frame: Any,
) -> SimulationResult:
    observable_traces = None
    if e_ops_meta is not None:
        from quchip.engine.stage3_observables import build_observable_traces

        observable_traces = build_observable_traces(
            solver_result, tlist, chip, dict_meta=e_ops_meta, resolved_frame=resolved_frame
        )
    return SimulationResult(
        solver_result=solver_result,
        backend=backend,
        dims=chip.dims,
        device_info=[(d.label, d.computational) for d in chip.devices],
        observable_traces=observable_traces,
    )


def wrap_solver_result(solver_result: SolverResult, problem: SolveProblem, backend: Backend) -> SimulationResult:
    """Wrap a raw backend :class:`SolverResult` into a user-facing :class:`SimulationResult`.

    The engine-side :class:`~quchip.engine.ir.SolveProblem` carries the
    metadata needed to rebuild dict-form observables
    (:func:`~quchip.engine.stage3_observables.build_observable_traces`)
    and to label devices for partial-trace helpers.
    """
    return _wrap(
        solver_result,
        backend,
        chip=problem.chip,
        tlist=problem.tlist,
        e_ops_meta=problem.e_ops_meta,
        resolved_frame=problem.resolved_frame,
    )


def wrap_solver_results_from_batch(
    solver_results: list[SolverResult],
    batch: SolveBatch,
    backend: Backend,
) -> list[SimulationResult]:
    """Wrap backend results from a batched solve that shares one :class:`SolveBatch` context.

    Avoids rematerializing a per-element :class:`~quchip.engine.ir.SolveProblem`
    (and its per-element :class:`~quchip.engine.ir.HamiltonianDescription`)
    just to read the shared fields.
    """
    chip = batch.chip
    tlist = batch.tlist
    e_ops_meta = batch.e_ops_meta
    resolved_frame = batch.resolved_frame
    return [
        _wrap(sr, backend, chip=chip, tlist=tlist, e_ops_meta=e_ops_meta, resolved_frame=resolved_frame)
        for sr in solver_results
    ]
