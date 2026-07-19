"""Array, basis-label, and reduction helpers shared across viz.

These helpers are deliberately small and backend-agnostic: they defer the
actual quantum-mechanical primitives (ket construction, ``dag``, ``ptrace``,
``expect``) to the active :class:`~quchip.backend.protocol.Backend` and only
assemble Python-level glue (represented-basis label strings, index
bookkeeping, computational-subspace filtering).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.axes import Axes

from quchip.backend.protocol import Backend
from quchip.devices.base import BaseDevice

if TYPE_CHECKING:
    from quchip.results.results import SimulationResult


def _to_dense_array(obj: Any, backend: Backend) -> np.ndarray:
    """Materialise a backend operator/state as a dense complex ``numpy`` array."""
    return np.asarray(backend.to_array(obj), dtype=complex)


def _basis_label(state: tuple[int, ...]) -> str:
    """Format a represented-basis state tuple as a Dirac ket string.

    A single-subsystem state ``(2,)`` becomes ``"|2>"``; a multi-subsystem
    state ``(0, 1, 2)`` is concatenated into ``"|012>"`` (this matches the
    tensor-product ordering used throughout ``quchip``).
    """
    if len(state) == 1:
        return f"|{state[0]}>"
    return f"|{''.join(str(level) for level in state)}>"


def _resolve_trace_out_indices(
    result: "SimulationResult",
    trace_out: str | BaseDevice | list[str | BaseDevice] | None,
) -> list[int]:
    """Resolve device specs (label or object) to unique subsystem indices, in order."""
    if trace_out is None:
        return []

    items: list[str | BaseDevice] = (
        [trace_out] if isinstance(trace_out, (str, BaseDevice)) else list(trace_out)
    )
    indices: list[int] = []
    for item in items:
        idx, _label = result._resolve_device_idx(item)
        if idx not in indices:
            indices.append(idx)
    return indices


def _keep_indices_after_trace(
    result: "SimulationResult",
    trace_out: str | BaseDevice | list[str | BaseDevice] | None,
) -> list[int]:
    """Return the subsystem indices that survive a partial trace.

    Raises ``ValueError`` if *trace_out* would remove every subsystem.
    """
    if trace_out is None:
        return list(range(len(result.dims)))
    trace_indices = set(_resolve_trace_out_indices(result, trace_out))
    keep = [idx for idx in range(len(result.dims)) if idx not in trace_indices]
    if not keep:
        raise ValueError("trace_out cannot remove every subsystem")
    return keep


def _basis_states(dims: list[int]) -> list[tuple[int, ...]]:
    """Tensor-product basis-state tuples in lexicographic order."""
    return list(itertools.product(*(range(dim) for dim in dims)))


def _device_info_for_indices(
    result: "SimulationResult",
    keep_indices: list[int],
) -> list[tuple[str, bool]]:
    """Return ``(label, computational)`` metadata for the selected subsystems."""
    if result.device_info is None:
        return [(f"subsystem_{idx}", False) for idx in keep_indices]
    return [result.device_info[idx] for idx in keep_indices]


def _filtered_basis_states(
    dims: list[int],
    device_info: list[tuple[str, bool]],
    computational: bool,
) -> list[tuple[int, ...]]:
    """Restrict computational subsystems to levels 0/1 when requested."""
    states = _basis_states(dims)
    if not computational:
        return states
    return [
        state for state in states
        if all(
            (not is_comp) or level in {0, 1}
            for level, (_label, is_comp) in zip(state, device_info)
        )
    ]


def _populations_from_states(
    result: "SimulationResult",
    states: list[Any],
    dims: list[int],
) -> dict[tuple[int, ...], np.ndarray]:
    """Compute ``Tr(|n><n| rho(t))`` for every basis state across time."""
    backend = result._backend
    n_times = len(result.times)
    populations: dict[tuple[int, ...], np.ndarray] = {}

    for basis_state in _basis_states(dims):
        kets = [backend.basis(dim, level) for dim, level in zip(dims, basis_state)]
        ket = kets[0] if len(kets) == 1 else backend.tensor_states(*kets)
        projector = backend.matmul(ket, backend.dag(ket))
        values = np.empty(n_times, dtype=float)
        for idx, state in enumerate(states):
            values[idx] = float(np.real(complex(backend.expect(projector, state))))
        populations[basis_state] = values

    return populations


def _ptrace_to_keep(
    result: "SimulationResult",
    state: Any,
    keep_indices: list[int],
) -> Any:
    """Partial-trace *state* down to *keep_indices* via the active backend."""
    keep_arg = keep_indices[0] if len(keep_indices) == 1 else keep_indices
    return result._backend.ptrace(state, keep_arg, result.dims)


def _reduce_result(
    result: "SimulationResult",
    trace_out: str | BaseDevice | list[str | BaseDevice] | None = None,
    *,
    computational: bool = False,
) -> tuple[np.ndarray, dict[tuple[int, ...], np.ndarray], list[int], list[tuple[str, bool]]]:
    """Return ``(times, populations, keep_indices, device_info)`` after tracing out.

    When *trace_out* is ``None`` the precomputed ``result.populations`` are
    reused; otherwise stored states are required and each is partial-traced
    before populations are re-derived.
    """
    keep_indices = _keep_indices_after_trace(result, trace_out)
    device_info = _device_info_for_indices(result, keep_indices)
    reduced_dims = [result.dims[idx] for idx in keep_indices]

    if trace_out is None:
        populations = result.populations
    else:
        if result.states is None:
            raise RuntimeError("No states stored — pass options={'store_states': True} to the solver.")
        reduced_states = [_ptrace_to_keep(result, state, keep_indices) for state in result.states]
        populations = _populations_from_states(result, reduced_states, reduced_dims)

    filtered_states = _filtered_basis_states(reduced_dims, device_info, computational)
    filtered_populations = {state: populations[state] for state in filtered_states}
    return result.times, filtered_populations, keep_indices, device_info


def _normalize_time_index(result: "SimulationResult", index: int) -> int:
    """Validate and normalize a stored-time/state index for *result*.

    Shared by :func:`~quchip.viz.results.plot_state` and
    :func:`~quchip.viz.results.plot_wigner` so both accept and reject the
    same range with the same message. Supports Python-style negative
    indexing (``-1`` is the last stored time/state): any *index* with
    ``-N <= index < N``, where ``N = len(result.times)``, is accepted and
    returned as its equivalent non-negative index; anything else raises
    ``IndexError``.
    """
    n = len(result.times)
    if not (-n <= index < n):
        raise IndexError(f"State index {index} out of range for {n} stored times")
    return index % n


def _reduce_state(
    result: Any,
    index: int,
    trace_out: str | BaseDevice | list[str | BaseDevice] | None,
) -> tuple[Any, list[int], list[tuple[str, bool]]]:
    """Partial-trace a single stored state at *index* to the kept subsystems."""
    if result.states is None:
        raise RuntimeError("No states stored — pass options={'store_states': True} to the solver.")

    keep_indices = _keep_indices_after_trace(result, trace_out)
    device_info = _device_info_for_indices(result, keep_indices)
    state = result.states[index]
    reduced = state if trace_out is None else _ptrace_to_keep(result, state, keep_indices)
    return reduced, keep_indices, device_info


def _project_density_matrix(
    dense_dm: np.ndarray,
    dims: list[int],
    device_info: list[tuple[str, bool]],
    computational: bool,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Restrict a density matrix to the computational subspace when requested.

    Performs a double pass over ``_basis_states(dims)`` — an O(D^2 N) set
    lookup where ``D = prod(dims)`` and ``N = len(plotted_states)``. This is
    fine for the tiny dimensions that visualisation ever sees (a handful of
    qubits with a few levels each); if this ever becomes a hot path, build a
    state→index dict first.
    """
    all_states = _basis_states(dims)
    plotted_states = _filtered_basis_states(dims, device_info, computational)
    if len(plotted_states) == len(all_states):
        return dense_dm, plotted_states

    plotted_set = set(plotted_states)
    keep_indices = [idx for idx, state in enumerate(all_states) if state in plotted_set]
    return dense_dm[np.ix_(keep_indices, keep_indices)], plotted_states


def _draw_energy_ladder(
    axis: Axes,
    entries: list[tuple[float, str]],
    *,
    color: Any,
    linewidth: float,
) -> None:
    """Draw a horizontal energy-level ladder — one ``hlines`` + label per entry.

    *entries* is a list of ``(energy, label)`` pairs; the caller owns the
    energy bookkeeping (bare eigenenergies, ground-shifted dressed levels,
    ...). Only the shared level lines, right-hand represented-basis
    labels, and the common ``x`` framing are rendered here.
    """
    for energy, label in entries:
        axis.hlines(energy, 0.0, 1.0, color=color, linewidth=linewidth)
        axis.text(1.03, energy, label, va="center")
    axis.set_xlim(0.0, 1.25)
    axis.set_xticks([])
