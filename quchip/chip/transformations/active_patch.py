"""Schedule-aware active-patch reduction: eliminate spectators, keep the driven patch.

The activity analysis uses scheduled targets plus ``hops`` coupling-graph
steps. It does not rank devices by amplitude, detuning, or an error bound. The
result records per-step Schrieffer-Wolff validity, and a poor fold emits a
``UserWarning``.
"""
# `_warn_on_poor_validity` emits the warning for a poor fold.

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quchip.chip.partition import _line_device_labels
from quchip.utils.jax_utils import contains_tracer

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.chip.transformations.result import EliminationResult
    from quchip.control.sequence import QuantumSequence


def coupling_adjacency(chip: "Chip") -> dict[str, set[str]]:
    """Device-label adjacency of the chip's coupling graph."""
    adjacency: dict[str, set[str]] = {d.label: set() for d in chip.devices}
    for coupling in chip.couplings:
        adjacency[coupling.device_a_label].add(coupling.device_b_label)
        adjacency[coupling.device_b_label].add(coupling.device_a_label)
    return adjacency


def active_labels(sequence: "QuantumSequence", *, hops: int = 1) -> set[str]:
    """Devices that the schedule touches, expanded ``hops`` coupling-graph steps.

    A pulse touches its target and the device whose frame its carrier follows,
    such as the target qubit of a cross-resonance tone.
    """
    chip = sequence._chip
    active: set[str] = set()
    for op in sequence.scheduled_ops:
        target = op.target_label
        if target in chip.coupling_map:
            coupling = chip.coupling_map[target]
            active.update((coupling.device_a_label, coupling.device_b_label))
        else:
            active.add(target)
        if op.frame is not None:
            active.add(op.frame)
    if not active:
        raise ValueError("The sequence has an empty schedule; there is no active patch to reduce to.")
    adjacency = coupling_adjacency(chip)
    frontier = set(active)
    for _ in range(hops):
        frontier = {n for label in frontier for n in adjacency[label]} - active
        if not frontier:
            break
        active |= frontier
    return active


def graph_distances(adjacency: dict[str, set[str]], sources: set[str]) -> dict[str, int]:
    """BFS distance from ``sources``, omitting unreachable labels."""
    distances = {label: 0 for label in sources}
    queue = deque(sources)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def _warn_on_poor_validity(step: "EliminationResult", target: str) -> None:
    """Warn (never raise) when a fold's Schrieffer-Wolff validity is poor.

    The reduction proceeds regardless, but a poor ``g/Δ`` indicates low
    approximation quality for that fold. ``is_valid`` may be a *traced* JAX
    boolean when ``eliminate`` runs under ``jit``/``grad``
    (:class:`~quchip.chip.transformations.result.EliminationResult`'s
    ``validity`` docstring) — branching a Python ``if`` on a tracer would
    concretize it, so the check is skipped entirely under tracing rather
    than forcing a concrete comparison.
    """
    for coupling_label, entry in step.validity.items():
        is_valid = entry.get("is_valid")
        if is_valid is None or contains_tracer(is_valid):
            continue
        if not is_valid:
            g_over_delta = entry.get("g_over_delta")
            warnings.warn(
                f"active_patch: eliminating '{target}' folds coupling '{coupling_label}' with "
                f"poor Schrieffer-Wolff validity (g/Δ={g_over_delta!r}, needs < 0.1); proceeding "
                "anyway — inspect this step's effective_params and validity before use.",
                UserWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class ActivePatchResult:
    """A reduced patch chip, the re-bound sequence, and the record of reduction validity.

    Attributes
    ----------
    chip
        The patch chip: schedule-active devices plus the spectators that
        :func:`active_patch` did not eliminate. They are unreachable or
        declined by elimination itself (see :attr:`notes`).
    sequence
        A :class:`~quchip.control.sequence.QuantumSequence` bound to
        :attr:`chip`, which replays the source sequence's entries verbatim.
    active_labels
        Sorted schedule-active device labels (never eliminated).
    eliminated_labels
        Device labels that were folded away, in elimination order
        (farthest-from-active first).
    steps
        The :class:`~quchip.chip.transformations.result.EliminationResult` from
        each successful :func:`~quchip.chip.transformations.eliminate` call,
        verbatim and in :attr:`eliminated_labels` order. Do not reshape them
        here. Read ``.validity``/``.effective_params`` from the step objects
        through the convenience properties below.
    notes
        Every explicitly dropped or deferred piece of physics, including
        unreachable spectators left in place, control lines stripped as unused,
        and every step's notes. If elimination cannot continue past a
        spectator, it also includes why the reduction stopped early.
    """

    chip: Any
    sequence: Any
    active_labels: tuple[str, ...]
    eliminated_labels: tuple[str, ...]
    steps: tuple
    notes: tuple[str, ...]

    @property
    def validity(self) -> dict[str, Any]:
        """``{eliminated label: that step's .validity}``, verbatim, with the per-coupling shape unchanged."""
        return {label: step.validity for label, step in zip(self.eliminated_labels, self.steps)}

    @property
    def effective_params(self) -> dict[str, Any]:
        """``{eliminated label: that step's .effective_params}``, verbatim, with the per-survivor shape unchanged."""
        return {label: step.effective_params for label, step in zip(self.eliminated_labels, self.steps)}

    @property
    def mapping(self) -> Any:
        """Compose the captured state/operator maps from the source into this patch."""
        from quchip.chip.transformations.result import ReductionMap
        from quchip.utils.values import DeferredValue
        from functools import reduce
        from math import prod
        import jax.numpy as jnp

        maps = tuple(step.mapping for step in self.steps)
        if not maps:
            labels, dims = tuple(d.label for d in self.chip.devices), tuple(self.chip.dims)
            return ReductionMap(labels, dims, labels, dims, self.chip.backend,
                                DeferredValue(lambda: jnp.eye(prod(dims))))
        return ReductionMap(maps[0].source_labels, maps[0].source_dims,
                            maps[-1].target_labels, maps[-1].target_dims, maps[0]._backend,
                            DeferredValue(lambda: reduce(jnp.matmul, (m.embedding for m in maps))))

    def simulate(self, **kwargs: Any) -> Any:
        """Solve the patch sequence.

        Parameters
        ----------
        **kwargs : Any
            Arguments forwarded to :meth:`QuantumSequence.simulate`.
        """
        return self.sequence.simulate(**kwargs)


def _split_reachable_spectators(
    chip: "Chip", spectators: list[str], active: set[str]
) -> tuple[list[str], list[str]]:
    """Split ``spectators`` into those the coupling graph can reach from ``active``.

    Returns ``(reachable, notes)``. Spectators the graph can't reach are
    left out of ``reachable`` — they stay on the patch chip untouched —
    and get one note explaining why, when there are any.
    """
    distances = graph_distances(coupling_adjacency(chip), active)
    reachable = [label for label in spectators if label in distances]
    unreachable = [label for label in spectators if label not in distances]
    notes: list[str] = []
    if unreachable:
        notes.append(
            f"spectators {sorted(unreachable)} share no coupling path with the active set; "
            "left in place for the exact partitioner to split off at solve time"
        )
    return reachable, notes


def _strip_dead_control_lines(chip: "Chip", sequence: "QuantumSequence", reachable: list[str]) -> tuple[Any, list[str]]:
    """Detach control lines that only ever target a spectator due for elimination.

    Returns the chip to eliminate on next (a clone only if a line was
    actually stripped, else ``chip`` itself, untouched) and a note listing
    what was dropped, when anything was.
    """
    equipment = chip.control_equipment
    if equipment is None:
        return chip, []
    scheduled_drives = {op.drive_label for op in sequence.scheduled_ops}
    doomed = set(reachable)
    unused = sorted(
        ln.label for ln in equipment.lines
        if ln.label not in scheduled_drives
        and any(lbl in doomed for lbl in _line_device_labels(chip, ln))
    )
    if not unused:
        return chip, []
    working = chip.clone()
    # chip.wire(*keep, signal_chain=...) is the natural-looking way to
    # strip lines, but chip.wire() treats an *empty* line list as "no
    # lines given" and reuses the existing connected set instead of
    # detaching everything — exactly backwards when every wired line
    # turns out to be unused. unwire() has no such trap: it removes one
    # line (and the signal-chain entries referencing it) at a time and
    # clears control_equipment to None once the last line is gone, so
    # it strips correctly whether one line or all of them are unused.
    for label in unused:
        working.unwire(label)
    return working, [f"dropped unused spectator control lines: {unused}"]


def _eliminate_spectators(
    chip: "Chip", active: set[str], reachable: list[str], method: str
) -> tuple[Any, list[Any], list[str], list[str]]:
    """Fold ``reachable`` spectators into ``chip`` one at a time, farthest-from-active first.

    Recomputes elimination order and reachability from the *current*
    working chip before every step — see :func:`active_patch`'s docstring
    for why this is safe and necessary. Stops early, without raising, if
    ``eliminate`` declines a step; everything folded before that stays
    folded.

    Returns ``(working_chip, steps, eliminated_labels, notes)``.
    """
    from quchip.chip.transformations import eliminate

    working = chip
    steps: list[Any] = []
    eliminated: list[str] = []
    notes: list[str] = []
    remaining = set(reachable)
    while remaining:
        current_adjacency = coupling_adjacency(working)
        current_distances = graph_distances(current_adjacency, active)
        working_labels = {d.label for d in working.devices}
        remaining &= working_labels
        if not remaining:
            break
        target = min(remaining, key=lambda label: (-current_distances.get(label, -1), label))
        try:
            # Unsupported physical reductions leave the remaining spectators
            # in place. Invalid configuration raises ValueError and propagates.
            step = eliminate(working, target, method=method)
        except NotImplementedError as exc:
            notes.append(
                f"stopped eliminating spectators at '{target}' ({exc}); "
                f"{sorted(remaining)} kept on the patch chip as-is"
            )
            break
        _warn_on_poor_validity(step, target)
        steps.append(step)
        eliminated.append(target)
        notes.extend(step.notes)
        working = step.chip
        remaining.discard(target)
    return working, steps, eliminated, notes


def active_patch(sequence: "QuantumSequence", *, hops: int = 1, method: str = "sw") -> ActivePatchResult:
    """Reduce a chip to its schedule-active patch by eliminating spectators.

    The spectators are eliminated one at a time through
    :func:`~quchip.chip.transformations.eliminate`, farthest-from-active first. The
    result keeps every step's validity metrics unchanged. The reduction is an
    explicit opt-in, because it approximates (Schrieffer-Wolff, or the exact
    dressed-spectrum route under ``method="exact"``), unlike the exact automatic
    partitioning that :meth:`~quchip.control.sequence.QuantumSequence.simulate`
    does internally at solve time.

    The elimination order and reachability are recalculated from the *current* reduced
    chip before every step. A fold can only add a bridging edge between the eliminated
    mode's neighbors, and never remove one. A label's distance from the active set can
    therefore shrink, but a reachable label never becomes unreachable.

    Bridging edges from earlier folds can be :class:`~quchip.chip.couplings.Capacitive`
    edges that carry ``g`` or :class:`~quchip.chip.couplings.TunableCapacitive` edges
    that carry ``g_0``. Both fold onward, so cycles among spectators reduce completely.
    ``eliminate`` can still decline an unsupported step, such as an unsupported
    accessible-field boundary, with its typed :class:`NotImplementedError`. The
    reduction then stops there, and everything eliminated until then stays folded. The
    remaining spectators, including the one that failed, stay unchanged on the patch
    chip, and :attr:`notes` records the reason.

    Parameters
    ----------
    sequence
        The schedule to reduce around.
    hops
        Coupling-graph hops by which the active set expands beyond the
        scheduled targets (see :func:`active_labels`).
    method
        Forwarded to :func:`~quchip.chip.transformations.eliminate` for
        every device elimination.

    Returns
    -------
    ActivePatchResult
    """
    # Rereading the graph at each step ensures that a folded label is never
    # considered again and that the order always shows what `eliminate` will
    # see.
    chip = sequence._chip
    active = active_labels(sequence, hops=hops)
    spectators = [d.label for d in chip.devices if d.label not in active]
    reachable, notes = _split_reachable_spectators(chip, spectators, active)
    working, strip_notes = _strip_dead_control_lines(chip, sequence, reachable)
    notes.extend(strip_notes)
    working, steps, eliminated, elimination_notes = _eliminate_spectators(working, active, reachable, method)
    notes.extend(elimination_notes)

    if working is chip:
        working = chip.clone()
    patch_sequence = sequence._copy_with_chip(working)
    return ActivePatchResult(
        chip=working,
        sequence=patch_sequence,
        active_labels=tuple(sorted(active)),
        eliminated_labels=tuple(eliminated),
        steps=tuple(steps),
        notes=tuple(notes),
    )
