"""Schedule-aware active-patch reduction: eliminate spectators, keep the driven patch.

The activity analysis is structural — scheduled targets plus ``hops``
coupling-graph steps. Amplitudes, detunings, and rigorous error bounds are
declared out-of-scope refinements; the honesty signal is the per-step
Schrieffer-Wolff validity carried on the result, plus a ``UserWarning`` (see
:func:`_warn_on_poor_validity`) raised at fold time for any step whose
validity comes back poor — the reduction still proceeds, but the caller is
told which fold to distrust.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    """Devices touched by the schedule, expanded ``hops`` coupling-graph steps."""
    chip = sequence._chip
    active: set[str] = set()
    for op in sequence.scheduled_ops:
        target = op.target_label
        if target in chip.coupling_map:
            coupling = chip.coupling_map[target]
            active.update((coupling.device_a_label, coupling.device_b_label))
        else:
            active.add(target)
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
    """BFS distance from ``sources``; labels unreachable from them are absent."""
    distances = {label: 0 for label in sources}
    queue = deque(sources)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def _line_targets(chip: "Chip", line: Any) -> tuple[str, ...]:
    """The device label(s) a control line's fate is tied to (one for a device line, two for an edge/pump line)."""
    target = line.target_label
    if target is None:
        return ()
    if getattr(line, "target_kind", "device") == "edge":
        coupling = chip.coupling_map[target]
        return (coupling.device_a_label, coupling.device_b_label)
    return (target,)


def _warn_on_poor_validity(step: "EliminationResult", target: str) -> None:
    """Warn (never raise) when a fold's Schrieffer-Wolff validity is poor.

    Honesty signal for :func:`active_patch`: the reduction proceeds
    regardless, but a poor ``g/Δ`` means that step's folded physics is a
    weaker approximation than usual. ``is_valid`` may be a *traced* JAX
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
                "anyway — treat this step's effective_params/validity as a weaker approximation.",
                UserWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class ActivePatchResult:
    """A reduced patch chip, the re-bound sequence, and the reduction's honesty record.

    Attributes
    ----------
    chip
        The patch chip: schedule-active devices plus whatever spectators
        :func:`active_patch` could not eliminate (unreachable, or where
        elimination itself declined — see :attr:`notes`).
    sequence
        A :class:`~quchip.control.sequence.QuantumSequence` bound to
        :attr:`chip`, replaying the source sequence's entries verbatim.
    active_labels
        Sorted schedule-active device labels (never eliminated).
    eliminated_labels
        Device labels actually folded away, in elimination order
        (farthest-from-active first).
    steps
        The :class:`~quchip.chip.transformations.result.EliminationResult`
        from each successful :func:`~quchip.chip.transformations.eliminate`
        call, verbatim and in :attr:`eliminated_labels` order — reshape
        nothing here; read ``.validity``/``.effective_params`` off the
        step objects through the convenience properties below.
    notes
        Every explicitly dropped or deferred piece of physics: unreachable
        spectators left in place, control lines stripped as unused, and
        (if elimination itself couldn't proceed past some spectator) why
        the reduction stopped early — plus every step's own notes.
    """

    chip: Any
    sequence: Any
    active_labels: tuple[str, ...]
    eliminated_labels: tuple[str, ...]
    steps: tuple
    notes: tuple[str, ...]

    @property
    def validity(self) -> dict[str, Any]:
        """``{eliminated label: that step's .validity}`` — verbatim, per-coupling shape untouched."""
        return {label: step.validity for label, step in zip(self.eliminated_labels, self.steps)}

    @property
    def effective_params(self) -> dict[str, Any]:
        """``{eliminated label: that step's .effective_params}`` — verbatim, per-survivor shape untouched."""
        return {label: step.effective_params for label, step in zip(self.eliminated_labels, self.steps)}

    def simulate(self, **kwargs: Any) -> Any:
        """Solve the patch sequence (automatic partitioning still applies inside)."""
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
        and any(lbl in doomed for lbl in _line_targets(chip, ln))
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
            # Narrow on purpose: NotImplementedError is eliminate()'s typed
            # signal that this particular elimination step is unsupported
            # (e.g. a Purcell decay fold onto a survivor that carries
            # thermal_population — eliminate_device.py has no collapse-channel
            # API to represent the resulting rate without inventing thermal
            # absorption that was never physically present). That is a model
            # limitation worth downgrading to a note. A bath explicitly
            # targeting this mode raises ValueError (eliminate_device.py's
            # fail-fast guard) and must propagate — an explicit user conflict
            # is not a graceful-stop candidate.
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

    Spectators are eliminated one at a time via
    :func:`~quchip.chip.transformations.eliminate`, farthest-from-active
    first; every step's validity metrics ride on the result untouched.
    Explicit opt-in — this approximates (Schrieffer-Wolff, or the exact
    dressed-spectrum route under ``method="exact"``), unlike the exact
    automatic partitioning :meth:`~quchip.control.sequence.QuantumSequence.simulate`
    performs internally at solve time.

    The elimination order and reachability are recomputed from the
    *current* reduced chip before every step, not just once up front: a
    fold can only ever add a bridging edge between the eliminated mode's
    neighbors, never remove one, so a label's distance from the active set
    can shorten but never make a previously reachable label unreachable —
    still, reading the graph fresh each time means a label that has
    already been folded away is never re-considered, and the order always
    reflects what ``eliminate`` will actually see. Bridging edges created
    by earlier folds may be :class:`~quchip.chip.couplings.Capacitive`
    edges carrying ``g`` or :class:`~quchip.chip.couplings.TunableCapacitive`
    edges carrying ``g_0``. Both fold onward, so cycles among spectators
    reduce all the way down. If ``eliminate`` still declines a step (its
    typed :class:`NotImplementedError` signal for an unsupported
    elimination — e.g. a Purcell decay fold onto a survivor that carries
    thermal occupation, which no collapse-channel API can represent without
    inventing physics that was never present), the reduction stops
    there: everything eliminated so far stays folded, and the remaining
    spectators — including the one that failed — are left on the patch
    chip untouched, with the reason recorded in :attr:`notes`.

    Parameters
    ----------
    sequence
        The schedule to reduce around.
    hops
        Coupling-graph hops the active set expands beyond the scheduled
        targets (see :func:`active_labels`).
    method
        Forwarded to :func:`~quchip.chip.transformations.eliminate` for
        every device elimination.

    Returns
    -------
    ActivePatchResult
    """
    from quchip.control.sequence import QuantumSequence as _Sequence

    chip = sequence._chip
    active = active_labels(sequence, hops=hops)
    spectators = [d.label for d in chip.devices if d.label not in active]
    if not spectators:
        return ActivePatchResult(
            chip=chip, sequence=sequence,
            active_labels=tuple(sorted(active)), eliminated_labels=(), steps=(), notes=(),
        )

    reachable, notes = _split_reachable_spectators(chip, spectators, active)
    working, strip_notes = _strip_dead_control_lines(chip, sequence, reachable)
    notes.extend(strip_notes)
    working, steps, eliminated, elimination_notes = _eliminate_spectators(working, active, reachable, method)
    notes.extend(elimination_notes)

    patch_sequence = _Sequence(working)
    # ``_entries`` is the single source of truth for scheduling and its
    # entries are immutable records replayed functionally — sharing them
    # (never deep-copying) is safe and required: an envelope may carry
    # traced pulse parameters that a deep copy would silently detach from
    # the surrounding trace.
    patch_sequence._entries = list(sequence._entries)
    return ActivePatchResult(
        chip=working,
        sequence=patch_sequence,
        active_labels=tuple(sorted(active)),
        eliminated_labels=tuple(eliminated),
        steps=tuple(steps),
        notes=tuple(notes),
    )
