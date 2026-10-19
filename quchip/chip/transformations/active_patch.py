"""Spectator elimination: reduce a chip to kept devices, or a schedule to its active patch.

:func:`eliminate_spectators` removes the spectators of a set of kept devices
one at a time, in an order that the patch structure and the transition
energies set (PHYSICS.md §10.8). :func:`active_patch` keeps the devices that a
schedule touches, plus ``hops`` coupling-graph steps, and eliminates the other
spectators in the same way. The activity analysis does not rank devices by
amplitude, detuning, or an error bound. The results record per-step
Schrieffer-Wolff validity, and a poor fold emits a ``UserWarning``.
"""
# `_warn_on_poor_validity` emits the warning for a poor fold.

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.chip.partition import _line_device_labels
from quchip.utils.jax_utils import contains_tracer
from quchip.utils.labeling import resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.chip.transformations.elimination_order import EliminationOrder
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


def _warn_on_poor_validity(step: "EliminationResult", target: str, caller: str) -> None:
    """Warn (never raise) when a validity entry of a fold is poor.

    The reduction proceeds regardless, but a poor ``g/Δ``, or a poor ratio of
    a kept section or response, indicates low approximation quality for that
    fold. ``is_valid`` may be a *traced* JAX
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
            if "g_over_delta" in entry:
                problem = (f"folds coupling '{coupling_label}' with poor Schrieffer-Wolff validity "
                           f"(g/Δ={entry['g_over_delta']!r}, needs < 0.1)")
            else:
                name = next(key for key in entry if key != "is_valid")
                problem = f"leaves '{coupling_label}' with {name}={entry[name]!r}, which needs < 0.1"
            warnings.warn(
                f"{caller}: eliminating '{target}' {problem}; proceeding "
                "anyway — inspect this step's effective_params and validity before use.",
                UserWarning,
                stacklevel=2,
            )


class _EliminationChain:
    """Views shared by the results of a chain of eliminations."""

    chip: Any
    eliminated_labels: tuple[str, ...]
    steps: tuple

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
        """Compose the captured state/operator maps from the source into this chip."""
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


@dataclass(frozen=True)
class SpectatorEliminationResult(_EliminationChain):
    """A chip reduced to kept devices, and the record of each elimination.

    Attributes
    ----------
    chip
        The reduced chip: the kept devices plus the spectators that
        :func:`eliminate_spectators` did not eliminate (see :attr:`notes`).
    kept_labels
        Kept device labels, in chip order.
    eliminated_labels
        Device labels that were folded away, in elimination order.
    steps
        The :class:`~quchip.chip.transformations.result.EliminationResult` of
        each :func:`~quchip.chip.transformations.eliminate` call, verbatim and
        in :attr:`eliminated_labels` order. Read ``.validity`` and
        ``.effective_params`` of all steps through the properties below.
    notes
        Every explicitly dropped or deferred piece of physics, including
        spectators left in place, each step that breaks the precedence of the
        default order, and every step's notes. If elimination cannot continue
        past a spectator, it also includes why the reduction stopped early.
    omitted_shifts
        For each kept device that a Schrieffer-Wolff step reaches through a
        device that stays on the chip, the estimated shift of its transition
        that the chain omits, in GHz: the sum over steps of
        ``|J|^2 / (E_k - E_v)`` (PHYSICS.md §10.8). The reduced transition is
        off by about minus this value. A value is ``None`` when the estimate
        does not apply, and :attr:`notes` says why. The attribute is ``None``
        when traced device parameters prevent the estimate. An empty dict
        means that no step found such a path, not that the chain omits
        nothing: the estimate reads number-conserving exchange elements only.
    """

    chip: Any
    kept_labels: tuple[str, ...]
    eliminated_labels: tuple[str, ...]
    steps: tuple
    notes: tuple[str, ...]
    omitted_shifts: dict[str, float | None] | None = field(default_factory=dict)


@dataclass(frozen=True)
class ActivePatchResult(_EliminationChain):
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
        Delays, barriers and virtual-Z entries on removed devices and
        stripped lines keep the source timing and have no other effect.
    active_labels
        Sorted schedule-active device labels (never eliminated).
    eliminated_labels
        Device labels that were folded away, in elimination order.
    steps
        The :class:`~quchip.chip.transformations.result.EliminationResult` from
        each successful :func:`~quchip.chip.transformations.eliminate` call,
        verbatim and in :attr:`eliminated_labels` order. Do not reshape them
        here. Read ``.validity``/``.effective_params`` from the step objects
        through the convenience properties below.
    notes
        Every explicitly dropped or deferred piece of physics, including
        unreachable spectators left in place, control lines stripped as unused,
        each step that breaks the precedence of the default order, and every
        step's notes. If elimination cannot continue past a spectator, it also
        includes why the reduction stopped early.
    omitted_shifts
        For each active device that a Schrieffer-Wolff step reaches through a
        device that stays on the chip, the estimated shift of its transition
        that the chain omits, in GHz: the sum over steps of
        ``|J|^2 / (E_k - E_v)`` (PHYSICS.md §10.8). The reduced transition is
        off by about minus this value. A value is ``None`` when the estimate
        does not apply, and :attr:`notes` says why. The attribute is ``None``
        when traced device parameters prevent the estimate. An empty dict
        means that no step found such a path, not that the chain omits
        nothing: the estimate reads number-conserving exchange elements only.
    """

    chip: Any
    sequence: Any
    active_labels: tuple[str, ...]
    eliminated_labels: tuple[str, ...]
    steps: tuple
    notes: tuple[str, ...]
    omitted_shifts: dict[str, float | None] | None = field(default_factory=dict)

    def simulate(self, **kwargs: Any) -> Any:
        """Solve the patch sequence.

        Parameters
        ----------
        **kwargs : Any
            Arguments forwarded to :meth:`QuantumSequence.simulate`.
        """
        return self.sequence.simulate(**kwargs)


def _plan(chip: "Chip", kept: set[str], order: Iterable[Any] | None,
          caller: str) -> tuple["EliminationOrder | None", list[str], list[str]]:
    """Return the default-order model or ``None``, the devices due for elimination, and notes.

    The default order eliminates every spectator that shares a chain of
    neighbours with the kept devices. An explicit order eliminates exactly the
    devices it lists. Other spectators stay in place, with a note.
    """
    from quchip.chip.transformations.elimination_order import EliminationOrder

    spectators = [d.label for d in chip.devices if d.label not in kept]
    notes: list[str] = []
    if order is None:
        model = EliminationOrder(chip, tuple(label for label in (d.label for d in chip.devices) if label in kept))
        doomed = [label for label in spectators if label in model.remaining]
        left = [label for label in spectators if label not in model.remaining]
        if left:
            notes.append(
                f"spectators {left} share no coupling, effective Hamiltonian or port path with the kept "
                "devices; left in place for the exact partitioner to split off at solve time"
            )
        return model, doomed, notes
    doomed = [resolve_label(target) for target in order]
    unknown = [label for label in doomed if label not in chip.device_map]
    if unknown:
        raise KeyError(f"{caller}: order names no device {unknown} on chip '{chip.label}'.")
    if len(set(doomed)) != len(doomed):
        raise ValueError(f"{caller}: order lists a device more than once: {doomed}.")
    named = [label for label in doomed if label in kept]
    if named:
        raise ValueError(f"{caller}: order names kept devices {named}.")
    left = [label for label in spectators if label not in doomed]
    if left:
        notes.append(f"spectators {left} are not in order; left in place")
    return None, doomed, notes


def _one_excitation_reader(chip: "Chip", method: str) -> Any:
    """Return a reader of the chip's single-excitation gaps and exchange elements, in GHz.

    It reads them at the approximation that the fold of ``method`` reads.
    """

    def read(labels: tuple[str, ...]) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
        from quchip.chip.transformations.eliminate_device import _one_excitation_elements
        from quchip.chip.transformations.methods import lookup_reduction_method
        from quchip.engine.assembly import _resolve_system

        # An unknown method leaves the chip's approximation here; eliminate() then rejects the method.
        reduction = lookup_reduction_method(method)
        approximation = chip.approximation if reduction is None else reduction.source_approximation(chip)
        return _one_excitation_elements(chip, _resolve_system(chip, chip.backend), labels, approximation, pairs=True)

    return read


def _eliminate_chain(chip: "Chip", model: "EliminationOrder | None", doomed: list[str], method: str,
                     local: bool, caller: str, kept: set[str]
                     ) -> tuple[Any, list[Any], list[str], list[str], dict[str, float | None] | None]:
    """Fold ``doomed`` into ``chip`` one at a time, in the default order of ``model`` or in the given order.

    Stops early, without raising, if ``eliminate`` declines a step;
    everything folded before that stays folded. Each Schrieffer-Wolff step
    adds the shift that it omits from the ``kept`` devices to the estimate
    (PHYSICS.md §10.8).

    Returns ``(working_chip, steps, eliminated_labels, notes, omitted)``.
    ``omitted`` is ``None`` when traced values prevent the estimate.
    """
    from quchip.chip.transformations import eliminate
    from quchip.chip.transformations.elimination_order import omitted_shifts

    working = chip
    steps: list[Any] = []
    eliminated: list[str] = []
    notes: list[str] = []
    omitted: dict[str, float | None] | None = {}
    pending = list(doomed)
    while pending:
        note = None
        estimate = method == "sw" and omitted is not None
        labels = tuple(device.label for device in working.devices)
        gaps, raw = _one_excitation_reader(working, method)(labels) if model is not None or estimate else ({}, {})

        def read(needed: tuple[str, ...], gaps: dict = gaps, raw: dict = raw) -> tuple[dict, dict]:
            return ({label: gaps[label] for label in needed},
                    {pair: value for pair, value in raw.items() if set(pair) <= set(needed)})

        if model is None:
            target = pending[0]
        else:
            target, note = model.choose(read, caller)
        try:
            # Unsupported physical reductions leave the remaining spectators
            # in place. Invalid configuration raises ValueError and propagates.
            step = eliminate(working, target, method=method, local=local)
        except NotImplementedError as exc:
            notes.append(
                f"stopped eliminating spectators at '{target}' ({exc}); "
                f"{pending} kept on the chip as-is"
            )
            break
        _warn_on_poor_validity(step, target, caller)
        if note is not None:
            notes.append(note)
        if estimate and contains_tracer((gaps, raw)):
            omitted = None
            notes.append("Omitted exchange shifts are not estimated, because device parameters are traced.")
        elif estimate:
            assert omitted is not None  # estimate implies it; keeps mypy's Optional narrow
            position = {label: index for index, label in enumerate(labels)}.__getitem__
            energies = {label: float(np.real(np.asarray(value))) for label, value in gaps.items()}
            elements = {pair: complex(np.asarray(value)) for pair, value in raw.items()}
            for label, (shift, mediators, reason) in omitted_shifts(target, kept, energies, elements,
                                                                    position).items():
                names = ", ".join(map(repr, mediators))
                if shift is None:
                    omitted[label] = None
                    notes.append(f"Eliminating {target!r} can omit a fourth-order exchange shift of the kept "
                                 f"{label!r} through {names}. It is not estimated, because {reason}. "
                                 "method='exact' with local=False keeps it.")
                    continue
                total = omitted.get(label, 0.0)
                omitted[label] = None if total is None else total + shift
                notes.append(f"Eliminating {target!r} omits an estimated {1e6 * shift:+.1f} kHz from the transition "
                             f"of the kept {label!r}, the fourth-order exchange shift through {names}. "
                             "method='exact' with local=False keeps it.")
        steps.append(step)
        eliminated.append(target)
        notes.extend(step.notes)
        working = step.chip
        pending.remove(target)
        if model is not None:
            model.remove(target)
    return working, steps, eliminated, notes, omitted


def eliminate_spectators(
    chip: "Chip", keep: Any, *, order: Iterable[Any] | None = None, method: str = "sw", local: bool = False,
) -> SpectatorEliminationResult:
    """Reduce a chip to the kept devices by eliminating the other devices one at a time.

    Each step is one :func:`~quchip.chip.transformations.eliminate` call, so
    each keeps second-order terms in its own couplings. The order of the steps
    sets which fourth-order terms the chain keeps and, with ``local=True``, how
    far the operators of the kept devices spread along the captured maps
    (PHYSICS.md §10.8). By default, a spectator waits while a spectator
    neighbour links it to a kept device that is at least two times closer to
    it in transition frequency than that neighbour. Among the others, a device
    whose neighbours are all kept goes last, and the step that keeps the
    widest operator of a kept device smallest goes first. A step that must
    break the wait is recorded in :attr:`~SpectatorEliminationResult.notes`.
    The order is a heuristic for dispersive chips and does not bound the
    error. Each Schrieffer-Wolff step also estimates the shift that it omits
    from a kept device, in :attr:`~SpectatorEliminationResult.omitted_shifts`.

    Spectators that share no coupling, effective Hamiltonian or port path
    with the kept devices stay on the reduced chip, for the exact partitioner
    to split off at solve time. If ``eliminate`` declines a step with its
    typed :class:`NotImplementedError`, the chain stops there, and everything
    eliminated until then stays folded.

    Parameters
    ----------
    chip
        Source chip (never mutated).
    keep
        The devices to keep, as labels or objects.
    order
        Devices to eliminate, in this order, as labels or objects. The other
        spectators stay on the reduced chip. The default order compares
        transition frequencies, so it needs concrete values. Under
        ``jax.jit`` or ``jax.grad`` with traced device parameters, pass the
        ``eliminated_labels`` of a call without tracing.
    method
        Forwarded to every :func:`~quchip.chip.transformations.eliminate` call.
    local
        Forwarded to every :func:`~quchip.chip.transformations.eliminate`
        call. Each step then computes its generator on the patch around the
        eliminated device.

    Returns
    -------
    SpectatorEliminationResult

    Examples
    --------
    >>> from quchip import Capacitive, Chip, DuffingTransmon, Resonator, eliminate_spectators
    >>> q = [DuffingTransmon(freq=5.0 + 0.1 * i, anharmonicity=-0.25, levels=3, label=f"q{i}") for i in range(3)]
    >>> c = [Resonator(freq=6.5 + 0.02 * i, levels=3, label=f"c{i}") for i in range(2)]
    >>> chip = Chip(q + c, [Capacitive(q[i + j], c[i], g=0.03) for i in range(2) for j in range(2)])
    >>> result = eliminate_spectators(chip, ["q0", "q2"], local=True)
    >>> [d.label for d in result.chip.devices]
    ['q0', 'q2']
    """
    from quchip.engine.routes import require_workflow

    require_workflow(chip, "eliminate_spectators()")
    if isinstance(keep, str) or not isinstance(keep, Iterable):
        keep = (keep,)
    kept = {resolve_label(device) for device in keep}
    unknown = sorted(label for label in kept if label not in chip.device_map)
    if unknown:
        raise KeyError(f"eliminate_spectators(): keep names no device {unknown} on chip '{chip.label}'.")
    if not kept:
        raise ValueError("eliminate_spectators(): keep names no device.")
    model, doomed, notes = _plan(chip, kept, order, "eliminate_spectators()")
    working, steps, eliminated, chain_notes, omitted = _eliminate_chain(chip, model, doomed, method, local,
                                                                        "eliminate_spectators()", kept)
    notes.extend(chain_notes)
    if working is chip:
        working = chip.clone()
    return SpectatorEliminationResult(
        chip=working,
        kept_labels=tuple(d.label for d in chip.devices if d.label in kept),
        eliminated_labels=tuple(eliminated),
        steps=tuple(steps),
        notes=tuple(notes),
        omitted_shifts=omitted,
    )


def _strip_dead_control_lines(chip: "Chip", sequence: "QuantumSequence", doomed: list[str]) -> tuple[Any, list[str]]:
    """Detach control lines that only ever target a spectator due for elimination.

    Returns the chip to eliminate on next (a clone only if a line was
    actually stripped, else ``chip`` itself, untouched) and a note listing
    what was dropped, when anything was.
    """
    equipment = chip.control_equipment
    if equipment is None:
        return chip, []
    scheduled_drives = {op.drive_label for op in sequence.scheduled_ops}
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


def active_patch(
    sequence: "QuantumSequence", *, hops: int = 1, method: str = "sw", local: bool = False,
    order: Iterable[Any] | None = None,
) -> ActivePatchResult:
    """Reduce a chip to its schedule-active patch by eliminating spectators.

    The spectators are eliminated one at a time through
    :func:`~quchip.chip.transformations.eliminate`, in the default order of
    :func:`eliminate_spectators` with the active devices kept. The result keeps
    every step's validity metrics unchanged. A kept device next to an active
    device, such as a coupler with ``hops=1``, can link a removed spectator to
    the active device. The step then omits their exchange shift, and
    :attr:`~ActivePatchResult.omitted_shifts` estimates it. The reduction is an explicit
    opt-in, because it approximates (Schrieffer-Wolff, or the exact
    dressed-spectrum route under ``method="exact"``), unlike the exact
    automatic partitioning that
    :meth:`~quchip.control.sequence.QuantumSequence.simulate` does internally
    at solve time.

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
    local
        Forwarded to :func:`~quchip.chip.transformations.eliminate`. Each
        step then computes its generator on the patch around the eliminated
        device.
    order
        Spectators to eliminate, in this order, as for
        :func:`eliminate_spectators`. The other spectators stay on the patch
        chip.

    Returns
    -------
    ActivePatchResult
    """
    from quchip.engine.routes import require_workflow

    chip = sequence._chip
    require_workflow(chip, "active_patch()")
    active = active_labels(sequence, hops=hops)
    model, doomed, notes = _plan(chip, active, order, "active_patch()")
    working, strip_notes = _strip_dead_control_lines(chip, sequence, doomed)
    notes.extend(strip_notes)
    working, steps, eliminated, elimination_notes, omitted = _eliminate_chain(working, model, doomed, method,
                                                                              local, "active_patch()", active)
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
        omitted_shifts=omitted,
    )
