"""Shared control-plane and chip-rebuild plumbing for :mod:`quchip.chip.transformations`.

Every transformation of shape ``Chip -> (Chip + diagnostics)`` faces the same
control-plane problem: a control line whose target has no image in the reduced
model (its device was eliminated, or its coupling touched the eliminated mode)
must either be converted by a registered retarget rule or fail fast. It also
rebuilds a reduced :class:`~quchip.chip.chip.Chip` from surviving devices and
couplings, reattaches the converted equipment, and detaches the intermediate
clone. These four steps are identical across the device and coupling reduction
paths, so they live here as single-purpose functions reused by any
``Chip -> EliminationResult`` transformation without touching dispatch — the
same shared-plumbing role :mod:`quchip.chip.retarget` plays for the rule
lookup itself.

The two reduction paths differ only in how a doomed line is *detected* and what
fail-fast message it raises; both feed a per-path ``classify`` closure into
:func:`plan_stranded_lines`, which owns the detection order and the rule
lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from quchip.chip.retarget import RetargetContext, lookup_retarget_rule
from quchip.control.equipment import ControlEquipment


@dataclass(frozen=True)
class StrandedLine:
    """One doomed control line's retarget target and its fail-fast message.

    Attributes
    ----------
    rule_target
        The eliminated target a converter would retarget the line onto — a
        coupling object for a coupling reduction, a device or an edge coupling
        for a device reduction. Its type drives the MRO lookup in
        :func:`~quchip.chip.retarget.lookup_retarget_rule`.
    missing_rule_message
        The exact ``ValueError`` message :func:`plan_stranded_lines` raises
        when no rule converts the line. Each reduction path builds its own
        message so the wording matches the target it eliminated.
    """

    rule_target: Any
    missing_rule_message: str


def plan_stranded_lines(
    equipment: Any,
    classify: Callable[[Any], StrandedLine | None],
    result_kind: str,
) -> tuple[list[Any], list[tuple[Any, Any]]]:
    """Partition control lines into survivors and a retarget plan, failing fast on missing rules.

    Each line is classified: a survivor (``classify`` returns ``None``) passes
    through untouched; a doomed line (``classify`` returns a
    :class:`StrandedLine`) has its converter looked up now, *before* any fold
    work, so a missing rule raises immediately. Rule *application* is deferred
    to :func:`reattach_equipment` (a converter may need an edge the reduction
    has not emitted yet), so this pass only detects and fails fast — the same
    ordering the device and coupling paths rely on.

    Parameters
    ----------
    equipment
        The source chip's :class:`~quchip.control.equipment.ControlEquipment`,
        or ``None`` when no equipment is wired.
    classify
        ``classify(line) -> StrandedLine | None``: ``None`` marks a survivor,
        a :class:`StrandedLine` marks a doomed line.
    result_kind
        The reduction's result kind (``"edge"``, ``"leaf-fold"``, or
        ``"crosskerr"``), matched exactly by the rule lookup.

    Returns
    -------
    tuple
        ``(survivor_lines, retarget_plan)`` in original line order:
        ``survivor_lines`` is the list of pass-through lines,
        ``retarget_plan`` is a list of ``(line, rule)`` pairs.

    Raises
    ------
    ValueError
        With the doomed line's ``missing_rule_message`` when no registered
        rule converts it.
    """
    survivor_lines: list[Any] = []
    retarget_plan: list[tuple[Any, Any]] = []
    if equipment is None:
        return survivor_lines, retarget_plan
    for line in equipment.lines:
        stranded = classify(line)
        if stranded is None:
            survivor_lines.append(line)
            continue
        rule = lookup_retarget_rule(type(line), type(stranded.rule_target), result_kind)
        if rule is None:
            raise ValueError(stranded.missing_rule_message)
        retarget_plan.append((line, rule))
    return survivor_lines, retarget_plan


def rebuild_chip(source_chip: Any, *, devices: Any, couplings: Any) -> Any:
    """Construct a reduced chip carrying the source chip's frame, RWA, backend, and baths.

    The label, frame (deep-copied when a per-device dict), RWA flag, backend,
    and baths are copied off ``source_chip``; ``devices`` and ``couplings`` are
    the reduction's survivors. An empty ``couplings`` becomes ``None`` (a chip
    with no couplings), matching the constructor's own convention.

    Parameters
    ----------
    source_chip
        The chip being reduced; never mutated.
    devices
        The surviving devices for the reduced chip.
    couplings
        The surviving (and emitted) couplings; an empty iterable yields a
        coupling-free chip.

    Returns
    -------
    Chip
        The reduced chip, before control equipment is attached.
    """
    from quchip.chip.chip import Chip

    return Chip(
        devices=list(devices),
        couplings=list(couplings) or None,
        label=source_chip.label,
        frame=dict(source_chip.frame) if isinstance(source_chip.frame, dict) else source_chip.frame,
        rwa=source_chip.rwa,
        backend=source_chip._backend,
        baths=[bath.copy() for bath in source_chip.baths] or None,
    )


def reattach_equipment(
    source_chip: Any,
    final_chip: Any,
    equipment: Any,
    survivor_lines: list[Any],
    retarget_plan: list[tuple[Any, Any]],
    *,
    mode_label: str,
    result_kind: str,
    edges: Any,
    notes: list[str],
) -> None:
    """Apply the retarget plan and attach the survivor plus converted lines to the reduced chip.

    A no-op when ``equipment is None``. Otherwise each ``(line, rule)`` in
    ``retarget_plan`` is converted against a
    :class:`~quchip.chip.retarget.RetargetContext` built from the reduction's
    ``mode_label``, ``result_kind``, and emitted ``edges``; every converter's
    note is appended to ``notes``. The survivor lines and the converted lines
    are then combined and connected to ``final_chip``.

    Parameters
    ----------
    source_chip
        The original (pre-reduction) chip the converters read from.
    final_chip
        The reduced chip the equipment is attached to.
    equipment
        The source chip's control equipment, or ``None``.
    survivor_lines
        Lines that pass through unchanged, from :func:`plan_stranded_lines`.
    retarget_plan
        ``(line, rule)`` pairs from :func:`plan_stranded_lines`.
    mode_label
        The eliminated device (or coupling) label, for the retarget context.
    result_kind
        The reduction's result kind, for the retarget context.
    edges
        The per-pair emitted-edge entries (``"edge"``/``"crosskerr"``) or
        ``None`` (``"leaf-fold"``), for the retarget context.
    notes
        The result's note list, extended in place with each converter's note.
    """
    if equipment is None:
        return
    retargeted_lines: list[Any] = []
    retargeted_transforms: list[Any] = []
    if retarget_plan:
        ctx = RetargetContext(
            chip=source_chip,
            reduced_chip=final_chip,
            mode_label=mode_label,
            result_kind=result_kind,
            edges=edges,
        )
        for line, rule in retarget_plan:
            converted = rule(line, ctx)
            retargeted_lines.extend(converted.lines)
            retargeted_transforms.extend(converted.transforms)
            if converted.note:
                notes.append(converted.note)
    # copy() clones the surviving lines and rebinds them to the reduced
    # chip's device (and coupling) map; retargeted lines already target
    # the reduced chip and are appended as-is. The signal chain is
    # copied verbatim — a retargeted line keeps its original label, so
    # any Crosstalk/Delay entry keyed by it stays valid. connect()
    # re-validates and swaps in the canonical instances.
    copied = ControlEquipment(lines=survivor_lines, signal_chain=equipment.signal_chain).copy(
        final_chip.device_map, final_chip.coupling_map
    )
    final_chip.connect(
        ControlEquipment(
            lines=copied.lines + retargeted_lines,
            signal_chain=copied.signal_chain + retargeted_transforms,
        )
    )


def detach_intermediate_clone(devices: Any, clone: Any) -> None:
    """Detach the intermediate clone from the surviving devices deterministically.

    Parameters
    ----------
    devices
        The surviving devices, now owned by the reduced chip.
    clone
        The intermediate clone the reduction built the survivors from.
    """
    # The intermediate clone is garbage from here on, but its chip↔analysis
    # reference cycle keeps it alive until the *cyclic* GC runs — long enough
    # to shadow the reduced chip as a second live owner of the surviving
    # devices when the engine resolves frames through them. Detach it
    # deterministically.
    for dev in devices:
        dev._detach_chip(clone)
