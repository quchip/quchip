"""Control retargeting and graph reconstruction shared by transformations.

Handlers identify removed targets and choose the reduced physics. These helpers
keep source settings and connect converted equipment to the rebuilt components.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from quchip.chip.retarget import RetargetContext, lookup_retarget_rule
from quchip.control.equipment import ControlEquipment
from quchip.utils.values import copy_value


@dataclass(frozen=True)
class StrandedLine:
    """A removed control target and the error when no retarget rule exists."""

    rule_target: Any
    missing_rule_message: str


def plan_stranded_lines(
    equipment: Any,
    classify: Callable[[Any], StrandedLine | None],
    result_kind: str,
) -> tuple[list[Any], list[tuple[Any, Any]]]:
    """Classify lines and resolve retarget rules before computing the reduction.

    ``classify(line)`` returns ``None`` for a survivor or a ``StrandedLine``
    for a removed target. ``result_kind`` selects the rule (``edge``,
    ``leaf-fold`` or ``crosskerr``). Missing rules raise the target's message.
    Return surviving lines and ``(line, rule)`` pairs in source order. The
    rules are applied only after the reduced model and the emitted edges exist."""
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


def inherited_notes(chip: Any) -> tuple[str, ...]:
    """Return the producer notes of ``chip``'s effective terms, prefixed by their labels.

    A reduction folds every earlier effective contribution into its own, so the
    approximations those contributions state also apply to the result."""
    return tuple(f"{terms.label}: {note}" for terms in chip.effective_terms for note in terms.notes)


def rebuild_chip(
    source_chip: Any,
    *,
    devices: Any,
    couplings: Any,
    port_replacements: dict[str, Any] | None = None,
    effective_terms: Any = None,
    baths: Any = None,
) -> Any:
    """Rebuild the retained graph with the source's calculation settings.

    Keep the backend, basis, approximation, label and surviving state shorthand.
    Filter per-device frames to retained labels. Copy baths and effective terms,
    unless replacements are supplied. Removed port targets need explicit
    replacements. Control equipment is attached separately."""
    from quchip.chip.chip import Chip

    device_list = list(devices)
    survivor_labels = {device.label for device in device_list}
    replacements = {} if port_replacements is None else dict(port_replacements)
    for port in source_chip.ports:
        targets = set(port.resolve_targets(source_chip))
        removed = targets - survivor_labels
        if removed and port.label not in replacements:
            raise NotImplementedError(
                f"Transformation removes {sorted(removed)}, targeted by port {port.label!r}. "
                "Keep the port-coupled device; an effective input-output port requires an explicit retarget rule."
            )
    network = (
        None
        if source_chip.port_network is None
        else source_chip.port_network._copy_with_port_replacements(replacements)
    )

    rebuilt = Chip(
        devices=device_list,
        couplings=list(couplings) or None,
        label=source_chip.label,
        frame=copy_value(
            {key: value for key, value in source_chip.frame.items() if key in survivor_labels}
            if isinstance(source_chip.frame, dict) else source_chip.frame
        ),
        approximation=source_chip.approximation,
        basis=source_chip.basis,
        backend=source_chip._backend,
        baths=[bath.copy() for bath in source_chip.baths] if baths is None else baths,
        port_network=network,
        effective_terms=source_chip.effective_terms if effective_terms is None else effective_terms,
    )
    from quchip.chip.states import copy_state_configuration

    copy_state_configuration(source_chip, rebuilt)
    return rebuilt


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
    """Apply retarget rules and reconnect equipment to the retained graph.

    Rules receive the source, reduced chip, removed target and emitted edges
    through ``RetargetContext``. Their notes are appended to ``notes``.
    Surviving lines and transforms are copied. Converted operators are marked
    as already using retained coordinates. Without equipment, do nothing."""
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
    if retargeted_lines:
        final_chip._effective_terms = tuple(
            replace(terms, projection=terms.projection.with_current_operators(
                tuple(f"drive:{line.label}" for line in retargeted_lines)))
            if terms.projection is not None else terms
            for terms in final_chip.effective_terms
        )
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
