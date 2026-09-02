"""Partition-aware dispatch for :func:`quchip.engine.simulate`.

Chip-structural logic lives in :mod:`quchip.chip.partition`; this module
only orchestrates: decide, split, run one pipeline per component, combine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _uses_field_boundary(drive_ops: list, e_ops: dict | None) -> bool:
    """Whether a solve needs the unsplit PortNetwork reference plane."""
    from quchip.engine.ir import CoherentOp
    from quchip.observables import is_output_observable

    if any(isinstance(operation, CoherentOp) for operation in drive_ops):
        return True
    if not e_ops:
        return False
    return any(is_output_observable(value) for value in e_ops.values())


def _scheduled_coupling_supports(chip: Any, drive_ops: list) -> tuple[tuple[str, ...], ...]:
    """Return coupling endpoints activated only by this solve's drives."""
    supports: list[tuple[str, ...]] = []
    for operation in drive_ops:
        coupling = chip.coupling_map.get(operation.target_label)
        if coupling is None:
            continue
        support = (coupling.device_a_label, coupling.device_b_label)
        if support not in supports:
            supports.append(support)
    return tuple(supports)


def maybe_simulate_partitioned(
    chip: Any,
    drive_ops: list,
    tlist: Any,
    *,
    solver: str | None,
    options: dict | None,
    e_ops: dict | None,
    initial_state: Any | None,
    check_truncation: bool,
    truncation_threshold: float,
    approximation: Any | None,
) -> Any | None:
    """Run per-component solves when the chip splits; ``None`` declines to the joint path."""
    if initial_state is not None and not isinstance(initial_state, Mapping):
        return None
    # Field inputs and outputs are defined at the complete network reference
    # plane. Component solves remain exact for internal observables, but the
    # current result combiner does not yet reconstruct a split field boundary;
    # keep these requests on the ordinary joint path.
    if _uses_field_boundary(drive_ops, e_ops):
        return None
    resolved = chip.resolve(approximation=approximation)
    from quchip.chip.partition import partition_chip

    part = partition_chip(
        chip,
        resolved=resolved,
        extra_supports=_scheduled_coupling_supports(chip, drive_ops),
    )
    if part.is_trivial:
        return None

    from quchip.chip.partition import split_drive_ops, split_e_ops, split_state_mapping
    from quchip.engine import simulate
    from quchip.results.partitioned import PartitionedSimulationResult

    per_ops = split_drive_ops(part, chip, drive_ops)
    per_eops, key_plan = split_e_ops(part, e_ops)
    per_state = (
        split_state_mapping(part, initial_state) if initial_state is not None
        else [None] * len(part)
    )

    results = []
    for comp, ops_i, eops_i, state_i in zip(part.components, per_ops, per_eops, per_state):
        results.append(simulate(
            comp.chip, ops_i, tlist,
            solver=solver, options=options,
            e_ops=eops_i or None,
            initial_state=state_i,
            check_truncation=check_truncation,
            truncation_threshold=truncation_threshold,
            partition=False,
            approximation=approximation,
        ))
    return PartitionedSimulationResult(results, part, key_plan)
