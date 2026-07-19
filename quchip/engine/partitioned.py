"""Partition-aware dispatch for :func:`quchip.engine.simulate`.

Chip-structural logic lives in :mod:`quchip.chip.partition`; this module
only orchestrates: decide, split, run one pipeline per component, combine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
) -> Any | None:
    """Run per-component solves when the chip splits; ``None`` declines to the joint path."""
    if initial_state is not None and not isinstance(initial_state, Mapping):
        return None
    part = chip.partition()
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
        ))
    return PartitionedSimulationResult(results, part, key_plan)
