"""Build and solve backend-neutral stationary Lindblad problems."""

from __future__ import annotations

import math
from typing import Any

from quchip.engine.ir import SteadyStateProblem
from quchip.engine.observables import decompose_eops
from quchip.results.steady_state import (
    SteadyStateBatchResult,
    SteadyStateResult,
    build_steady_state_result,
)
from quchip.utils.jax_utils import maybe_concrete_scalar


def build_steadystate_problem(
    chip: Any,
    *,
    e_ops: dict | None = None,
    options: dict | None = None,
    frame: Any | None = None,
    approximation: Any | None = None,
) -> SteadyStateProblem:
    """Resolve a chip into one static Lindblad steady-state request."""
    if e_ops is not None and not isinstance(e_ops, dict):
        raise TypeError("e_ops must be dict or None")
    engine_result = chip.resolve(frame=frame, approximation=approximation)
    if engine_result.dynamic_terms:
        raise ValueError(
            "steadystate() requires a static resolved Hamiltonian; "
            f"found {len(engine_result.dynamic_terms)} dynamic term(s). Use QuantumSequence for time evolution."
        )
    if not engine_result.collapse_terms and math.prod(engine_result.dims) > 1:
        raise ValueError(
            "steadystate() requires a unique stationary state; this closed system has no collapse channels."
        )

    solver_e_ops = None
    e_ops_meta = None
    if e_ops is not None:
        solver_e_ops, e_ops_meta = decompose_eops(e_ops, chip, chip.backend, engine_result.bases)
    return SteadyStateProblem(
        chip=chip,
        engine_result=engine_result,
        e_ops=solver_e_ops,
        e_ops_meta=e_ops_meta,
        resolved_frame=engine_result.resolved_frame,
        options={} if options is None else options,
    )


def steadystate(
    chip: Any,
    *,
    e_ops: dict | None = None,
    options: dict | None = None,
    frame: Any | None = None,
    approximation: Any | None = None,
) -> SteadyStateResult:
    """Solve a chip's unique static Lindblad steady state."""
    problem = build_steadystate_problem(
        chip,
        e_ops=e_ops,
        options=options,
        frame=frame,
        approximation=approximation,
    )
    return solve_steadystate_problem(problem)


def solve_steadystate_problem(problem: SteadyStateProblem) -> SteadyStateResult:
    """Solve an already assembled stationary problem and enforce uniqueness."""
    backend = problem.chip.backend
    backend_result = backend.steadystate(problem)
    nullity = maybe_concrete_scalar(backend_result.nullity)
    if nullity is not None and int(nullity) != 1:
        raise ValueError(
            "steadystate() requires a unique stationary state; "
            f"the resolved Liouvillian has nullity {int(nullity)}."
        )
    return build_steady_state_result(backend_result, problem, backend)


def steadystate_batch(
    chip: Any,
    *axes: Any,
    e_ops: dict | None = None,
    options: dict | None = None,
    frame: Any | None = None,
    approximation: Any | None = None,
    progress: bool = True,
) -> SteadyStateBatchResult:
    """Solve stationary states over Cartesian or zipped parameter axes."""
    from quchip.sweep import _axis_metadata, _iter_axis_points

    shape, expanded = _iter_axis_points(axes)
    iterator = expanded
    if progress:
        from tqdm import tqdm

        iterator = tqdm(expanded, desc="Steady state")

    results = [
        steadystate(
            chip.with_params(params),
            e_ops=e_ops,
            options=options,
            frame=frame,
            approximation=approximation,
        )
        for _, params in iterator
    ]
    return SteadyStateBatchResult(
        results,
        shape=shape,
        axes=_axis_metadata(axes),
    )
