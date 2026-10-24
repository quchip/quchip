"""Build and solve backend-neutral stationary Lindblad problems."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any

import numpy as np

from quchip.backend.containers import PreparedStationary
from quchip.engine.ir import SteadyStateProblem
from quchip.engine.observables import decompose_eops, require_eops_mapping
from quchip.engine.routes import WORKFLOWS, liouvillian_route, plan_sectors, require_workflow, sector_route
from quchip.results.steady_state import (
    SteadyStateBatchResult,
    SteadyStateResult,
    build_steady_state_result,
    stationary_expectations,
)
from quchip.utils.jax_utils import maybe_concrete_scalar

_VACUUM = ("Without a drive the vacuum is stationary, and the lowering channels empty every sector with N >= 1, "
           "so the vacuum is the only stationary state.")


def build_steadystate_problem(
    chip: Any,
    *,
    e_ops: dict | None = None,
    options: dict | None = None,
    frame: Any | None = None,
    approximation: Any | None = None,
) -> SteadyStateProblem:
    """Resolve a chip into one static Lindblad steady-state request.

    Under an excitation cutoff, the request holds the projected model on the
    retained labels.
    """
    require_workflow(chip, "steadystate()")
    require_eops_mapping(e_ops)
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
        solver_e_ops, e_ops_meta = decompose_eops(e_ops, chip, chip.backend, engine_result.bases, engine_result)
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
    """Solve the unique static Lindblad steady state of a chip.

    A conserving chip with an excitation cutoff, or with more than 64 product
    states, returns the vacuum on its labels with ``N <= 1`` when the lowering
    channels prove it unique. Other chips, and every call with ``options``,
    solve the backend's stationary Liouvillian.
    """
    from quchip.observables import is_output_field

    require_workflow(chip, "steadystate()")
    require_eops_mapping(e_ops)
    workflow = WORKFLOWS["steadystate()"]
    if approximation is not None and approximation != chip.approximation:
        reason = "An approximation other than the chip's selects the backend's stationary Liouvillian."
    elif e_ops is not None and any(is_output_field(value) for value in e_ops.values()):
        reason = "Output observables need the resolved model, so the backend's stationary Liouvillian solves it."
    else:
        # The planner runs first, because the vacuum needs neither a solve nor the resolved product model.
        model, reason = plan_sectors(chip, workflow, labels=(), options=options)
        if model is not None:
            if model.unique(model.max_total if model.sectors.cutoff is None else model.sectors.cutoff):
                return _vacuum_result(chip, e_ops, model, frame,
                                      sector_route(model, 1, f"{reason} {_VACUUM}", solved=False))
            reason = (f"{reason} The lowering channels do not prove one stationary state, so the backend's "
                      "stationary Liouvillian solves it.")
    problem = build_steadystate_problem(
        chip,
        e_ops=e_ops,
        options=options,
        frame=frame,
        approximation=approximation,
    )
    return solve_steadystate_problem(problem, reason=reason)


def solve_steadystate_problem(
    problem: SteadyStateProblem, *, prepared: PreparedStationary | None = None, guess: Any | None = None,
    reason: str = "The request names the backend's stationary Liouvillian.",
) -> SteadyStateResult:
    """Solve an already assembled stationary problem and enforce uniqueness.

    ``guess`` is a native stationary state of a related problem. The backend
    can reuse it when it is also stationary here. ``reason`` explains the
    route in the result's diagnostics.
    """
    backend = problem.backend
    options = {"prepared": prepared, "guess": guess}
    backend_result = backend.steadystate(problem, **{key: value for key, value in options.items() if value is not None})
    nullity = maybe_concrete_scalar(backend_result.nullity)
    if nullity is not None and int(nullity) != 1:
        raise ValueError(
            "steadystate() requires a unique stationary state; "
            f"the resolved Liouvillian has nullity {int(nullity)}."
        )
    route = liouvillian_route(backend, problem.engine_result, reason)
    return build_steady_state_result(backend_result, problem, backend, diagnostics={"solver": route.name,
                                                                                     "route": route})


def _vacuum_result(chip: Any, e_ops: dict | None, model: Any, frame: Any, route: Any) -> SteadyStateResult:
    """Return the vacuum on the labels with ``N <= 1``, with observables on those labels.

    The vacuum is the same in every frame, so the result resolves only the
    frame record, not the model.
    """
    from quchip.engine.assembly import _prepare_engine_assembly
    from quchip.engine.pulse_space import PulseSpace

    backend, bases = chip.backend, model.sectors.bases
    from quchip.engine.frames import static_frame

    _, resolved_frame = _prepare_engine_assembly(chip, static_frame(chip, frame), chip.approximation)
    # A one-label density matrix reads as a ket in some backends, so the space keeps N = 1 as well.
    space = PulseSpace.create(model.dims, tuple(device.label for device in chip.devices), (0, 1), bases)
    vacuum = np.zeros((space.size, space.size), dtype=complex)
    vacuum[0, 0] = 1.0
    state = backend.from_array(vacuum, dims=[[space.size], [space.size]])
    operators, meta = (None, None) if e_ops is None else decompose_eops(e_ops, chip, backend, bases, space=space)
    values = None if operators is None else [backend.expect(operator, state) for operator in operators]
    return SteadyStateResult(
        state=state,
        residual=0.0,
        nullity=1,
        _condition_number=None,
        dims=space.dims,
        device_info=tuple((device.label, device.computational) for device in chip.devices),
        stats=MappingProxyType({"uniqueness_checked": True}),
        _backend=backend,
        _expectations=stationary_expectations(values, meta, resolved_frame, backend),
        frame=resolved_frame,
        excitation_space=space,
        diagnostics=MappingProxyType({"solver": route.name, "route": route, "retained_dimension": space.size}),
    )


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
