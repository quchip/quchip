"""Stage 4: pack engine physics and solve inputs into a frozen :class:`SolveProblem`.

Responsibilities
----------------
* Ensure the chip is dressed and run stage 1 (:func:`resolve_frame`).
* Run stage 3 (:func:`decompose_eops`) to flatten ``e_ops`` into
  solver-ready bands.
* Run stage 2 (:func:`build_engine_result`) for each variant
  and pack into a single :class:`SolveProblem`, or merge homogeneous
  variants into a :class:`SolveBatch` (``N`` identical skeletons with
  per-element :class:`ScalarModulation` signals).

Collapse operators enter the standard Lindblad master equation
``dρ/dt = −i[H, ρ] + Σₖ D[Lₖ]ρ``. Rates are stored in 1/ns.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.engine.ir import (
    DriveOp,
    EngineResult,
    ScalarModulation,
    SolveBatch,
    SolveProblem,
    _aggregate_batch_metadata,
)
from quchip.engine.stage2_assembly import build_engine_result
from quchip.engine.stage3_observables import decompose_eops
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


@dataclass(frozen=True)
class SolveProblemContext:
    """Shared solve metadata reused across a homogeneous problem batch.

    Built once by :func:`prepare_solve_problem_context` so sweep points can
    skip redundant observable normalization.
    """

    chip: Chip
    tlist: Any
    e_ops: Any
    e_ops_meta: Any
    resolved_frame: Any
    _base_result: EngineResult | None
    solver: str | None
    options: dict[str, Any]
    default_initial_state: Any

    @classmethod
    def from_problem(cls, ref: SolveProblem) -> "SolveProblemContext":
        """Reconstruct a shared context from an existing :class:`SolveProblem`.

        Mirrors :meth:`SolveBatch.element`: it lifts a single concrete problem
        back into the context shape so a homogeneous group of problems can be
        re-batched. The reference problem's already-built ``initial_state``
        becomes the default state specification, and ``options`` is
        defensively copied.
        """
        return cls(
            chip=ref.chip,
            tlist=ref.tlist,
            e_ops=ref.e_ops,
            e_ops_meta=ref.e_ops_meta,
            resolved_frame=ref.resolved_frame,
            _base_result=None,
            solver=ref.solver,
            options=dict(ref.options),
            default_initial_state=ref.initial_state,
        )


def _validate_tlist(tlist: Any) -> None:
    """Validate a solve-time grid; shape checks run regardless of tracing.

    ``tlist`` must be one-dimensional and hold at least two points -- both
    static array facts available on a JAX tracer's abstract shape without
    forcing concretization, so they raise even under ``jax.jit``.
    Finiteness and strict monotonicity are value-dependent and validated
    only when ``tlist`` is concrete.
    """
    if tlist.ndim != 1:
        raise ValueError(f"tlist must be one-dimensional; got shape {tlist.shape}.")
    if tlist.shape[0] < 2:
        raise ValueError(f"tlist must have at least two points; got {tlist.shape[0]}.")

    if contains_tracer(tlist):
        return
    arr = np.asarray(tlist)
    if not np.all(np.isfinite(arr)):
        raise ValueError("tlist must be finite everywhere.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("tlist must be strictly increasing.")


def _validate_drive_op_window(drive_op: DriveOp, tlist: Any) -> None:
    """Require *drive_op*'s pulse window to overlap ``tlist`` with positive measure.

    Raises ``ValueError`` unless
    ``start_time + envelope.duration > tlist[0]`` and
    ``start_time < tlist[-1]`` -- a window that only touches an endpoint
    contributes no evolution. Concrete-only: skipped when the drive's
    ``start_time``/``duration`` or either ``tlist`` endpoint is traced.
    """
    start = maybe_concrete_scalar(drive_op.start_time)
    duration = maybe_concrete_scalar(drive_op.envelope.duration)
    t_start = maybe_concrete_scalar(tlist[0])
    t_stop = maybe_concrete_scalar(tlist[-1])
    if start is None or duration is None or t_start is None or t_stop is None:
        return
    stop = start + duration
    if not (stop > t_start and start < t_stop):
        raise ValueError(
            f"Drive '{drive_op.drive_label}' on target '{drive_op.target_label}' has pulse window "
            f"[{start}, {stop}] with no positive-measure overlap with the solve interval "
            f"[{t_start}, {t_stop}]."
        )


def validate_drive_ops_window(drive_ops: list[DriveOp], tlist: Any) -> None:
    """Validate every ``DriveOp`` in *drive_ops* against :func:`_validate_drive_op_window`."""
    for drive_op in drive_ops:
        _validate_drive_op_window(drive_op, tlist)


def prepare_solve_problem_context(
    chip: Chip,
    tlist: Any,
    *,
    solver: str | None = None,
    options: dict | None = None,
    e_ops: dict | None = None,
    drive_ops: list[DriveOp] | None = None,
) -> SolveProblemContext:
    """Resolve the frame and retain authored observables and state specifications.

    Observables and states are materialized only after stage 2 resolves every
    local solver basis. The default ground state remains lazy, so callers that
    provide an explicit state do not pay for unused state construction.

    ``tlist`` is validated by :func:`_validate_tlist`. When *drive_ops* is
    given, each entry's pulse window is checked against ``tlist`` via
    :func:`validate_drive_ops_window`; omit it (the default) when the
    caller validates its own per-variant drive ops elsewhere (see
    :meth:`~quchip.control.sequence.QuantumSequence.build_batch`).
    """
    backend = chip.backend
    tlist_arr = backend.array_module.asarray(tlist, dtype=float)
    _validate_tlist(tlist_arr)
    if drive_ops is not None:
        validate_drive_ops_window(drive_ops, tlist_arr)

    merged_options: dict[str, Any] = {"store_states": True, "store_final_state": True}
    if options is not None:
        merged_options.update(options)

    if e_ops is not None and not isinstance(e_ops, dict):
        raise TypeError("e_ops must be dict or None")
    base_result = chip.resolve()
    return SolveProblemContext(
        chip=chip,
        tlist=tlist_arr,
        e_ops=e_ops,
        e_ops_meta=None,
        resolved_frame=base_result.resolved_frame,
        _base_result=base_result,
        solver=solver,
        options=merged_options,
        default_initial_state=None,
    )


def _materialize_context_state(
    context: SolveProblemContext,
    state_spec: Any,
    engine_result: EngineResult,
) -> Any:
    """Materialize one explicit or default state against its own result bases."""
    from quchip.chip.states import materialize_state_spec

    selected = context.default_initial_state if state_spec is None else state_spec
    return materialize_state_spec(context.chip, selected, engine_result.bases)


def _prepare_context_eops(
    context: SolveProblemContext,
    engine_result: EngineResult,
) -> tuple[Any, Any]:
    """Project raw observable specs once the engine basis maps are available."""
    if not isinstance(context.e_ops, dict):
        return context.e_ops, context.e_ops_meta
    return decompose_eops(
        context.e_ops,
        context.chip,
        context.chip.backend,
        engine_result.bases,
    )


def build_solve_batch_from_results(
    context: SolveProblemContext,
    engine_results: list[EngineResult],
    *,
    initial_states: list[Any] | None = None,
) -> SolveBatch:
    """Package homogeneous :class:`EngineResult`s as one :class:`SolveBatch`.

    All results must share ``static_terms`` identity, the same number
    of dynamic terms, and matching operator payloads per slot (by identity
    or by canonical fingerprint — crosstalk rebuilds equal-by-value
    operators on every instantiation). ``initial_states=None`` fills every
    element with ``context.default_initial_state``.
    """
    if not engine_results:
        raise ValueError("build_solve_batch_from_results requires at least one engine result")

    ref = engine_results[0]
    batch_size = len(engine_results)
    n_dyn = len(ref.dynamic_terms)
    _prefix = "build_solve_batch_from_results: "

    # --- Skeleton checks: static terms, dim shape, dynamic term count ---
    for idx, result in enumerate(engine_results):
        if result.static_terms is not ref.static_terms:
            raise ValueError(
                _prefix + "all engine results must share identical static_terms (by identity); "
                f"element {idx} differs."
            )
        if len(result.dynamic_terms) != n_dyn:
            raise ValueError(
                _prefix + "all engine results must have the same number of dynamic terms; "
                f"element {idx} has {len(result.dynamic_terms)}, expected {n_dyn}."
            )
        if tuple(result.dims) != tuple(ref.dims):
            raise ValueError(
                _prefix + "all engine results must share identical dims; "
                f"element {idx} has {tuple(result.dims)}, expected {tuple(ref.dims)}."
            )

    # --- Per-slot compatibility + signal collection ---
    # Per element, verify the dynamic term matches the reference in
    # (operator payload, origin, tag) and is a ScalarModulation; collect
    # its signal. Crosstalk rebuilds operators every instantiation, so
    # equality is by canonical fingerprint, not by object identity.
    for slot in range(n_dyn):
        ref_term = ref.dynamic_terms[slot]
        shared_operator = ref_term.operator
        canonical_key = shared_operator.fingerprint()
        for idx, result in enumerate(engine_results):
            term = result.dynamic_terms[slot]
            where = f"slot {slot}, element {idx}"

            if not isinstance(term.time_dependence, ScalarModulation):
                raise ValueError(
                    _prefix + f"only ScalarModulation time dependencies are supported ({where})."
                )
            if term.origin != ref_term.origin:
                raise ValueError(
                    _prefix + f"dynamic term origin differs at {where} "
                    f"({term.origin!r} vs {ref_term.origin!r})."
                )
            if term.tag != ref_term.tag:
                raise ValueError(
                    _prefix + f"dynamic term tag differs at {where} "
                    f"({term.tag!r} vs {ref_term.tag!r})."
                )
            if term.operator is not shared_operator and term.operator.fingerprint() != canonical_key:
                raise ValueError(
                    _prefix + f"dynamic operator {where} differs from the slot reference; "
                    "batched IR requires equivalent operator payloads across the batch."
                )

    if initial_states is None:
        states: tuple[Any, ...] = tuple(
            _materialize_context_state(context, None, result)
            for result in engine_results
        )
    elif len(initial_states) != batch_size:
        raise ValueError(
            f"initial_states length {len(initial_states)} does not match batch_size {batch_size}"
        )
    else:
        states = tuple(
            _materialize_context_state(context, state_spec, result)
            for state_spec, result in zip(initial_states, engine_results)
        )

    shared_metadata = _aggregate_batch_metadata(engine_results)
    e_ops, e_ops_meta = _prepare_context_eops(context, ref)
    problems: list[SolveProblem] = []
    for state, result in zip(states, engine_results):
        problems.append(
            SolveProblem(
                chip=context.chip,
                engine_result=replace(result, metadata=shared_metadata),
                initial_state=state,
                tlist=context.tlist,
                e_ops=e_ops,
                e_ops_meta=e_ops_meta,
                resolved_frame=context.resolved_frame,
                solver=context.solver,
                options=context.options,
            )
        )
    return SolveBatch(chip=context.chip, problems=tuple(problems))


def build_solve_problem(
    chip: Chip,
    drive_ops: list[DriveOp],
    tlist: Any,
    *,
    solver: str | None = None,
    options: dict | None = None,
    e_ops: dict | None = None,
    initial_state: Any | None = None,
) -> SolveProblem:
    """Run stages 1-4 end-to-end and return a frozen :class:`SolveProblem`.

    Equivalent to :func:`prepare_solve_problem_context` followed by
    :func:`build_engine_result`. For many variants sharing one
    chip configuration, prefer that two-step form with
    :func:`build_solve_batch_from_results`.
    """
    context = prepare_solve_problem_context(
        chip, tlist, solver=solver, options=options, e_ops=e_ops, drive_ops=drive_ops,
    )
    engine_result = build_engine_result(
        chip,
        drive_ops,
        resolved_frame=context.resolved_frame,
        _base_result=context._base_result,
    )
    e_ops_solver, e_ops_meta = _prepare_context_eops(context, engine_result)
    return SolveProblem(
        chip=context.chip,
        engine_result=engine_result,
        initial_state=_materialize_context_state(context, initial_state, engine_result),
        tlist=context.tlist,
        e_ops=e_ops_solver,
        e_ops_meta=e_ops_meta,
        resolved_frame=context.resolved_frame,
        solver=context.solver,
        options=context.options,
    )


def solve_problem_list(
    problems: list[SolveProblem],
    backend: Any,
    *,
    progress: bool = True,
) -> Any:
    """Group problems by shared operator skeleton and dispatch as :class:`SolveBatch`es.

    Problems that share an operator skeleton are merged into one batched solve.
    A backend may dispatch a large heterogeneous list independently; otherwise
    each structural group follows the normal batch path, with incompatible
    results falling back to per-problem ``backend.solve_problem`` calls.

    Grouping is a two-stage filter. The cheap identity-based prefilter here
    (:func:`_skeleton_prefilter_key`) buckets problems by ``id()`` of their
    shared operators/metadata so that obviously-incompatible problems are never
    compared by value. The canonical by-value compatibility check is intentionally
    a *separate* concern that lives inside
    :func:`build_solve_batch_from_results` (operator ``fingerprint``):
    the prefilter is an identity prefilter, the fingerprint is the value check.
    Returns a :class:`~quchip.results.results.SimulationBatchResult`.
    """
    from quchip.results.results import (
        SimulationBatchResult,
        wrap_solver_result,
        wrap_solver_results_from_batch,
    )

    _tlist_cache: dict[int, tuple] = {}

    def _options_key(opts: dict) -> tuple:
        items = []
        for key in sorted(opts.keys(), key=str):
            val = opts[key]
            try:
                hash(val)
                items.append((str(key), val))
            except TypeError:
                items.append((str(key), repr(val)))
        return tuple(items)

    def _tlist_key(tlist: Any) -> tuple:
        if tlist is None:
            return ("none",)
        obj_id = id(tlist)
        cached = _tlist_cache.get(obj_id)
        if cached is not None:
            return cached
        arr = np.asarray(tlist)
        key = ("tlist", arr.shape, str(arr.dtype), arr.tobytes())
        _tlist_cache[obj_id] = key
        return key

    def _op_list_key(ops: Any) -> tuple:
        if ops is None:
            return ("none",)
        return ("ops", tuple(id(o) for o in ops))

    def _skeleton_prefilter_key(problem: SolveProblem) -> tuple:
        desc = problem.engine_result
        solver_name = problem.solver or ("mesolve" if desc.collapse_terms else "sesolve")
        return (
            solver_name,
            id(desc.static_terms),
            tuple(id(term.operator) for term in desc.dynamic_terms),
            tuple(term.origin for term in desc.dynamic_terms),
            tuple(term.tag for term in desc.dynamic_terms),
            _tlist_key(problem.tlist),
            _op_list_key(problem.e_ops),
            tuple(id(term.operator) for term in desc.collapse_terms),
            _options_key(problem.options),
            id(problem.resolved_frame),
        )

    groups: dict[tuple, list[tuple[int, SolveProblem]]] = {}
    for idx, problem in enumerate(problems):
        groups.setdefault(_skeleton_prefilter_key(problem), []).append((idx, problem))

    if len(groups) > 1:
        parallel_results = backend.parallel_solve_problems(problems, progress=progress)
        if parallel_results is not None:
            if len(parallel_results) != len(problems):
                raise RuntimeError(
                    "parallel_solve_problems returned "
                    f"{len(parallel_results)} results for {len(problems)} problems."
                )
            wrapped = [
                wrap_solver_result(result, problem, backend)
                for problem, result in zip(problems, parallel_results)
            ]
            return SimulationBatchResult(wrapped)

    ordered_results: list[Any] = [None] * len(problems)
    for group in groups.values():
        indices = [i for i, _ in group]
        group_problems = [p for _, p in group]
        ref = group_problems[0]
        try:
            ctx = SolveProblemContext.from_problem(ref)
            batch = build_solve_batch_from_results(
                ctx,
                [p.engine_result for p in group_problems],
                initial_states=[p.initial_state for p in group_problems],
            )
        except ValueError:
            for idx_original, problem in zip(indices, group_problems):
                result = backend.solve_problem(problem)
                ordered_results[idx_original] = wrap_solver_result(result, problem, backend)
            continue

        solver_results = backend.solve_batch(batch, progress=progress)
        for idx_original, wrapped_result in zip(
            indices, wrap_solver_results_from_batch(solver_results, batch, backend)
        ):
            ordered_results[idx_original] = wrapped_result

    missing = [idx for idx, result in enumerate(ordered_results) if result is None]
    if missing:
        raise RuntimeError(
            f"solve_problem_list failed to populate results for problem indices {missing}; "
            "backend returned incomplete results."
        )
    return SimulationBatchResult(ordered_results)
