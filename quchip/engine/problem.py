"""Package engine physics and solve inputs into frozen solve requests.

Responsibilities
----------------
* Resolve the chip frame.
* Flatten ``e_ops`` into solver-ready bands with :func:`decompose_eops`.
* Build an :class:`EngineResult` for each variant
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

from quchip.approximations import Approximation
from quchip.backend import BatchSolveError
from quchip.chip.states import materialize_state_spec
from quchip.engine.ir import (
    ControlOp,
    EngineResult,
    ScalarModulation,
    SolveBatch,
    SolveProblem,
    StateStorage,
    _aggregate_batch_metadata,
)
from quchip.engine.assembly import build_engine_result
from quchip.engine.frames import resolve_for_operations
from quchip.engine.observables import decompose_eops
from quchip.engine.sampling import AutomaticTimeGrid, sample_problems
from quchip.utils.jax_utils import contains_tracer

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
    resolved_frame: Any
    approximation: Approximation
    _base_result: EngineResult | None
    solver: str | None
    options: dict[str, Any]
    run_args: dict[str, Any]
    states: StateStorage | None = None
    dissipation: bool = True
    automatic_sampling: bool = False



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


def prepare_solve_problem_context(
    chip: Chip,
    tlist: Any,
    *,
    solver: str | None = None,
    options: dict | None = None,
    e_ops: dict | None = None,
    drive_ops: list[ControlOp] | None = None,
    approximation: Approximation | None = None,
    frame: Any = None,
    states: StateStorage | None = None,
    dissipation: bool = True,
    run_args: dict | None = None,
) -> SolveProblemContext:
    """Resolve the frame and retain authored observables and state specifications.

    Observables and states are materialized only after assembly resolves every
    local solver basis. The default ground state remains lazy, so callers that
    provide an explicit state do not pay for unused state construction.

    ``frame`` overrides the chip's declared frame. For ``"auto"``, the solve
    window ``tlist[-1] - tlist[0]`` supplies the duration used for weights.

    ``tlist`` defines the actual interval. Scheduled signals retain their
    absolute times, including operations partly or wholly outside this interval.
    """
    if not isinstance(dissipation, bool):
        raise TypeError("dissipation must be a boolean calculation choice.")
    backend = chip.backend
    automatic = isinstance(tlist, AutomaticTimeGrid)
    tlist_arr = backend.array_module.asarray(tlist.bounds if automatic else tlist, dtype=float)
    _validate_tlist(tlist_arr)

    if e_ops is not None and not isinstance(e_ops, dict):
        raise TypeError("e_ops must be dict or None")
    base_result = resolve_for_operations(
        chip,
        drive_ops or [],
        frame=frame,
        approximation=approximation,
        solve_window=(tlist_arr[0], tlist_arr[-1]),
    )
    return SolveProblemContext(
        chip=chip,
        tlist=tlist if isinstance(tlist, tuple) and solver in ("jssesolve", "dssesolve") else tlist_arr,
        e_ops=e_ops,
        resolved_frame=base_result.resolved_frame,
        approximation=base_result.approximation,
        _base_result=base_result,
        solver=solver,
        options={} if options is None else options,
        run_args={} if run_args is None else run_args,
        automatic_sampling=automatic,
        states=states,
        dissipation=dissipation,
    )


def _prepare_context_eops(
    context: SolveProblemContext,
    engine_result: EngineResult,
) -> tuple[Any, Any]:
    """Project raw observable specs once the engine basis maps are available."""
    if context.e_ops is None:
        return None, None
    return decompose_eops(
        context.e_ops,
        context.chip,
        context.chip.backend,
        engine_result.bases,
        engine_result=engine_result,
    )


def _validate_batch_skeleton(engine_results: list[EngineResult]) -> None:
    """Require equivalent Hamiltonian skeletons without rebuilding solve inputs."""
    ref = engine_results[0]
    n_dyn = len(ref.dynamic_terms)
    _prefix = "build_solve_batch_from_results: "

    # --- Skeleton checks: static terms, dim shape, dynamic term count ---
    for idx, result in enumerate(engine_results):
        if result.approximation != ref.approximation:
            raise ValueError(
                _prefix + "all engine results must use the same approximation; "
                f"element {idx} uses {type(result.approximation).__name__}, "
                f"expected {type(ref.approximation).__name__}."
            )
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
    operators on every instantiation). ``initial_states=None`` constructs
    each element's default ground state in its resolved basis.
    """
    if not engine_results:
        raise ValueError("build_solve_batch_from_results requires at least one engine result")

    _validate_batch_skeleton(engine_results)
    ref = engine_results[0]
    batch_size = len(engine_results)

    if initial_states is None:
        states: tuple[Any, ...] = tuple(
            materialize_state_spec(context.chip, None, result.bases)
            for result in engine_results
        )
    elif len(initial_states) != batch_size:
        raise ValueError(
            f"initial_states length {len(initial_states)} does not match batch_size {batch_size}"
        )
    else:
        states = tuple(
            materialize_state_spec(context.chip, state_spec, result.bases)
            for state_spec, result in zip(initial_states, engine_results)
        )

    shared_metadata = _aggregate_batch_metadata(engine_results)
    e_ops, e_ops_meta = _prepare_context_eops(context, ref)
    problems: list[SolveProblem] = []
    for state, result in zip(states, engine_results):
        problems.append(
            SolveProblem(
                chip=context.chip,
                engine_result=replace(result, metadata=shared_metadata, dissipation=context.dissipation),
                initial_state=state,
                tlist=context.tlist,
                e_ops=e_ops,
                e_ops_meta=e_ops_meta,
                resolved_frame=context.resolved_frame,
                solver=context.solver,
                options=context.options,
                run_args=context.run_args,
                states=context.states,
            )
        )
    if context.automatic_sampling:
        problems = sample_problems(problems)
    return SolveBatch(chip=context.chip, problems=tuple(problems))


def build_solve_problem(
    chip: Chip,
    drive_ops: list[ControlOp],
    tlist: Any,
    *,
    solver: str | None = None,
    options: dict | None = None,
    e_ops: dict | None = None,
    initial_state: Any | None = None,
    approximation: Approximation | None = None,
    frame: Any = None,
    states: StateStorage | None = None,
    dissipation: bool = True,
    run_args: dict | None = None,
) -> SolveProblem:
    """Resolve, assemble, and package a frozen :class:`SolveProblem`.

    Equivalent to :func:`prepare_solve_problem_context` followed by
    :func:`build_engine_result`. For many variants sharing one
    chip configuration, prefer that two-step form with
    :func:`build_solve_batch_from_results`.

    ``frame`` overrides the chip's declared frame and follows the same
    operation-aware ``"auto"`` resolution as
    :func:`prepare_solve_problem_context`.
    """
    context = prepare_solve_problem_context(
        chip,
        tlist,
        solver=solver,
        options=options, run_args=run_args,
        e_ops=e_ops,
        drive_ops=drive_ops,
        approximation=approximation,
        frame=frame,
        states=states,
        dissipation=dissipation,
    )
    engine_result = build_engine_result(
        chip,
        drive_ops,
        resolved_frame=context.resolved_frame,
        approximation=context.approximation,
        _base_result=context._base_result,
    )
    engine_result = replace(engine_result, dissipation=context.dissipation)
    e_ops_solver, e_ops_meta = _prepare_context_eops(context, engine_result)
    problem = SolveProblem(
        chip=context.chip,
        engine_result=engine_result,
        initial_state=materialize_state_spec(context.chip, initial_state, engine_result.bases),
        tlist=context.tlist,
        e_ops=e_ops_solver,
        e_ops_meta=e_ops_meta,
        resolved_frame=context.resolved_frame,
        solver=context.solver,
        options=context.options,
        run_args=context.run_args,
        states=context.states,
    )
    return sample_problems([problem])[0] if context.automatic_sampling else problem


def solve_problem_list(problems: list[SolveProblem], *, progress: bool = True,
                       parameters: tuple[dict[str, Any], ...] | None = None) -> Any:
    """Dispatch each captured backend's requests and restore original point order."""
    from quchip.results.results import SimulationBatchResult

    problems = assign_point_noise(problems)
    groups: dict[int, list[tuple[int, SolveProblem]]] = {}
    for index, problem in enumerate(problems):
        groups.setdefault(id(problem.backend), []).append((index, problem))
    ordered: list[Any] = [None] * len(problems)
    for entries in groups.values():
        try:
            results = _solve_backend_problems([problem for _, problem in entries],
                entries[0][1].backend, progress=progress,
                failure_context=tuple((index, parameters[index] if parameters is not None else {})
                                      for index, _ in entries))
        except BatchSolveError as exc:
            raise BatchSolveError(entries[exc.index][0], exc.detail, exc.parameters) from exc
        for (index, _), result in zip(entries, results, strict=True):
            ordered[index] = result
    return SimulationBatchResult(ordered)


def _solve_backend_problems(
    problems: list[SolveProblem],
    backend: Any,
    *,
    progress: bool = True,
    failure_context: tuple[tuple[int, dict[str, Any]], ...] = (),
) -> Any:
    """Group problems by shared operator skeleton and dispatch as :class:`SolveBatch`es.

    Problems that share an operator skeleton are merged into one batched solve.
    A backend may dispatch a large heterogeneous list independently; otherwise
    each structural group follows the normal batch path, with incompatible
    results falling back to per-problem ``backend.solve_problem`` calls.

    Grouping first checks shared Hamiltonian/metadata identity and compares
    captured observable values, grids and options. Canonical operator
    fingerprints then establish compatibility within each group.
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
        if contains_tracer(tlist):
            return ("traced_tlist", obj_id)
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
        from quchip.utils.values import value_fingerprint

        try:
            return ("ops", tuple(value_fingerprint(backend.to_array(op)) for op in ops))
        except ValueError:
            return ("opaque_ops", object())

    def _skeleton_prefilter_key(problem: SolveProblem) -> tuple:
        desc = problem.engine_result
        solver_name = problem.solver_name(problem.backend)
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
            _options_key({key: value for key, value in problem.run_args.items() if key not in ("keys", "seeds")}),
            problem.states,
            _options_key(problem.monitoring or {}),
            id(problem.resolved_frame),
        )

    groups: dict[tuple, list[tuple[int, SolveProblem]]] = {}
    for idx, problem in enumerate(problems):
        groups.setdefault(_skeleton_prefilter_key(problem), []).append((idx, problem))

    if len(groups) > 1 and not any(problem.stochastic for problem in problems):
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
            _validate_batch_skeleton([problem.engine_result for problem in group_problems])
            batch = SolveBatch(chip=ref.chip, problems=tuple(group_problems),
                _failure_context=tuple(failure_context[index] for index in indices) if failure_context else ())
        except ValueError:
            for idx_original, problem in zip(indices, group_problems):
                try:
                    result = backend.solve_problem(problem)
                except Exception as exc:
                    raise BatchSolveError(idx_original, f"{type(exc).__name__}: {exc}") from exc
                ordered_results[idx_original] = wrap_solver_result(result, problem, backend)
            continue

        try:
            solver_results = backend.solve_batch(batch, progress=progress)
        except BatchSolveError as exc:
            raise BatchSolveError(indices[exc.index], exc.detail, exc.parameters) from exc
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


def assign_point_noise(problems: list[SolveProblem], *, split_keys: bool = False) -> list[SolveProblem]:
    """Assign omitted QuTiP seeds in logical point order before grouping.

    Explicit native seeds/keys are retained, including deliberately shared noise.
    Dynamiqs keys remain a required native input.
    """
    if split_keys and any("keys" in problem.run_args for problem in problems):
        import jax
        from jax import random

        assigned = []
        for index, problem in enumerate(problems):
            keys = problem.run_args.get("keys")
            if problem.stochastic and keys is not None:
                typed = jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key)
                explicit = keys.ndim == (2 if typed else 3)
                point_keys = keys[index] if explicit else jax.vmap(lambda key: random.fold_in(key, index))(keys)
                problem = replace(problem, run_args={**problem.run_args, "keys": point_keys})
            assigned.append(problem)
        problems = assigned
    if not any(p.solver in ("mcsolve", "ssesolve", "smesolve") and "seeds" not in p.run_args for p in problems):
        return problems
    seeds = np.random.SeedSequence().spawn(len(problems))
    return [replace(problem, run_args={**problem.run_args, "seeds": seed})
            if problem.solver in ("mcsolve", "ssesolve", "smesolve") and "seeds" not in problem.run_args
            else problem for problem, seed in zip(problems, seeds)]
