"""Package engine physics and solve inputs into frozen solve requests.

Responsibilities
----------------
* Resolve the chip frame.
* Flatten ``e_ops`` into solver-ready bands with :func:`decompose_eops`.
* Build an :class:`EngineResult` for each variant and pack it into a single
  :class:`SolveProblem`, or merge homogeneous variants into a
  :class:`SolveBatch` (``N`` identical skeletons with per-element
  :class:`ScalarModulation` signals).

Collapse operators enter the standard Lindblad master equation
``dρ/dt = −i[H, ρ] + Σₖ D[Lₖ]ρ``. Rates are stored in 1/ns.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from types import MappingProxyType

import numpy as np

from quchip.approximations import Approximation
from quchip.backend import BatchSolveError
from quchip.engine.states import (
    CompactState,
    compatible_values,
    label_state,
    materialize_compact,
    materialize_state_spec,
)
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
from quchip.engine.frames import FrameMap, IncompatibleFrameError, resolve_for_operations
from quchip.engine.observables import decompose_eops
from quchip.engine.sampling import AutomaticTimeGrid, sample_problems
from quchip.utils.jax_utils import array_namespace, contains_tracer, maybe_concrete_scalar
from quchip.utils.stack import user_stacklevel

if TYPE_CHECKING:
    from quchip.chip.chip import Chip

# Fixed texts, so the default warning filter shows each once per call site.
DEFAULT_FRAME_WARNINGS: Mapping[str, str] = MappingProxyType({
    "rotating": ("The chip sets no frame, so this solve uses the rotating frame at each device's reference_freq "
                 "(by default its dressed 0 -> 1 frequency). Set Chip(frame=...) to choose a frame."),
    "auto": ("The chip sets no frame, and a collapse or port operator does not fit the rotating frame, so this "
             "solve uses the frame that frame='auto' plans. Set Chip(frame=...) to choose a frame."),
    "lab": ("The chip sets no frame, and a collapse or port operator fits neither the rotating frame nor the "
            "frame='auto' plan, so this solve uses the lab frame. Set Chip(frame=...) to choose a frame."),
})


def _name_list(names: Sequence[str]) -> str:
    """Join names as ``'a', 'b' and 'c'``, and shorten more than four to three and a count."""
    shown = list(names) if len(names) <= 4 else [*names[:3], f"{len(names) - 3} more"]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + f" and {shown[-1]}"


# A solve on a chip without a frame uses the first of these frames that fits it, and a batch the first that fits
# every point.
DEFAULT_FRAME_ORDER = ("rotating", "auto", "lab")


def warn_default_frame(mode: str) -> None:
    """Warn which frame a solve on a chip without a frame uses, at the first caller outside quchip."""
    warnings.warn(DEFAULT_FRAME_WARNINGS[mode], UserWarning, stacklevel=user_stacklevel())


def _choose_default_frame(chip: Chip, resolve: Any, resolution: Any) -> tuple[str, str | None, Any]:
    """Resolve a chip without a frame in the first frame of :data:`DEFAULT_FRAME_ORDER` that fits it.

    Returns the frame mode, the reason for a fallback (``None`` for the
    rotating frame), and the output of ``resolve``. Raises when a dressed
    ``0 -> 1`` reference of the rotating frame has no reliable label
    assignment.
    """
    if unreliable := chip.analysis._unreliable_references(_local_resolution=resolution):
        names = [name for name, _ in unreliable]
        raise ValueError(
            "The chip sets no frame, and the rotating frame of a solve needs the dressed 0 -> 1 frequency of each "
            f"device, but the dressed label assignment is unreliable for {_name_list(names)} ({names[0]}: "
            f"{unreliable[0][1]}). Choose a frame with Chip(frame='lab') or chip.set_frame('lab'). frame='auto' "
            "may also work: it gives exchange-coupled devices one frequency. A reference_freq on each of these "
            "devices gives the rotating frame its frequencies instead."
        )
    try:
        return "rotating", None, resolve("rotating")
    except IncompatibleFrameError as error:
        reason = str(error)
    try:
        return "auto", reason, resolve("auto")
    except IncompatibleFrameError as error:
        # The planner sets frames from couplings, cascades and tones, but not from collapse operators.
        return "lab", str(error), resolve("lab")


def build_default_frame_problems(builders: Sequence[Any]) -> tuple[list[SolveProblem], list[Any]]:
    """Build batch points of a chip without a frame, all in one frame, and warn once.

    Each builder is a partial :func:`build_solve_problem`. Every point checks
    the frames of :data:`DEFAULT_FRAME_ORDER`, and the batch uses the first
    frame that fits every point. A point that its own solve would put in an
    earlier frame reads its initial state in that frame, so it starts in the
    same physical state. Returns the problems and the builders bound to the
    batch frame.
    """
    notes: list[Any] = []
    problems = [build(_default_frame_notes=notes) for build in builders]

    def bind(mode: str) -> list[Any]:
        return [functools.partial(build, frame=mode,
                                  _initial_state_frame=None if note[0] == mode else problem.resolved_frame)
                for build, note, problem in zip(builders, notes, problems, strict=True)]

    mode, _ = max(notes, key=lambda note: DEFAULT_FRAME_ORDER.index(note[0]))
    bound = bind(mode)
    try:
        rebuilt = [problem if note[0] == mode else build()
                   for problem, note, build in zip(problems, notes, bound, strict=True)]
    except IncompatibleFrameError:
        # The frame that "auto" plans at one point need not fit the operators of another point.
        mode, bound = "lab", bind("lab")
        rebuilt = [problem if note[0] == mode else build()
                   for problem, note, build in zip(problems, notes, bound, strict=True)]
    warn_default_frame(mode)
    return rebuilt, bound


@dataclass(frozen=True)
class SolveProblemContext:
    """Shared solve metadata reused across a homogeneous problem batch.

    Built once by :func:`prepare_solve_problem_context`, so sweep points can
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
    initial_states: list[Any] | None = None,
    _product_space: bool = False,
    _default_frame_notes: list[Any] | None = None,
) -> SolveProblemContext:
    """Resolve the frame and keep authored observables and state specifications.

    Observables and states are materialized only after assembly resolves every
    local solver basis. The default ground state stays lazy, so callers that
    supply an explicit state do not pay for unused state construction.

    ``frame`` overrides the chip's declared frame. For ``"auto"``, the solve
    window ``tlist[-1] - tlist[0]`` gives the duration used for weights. When
    neither sets a frame, the solve uses ``"rotating"`` and warns. If a collapse
    or port operator does not fit that frame, the solve uses the frame that
    ``"auto"`` plans, or the lab frame when that frame does not fit either, and
    warns. If a dressed ``0 -> 1`` reference has an unreliable label
    assignment, the solve raises and asks for a frame. A list passed as
    ``_default_frame_notes`` receives the chosen mode and the reason instead of
    the warning.

    ``tlist`` defines the actual interval. Scheduled signals keep their absolute
    times, including operations partly or fully outside this interval.
    """
    if not isinstance(dissipation, bool):
        raise TypeError("dissipation must be a boolean calculation choice.")
    backend = chip.backend
    automatic = isinstance(tlist, AutomaticTimeGrid)
    tlist_arr = backend.array_module.asarray(tlist.bounds if automatic else tlist, dtype=float)
    _validate_tlist(tlist_arr)

    if e_ops is not None and not isinstance(e_ops, dict):
        raise TypeError("e_ops must be dict or None")
    from math import prod
    from quchip.engine.assembly import _resolve_system, _prepare_engine_assembly
    from quchip.engine.frames import plan_for_operations
    from quchip.engine.pulse_space import plan_pulse_space
    from quchip.engine.excitation import sector_size
    from quchip.approximations import require_approximation
    from quchip.cutoff import active_cutoff

    strategy = chip.approximation if approximation is None else require_approximation(approximation)
    resolution = _resolve_system(chip, backend)
    policy = active_cutoff(chip)
    deterministic = solver in (None, "sesolve", "mesolve")
    if policy is not None and not deterministic:
        raise ValueError(
            f"Stochastic trajectory solvers do not support an excitation cutoff, and this chip keeps only "
            f"N <= {policy.max_total}. Use sesolve or mesolve, or chip.with_space(None) for the full product space."
        )
    space, reason, boundary = (None, "Trajectory solvers use the product space.", None)
    if _product_space and policy is None:
        reason = "Batch points prove different exact spaces."
    elif deterministic:
        # Under a cutoff, batch points that prove different spaces all keep every label with N <= K.
        space, reason, boundary = plan_pulse_space(chip, drive_ops or [], initial_states or [None], resolution,
                                                   strategy, dissipation, whole=_product_space)
    # A solve without dissipation leaves the collapse channels out, so they need not fit its frame.
    def resolve(frame_spec: Any) -> tuple[Any, Any]:
        if space is None:
            return resolve_for_operations(
                chip, drive_ops or [], frame=frame_spec, approximation=strategy,
                solve_window=(tlist_arr[0], tlist_arr[-1]), _check_collapse_frames=dissipation,
            ), resolution
        spec = plan_for_operations(chip, frame_spec, drive_ops or [], approximation=strategy,
                                   solve_window=(tlist_arr[0], tlist_arr[-1]), _resolution=resolution)
        prepared, resolved_frame = _prepare_engine_assembly(chip, spec, strategy, resolution=resolution)
        return build_engine_result(chip, [], resolved_frame=resolved_frame, approximation=strategy,
                                   _local_resolution=replace(prepared, space=space),
                                   _check_collapse_frames=dissipation), prepared

    if frame is not None or chip.frame is not None:
        base_result, resolution = resolve(chip.frame if frame is None else frame)
    else:
        mode, why, (base_result, resolution) = _choose_default_frame(chip, resolve, resolution)
        if _default_frame_notes is None:
            warn_default_frame(mode)
        else:
            _default_frame_notes.append((mode, why))
    base_result = replace(base_result, metadata={**base_result.metadata, "excitation_reduction": {
        "exact": space is not None, "reason": reason,
        "full_dimension": prod(resolution.dims), "dimension": prod(base_result.dims),
        "totals": None if space is None else space.totals,
        "cutoff": None if policy is None else policy.max_total,
        "cutoff_dimension": None if policy is None else sum(
            sector_size(resolution.dims, total) for total in range(policy.max_total + 1)),
        "boundary": boundary,
    }})
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
    static_term_ids = ref._static_term_ids
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
        if result._static_term_ids != static_term_ids:
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

    All results must share the same static term objects and the same number of
    dynamic terms. Each slot's operator payloads must match, by identity or by
    canonical fingerprint. ``initial_states=None`` gives every point one shared
    default start. Individual ``None`` entries reuse that start. The batch
    shares one static Hamiltonian, frame, and start time.
    """
    # Crosstalk rebuilds equal-by-value operators on every instantiation.
    if not engine_results:
        raise ValueError("build_solve_batch_from_results requires at least one engine result")

    _validate_batch_skeleton(engine_results)
    ref = engine_results[0]
    batch_size = len(engine_results)

    if initial_states is None:
        initial_states = [None] * batch_size
    elif len(initial_states) != batch_size:
        raise ValueError(
            f"initial_states length {len(initial_states)} does not match batch_size {batch_size}"
        )
    _require_product_state(context, ref)
    # Materialize the shared default only when at least one point omits its state.
    ground = (
        _initial_state(context.chip, None, ref, context.tlist[0])
        if any(state_spec is None for state_spec in initial_states) else None
    )
    states = tuple(
        ground if state_spec is None else _initial_state(context.chip, state_spec, result, context.tlist[0])
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


def _initial_state(chip, spec, result, start):
    if spec is None or isinstance(spec, (Mapping, str)):
        return label_state(chip, spec, result, start)
    if result.excitation_space is not None:
        return materialize_compact(chip, spec, result.excitation_space)
    if isinstance(spec, CompactState):
        backend = chip.backend
        values = compatible_values(spec, tuple(d.label for d in chip.devices), result.dims, result.bases, backend)
        native = backend.from_array(values, dims=[[spec.space.size],
                                    [1] if spec.values.shape[-1] == 1 else [spec.space.size]])
        spec = spec.space.lift(native, backend)
    return materialize_state_spec(chip, spec, result.bases)


def _reframe_state(chip, state, result, frame, start):
    """Map a ket or density matrix of ``frame`` at ``start`` into the solve frame of ``result``.

    A level label is a dressed state in every frame and needs no map.
    """
    if maybe_concrete_scalar(start) == 0.0:
        return state
    backend = chip.backend
    values = backend.to_array(state)
    space = result.excitation_space
    labels = tuple(device.label for device in chip.devices)
    bases = result.bases if space is None else space.bases
    xp = array_namespace(values)
    source = FrameMap.create(frame, labels, bases, xp, space=space)
    target = FrameMap.create(result.resolved_frame, labels, bases, xp, space=space)
    dims = [space.size] if space is not None else list(result.dims)
    if values.shape[-1] == 1:
        values = target.kets(source.kets(values[:, 0], start, to_lab=True), start)[:, None]
        return backend.from_array(values, dims=[dims, [1] * len(dims)])
    values = target.operators(source.operators(values, start, to_lab=True), start)
    return backend.from_array(values, dims=[dims, dims])


def _require_product_state(context, result):
    """Raise before a dissipative product-space solve evolves a density matrix that cannot fit."""
    dissipative = context.dissipation and bool(result.slh.collapse_terms)
    if result.excitation_space is None and dissipative and context.solver in (None, "mesolve"):
        from math import prod
        from quchip.utils.memory import dense_bytes, require_memory

        dimension = prod(result.dims)
        require_memory(dense_bytes(dimension, dimension),
                       task=f"The product-space density matrix at dimension {dimension}",
                       remedy="Use a model and initial state with a smaller proven invariant space.")


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
    _initial_states: list[Any] | None = None,
    _product_space: bool = False,
    _default_frame_notes: list[Any] | None = None,
    _initial_state_frame: Any = None,
) -> SolveProblem:
    """Resolve, assemble, and package a frozen :class:`SolveProblem`.

    Equivalent to :func:`prepare_solve_problem_context` followed by
    :func:`build_engine_result`. For many variants that share one chip
    configuration, use that two-step form with
    :func:`build_solve_batch_from_results`.

    ``frame`` overrides the chip's declared frame and follows the same
    operation-aware ``"auto"`` resolution as
    :func:`prepare_solve_problem_context`. ``_initial_state_frame`` is the
    resolved frame that a ket or density matrix ``initial_state`` is a vector
    of, when it is not the solve frame.
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
        initial_states=[initial_state] if _initial_states is None else _initial_states,
        _product_space=_product_space,
        _default_frame_notes=_default_frame_notes,
    )
    return _problem_from_context(context, drive_ops, initial_state, state_frame=_initial_state_frame)


def _problem_from_context(context, drive_ops, initial_state, *, state_frame=None):
    """Instantiate one solve request on a prepared context."""
    engine_result = build_engine_result(
        context.chip,
        drive_ops,
        resolved_frame=context.resolved_frame,
        approximation=context.approximation,
        _base_result=context._base_result,
    )
    engine_result = replace(engine_result, dissipation=context.dissipation)
    _require_product_state(context, engine_result)
    e_ops_solver, e_ops_meta = _prepare_context_eops(context, engine_result)
    state = _initial_state(context.chip, initial_state, engine_result, context.tlist[0])
    if state_frame is not None and not (initial_state is None or isinstance(initial_state, (Mapping, str))):
        state = _reframe_state(context.chip, state, engine_result, state_frame, context.tlist[0])
    problem = SolveProblem(
        chip=context.chip,
        engine_result=engine_result,
        initial_state=state,
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
                       parameters: tuple[dict[str, Any], ...] | None = None,
                       max_workers: int | None = None) -> Any:
    """Dispatch each captured backend's requests and restore the original point order."""
    from quchip.results.results import SimulationBatchResult

    problems = assign_point_noise(problems)
    groups: dict[int, list[tuple[int, SolveProblem]]] = {}
    for index, problem in enumerate(problems):
        groups.setdefault(id(problem.backend), []).append((index, problem))
    ordered: list[Any] = [None] * len(problems)
    for entries in groups.values():
        try:
            results = _solve_backend_problems([problem for _, problem in entries],
                entries[0][1].backend, progress=progress, max_workers=max_workers,
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
    max_workers: int | None = None,
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
        capture_label_targets,
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
            desc._static_term_ids,
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
        parallel_results = backend.parallel_solve_problems(problems, progress=progress, max_workers=max_workers)
        if parallel_results is not None:
            if len(parallel_results) != len(problems):
                raise RuntimeError(
                    "parallel_solve_problems returned "
                    f"{len(parallel_results)} results for {len(problems)} problems."
                )
            captured: dict[int, list[tuple[Any, Any, Any]]] = {}
            wrapped = [
                wrap_solver_result(result, problem, backend, label_targets=capture_label_targets(problem, captured))
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
            solver_results = backend.solve_batch(batch, progress=progress, max_workers=max_workers)
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

    Explicit native seeds/keys are kept, including intentionally shared noise.
    Dynamiqs keys stay a necessary native input.
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
