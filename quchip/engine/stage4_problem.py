"""Stage 4: pack stage outputs + collapse operators into a frozen :class:`SolveProblem`.

Responsibilities
----------------
* Ensure the chip is dressed and run stage 1 (:func:`resolve_frame`).
* Run stage 3 (:func:`decompose_eops`) to flatten ``e_ops`` into
  solver-ready bands.
* Collect and embed every Lindblad collapse operator contributed by
  devices, drive lines, and couplings (see :func:`_collect_c_ops`).
* Run stage 2 (:func:`build_hamiltonian_description`) for each variant
  and pack into a single :class:`SolveProblem`, or merge homogeneous
  variants into a :class:`SolveBatch` (``N`` identical skeletons with
  per-element :class:`ScalarModulation` signals).

Collapse operators enter the standard Lindblad master equation
``dρ/dt = −i[H, ρ] + Σₖ D[Lₖ]ρ``. Rates are stored in 1/ns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from quchip.backend import _backend_context
from quchip.engine.bands import embed_on_support
from quchip.engine.ir import (
    BatchedHamiltonianDescription,
    DriveOp,
    HamiltonianDescription,
    ScalarModulation,
    SolveBatch,
    SolveProblem,
)
from quchip.engine.stage1_frames import resolve_frame
from quchip.engine.stage2_assembly import build_hamiltonian_description
from quchip.engine.stage3_observables import decompose_eops
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


@dataclass(frozen=True)
class SolveProblemContext:
    """Shared solve metadata reused across a homogeneous problem batch.

    Built once by :func:`prepare_solve_problem_context` so sweep points can
    skip redundant collapse-operator collection and e_ops normalization.
    """

    chip: Chip
    tlist: Any
    c_ops: tuple[Any, ...]
    e_ops: Any
    e_ops_meta: Any
    resolved_frame: Any
    solver: str | None
    options: dict[str, Any]
    default_initial_state: Any

    @classmethod
    def from_problem(cls, ref: SolveProblem) -> "SolveProblemContext":
        """Reconstruct a shared context from an existing :class:`SolveProblem`.

        Mirrors :meth:`SolveBatch.element`: it lifts a single concrete problem
        back into the context shape so a homogeneous group of problems can be
        re-batched. The reference problem's already-built ``initial_state`` is
        wrapped in a :class:`PrematerializedState` (no chip ground state is
        recomputed), and ``options`` is defensively copied.
        """
        return cls(
            chip=ref.chip,
            tlist=ref.tlist,
            c_ops=ref.c_ops,
            e_ops=ref.e_ops,
            e_ops_meta=ref.e_ops_meta,
            resolved_frame=ref.resolved_frame,
            solver=ref.solver,
            options=dict(ref.options),
            default_initial_state=PrematerializedState(ref.initial_state),
        )


def _collect_c_ops(chip: Chip) -> tuple[Any, ...]:
    """Embed every collapse operator the chip's components contribute.

    The chip enumerates its component families
    (:meth:`~quchip.chip.chip.Chip.collapse_contributions`); this stage only
    embeds each operator by its support arity, so a new physics-bearing
    component family never touches the engine.

    Collapse operators arrive in Lindblad-ready units (rates in 1/ns). No
    additional scaling happens here: each device owns its physics, including
    any 2π that belongs to a rate formula (e.g. κ = 2πf/Q). This is
    deliberately distinct from the Hamiltonian path, where the blanket 2π
    conversion lives at the stage-2 boundary.
    """
    backend = chip.backend
    with _backend_context(backend):
        return tuple(
            embed_on_support(backend, c_op, support, chip.dims)
            for c_op, support in chip.collapse_contributions()
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
    """Resolve the frame, normalize e_ops, collect c_ops, and prepare a default state.

    The dressed dict-keyed analysis is deliberately *not* triggered here: anything
    downstream that needs dressed quantities must use the array-only kernel
    (``chip.freq``, ``chip.energy``, etc.) so this stage stays JAX-traceable. The
    default initial state is computed lazily via a thunk so callers that always
    pass ``initial_state`` never pay for it (and so the thunk's eager dict-based
    state construction never runs under ``jax.jit``).

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

    if e_ops is None:
        e_ops_solver, dict_meta = None, None
    elif isinstance(e_ops, dict):
        e_ops_solver, dict_meta = decompose_eops(e_ops, chip, backend)
    else:
        raise TypeError("e_ops must be dict or None")
    return SolveProblemContext(
        chip=chip,
        tlist=tlist_arr,
        c_ops=_collect_c_ops(chip),
        e_ops=e_ops_solver,
        e_ops_meta=dict_meta,
        resolved_frame=resolve_frame(chip, chip.frame),
        solver=solver,
        options=merged_options,
        default_initial_state=_LazyDefaultState(chip),
    )


class _LazyDefaultState:
    """Wraps ``chip.state()`` so ground-state computation is deferred until ``materialize()`` is called.

    Keeps :func:`prepare_solve_problem_context` cheap: callers that always
    pass ``initial_state`` never pay for the dressed ground state. Under
    tracing ``chip.state()`` routes through the array kernel; the result is
    never memoized — caching a tracer would leak it into a later trace.
    """

    __slots__ = ("_chip", "_cached")

    def __init__(self, chip: Chip) -> None:
        self._chip = chip
        self._cached: Any = None

    def materialize(self) -> Any:
        if self._cached is not None:
            return self._cached
        state = self._chip.state()
        if not contains_tracer(state):
            self._cached = state
        return state


class PrematerializedState:
    """Adapter giving an already-built state the ``materialize()`` surface.

    Used where a :class:`SolveProblemContext` is assembled from an existing
    :class:`SolveProblem` (whose ``initial_state`` is already a concrete
    ket/DM) rather than from a chip.
    """

    __slots__ = ("_state",)

    def __init__(self, state: Any) -> None:
        self._state = state

    def materialize(self) -> Any:
        return self._state


def _aggregate_batch_metadata(descriptions: list[HamiltonianDescription]) -> dict[str, Any]:
    """Aggregate advisory solver-hint metadata across every description in a batch.

    Copying the reference element's metadata verbatim is unsafe once
    durations or frequencies are swept per element: each variant's own
    ``max_carrier_freq_ghz`` / ``spectral_bound_ghz`` / ``max_step_ns`` can
    differ. Carrier and spectral bounds take the maximum across elements
    (the batch as a whole is bounded by its fastest/most-spread element);
    ``max_step_ns`` takes the minimum (bounded by its narrowest pulse) and
    is present only when every element reports one -- a single traced
    window anywhere in the batch means the ceiling is incomplete for the
    whole batch. Non-advisory keys (e.g. ``"frame"``) are carried through
    from the reference element, since batching already requires a shared
    template.
    """
    metadata = dict(descriptions[0].metadata)
    for key in ("max_carrier_freq_ghz", "spectral_bound_ghz", "max_step_ns"):
        metadata.pop(key, None)

    carrier_values = [d.metadata["max_carrier_freq_ghz"] for d in descriptions if "max_carrier_freq_ghz" in d.metadata]
    if carrier_values:
        metadata["max_carrier_freq_ghz"] = max(carrier_values)

    spectral_values = [d.metadata["spectral_bound_ghz"] for d in descriptions if "spectral_bound_ghz" in d.metadata]
    if spectral_values:
        metadata["spectral_bound_ghz"] = max(spectral_values)

    step_values = [d.metadata.get("max_step_ns") for d in descriptions]
    non_none = [v for v in step_values if v is not None]
    if len(non_none) == len(step_values) and non_none:
        metadata["max_step_ns"] = min(non_none)

    return metadata


def build_solve_batch_from_descriptions(
    context: SolveProblemContext,
    descriptions: list[HamiltonianDescription],
    *,
    initial_states: list[Any] | None = None,
) -> SolveBatch:
    """Merge N homogeneous :class:`HamiltonianDescription`s into one :class:`SolveBatch`.

    All descriptions must share ``static_terms`` identity, the same number
    of dynamic terms, and matching operator payloads per slot (by identity
    or by canonical fingerprint — crosstalk rebuilds equal-by-value
    operators on every instantiation). ``initial_states=None`` fills every
    element with ``context.default_initial_state``.
    """
    if not descriptions:
        raise ValueError("build_solve_batch_from_descriptions requires at least one description")

    ref = descriptions[0]
    batch_size = len(descriptions)
    n_dyn = len(ref.dynamic_terms)
    _prefix = "build_solve_batch_from_descriptions: "

    # --- Skeleton checks: static terms, dim shape, dynamic term count ---
    for idx, desc in enumerate(descriptions):
        if desc.static_terms is not ref.static_terms:
            raise ValueError(
                _prefix + "all descriptions must share identical static_terms (by identity); "
                f"element {idx} differs."
            )
        if len(desc.dynamic_terms) != n_dyn:
            raise ValueError(
                _prefix + "all descriptions must have the same number of dynamic terms; "
                f"element {idx} has {len(desc.dynamic_terms)}, expected {n_dyn}."
            )
        if tuple(desc.dims) != tuple(ref.dims):
            raise ValueError(
                _prefix + "all descriptions must share identical dims; "
                f"element {idx} has {tuple(desc.dims)}, expected {tuple(ref.dims)}."
            )

    # --- Per-slot compatibility + signal collection ---
    # Per element, verify the dynamic term matches the reference in
    # (operator payload, origin, tag) and is a ScalarModulation; collect
    # its signal. Crosstalk rebuilds operators every instantiation, so
    # equality is by canonical fingerprint, not by object identity.
    dynamic_operators: list[Any] = []
    dynamic_origins: list[Any] = []
    dynamic_tags: list[str | None] = []
    dynamic_signals: list[list[ScalarModulation]] = []
    for slot in range(n_dyn):
        ref_term = ref.dynamic_terms[slot]
        shared_operator = ref_term.operator
        canonical_key = shared_operator.fingerprint()
        dynamic_origins.append(ref_term.origin)
        dynamic_tags.append(ref_term.tag)

        slot_signals: list[ScalarModulation] = []
        for idx, desc in enumerate(descriptions):
            term = desc.dynamic_terms[slot]
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

            slot_signals.append(term.time_dependence)

        dynamic_operators.append(shared_operator)
        dynamic_signals.append(slot_signals)

    batched = BatchedHamiltonianDescription(
        batch_size=batch_size,
        static_terms=ref.static_terms,
        dynamic_operators=tuple(dynamic_operators),
        dynamic_origins=tuple(dynamic_origins),
        dynamic_tags=tuple(dynamic_tags),
        dynamic_signals=tuple(tuple(sigs) for sigs in dynamic_signals),
        dims=ref.dims,
        metadata=_aggregate_batch_metadata(descriptions),
        dropped_terms_by_element=tuple(d.dropped_terms for d in descriptions),
    )

    if initial_states is None:
        default = context.default_initial_state.materialize()
        states: tuple[Any, ...] = tuple(default for _ in range(batch_size))
    elif len(initial_states) != batch_size:
        raise ValueError(
            f"initial_states length {len(initial_states)} does not match batch_size {batch_size}"
        )
    else:
        default = context.default_initial_state
        states = tuple(
            (default.materialize() if s is None else s) for s in initial_states
        )

    return SolveBatch(
        chip=context.chip,
        hamiltonian=batched,
        initial_states=states,
        tlist=context.tlist,
        c_ops=context.c_ops,
        e_ops=context.e_ops,
        e_ops_meta=context.e_ops_meta,
        resolved_frame=context.resolved_frame,
        solver=context.solver,
        options=context.options,
    )


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
    :func:`build_hamiltonian_description`. For many variants sharing one
    chip configuration, prefer that two-step form with
    :func:`build_solve_batch_from_descriptions`.
    """
    context = prepare_solve_problem_context(
        chip, tlist, solver=solver, options=options, e_ops=e_ops, drive_ops=drive_ops,
    )
    description = build_hamiltonian_description(
        chip, drive_ops, resolved_frame=context.resolved_frame,
    )
    return SolveProblem(
        chip=context.chip,
        hamiltonian=description,
        initial_state=context.default_initial_state.materialize() if initial_state is None else initial_state,
        tlist=context.tlist,
        c_ops=context.c_ops,
        e_ops=context.e_ops,
        e_ops_meta=context.e_ops_meta,
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

    Problems that share an operator skeleton are merged into one batched solve;
    those that cannot be structurally batched fall back to per-problem
    ``backend.solve_problem`` calls.

    Grouping is a two-stage filter. The cheap identity-based prefilter here
    (:func:`_skeleton_prefilter_key`) buckets problems by ``id()`` of their
    shared operators/metadata so that obviously-incompatible problems are never
    compared by value. The canonical by-value compatibility check is intentionally
    a *separate* concern that lives inside
    :func:`build_solve_batch_from_descriptions` (operator ``fingerprint``):
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
        desc = problem.hamiltonian
        solver_name = problem.solver or ("mesolve" if problem.c_ops else "sesolve")
        return (
            solver_name,
            id(desc.static_terms),
            tuple(id(term.operator) for term in desc.dynamic_terms),
            tuple(term.origin for term in desc.dynamic_terms),
            tuple(term.tag for term in desc.dynamic_terms),
            _tlist_key(problem.tlist),
            _op_list_key(problem.e_ops),
            _op_list_key(problem.c_ops),
            _options_key(problem.options),
            id(problem.resolved_frame),
        )

    groups: dict[tuple, list[tuple[int, SolveProblem]]] = {}
    for idx, problem in enumerate(problems):
        groups.setdefault(_skeleton_prefilter_key(problem), []).append((idx, problem))

    ordered_results: list[Any] = [None] * len(problems)
    for group in groups.values():
        indices = [i for i, _ in group]
        group_problems = [p for _, p in group]
        ref = group_problems[0]
        try:
            ctx = SolveProblemContext.from_problem(ref)
            batch = build_solve_batch_from_descriptions(
                ctx,
                [p.hamiltonian for p in group_problems],
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
