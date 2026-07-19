"""Engine pipeline: ``Chip → ResolvedFrame → HamiltonianDescription → SolveProblem``.

The engine is the physics-to-solver layer. It owns no solvers and no
backend-specific types; it produces structured, backend-agnostic
descriptions that each backend converts to its own optimal form.
The single 2π boundary lives in :mod:`quchip.engine.stage2_assembly`
and nowhere else.

Pipeline
--------
* **Stage 1** (:mod:`quchip.engine.stage1_frames`) — resolve a
  ``FrameSpec`` into :class:`~quchip.engine.ir.ResolvedFrame`
  (per-device frame frequencies, demodulation frequencies, and the
  frame mode).
* **Stage 2** (:mod:`quchip.engine.stage2_assembly`) — assemble a
  :class:`~quchip.engine.ir.HamiltonianDescription` with static terms,
  dynamic terms, and their :class:`~quchip.engine.ir.ScalarModulation`
  signal programs. Applies 2π, rotating-frame subtraction, RWA band
  decomposition (Jaynes & Cummings 1963; Gambetta et al., *PRA* **74**,
  042318 (2006)).
* **Stage 3** (:mod:`quchip.engine.stage3_observables`) — decompose
  dict-form ``e_ops`` into solver-ready bands; post-solve, demodulate
  expectations back into the lab/control frame.
* **Stage 4** (:mod:`quchip.engine.stage4_problem`) — pack everything
  (including collapse operators) into a frozen
  :class:`~quchip.engine.ir.SolveProblem` or
  :class:`~quchip.engine.ir.SolveBatch`.

Public API
----------
* :func:`simulate` — full pipeline + solve + wrap result.
* :func:`build_problem` — stages 1-4, returns a ``SolveProblem``.
* :func:`solve_problem` — dispatch a ``SolveProblem`` through the chip's backend.
* :func:`solve_batch` / :func:`solve_many` — batched dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quchip.control.batch import ProblemBatch
    from quchip.results.results import SimulationBatchResult, SimulationResult

from quchip.engine.ir import (
    BatchedHamiltonianDescription,
    CanonicalOperator,
    Carrier,
    DroppedTerm,
    DynamicTerm,
    HamiltonianDescription,
    ScalarModulation,
    SolveBatch,
    SolveProblem,
    StaticTerm,
)

__all__ = [
    "simulate",
    "build_problem",
    "solve_problem",
    "solve_many",
    "solve_batch",
    "build_hamiltonian_description",
    "BatchedHamiltonianDescription",
    "CanonicalOperator",
    "Carrier",
    "DroppedTerm",
    "DynamicTerm",
    "HamiltonianDescription",
    "ScalarModulation",
    "SolveBatch",
    "SolveProblem",
    "StaticTerm",
]


# Wrapper bodies defer the heavy imports so package import stays cheap and
# order-tolerant. ``chip.analysis`` (pulled by ``quchip.chip.__init__``)
# imports :func:`resolve_frame` from ``engine.stage1_frames``, which triggers
# this ``engine/__init__`` while ``quchip.chip`` is still partially
# initialized. ``stage2_assembly`` / ``stage4_problem`` import ``Chip`` only
# under TYPE_CHECKING, but they pull the backend and control stacks;
# deferring those into the wrapper bodies keeps their cost off the
# package-import path.


def build_problem(
    chip: Any,
    drive_ops: list,
    tlist: Any,
    *,
    solver: str | None = None,
    options: dict | None = None,
    e_ops: dict | None = None,
    initial_state: Any | None = None,
) -> SolveProblem:
    """Run stages 1-4 and return a frozen :class:`SolveProblem`.

    Returns an immutable request that can be passed to
    :func:`solve_problem`, batched with :func:`solve_many`, or
    serialized. No solver is invoked.

    Parameters
    ----------
    chip : Chip
        The chip whose Hamiltonian, frame, and backend are assembled.
    drive_ops : list of DriveOp
        Scheduled drive operations, typically produced by a
        :class:`~quchip.control.sequence.QuantumSequence`.
    tlist : array_like
        Solver time grid in ns.
    solver : {"sesolve", "mesolve"}, optional
        Solver selection; ``None`` auto-selects ``mesolve`` when collapse
        operators are present, else ``sesolve``.
    options : dict, optional
        Backend solver options. Must not contain a ``"backend"`` key
        (backend selection is chip-owned).
    e_ops : dict, optional
        Observables keyed by device label (or a 2-tuple of labels for a
        two-body observable).
    initial_state : optional
        Initial state; ``None`` defaults to the chip ground state.

    Returns
    -------
    SolveProblem
        The frozen request handed to a backend.

    Raises
    ------
    ValueError
        If ``tlist`` is not one-dimensional, finite, strictly increasing,
        and at least two points long, or if any ``drive_ops`` entry's
        pulse window ``[start_time, start_time + envelope.duration]``
        does not overlap ``tlist`` with positive measure. Both checks are
        concrete-only and skip silently under JAX tracing.

    Examples
    --------
    >>> import numpy as np
    >>> from quchip import Chip, DuffingTransmon, ChargeDrive, Gaussian, QuantumSequence
    >>> from quchip.engine import build_problem, solve_problem
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3)
    >>> chip = Chip([q], frame="rotating", rwa=True)
    >>> ctrl = ChargeDrive(target=q)
    >>> chip.wire(ctrl)
    >>> seq = QuantumSequence(chip)
    >>> _ = seq.schedule(ctrl, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q))
    >>> problem = build_problem(chip, list(seq.scheduled_ops), np.linspace(0.0, 20.0, 41))
    >>> result = solve_problem(problem)
    """
    from quchip.engine.stage4_problem import build_solve_problem as _build

    return _build(
        chip, drive_ops, tlist,
        solver=solver, options=options, e_ops=e_ops, initial_state=initial_state,
    )


def build_hamiltonian_description(chip: Any, drive_ops: list, **kwargs: Any) -> HamiltonianDescription:
    """Build a :class:`HamiltonianDescription` (stages 1-2 only).

    Parameters
    ----------
    chip : Chip
        The chip whose device, coupling, and drive Hamiltonians are assembled.
    drive_ops : list of DriveOp
        Scheduled drive operations to embed as dynamic terms.
    **kwargs
        Forwarded to
        :func:`quchip.engine.stage2_assembly.build_hamiltonian_description`
        (notably ``resolved_frame``).

    Returns
    -------
    HamiltonianDescription
        Static and dynamic terms plus dropped-term records.
    """
    from quchip.engine.stage2_assembly import build_hamiltonian_description as _build

    return _build(chip, drive_ops, **kwargs)


def simulate(
    chip: Any,
    drive_ops: list,
    tlist: Any,
    *,
    solver: str | None = None,
    options: dict | None = None,
    e_ops: dict | None = None,
    initial_state: Any | None = None,
    check_truncation: bool = True,
    truncation_threshold: float = 1e-3,
    partition: bool = True,
) -> "SimulationResult":
    """Build a :class:`SolveProblem`, dispatch it, and wrap the solver output.

    Parameters mirror :func:`build_problem`. ``solver`` is ``"sesolve"``
    or ``"mesolve"``; ``None`` auto-selects ``mesolve`` when collapse
    operators exist. ``e_ops`` is dict-form, keyed by device label (or
    a 2-tuple of labels for two-body observables), and favors object
    references via :func:`~quchip.utils.labeling.resolve_label`. The
    Hilbert-truncation safety net is inherited from :func:`solve_problem`;
    ``check_truncation`` / ``truncation_threshold`` are threaded down.

    Parameters
    ----------
    chip : Chip
        The chip to simulate.
    drive_ops : list of DriveOp
        Scheduled drive operations, typically produced by a
        :class:`~quchip.control.sequence.QuantumSequence`.
    tlist : array_like
        Solver time grid in ns.
    solver : {"sesolve", "mesolve"}, optional
        Solver selection; ``None`` auto-selects ``mesolve`` when collapse
        operators are present, else ``sesolve``.
    options : dict, optional
        Backend solver options. Must not contain a ``"backend"`` key
        (backend selection is chip-owned).
    e_ops : dict, optional
        Observables keyed by device label (or a 2-tuple of labels for a
        two-body observable).
    initial_state : optional
        Initial state. ``None`` defaults to the chip ground state. A
        ``Mapping`` (device label/object -> Fock index, e.g.
        ``{"q0": 1}``) resolves through :meth:`Chip.state` — the same
        dressed-state semantics
        :meth:`~quchip.control.sequence.QuantumSequence._resolve_initial_state_spec`
        uses — on both the joint and the partitioned path. Any other
        value (a raw backend state) is passed through unchanged.
    check_truncation : bool, default True
        Screen the result for over-populated top Fock levels.
    truncation_threshold : float, default 1e-3
        Top-level population above which the truncation check warns.
    partition : bool, default True
        When the chip splits into independent sub-chips (see
        :meth:`Chip.partition`), dispatch one solve per component and
        combine them into a :class:`~quchip.results.partitioned.PartitionedSimulationResult`
        instead of solving the full tensor-product space. Declines back
        to the joint solve (returning a plain
        :class:`~quchip.results.results.SimulationResult`) when the
        partition is trivial or ``initial_state`` is a raw backend state
        rather than ``None``/a ``Mapping``. Set ``False`` to force the
        joint solve unconditionally. ``simulate_batch``/``solve_many``
        never partition in v1 — batched dispatch always solves the full
        chip.

    Returns
    -------
    SimulationResult or PartitionedSimulationResult
        The wrapped solver output.

    Raises
    ------
    ValueError
        If ``solver`` is neither ``"sesolve"`` nor ``"mesolve"``, if
        ``tlist`` is not one-dimensional, finite, strictly increasing,
        and at least two points long, or if any ``drive_ops`` entry's
        pulse window does not overlap ``tlist`` with positive measure
        (both concrete-only checks; see :func:`build_problem`).
    RuntimeError
        If the backend solve fails.

    Examples
    --------
    >>> import numpy as np
    >>> from quchip import Chip, DuffingTransmon, ChargeDrive, Gaussian, QuantumSequence
    >>> from quchip.engine import simulate
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3)
    >>> chip = Chip([q], frame="rotating", rwa=True)
    >>> ctrl = ChargeDrive(target=q)
    >>> chip.wire(ctrl)
    >>> seq = QuantumSequence(chip)
    >>> _ = seq.schedule(ctrl, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q))
    >>> tlist = np.linspace(0.0, 20.0, 41)
    >>> result = simulate(chip, list(seq.scheduled_ops), tlist, e_ops={q: q.number_operator()})
    >>> populations = result.expect(q)
    """
    valid_solvers = ("sesolve", "mesolve")
    if solver is not None and solver not in valid_solvers:
        raise ValueError(f"Unknown solver '{solver}'. Must be one of {valid_solvers}.")

    from collections.abc import Mapping as _Mapping

    if partition:
        from quchip.engine.partitioned import maybe_simulate_partitioned

        partitioned = maybe_simulate_partitioned(
            chip, drive_ops, tlist,
            solver=solver, options=options, e_ops=e_ops, initial_state=initial_state,
            check_truncation=check_truncation, truncation_threshold=truncation_threshold,
        )
        if partitioned is not None:
            return partitioned

    if isinstance(initial_state, _Mapping):
        initial_state = chip.state(initial_state)

    problem = build_problem(
        chip, drive_ops, tlist,
        solver=solver, options=options, e_ops=e_ops, initial_state=initial_state,
    )
    chosen_solver = problem.solver or ("mesolve" if problem.c_ops else "sesolve")

    try:
        return solve_problem(
            problem,
            check_truncation=check_truncation,
            truncation_threshold=truncation_threshold,
        )
    except Exception as e:
        tlist_arr = chip.backend.array_module.asarray(problem.tlist, dtype=float)
        raise RuntimeError(
            f"Solver '{chosen_solver}' failed. "
            f"Devices: {[d.label for d in chip.devices]}, "
            f"time: {float(tlist_arr[0]):.1f}-{float(tlist_arr[-1]):.1f} ns, "
            f"c_ops: {len(problem.c_ops)}."
        ) from e


def solve_problem(
    problem: SolveProblem,
    *,
    check_truncation: bool = True,
    truncation_threshold: float = 1e-3,
) -> "SimulationResult":
    """Dispatch a :class:`SolveProblem` through its chip backend.

    This is the common single-solve chokepoint, so the Hilbert-truncation
    safety net lives here: unless ``check_truncation=False``, the wrapped
    result is screened for over-populated top Fock levels (warning above
    ``truncation_threshold``). Every example-facing single-solve path
    (``chip.solve``, ``seq.simulate``) inherits the check by routing through here.
    """
    from quchip.results.results import wrap_solver_result

    backend = problem.chip.backend
    result = wrap_solver_result(backend.solve_problem(problem), problem, backend)
    if check_truncation:
        result.check_truncation(threshold=truncation_threshold)
    return result


def solve_batch(batch: "SolveBatch", *, progress: bool = True) -> "SimulationBatchResult":
    """Dispatch a :class:`SolveBatch` through its chip backend.

    The backend converts each shared operator exactly once and stitches
    per-element coefficient data before running the parallel solve.
    """
    from quchip.results.results import SimulationBatchResult, wrap_solver_results_from_batch

    if batch.batch_size == 0:
        return SimulationBatchResult([])

    backend = batch.chip.backend
    solver_results = backend.solve_batch(batch, progress=progress)
    return SimulationBatchResult(wrap_solver_results_from_batch(solver_results, batch, backend))


def solve_many(
    batch_or_problems: "ProblemBatch | SolveBatch | list[SolveProblem]",
    *,
    progress: bool = True,
) -> "SimulationBatchResult":
    """Batch-dispatch typed solve requests that share one chip configuration.

    Accepts a :class:`SolveBatch`, a :class:`~quchip.control.batch.ProblemBatch`,
    or a flat list of :class:`SolveProblem` objects. The batched paths are
    preferred: backends convert shared operators exactly once and stitch
    per-element coefficients into one parallel solve.
    """
    from quchip.control.batch import ProblemBatch

    if isinstance(batch_or_problems, ProblemBatch):
        return solve_batch(batch_or_problems.batch, progress=progress).with_sweep_metadata(
            shape=batch_or_problems.shape,
            axes=batch_or_problems.axes,
        )

    if isinstance(batch_or_problems, SolveBatch):
        return solve_batch(batch_or_problems, progress=progress)

    problems = list(batch_or_problems)
    from quchip.results.results import SimulationBatchResult

    if not problems:
        return SimulationBatchResult([])

    for i, problem in enumerate(problems):
        if not hasattr(problem, "hamiltonian") or not hasattr(problem, "chip"):
            raise TypeError(f"problems[{i}]: expected SolveProblem, got {type(problem).__name__}")

    chip = problems[0].chip
    for i, problem in enumerate(problems[1:], start=1):
        if problem.chip is not chip:
            raise ValueError(
                f"problems[{i}] was built for a different chip. "
                "All problems in solve_many() must share the same chip instance."
            )

    from quchip.engine.stage4_problem import solve_problem_list

    return solve_problem_list(problems, chip.backend, progress=progress)
