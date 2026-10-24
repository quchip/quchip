"""Captured physical cutoff checks, sampled without storing state histories."""

import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from quchip.devices.spaces import TruncationBoundary
from quchip.engine.bands import local_mode_bands
from quchip.engine.observables import BandMeta, recombine_expect
from quchip.utils.values import copy_value
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar
from quchip.utils.stack import user_stacklevel

DEFAULT_TRUNCATION_THRESHOLD = 1e-3


@dataclass(frozen=True)
class BoundaryCheck:
    label: str
    boundary: TruncationBoundary
    basis: Any
    energy: bool = False


@dataclass(frozen=True)
class TruncationPlan:
    checks: tuple[BoundaryCheck, ...]
    unavailable: tuple[str, ...]
    dims: tuple[int, ...]
    labels: tuple[str, ...]
    frame_frequencies: dict[str, Any]
    operators: tuple[Any, ...] = ()
    metadata: tuple[BandMeta, ...] = ()
    sampled: bool = False
    space: Any = None
    # Excitation cutoff K and the retained totals that a generator change can move above K.
    excitation: tuple[int, tuple[int, ...]] | None = None


def capture_truncation(chip: Any, engine_result: Any) -> TruncationPlan:
    """Capture boundary selections and bases, with no work that grows with the space."""
    checks = []
    unavailable = []
    for device in chip.devices:
        basis = engine_result.bases[device.label]
        try:
            boundary = device.truncation_boundary()
        except NotImplementedError as exc:
            unavailable.append(f"Device {device.label!r}: {exc}.")
        else:
            if boundary is not None:
                if not isinstance(boundary, TruncationBoundary):
                    raise TypeError("truncation_boundary() must return TruncationBoundary or None")
                if any(not isinstance(i, int) or not 0 <= i < basis.native_dim for i in boundary.indices):
                    raise ValueError(f"Device {device.label!r} has invalid truncation boundary indices")
                checks.append(BoundaryCheck(device.label, boundary, basis))
        if basis.resolved_dim < basis.native_dim:
            checks.append(BoundaryCheck(device.label, TruncationBoundary(
                (basis.resolved_dim - 1,), "retained energy boundary",
                "Increase retained levels and compare observables.",
            ), basis, energy=True))
    reduction = engine_result.metadata.get("excitation_reduction") or {}
    cutoff, excitation = reduction.get("cutoff"), None
    if cutoff is not None:
        boundary = reduction.get("boundary")
        if boundary is None:
            unavailable.append(f"Excitation cutoff N <= {cutoff}: the generator's excitation changes are unknown, "
                               "so the sectors that connect above the cutoff are unknown.")
        else:
            excitation = (cutoff, tuple(boundary))
    return TruncationPlan(
        tuple(checks), tuple(unavailable), tuple(record.resolved_dim for record in engine_result.bases.values()),
        tuple(d.label for d in chip.devices),
        copy_value(engine_result.resolved_frame.frequencies, readonly=True),
        space=engine_result.excitation_space,
        excitation=excitation,
    )


def excitation_rows(plan: TruncationPlan) -> Any:
    """Compact rows of the retained totals that a generator change can move above the cutoff."""
    assert plan.excitation is not None
    return np.flatnonzero(np.isin(plan.space.levels.sum(axis=1), plan.excitation[1]))


def boundary_rows(check: BoundaryCheck, xp: Any) -> Any:
    """Rows defining a captured cutoff projector in the local solver basis."""
    indices = xp.asarray(check.boundary.indices, dtype=int)
    if check.energy:
        return xp.eye(check.basis.resolved_dim, dtype=complex)[indices]
    return xp.asarray(check.basis.vectors)[indices]


def boundary_sampling_frequency(plan: TruncationPlan | None) -> float | None:
    """Fastest nonzero frame band in local cutoff reconstruction, in GHz."""
    import numpy as np
    from quchip.engine.bands import decompose_bands

    maximum = 0.0
    if plan is None:
        return maximum
    for check in plan.checks:
        frequency = maybe_concrete_scalar(plan.frame_frequencies[check.label])
        if frequency is None:
            return None
        if frequency == 0:
            continue
        if contains_tracer((check.basis.vectors, check.basis.energy_vectors)):
            return None
        rows = boundary_rows(check, np)
        transform = check.basis.energy_to_solver()
        if transform is not None:
            rows = rows @ np.asarray(transform)
        matrix = rows.conj().T @ rows
        for weight in decompose_bands(matrix, check.basis.resolved_dim):
            maximum = max(maximum, abs(frequency * weight))
    return maximum


def compile_truncation(plan: TruncationPlan, backend: Any) -> TruncationPlan:
    """Lower captured cutoff projectors through the ordinary frame-band machinery."""
    if plan.sampled:
        return plan
    operators = []
    metadata = []
    xp = backend.array_module
    for index, check in enumerate(plan.checks):
        basis = check.basis
        rows = boundary_rows(check, xp)
        matrix = xp.conj(rows.T) @ rows
        local = backend.from_array(matrix, dims=[[basis.resolved_dim], [basis.resolved_dim]])
        device_index = plan.labels.index(check.label)
        if maybe_concrete_scalar(plan.frame_frequencies[check.label]) == 0.0:
            bands = [(0, local)]
        else:
            bands = local_mode_bands(
                backend, local, dim=basis.resolved_dim,
                label=check.label, semantic_to_solver=basis.energy_to_solver(),
            )
        for weight, operator in bands:
            operators.append(backend.embed(operator, device_index, plan.dims) if plan.space is None
                             else plan.space.embed(backend, operator, (device_index,)))
            metadata.append(BandMeta(key=str(index), weight=weight, device_labels=check.label))
    if plan.excitation is not None:
        # The projector is diagonal in product energy labels, so every frame leaves it unchanged.
        diagonal = np.zeros(plan.space.size)
        diagonal[excitation_rows(plan)] = 1.0
        operators.append(backend.diag(diagonal, dims=[[plan.space.size], [plan.space.size]]))
        metadata.append(BandMeta(key="excitation", weight=0, device_labels=plan.labels[0]))
    return replace(plan, operators=tuple(operators), metadata=tuple(metadata), sampled=True)


def with_truncation(problem: Any) -> Any:
    """Return a request collecting physical cutoff populations without saving states.

    Parameters
    ----------
    problem : SolveProblem or SolveBatch
        Captured request or parameter batch. Its existing time grid is kept. Samples
        can miss excursions between save times and do not prove convergence.

    Returns
    -------
    SolveProblem or SolveBatch
        Request with additional diagnostic observables. No solve occurs.
    """
    from quchip.engine.ir import SolveBatch

    if isinstance(problem, SolveBatch):
        return replace(problem, problems=tuple(with_truncation(p) for p in problem.problems))
    plan = problem.truncation
    if plan is None or plan.sampled:
        return problem
    plan = compile_truncation(plan, problem.backend)
    return replace(problem, e_ops=list(problem.e_ops or ()) + list(plan.operators), truncation=plan)


def boundary_traces(plan: TruncationPlan, values: Any, times: Any, backend: Any) -> tuple[Any, ...]:
    """Reconstruct boundary populations in the physical frame, independent of readout LO.

    The traces follow ``plan.checks``. An excitation cutoff with known
    boundary sectors adds their population as the last trace.
    """
    _, traces = recombine_expect(
        flat_expect=values, meta_list=list(plan.metadata), tlist=times,
        frame_freqs={label: -frequency for label, frequency in plan.frame_frequencies.items()},
        direction="demodulate",
    )
    xp = backend.array_module
    keys = [str(index) for index in range(len(plan.checks))] + (["excitation"] if plan.excitation else [])
    return tuple(xp.real(traces.get(key, xp.zeros_like(times))) for key in keys)


def evaluate_boundaries(result: Any) -> tuple[TruncationPlan, tuple[Any, ...]]:
    """Use sampled traces or derive them from a kept full history without a rerun."""
    plan = result._truncation
    if plan is None:
        raise RuntimeError("Truncation context is unavailable; build the calculation through quchip.")
    if result._boundary_traces is not None:
        return plan, result._boundary_traces
    if not plan.checks and plan.excitation is None:
        return plan, ()
    if result._states is None and result._final_state is None:
        raise RuntimeError('Boundary samples unavailable; prepare with_truncation(problem) or retain states="all".')
    backend = result._backend
    final_only = result._states is None
    states = backend.stack_states([result._final_state]) if final_only else result._stacked_states()
    times = result.times[-1:] if final_only else result.times
    devices = compile_truncation(replace(plan, excitation=None), backend)
    values = [backend.expect_over_time(operator, states) for operator in devices.operators]
    traces = boundary_traces(devices, values, times, backend) if devices.operators else ()
    if plan.excitation is not None:
        # The projector is diagonal in the compact labels, so the saved diagonals give it in every frame.
        populations = backend.populations_over_time(states)
        traces += (backend.array_module.sum(populations[:, excitation_rows(plan)], axis=1),)
    result.stats["truncation_sampling"] = "final state only" if final_only else "saved time grid"
    if not contains_tracer(traces):
        result._boundary_traces = traces
    return plan, traces


def check_truncation(result: Any, *, threshold: float = DEFAULT_TRUNCATION_THRESHOLD) -> dict[Any, Any]:
    """Report the maximum sampled boundary population of each device in a solved result.

    Each device model declares the levels at its cutoff, and a basis with fewer
    retained than native levels adds its highest retained energy level. The
    check reads these populations from the saved states, or from the samples
    that :func:`with_truncation` adds to a request, and warns for each value
    above ``threshold``. This warning heuristic can miss excursions between
    samples, and it is not an error bound. To confirm convergence, increase the
    applicable cutoff and compare observables. Native arrays stay
    differentiable. Warning thresholds are evaluated only for concrete results.

    Under an :class:`~quchip.cutoff.ExcitationCutoff`, the report also holds
    the policy as a key. Its value is the maximum population in the retained
    sectors that a Hamiltonian term, drive, or jump can move above the cutoff.
    To check convergence, compare observables at a larger ``max_total``.

    Parameters
    ----------
    result : SimulationResult, TrajectoryResult, or PartitionedSimulationResult
        Time-domain result built by quchip. It needs saved states
        (``states="all"`` or ``states="final"``) or the samples of a request
        prepared with :func:`with_truncation`. A final state alone covers only
        the final time. A trajectory result needs each run's states or samples.
    threshold : float, default=1e-3
        Maximum accepted boundary population.

    Returns
    -------
    dict
        For a :class:`~quchip.results.results.SimulationResult`, the maximum
        population at each device label, and at the ``ExcitationCutoff``
        policy if one applies. For a
        :class:`~quchip.results.trajectories.TrajectoryResult`, ``runs`` holds
        the report of each run, ``deterministic`` the report of each no-click
        run, and ``maximum`` the largest value of each key over both. For a
        :class:`~quchip.results.partitioned.PartitionedSimulationResult`, the
        reports of the components merge into one per-device report.

    Raises
    ------
    TypeError
        If ``result`` is not one of these result types.
    RuntimeError
        If the result holds neither saved states nor diagnostic samples.
    ValueError
        If ``threshold`` is negative or not finite.
    """
    from quchip.results.partitioned import PartitionedSimulationResult
    from quchip.results.results import SimulationResult
    from quchip.results.trajectories import TrajectoryResult

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("truncation threshold must be finite and nonnegative")
    if isinstance(result, TrajectoryResult):
        return _check_runs(result, threshold)
    if isinstance(result, PartitionedSimulationResult):
        merged: dict[Any, Any] = {}
        for component in result.components:
            merged.update(_check_result(component, threshold))
        return merged
    if isinstance(result, SimulationResult):
        return _check_result(result, threshold)
    from quchip.results._batch import BatchResult

    hint = " Check each result of the batch." if isinstance(result, BatchResult) else ""
    raise TypeError("check_truncation() needs a SimulationResult, TrajectoryResult, or PartitionedSimulationResult, "
                    f"not {type(result).__name__}.{hint}")


def _check_result(result: Any, threshold: float) -> dict[Any, Any]:
    plan, traces = evaluate_boundaries(result)
    for reason in plan.unavailable:
        warnings.warn(f"Truncation diagnostic unavailable: {reason}", UserWarning, stacklevel=user_stacklevel())
    xp = result._backend.array_module
    observed: dict[Any, Any] = {}
    maxima = tuple(xp.maximum(0.0, xp.max(trace)) for trace in traces)
    maxima, excitation = maxima[:len(plan.checks)], maxima[len(plan.checks):]
    traced = contains_tracer(maxima)
    if traced:
        warnings.warn(
            "Truncation boundary populations are traced, so the thresholds are not checked. Check the concrete "
            "result.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    sampling = result.stats.get("truncation_sampling", "saved time grid")
    for check, maximum in zip(plan.checks, maxima, strict=True):
        observed[check.label] = xp.maximum(observed.get(check.label, 0.0), maximum)
        if not traced and float(maximum) > threshold:
            warnings.warn(
                f"Device {check.label!r}: maximum sampled {check.boundary.description} population "
                f"({sampling}) {float(maximum):.3g} > threshold {threshold:.3g}. "
                f"{check.boundary.convergence_hint} This is a sampling heuristic, not an error bound.",
                UserWarning, stacklevel=user_stacklevel(),
            )
    if excitation:
        from quchip.cutoff import ExcitationCutoff

        assert plan.excitation is not None
        (maximum,) = excitation
        observed[ExcitationCutoff(plan.excitation[0])] = maximum
        if not traced and float(maximum) > threshold:
            warnings.warn(
                f"Excitation cutoff N <= {plan.excitation[0]}: maximum sampled population in sectors that "
                f"connect above the cutoff ({sampling}) {float(maximum):.3g} > threshold {threshold:.3g}. "
                "Raise max_total and compare observables. This is a sampling heuristic, not an error bound.",
                UserWarning, stacklevel=user_stacklevel(),
            )
    return observed


def _check_runs(result: Any, threshold: float) -> dict[str, Any]:
    native = result.native
    count = len(native.trajectories) if hasattr(native, "trajectories") else len(native.keys)
    if not count:
        raise RuntimeError("Per-run truncation data unavailable; retain native trajectory results.")
    runs = tuple(_check_result(result.run(i), threshold) for i in range(count))
    deterministic_count = (len(native.deterministic_trajectories) if hasattr(native, "deterministic_trajectories")
                           else int(getattr(native.method, "smart_sampling", False)))
    deterministic = tuple(_check_result(result._view(i, deterministic=True), threshold)
                          for i in range(deterministic_count))
    xp = result._problem.backend.array_module
    maximum = {key: xp.max(xp.stack([run[key] for run in runs + deterministic])) for key in runs[0]}
    return {"runs": runs, "deterministic": deterministic, "maximum": maximum}
