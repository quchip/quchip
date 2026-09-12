"""Captured physical cutoff checks, sampled without retaining state histories."""

from dataclasses import dataclass, replace
from typing import Any

from quchip.devices.spaces import TruncationBoundary
from quchip.engine.bands import embed_single_mode_bands
from quchip.engine.observables import BandMeta, recombine_expect
from quchip.utils.values import copy_value
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar


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


def capture_truncation(chip: Any, engine_result: Any) -> TruncationPlan:
    """Capture boundary selections and bases, without allocating observable matrices."""
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
    return TruncationPlan(
        tuple(checks), tuple(unavailable), tuple(engine_result.dims), tuple(d.label for d in chip.devices),
        copy_value(engine_result.resolved_frame.frequencies, readonly=True),
    )


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
            bands = [(0, backend.embed(local, device_index, plan.dims))]
        else:
            bands = embed_single_mode_bands(
                backend, local, device_index=device_index, dim=basis.resolved_dim,
                label=check.label, dims=plan.dims, semantic_to_solver=basis.energy_to_solver(),
            )
        for weight, operator in bands:
            operators.append(operator)
            metadata.append(BandMeta(key=str(index), weight=weight, device_labels=check.label))
    return replace(plan, operators=tuple(operators), metadata=tuple(metadata), sampled=True)


def with_truncation(problem: Any) -> Any:
    """Return a request collecting physical cutoff populations without saving states.

    Parameters
    ----------
    problem : SolveProblem or SolveBatch
        Captured request or parameter batch. Its existing time grid is preserved;
        samples can miss excursions between save times and do not prove convergence.

    Returns
    -------
    SolveProblem or SolveBatch
        Request with additional diagnostic observables. No solve is performed.
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
    """Reconstruct boundary populations in the physical frame, independent of readout LO."""
    _, traces = recombine_expect(
        flat_expect=values, meta_list=list(plan.metadata), tlist=times,
        frame_freqs={label: -frequency for label, frequency in plan.frame_frequencies.items()},
        direction="demodulate",
    )
    xp = backend.array_module
    return tuple(xp.real(traces.get(str(index), xp.zeros_like(times))) for index in range(len(plan.checks)))


def evaluate_boundaries(result: Any) -> tuple[TruncationPlan, tuple[Any, ...]]:
    """Use sampled traces or derive them from a retained full history without rerunning."""
    plan = result._truncation
    if plan is None:
        raise RuntimeError("Truncation context is unavailable; build the calculation through quchip.")
    if result._boundary_traces is not None:
        return plan, result._boundary_traces
    if not plan.checks:
        return plan, ()
    if result._states is None and result._final_state is None:
        raise RuntimeError('Boundary samples unavailable; prepare with_truncation(problem) or retain states="all".')
    plan = compile_truncation(plan, result._backend)
    final_only = result._states is None
    states = result._backend.stack_states([result._final_state]) if final_only else result._stacked_states()
    times = result.times[-1:] if final_only else result.times
    values = [result._backend.expect_over_time(operator, states) for operator in plan.operators]
    traces = boundary_traces(plan, values, times, result._backend)
    result.stats["truncation_sampling"] = "final state only" if final_only else "saved time grid"
    if not contains_tracer(traces):
        result._boundary_traces = traces
    return plan, traces
