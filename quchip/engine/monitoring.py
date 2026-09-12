"""Select declared physical channels for continuous diffusive monitoring."""

from dataclasses import replace
from typing import Any, Mapping

from quchip.engine.ir import SolveBatch
from quchip.engine.slh import _index
from quchip.utils.jax_utils import maybe_concrete_scalar


def with_monitoring(problem: Any, efficiencies: Mapping[Any, Any], *, phases: Mapping[Any, Any] | None = None) -> Any:
    r"""Return a diffusive request monitoring selected declared channels.

    Parameters
    ----------
    problem : SolveProblem or SolveBatch
        Captured diffusive solve request or parameter batch.
    efficiencies : mapping
        Resolved channel key or SLHChannel to efficiency in [0, 1]. Unlisted
        channels remain unobserved. Inspect ``problem.engine_result.slh.channels``.
    phases : mapping or None, default None
        Channel key or SLHChannel to homodyne phase in radians, relative to the
        captured integration frame. Omitted phases are zero. Channel coupling
        phases are retained. This does not model downstream receiver processing.

    Returns
    -------
    SolveProblem or SolveBatch
        Captured selection. Pure-state SSE requires all losses to be monitored
        with unit efficiency. Native SME owns inefficient state evolution.

    Notes
    -----
    QuTiP uses S = sqrt(eta) exp(-i phase) L and C = sqrt(1-eta) L,
    preserving D[S] + D[C] = D[L]. Dynamiqs receives L and eta directly.
    See Wiseman and Milburn, Quantum Measurement and Control, chapter 4
    (https://doi.org/10.1017/CBO9780511813948).
    """
    if isinstance(problem, SolveBatch):
        return replace(
            problem, problems=tuple(with_monitoring(p, efficiencies, phases=phases) for p in problem.problems)
        )
    if problem.solver not in ("ssesolve", "smesolve", "dssesolve", "dsmesolve"):
        raise ValueError("Monitoring requires an explicitly selected native diffusive solver.")
    if not problem.dissipation:
        raise ValueError("Monitoring requires declared dissipation.")
    slh = problem.engine_result.slh
    selected = {_index(slh, key): (eta, 0.0) for key, eta in efficiencies.items()}
    for key, phase in (phases or {}).items():
        index = _index(slh, key)
        if index not in selected:
            raise ValueError("A monitor phase requires a selected channel.")
        selected[index] = (selected[index][0], phase)
    for eta, _ in selected.values():
        concrete = maybe_concrete_scalar(eta)
        if concrete is not None and not 0 <= concrete <= 1:
            raise ValueError("Monitor efficiencies must lie in [0, 1].")
    return replace(problem, monitoring=selected)


def monitored_operators(problem: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """Lower physical couplings once; retain selected channel order and phases."""
    backend = problem.backend
    xp = backend.array_module
    pure = problem.solver in ("ssesolve", "dssesolve")
    channels = problem.engine_result.slh.channels if problem.dissipation else ()
    selection = problem.monitoring
    if selection is None:
        selection = {i: (1.0, 0.0) for i in range(len(channels))} if pure else {}
    unobserved, monitored, efficiencies = [], [], []
    for index, channel in enumerate(channels):
        operator = backend.from_canonical_operator(channel.coupling)
        if index not in selection:
            if pure:
                raise ValueError("Pure-state SSE cannot discard unobserved losses; use SME.")
            unobserved.append(operator)
            continue
        eta, phase = selection[index]
        if pure and maybe_concrete_scalar(eta) != 1.0:
            raise ValueError("Pure-state SSE requires concrete unit monitor efficiencies; use SME.")
        monitored.append(xp.exp(-1j * xp.asarray(phase)) * operator)
        efficiencies.append(eta)
    return unobserved, monitored, efficiencies
