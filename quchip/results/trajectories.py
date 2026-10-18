"""Native trajectories with captured model context and lazy analysis views."""

from typing import Any

from quchip.backend.containers import SolverResult
from quchip.results.results import SimulationResult, wrap_solver_result


class TrajectoryResult(SimulationResult):
    """Keep a native stochastic result with its storage and weights unchanged.

    Parameters
    ----------
    native : object
        QuTiP or Dynamiqs stochastic result, including seeds/keys and records.
    problem : SolveProblem
        Captured model, observables, channel order, and integration frame.

    Attributes
    ----------
    native : object
        Unmodified native result, with native record conventions and event padding.
    times : array_like, shape (T,)
        Requested save times in ns.
    channel_labels : tuple of str
        Unique resolved channel keys in native jump-operator order.
    monitor_labels : tuple of str or None
        Native record channel order for an explicit selection, or None when unknown.
    """

    def __init__(self, native: Any, problem: Any):
        super().__init__(SolverResult(times=problem.tlist, solver=problem.solver),
                         problem.backend, problem.engine_result.dims, device_info=problem.device_info,
                         bases=problem.engine_result.bases, dissipation=problem.dissipation)
        self.native = native
        self._problem = problem
        labels = tuple(channel.key for channel in problem.engine_result.slh.channels) if problem.dissipation else ()
        selection = problem.monitoring
        if selection is None and problem.solver in ("ssesolve", "dssesolve"):
            selection = {i: (1.0, 0.0) for i in range(len(labels))}
        self.channel_labels = labels
        self.monitor_labels = None
        if selection is not None:
            from quchip.utils.jax_utils import maybe_concrete_scalar

            selected = tuple(i for i in range(len(labels)) if i in selection)
            if problem.solver in ("dssesolve", "dsmesolve"):
                self.channel_labels = tuple(labels[i] for i in range(len(labels)) if i not in selection)
                self.channel_labels += tuple(labels[i] for i in selected)
                if any(maybe_concrete_scalar(selection[i][0]) is None for i in selected):
                    return
                selected = tuple(i for i in selected if maybe_concrete_scalar(selection[i][0]) != 0)
            self.monitor_labels = tuple(labels[i] for i in selected)

    def _view(self, index: int | None, *, states: bool = True, deterministic: bool = False) -> SimulationResult:
        values, history, final = self._backend.trajectory_view(
            self.native, self._problem, index=index, states=states, deterministic=deterministic)
        payload = SolverResult(times=self.times, states=history, final_state=final,
                               expect=list(values) if values is not None else None, solver=self.solver)
        return wrap_solver_result(payload, self._problem, self._problem.backend)

    def average(self) -> SimulationResult:
        """Return native ensemble averages, including native trajectory weights."""
        return self._view(None)

    def run(self, index: int) -> SimulationResult:
        """Return the analysis of one kept conditional trajectory.

        Parameters
        ----------
        index : int
            Zero-based trajectory index of a run that native storage keeps.
        """
        return self._view(index)

    def expect(self, key: Any, index: int | None = None) -> Any:
        """Return a named native ensemble expectation, shape (T,).

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, default None
            Entry in a list-valued observable. Trajectory selection uses run().
        """
        return self._view(None, states=False).expect(key, index)
