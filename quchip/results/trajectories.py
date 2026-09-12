"""Native trajectories with captured model context and lazy analysis views."""

from typing import Any

from quchip.backend.containers import SolverResult
from quchip.results.results import SimulationResult, wrap_solver_result


class TrajectoryResult(SimulationResult):
    """Retain a native stochastic result without changing its storage or weights.

    Parameters
    ----------
    native : object
        QuTiP or Dynamiqs stochastic result, including seeds/keys and records.
    problem : SolveProblem
        Captured model, observables, channel order, and integration frame.

    Attributes
    ----------
    native : object
        Unmodified native result. Record conventions and event padding are native.
    times : array_like, shape (T,)
        Requested save times in ns.
    channel_labels : tuple of str
        Unique resolved channel keys in native jump-operator order.
    monitor_labels : tuple of str or None
        Native record channel order for an explicit selection; None when unknown.
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
        native = self.native
        if self.solver in ("mcsolve", "ssesolve", "smesolve"):
            if index is None:
                values = native.average_expect
                history = native.average_states if states else None
                final = native.average_final_state if states else None
            else:
                runs = native.deterministic_trajectories if deterministic else native.trajectories
                if not runs:
                    raise RuntimeError("Individual runs require native keep_runs_results=True.")
                run = runs[index]
                values, history, final = run.expect, run.states if states else None, run.final_state if states else None
            history = history or None
        else:
            if deterministic:
                values = native.infos.noclick_expects
                history = native.infos.noclick_states if states else None
            elif index is None:
                values = native.mean_expects()
                history = native.mean_states() if states else None
            else:
                values = native.expects[index] if native.expects is not None else None
                history = native.states[index] if states else None
            final = history[-1] if history is not None else None
            retain = (
                self._problem.states == "all"
                if self._problem.states is not None
                else self._problem.options.get("save_states", True)
            )
            if not retain and not deterministic:
                history = None
        if self.solver == "jssesolve" and not deterministic and any(
            value is not None for value in (values, history, final)
        ):
            import dynamiqs as dq
            import equinox as eqx

            if isinstance(native.method, dq.method.Event):
                xp = self._backend.array_module
                nclicks = native.nclicks if index is None else native.nclicks[index]
                values, history, final = eqx.error_if(
                    (values, history, final), xp.any(xp.sum(nclicks, axis=-1) >= native.options.nmaxclick),
                    "Native Event exhausted nmaxclick before completing evolution; increase the buffer. "
                    "The incomplete native result remains accessible through result.native.",
                )
        payload = SolverResult(times=self.times, states=history, final_state=final,
                               expect=list(values) if values is not None else None, solver=self.solver)
        return wrap_solver_result(payload, self._problem, self._problem.backend)

    def average(self) -> SimulationResult:
        """Return native ensemble averages, including native trajectory weights."""
        return self._view(None)

    def run(self, index: int) -> SimulationResult:
        """Return analysis of one retained conditional trajectory.

        Parameters
        ----------
        index : int
            Zero-based trajectory index. Native storage must retain this run.
        """
        return self._view(index)

    def expect(self, key: Any, index: int | None = None) -> Any:
        """Return a named native ensemble expectation, shape (T,).

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, default None
            Entry within a list-valued observable; trajectory selection uses run().
        """
        return self._view(None, states=False).expect(key, index)

    def check_truncation(self, *, threshold: float = 1e-3) -> dict[str, Any]:
        """Report per-run cutoff maxima and their largest value across retained runs.

        Parameters
        ----------
        threshold : float, default 1e-3
            Warning threshold for each run's sampled boundary population.

        Returns
        -------
        dict
            ``runs`` contains each run's device maxima; ``maximum`` aggregates them.
            Missing per-run states and diagnostic samples raise instead of checking
            an ensemble mean. Unsaved or unsampled excursions remain unknown.
        """
        count = len(self.native.trajectories) if hasattr(self.native, "trajectories") else len(self.native.keys)
        if not count:
            raise RuntimeError("Per-run truncation data unavailable; retain native trajectory results.")
        runs = tuple(self.run(i).check_truncation(threshold=threshold) for i in range(count))
        deterministic_count = (len(self.native.deterministic_trajectories)
                               if hasattr(self.native, "deterministic_trajectories")
                               else int(getattr(self.native.method, "smart_sampling", False)))
        deterministic = tuple(self._view(i, deterministic=True).check_truncation(threshold=threshold)
                              for i in range(deterministic_count))
        xp = self._problem.backend.array_module
        maximum = {key: xp.max(xp.stack([run[key] for run in runs + deterministic])) for key in runs[0]}
        return {"runs": runs, "deterministic": deterministic, "maximum": maximum}
