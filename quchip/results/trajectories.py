"""Native trajectories with captured model context and lazy analysis views."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from quchip.backend.containers import SolverResult
from quchip.results.results import SimulationResult, wrap_solver_result


@dataclass(frozen=True)
class MeasurementRecords:
    """Diffusive measurement currents of a stochastic solve, in one layout on both backends.

    A monitored channel with collapse operator L, efficiency η and phase φ
    records the current ``2 Re⟨c⟩ + ξ`` on each interval, in 1/sqrt(ns), with
    ``c = √η e^(−iφ) L``. Here L includes the square root of its rate. The
    homodyne noise ξ has variance 1/Δt. A QuTiP heterodyne solve records the
    quadratures ``2 Re⟨c⟩ + ξ_X`` and ``2 Im⟨c⟩ + ξ_Y``, and each noise has
    variance 2/Δt. The values keep this native normalization and the native
    sampling of the signal on each interval.

    Attributes
    ----------
    values : array_like
        Current of each run, channel and interval, with shape
        ``(runs, channels, intervals)``. A QuTiP heterodyne solve adds a
        quadrature axis of length 2 before the intervals.
    channels : tuple of str or None
        Channel of each row, as :attr:`TrajectoryResult.monitor_labels` gives
        them. None when a traced efficiency leaves the native order unknown.
    edges : array_like, shape (intervals + 1,)
        Interval edges in ns: the save times of the solve.
    sampling : {"start", "middle", "end", "mean"}
        Signal sample of each interval. QuTiP takes the signal at the start or
        the end of an interval, or the mean of both, as its option
        ``store_measurement`` selects, and adds the noise increment divided by
        Δt. Dynamiqs divides the increment of the integrated record by Δt, so
        the signal is its mean over the interval.
    efficiencies : tuple
        Efficiency η of each row.
    phases : tuple
        Phase φ of each row, in rad, relative to the integration frame.
    heterodyne : bool
        Whether :attr:`values` has a quadrature axis.
    """

    values: Any
    channels: tuple[str, ...] | None
    edges: Any
    sampling: str
    efficiencies: tuple[Any, ...]
    phases: tuple[Any, ...]
    heterodyne: bool


@dataclass(frozen=True)
class ClickRecords:
    """Jump events of a stochastic solve, in one layout on both backends.

    An ensemble mean over runs is the weighted sum ``Σ_r w_r x_r``, plus the
    no-click run with its weight :attr:`no_click_weight` when the native
    sampling solves that run apart: QuTiP ``improved_sampling`` and the
    dynamiqs ``Event(smart_sampling=True)``. The weights then sum to
    ``1 − no_click_weight``.

    Attributes
    ----------
    times : tuple of array_like
        Click times of each run in ns, in increasing order.
    channels : tuple of tuple of str
        Channel of each click, from :attr:`TrajectoryResult.channel_labels`.
    weights : array_like, shape (runs,)
        Sampling weight of each run.
    no_click_weight : float
        Weight of the no-click run that the native sampling solves apart, or 0.
    full : array_like of bool, shape (runs,)
        Whether a run filled its dynamiqs ``nmaxclick`` buffer. Its evolution
        may stop before the end of the solve. QuTiP has no buffer.
    """

    times: tuple[Any, ...]
    channels: tuple[tuple[str, ...], ...]
    weights: Any
    no_click_weight: float
    full: Any

    def counts(self, channel: Any = None) -> Any:
        """Return the number of clicks of each run, shape ``(runs,)``.

        Parameters
        ----------
        channel : str or None, default=None
            Count only this channel. None counts every channel.
        """
        from quchip.utils.labeling import resolve_label

        label = None if channel is None else resolve_label(channel)
        return np.asarray([sum(1 for name in names if label is None or name == label) for names in self.channels])


class TrajectoryResult(SimulationResult):
    """Keep a native stochastic result with its storage and weights unchanged.

    Parameters
    ----------
    native : object
        QuTiP or Dynamiqs stochastic result, including seeds/keys and records.
    problem : SolveProblem
        Captured model, observables, channel order, and integration frame.
    label_targets : LabelTargets or None, optional
        Snapshot of the solve's chip that prepares label targets.

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

    def __init__(self, native: Any, problem: Any, *, label_targets: Any = None):
        super().__init__(SolverResult(times=problem.tlist, solver=problem.solver),
                         problem.backend, problem.engine_result.dims, device_info=problem.device_info,
                         bases=problem.engine_result.bases, dissipation=problem.dissipation,
                         frame=problem.resolved_frame, label_targets=label_targets)
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
        return wrap_solver_result(payload, self._problem, self._problem.backend, label_targets=self._label_targets)

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

    def records(self) -> MeasurementRecords:
        """Return the diffusive measurement currents of every kept run.

        QuTiP keeps them with the native option ``store_measurement``.
        :attr:`result.native <native>` keeps the unmodified native fields.
        """
        values, sampling = self._backend.trajectory_records(self.native, self._problem)
        selection = self._problem.monitoring
        if selection is None:
            selection = {i: (1.0, 0.0) for i in range(len(self.channel_labels))}
        keys = tuple(channel.key for channel in self._problem.engine_result.slh.channels)
        rows = [] if self.monitor_labels is None else [selection[keys.index(label)] for label in self.monitor_labels]
        return MeasurementRecords(
            values=values,
            channels=self.monitor_labels,
            edges=self.times,
            sampling=sampling,
            efficiencies=tuple(eta for eta, _ in rows),
            phases=tuple(phase for _, phase in rows),
            heterodyne=bool(np.ndim(values) == 4),
        )

    def clicks(self) -> ClickRecords:
        """Return the jump events of every kept run, with the sampling weight of each run."""
        events, weights, no_click, full = self._backend.trajectory_clicks(self.native, self._problem)
        return ClickRecords(
            times=tuple(times for times, _ in events),
            channels=tuple(tuple(self.channel_labels[int(index)] for index in which) for _, which in events),
            weights=weights,
            no_click_weight=no_click,
            full=full,
        )

    def run_expect(self, key: Any, index: int | None = None) -> Any:
        """Return a named expectation of every kept run, shape ``(runs, T)``.

        Row i equals ``run(i).expect(key, index)``. The native sampling
        weights do not enter; :meth:`expect` gives the weighted ensemble mean.

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, default None
            Entry in a list-valued observable.
        """
        values = self._backend.trajectory_run_expect(self.native, self._problem)
        payload = SolverResult(times=self.times, expect=list(values), solver=self.solver)
        return wrap_solver_result(payload, self._problem, self._backend, label_targets=self._label_targets).expect(
            key, index)
