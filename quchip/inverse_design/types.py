from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from quchip.chip import Chip
    from quchip.devices.base import BaseDevice


@dataclass(frozen=True)
class ObservableReport:
    """Per-target record from a ``fit_a_dress`` run.

    Attributes
    ----------
    kind
        Canonical desired-chip kind (``"freq"``, ``"anharmonicity"``,
        ``"cross_kerr"``, ``"exchange_rate"``, or
        ``"coupling_strength"``), or its deprecated compatibility counterpart.
    label
        Target locator — a device label for single-device observables,
        a ``(label_a, label_b)`` tuple for pair observables, or a
        coupling label for coupling-keyed observables.
    target
        The value the optimizer tried to match (GHz).
    initial
        Observable value at the seed chip, before optimization (GHz).
    final
        Observable value at the fitted chip, after optimization (GHz).
    evaluator
        ``"full"`` if this target was evaluated on the whole chip or
        ``"local"`` if it was evaluated on a one-hop subsystem (see
        ``max_hilbert_dim`` in :func:`fit_a_dress`).
    source
        ``"component default"`` or ``"explicit"`` for the desired-chip
        contract; ``"legacy"`` for the deprecated compatibility path.
    """

    kind: str
    label: Any
    target: float
    initial: float
    final: float
    evaluator: str
    source: str = "legacy"

    @property
    def residual(self) -> float:
        """Final signed error, ``final - target`` (GHz)."""
        return self.final - self.target

    @property
    def relative_residual(self) -> float:
        """Final residual on the fitter's normalized objective scale."""
        return self.residual / max(abs(self.target), 1e-9)


@dataclass(frozen=True)
class FitParameterReport:
    """Starting point, bounds, result, and provenance for one bare parameter."""

    name: str
    initial: float
    final: float
    lower_bound: float
    upper_bound: float
    seed_source: str
    sign_choice: str | None = None

    @property
    def delta(self) -> float:
        """Signed optimizer displacement, ``final - initial`` (GHz)."""
        return self.final - self.initial


@dataclass(frozen=True)
class FitADressResult:
    """Result of a :func:`fit_a_dress` optimization run.

    Attributes
    ----------
    chip
        Fitted chip: a clone of the desired specification (or compatibility seed)
        with updated device and coupling parameters. The input chip is never
        mutated. Exposing ``.chip`` makes this satisfy
        :class:`~quchip.chip.transformations.ChipTransform` structurally,
        with no inheritance required.
    loss
        Final objective (sum of squared, scale-normalized residuals).
    history
        One-dimensional ``numpy`` array containing the normalized objective
        at every distinct parameter vector passed to the residual function.
        The first entry is the seed and the last is :attr:`loss`. With a
        numerical Jacobian, the intermediate entries include finite-difference
        probes as well as accepted solver iterates; use
        ``numpy.minimum.accumulate(history)`` for a monotone best-so-far
        convergence curve.
    initial_targets
        One :class:`ObservableReport` per target, evaluated on the
        optimizer's initial candidate.
    final_targets
        One :class:`ObservableReport` per target, evaluated on the
        fitted chip.
    initial_params
        ``{parameter_name: seed_value}`` — the starting point passed
        to the optimizer.
    final_params
        ``{parameter_name: fitted_value}`` — the optimizer output.
        Parameter names follow ``"<device>.freq"``,
        ``"<device>.anharmonicity"``, and
        ``"<coupling>.<coupling_strength_name>"`` — ``"<coupling>.g"`` for
        :class:`~quchip.chip.couplings.Capacitive`, ``"<coupling>.g_0"``
        for :class:`~quchip.chip.couplings.TunableCapacitive`,
        ``"<coupling>.chi"`` for :class:`~quchip.chip.couplings.CrossKerr`.
    parameter_reports
        One :class:`FitParameterReport` per varied bare parameter, including
        its bounds, starting-point source, and any coupling-sign choice.
    solver_info
        ``scipy`` solver metadata (``method``, ``status``,
        ``message``, ``nfev``, ``jacobian``), plus the identifiability
        receipt recorded for every :func:`~quchip.inverse_design.fit.fit_a_dress`
        call: ``n_free_parameters`` (length of ``final_params``),
        ``n_target_residuals`` (length of ``final_targets``), and
        ``underdetermined_by_count`` (``True`` when the former exceeds the
        latter — a necessary, not sufficient, identifiability condition),
        final scaled-Jacobian rank, condition number, singular values, and
        any weak parameter directions. Rank uses normalized residuals in the
        solver's scaled parameter coordinates. ``history_axis`` names the
        sampling axis used by :attr:`history`, and ``n_recorded_evaluations``
        gives its length. ``jacobian`` is ``"jax"``
        when a JAX-native backend supplies the exact residual Jacobian and
        ``"finite-difference"`` otherwise.
    """

    chip: Chip
    loss: float
    history: Any
    initial_targets: tuple[ObservableReport, ...]
    final_targets: tuple[ObservableReport, ...]
    initial_params: dict[str, float]
    final_params: dict[str, float]
    solver_info: dict[str, Any]
    parameter_reports: tuple[FitParameterReport, ...] = ()

    def summary(self) -> str:
        """Return a compact target, parameter, and identifiability receipt."""
        converged = int(self.solver_info.get("status", 0)) > 0
        rank = self.solver_info.get("jacobian_rank", "?")
        n_parameters = self.solver_info.get("n_free_parameters", len(self.final_params))
        condition = self.solver_info.get("jacobian_condition_number")
        if condition is None:
            condition_text = "unknown"
        elif condition == float("inf"):
            condition_text = "inf"
        else:
            condition_text = f"{float(condition):.3g}"
        lines = [
            f"fit_a_dress: {'converged' if converged else 'stopped'} | loss {self.loss:.3g} | "
            f"targets: {len(self.final_targets)} | parameters: {n_parameters}",
            f"identifiability: rank {rank}/{n_parameters} | condition {condition_text}",
        ]
        if self.final_targets:
            lines.append("targets (GHz):")
            for report in self.final_targets:
                locator = (
                    " <-> ".join(str(part) for part in report.label)
                    if isinstance(report.label, tuple)
                    else str(report.label)
                )
                lines.append(
                    f"  {locator}.{report.kind} [{report.source}]: "
                    f"{report.target:.6g} -> {report.final:.6g} "
                    f"(error {report.residual:+.2g})"
                )
        if self.parameter_reports:
            lines.append("bare parameters (GHz):")
            for report in self.parameter_reports:
                choice = f"; {report.sign_choice}" if report.sign_choice else ""
                lines.append(
                    f"  {report.name}: {report.initial:.6g} -> {report.final:.6g} [{report.seed_source}{choice}]"
                )
        return "\n".join(lines)

    @overload
    def rebind(self, seed: BaseDevice | str, /) -> BaseDevice: ...
    @overload
    def rebind(self, seed: BaseDevice | str, /, *more: BaseDevice | str) -> tuple[BaseDevice, ...]: ...
    def rebind(self, *seeds: BaseDevice | str) -> Any:
        """Look up the fitted clones matching one or more seed devices.

        Use ``fit.rebind(qb, tc, cr)`` to retrieve the fitted clones
        corresponding to the seed devices.

        Parameters
        ----------
        *seeds : BaseDevice or str
            One or more devices (or their labels) from the *seed* chip
            passed to :func:`~quchip.inverse_design.fit.fit_a_dress`. At
            least one is required.

        Returns
        -------
        BaseDevice or tuple[BaseDevice, ...]
            The matching device(s) on :attr:`chip` (the fitted clone), in
            input order. A single positional ``seed`` returns that device
            directly; two or more return a tuple.

        Raises
        ------
        ValueError
            No seeds were given.
        """
        if not seeds:
            raise ValueError("rebind requires at least one seed device or label")
        fitted = tuple(self.chip[s] for s in seeds)
        return fitted[0] if len(fitted) == 1 else fitted
