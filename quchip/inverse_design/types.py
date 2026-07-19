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
        Observable kind: ``"freq"``, ``"anharmonicity"``, ``"chi"``,
        ``"zz"``, ``"exchange"``, or ``"g"``.
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
    """

    kind: str
    label: Any
    target: float
    initial: float
    final: float
    evaluator: str


@dataclass(frozen=True)
class FitADressResult:
    """Result of a :func:`fit_a_dress` optimization run.

    Attributes
    ----------
    chip
        Fitted chip — a clone of the seed with updated
        ``freq``/``anharmonicity`` and coupling-strength values. The seed
        chip is never mutated. Exposing ``.chip`` makes this satisfy
        :class:`~quchip.chip.transformations.ChipTransform` structurally,
        with no inheritance required.
    loss
        Final objective (sum of squared, scale-normalized residuals).
    history
        1-D ``numpy`` array holding ``[loss_initial, loss_final]``.
        A compact record of how far the solver moved; it is not a
        per-iteration trace because ``scipy.optimize.least_squares``
        does not expose one.
    initial_targets
        One :class:`ObservableReport` per target, evaluated on the
        seed chip.
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
    solver_info
        ``scipy`` solver metadata (``method``, ``status``,
        ``message``, ``nfev``, ``jacobian``), plus the identifiability
        receipt recorded for every :func:`~quchip.inverse_design.fit.fit_a_dress`
        call: ``n_free_parameters`` (length of ``final_params``),
        ``n_target_residuals`` (length of ``final_targets``), and
        ``underdetermined_by_count`` (``True`` when the former exceeds the
        latter — a necessary, not sufficient, identifiability condition; no
        Jacobian-rank analysis is performed). ``jacobian`` is ``"jax"``
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

    @overload
    def rebind(self, seed: BaseDevice | str, /) -> BaseDevice: ...
    @overload
    def rebind(self, seed: BaseDevice | str, /, *more: BaseDevice | str) -> tuple[BaseDevice, ...]: ...
    def rebind(self, *seeds: BaseDevice | str) -> Any:
        """Look up the fitted clones matching one or more seed devices.

        ``fit.rebind(qb, tc, cr)`` replaces the
        ``qb_f = chip.device_map[qb.label]`` triple most tutorials opened
        with.

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
