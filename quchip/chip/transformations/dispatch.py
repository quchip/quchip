"""Elimination target registry and the public ``eliminate`` dispatcher.

The unifying abstraction is the :class:`~quchip.chip.chip.Chip` type itself,
not a base class: a transformation consumes a chip and produces a new one, so
transformations compose through ``Chip`` (e.g.
``eliminate(fit_a_dress(chip).chip, "r").chip``). :class:`ChipTransform` is a
thin *structural* protocol capturing only the ``.chip`` output that every
transformation result exposes;
:class:`~quchip.inverse_design.types.FitADressResult` already satisfies it with
no changes.

``eliminate`` performs **model reduction**: it removes a far-detuned mode (or
an edge coupling) and replaces its effect on the survivors with ordinary
owned-Hamiltonian physics — a Lamb shift folded into ``freq``, Purcell decay
folded into ``T1``, mediated exchange folded into a coupling — so the engine
never special-cases the reduction. The approximation
is declared in :attr:`EliminationResult.notes` and
:attr:`EliminationResult.validity`.

Each target *kind* is an :class:`EliminationTarget` — a pair of ``(claims,
reduce)`` closures registered in :data:`_ELIMINATION_TARGETS`. The dispatcher
scans the registry, hands the target to the first kind that claims it, and
falls through to a clear error otherwise. The reductions themselves live in the
sibling handler modules (:mod:`quchip.chip.transformations.eliminate_device`,
:mod:`quchip.chip.transformations.eliminate_coupling`), which register at
import time; the generic P/Q partitioning physics lives in
:mod:`quchip.chip.sw`. This module owns only the registry and the dispatch, so
a new reducible target kind registers here without touching either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from quchip.chip.transformations.methods import reduction_method_names
from quchip.chip.transformations.result import EliminationResult
from quchip.utils.labeling import resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


@dataclass(frozen=True)
class EliminationTarget:
    """One registered elimination target kind — how it recognizes a target and reduces it.

    A new kind registers an instance with :func:`register_elimination_target`
    at its handler module's bottom; that module must be imported for the
    side effect (the package ``__init__`` does this for the shipped
    handlers) — the same registration ritual as
    :func:`~quchip.chip.retarget.register_retarget_rule`.

    Attributes
    ----------
    kind
        A short label for the target kind (``"device"``, ``"coupling"``),
        for diagnostics.
    claims
        ``claims(chip, target) -> bool``: whether this kind owns ``target`` on
        ``chip`` (e.g. the label names a device, or a coupling). The device and
        coupling namespaces are disjoint by construction, so at most one kind
        claims any target.
    reduce
        ``reduce(chip, target, method) -> EliminationResult``: performs the
        reduction. ``method`` is the already-validated route string (a handler
        that has no notion of a route ignores it).
    """

    kind: str
    claims: Callable[[Any, Any], bool]
    reduce: Callable[[Any, Any, str], EliminationResult]


_ELIMINATION_TARGETS: list[EliminationTarget] = []


def register_elimination_target(target: EliminationTarget) -> None:
    """Register an :class:`EliminationTarget` kind for :func:`eliminate` to dispatch on."""
    _ELIMINATION_TARGETS.append(target)


def eliminate(chip: "Chip", target: Any, *, method: str = "sw") -> EliminationResult:
    """Reduce a far-detuned device or an edge coupling, returning a reduced chip.

    ``target`` is resolved against the chip's device and coupling namespaces
    (disjoint by construction — :class:`~quchip.chip.chip.Chip` rejects a
    coupling label that collides with a device label) and dispatches to one
    of two model reductions:

    - **Device target** — adiabatic elimination of a far-detuned mode, via
      :mod:`quchip.chip.sw`. A mode touching **one** survivor folds into a
      Lamb shift (and a Purcell channel when the mode dissipates). A mode
      touching **two or more** survivors — bus / tunable-coupler (bridge) or
      several at once — additionally induces a mediated exchange ``J =
      g_a g_b / 2 · (1/Δ_a + 1/Δ_b)`` between every survivor pair (with
      ``∂J/∂ω_c`` recorded alongside it), folded into the direct coupling
      between that pair when one exists or added as a new edge otherwise.
      A fixed eliminated mode emits a
      :class:`~quchip.chip.couplings.Capacitive`; a mode that declares
      frequency control, has a retargeted flux line, or folds into an
      already-modulable edge emits a
      :class:`~quchip.chip.couplings.TunableCapacitive`. At a tunable
      coupler's idle point, its ``g_0`` is the net coupling the pair feels,
      including any direct-edge cancellation. Eliminating several couplers
      is sequential composition:
      ``eliminate(eliminate(chip, "TC1").chip, "TC2")``.
    - **Coupling target** — dispersive reduction of an exchange edge to its
      dressed cross-Kerr shift: both endpoint devices survive (Lamb-shifted),
      and the coupling itself is replaced by a
      :class:`~quchip.chip.couplings.CrossKerr` carrying the dressed pull
      (see :func:`~quchip.chip.transformations.eliminate_coupling.reduce_coupling`).
      This is the effective-readout-chip flow — reduce a qubit-resonator
      exchange edge to the diagonal interaction an ordinary charge line
      probes. ``method`` has no effect here: no mode is removed, so the
      reduction always reads the chip's exact dressed spectrum.

    Parameters
    ----------
    chip
        Source chip (never mutated).
    target
        The device or coupling to eliminate — label string or object.
    method
        Device targets only. ``"sw"`` (default) is the 2nd-order
        Schrieffer-Wolff reduction (Bravyi, DiVincenzo & Loss, Ann. Phys. 326,
        2793 (2011)) — cheap, differentiable, and what every effective
        parameter above is derived from perturbatively. ``"exact"`` instead
        reads the reduced parameters off the chip's exact dressed spectrum
        (exact-from-dressing, :func:`quchip.chip.sw.exact_reduction`) —
        exact kept-block energies (what residual ZZ needs) at the cost of a
        full diagonalization, and it raises when near-degenerate dressed
        states make the bare labeling ambiguous. Any other value raises
        ``ValueError``.

    Returns
    -------
    EliminationResult

    Examples
    --------
    >>> from quchip import DuffingTransmon, Resonator, Capacitive, Chip
    >>> from quchip.chip.transformations import eliminate
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    >>> r = Resonator(freq=7.0, levels=5, label="r")
    >>> chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05)])
    >>> result = eliminate(chip, r)          # r removed; Lamb shift folded into q.freq
    >>> reduced = result.chip
    >>> [d.label for d in reduced.devices]
    ['q']
    """
    if method not in reduction_method_names():
        expected = " or ".join(repr(n) for n in sorted(reduction_method_names()))
        raise ValueError(f"Unknown method {method!r} for eliminate(); expected {expected}.")
    for spec in _ELIMINATION_TARGETS:
        if spec.claims(chip, target):
            return spec.reduce(chip, target, method)
    raise KeyError(f"'{resolve_label(target)}' is neither a device nor a coupling on chip '{chip.label}'.")
