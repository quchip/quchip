"""Elimination target registry and the public ``eliminate`` dispatcher.

A transformation consumes a chip and produces a new one, so transformations
compose through ``Chip`` (for example
``eliminate(fit_a_dress(chip).chip, "r").chip``). :class:`ChipTransform` is a
thin *structural* protocol that captures only the ``.chip`` output that every
transformation result exposes.
:class:`~quchip.inverse_design.types.FitADressResult` already satisfies it.

``eliminate`` removes a mode or edge, or diagonalizes effective terms, and
keeps its calculated correction in ``EffectiveTerms``. Surviving authored
parameters stay unchanged. The result reports the approximation, diagnostics
and captured coordinate map.

Each target *kind* is an :class:`EliminationTarget`: a pair of
``(claims, reduce)`` closures. The dispatcher scans the registry and gives the
target to the first kind that claims it, or raises a clear error if none does.

A new reducible target kind registers here and does not change the handler
modules or :mod:`quchip.chip.sw`.
"""
# The unifying abstraction is the `Chip` type itself, not a base class.
#
# Elimination target kinds are registered in `_ELIMINATION_TARGETS`.
#
# The reductions are in the sibling handler modules
# (`quchip.chip.transformations.eliminate_device`,
# `quchip.chip.transformations.eliminate_coupling`,
# `quchip.chip.transformations.eliminate_effective`), which register at import
# time. The generic P/Q partitioning physics is in `quchip.chip.sw`. This module
# owns only the registry and the dispatch.

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
    """One registered elimination target kind: how it recognizes a target and reduces it.

    A new kind registers an instance with :func:`register_elimination_target`
    at the bottom of its handler module. That module must be imported for the
    side effect, which the package ``__init__`` does for the shipped handlers.
    This is the same registration ritual as
    :func:`~quchip.chip.retarget.register_retarget_rule`.

    Attributes
    ----------
    kind
        A short label for the target kind (``"device"``, ``"coupling"``,
        ``"effective terms"``), for diagnostics.
    claims
        ``claims(chip, target) -> bool``: whether this kind owns ``target`` on
        ``chip`` (for example, the label names a device, a coupling or
        effective terms). Chip rejects colliding device, coupling and
        effective-term labels, so at most one shipped kind claims a target.
    reduce
        ``reduce(chip, target, method) -> EliminationResult``: does the
        reduction. ``method`` is the already validated route string. A handler
        that has no concept of a route ignores it.
    """

    kind: str
    claims: Callable[[Any, Any], bool]
    reduce: Callable[[Any, Any, str], EliminationResult]


_ELIMINATION_TARGETS: list[EliminationTarget] = []


def register_elimination_target(target: EliminationTarget) -> None:
    """Register an elimination target kind.

    Parameters
    ----------
    target : EliminationTarget
        Target handler added to the dispatch registry.
    """
    _ELIMINATION_TARGETS.append(target)


def eliminate(chip: "Chip", target: Any, *, method: str = "sw") -> EliminationResult:
    """Reduce a far-detuned device, an edge coupling or effective terms, and return a reduced chip.

    ``target`` is resolved against the chip's device, coupling and
    effective-term labels. These labels are disjoint by construction, because
    :class:`~quchip.chip.chip.Chip` rejects colliding labels. ``target``
    dispatches to one of three model reductions:

    - **Device target**: remove a far-detuned mode and keep its calculated
      Hamiltonian correction, transformed channels and interpretation map.
      Surviving devices and direct couplings keep their authored values. A mode
      that connects several survivors contributes a separate mediated exchange edge
      per pair. Each edge reports ``∂J/∂ω_c`` for control retargeting. Capacitive
      legs emit :class:`~quchip.chip.couplings.Capacitive` or
      :class:`~quchip.chip.couplings.TunableCapacitive` when controls require it.
      Other legs emit a first-transition exchange edge. Direct and mediated
      contributions can cancel in the complete Hamiltonian. Successive reductions
      compose through ``eliminate(eliminate(chip, "TC1").chip, "TC2")``.
    - **Coupling target**: keep both endpoints and remove the selected edge. A
      coordinate change from that isolated pair acts on the entire Hamiltonian,
      including parallel and spectator interactions. The exact route keeps a
      full unitary transformation. SW keeps terms through second order. The
      correction keeps per-level shifts and does not fold them into endpoint
      frequencies or a uniform cross-Kerr coefficient.
    - **Effective-terms target**: keep every device and edge. Diagonalize the
      selected :class:`~quchip.chip.effective.EffectiveTerms` exactly with the
      local Hamiltonians of the devices they act on, for example the junction
      cosine of an energy-participation chip. The rotation acts on the entire
      Hamiltonian, and the correction keeps each level's shift, so dressed
      queries on the reduced chip return the source spectrum. Only
      ``method="exact"`` is implemented.

    Parameters
    ----------
    chip
        Source chip (never mutated).
    target
        The device, coupling or effective terms to eliminate, as a label string
        or object.
    method
        ``"sw"`` (default) keeps second-order Schrieffer-Wolff terms with the
        chip's approximation. ``"exact"`` uses the unapproximated static
        Hamiltonian. For device targets, it diagonalizes the full chip and
        selects a kept subspace, and it rejects ambiguous computational labels.
        For edge targets, it diagonalizes the selected isolated pair and
        transforms the full chip through that unitary. Surviving noise operators
        follow the captured map, and components keep rate ownership. Control
        operators are not transformed yet.

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
    >>> result = eliminate(chip, r)          # r removed; its correction is retained
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
    raise KeyError(
        f"'{resolve_label(target)}' names no device, coupling or effective terms on chip '{chip.label}'."
    )
