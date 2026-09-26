"""Elimination target registry and the public ``eliminate`` dispatcher.

The unifying abstraction is the :class:`~quchip.chip.chip.Chip` type itself,
not a base class: a transformation consumes a chip and produces a new one, so
transformations compose through ``Chip`` (e.g.
``eliminate(fit_a_dress(chip).chip, "r").chip``). :class:`ChipTransform` is a
thin *structural* protocol capturing only the ``.chip`` output that every
transformation result exposes;
:class:`~quchip.inverse_design.types.FitADressResult` already satisfies it with
no changes.

``eliminate`` removes a mode or edge and retains its computed correction in
``EffectiveTerms``. Surviving authored parameters stay unchanged. The result
reports the approximation, diagnostics and captured coordinate map.

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
    """Register an elimination target kind.

    Parameters
    ----------
    target : EliminationTarget
        Target handler added to the dispatch registry.
    """
    _ELIMINATION_TARGETS.append(target)


def eliminate(chip: "Chip", target: Any, *, method: str = "sw") -> EliminationResult:
    """Reduce a far-detuned device or an edge coupling, returning a reduced chip.

    ``target`` is resolved against the chip's device and coupling namespaces
    (disjoint by construction — :class:`~quchip.chip.chip.Chip` rejects a
    coupling label that collides with a device label) and dispatches to one
    of two model reductions:

    - **Device target** — remove a far-detuned mode while retaining its
      computed Hamiltonian correction, transformed channels and interpretation
      map. Surviving devices and direct couplings retain their authored values.
      A mode connecting several survivors contributes a separate mediated
      exchange edge per pair, with ``∂J/∂ω_c`` reported for control retargeting.
      Capacitive legs emit :class:`~quchip.chip.couplings.Capacitive` or
      :class:`~quchip.chip.couplings.TunableCapacitive` when controls require it;
      other legs emit a first-transition exchange edge. Direct and mediated
      contributions can cancel in the complete Hamiltonian. Successive
      reductions compose through ``eliminate(eliminate(chip, "TC1").chip, "TC2")``.
    - **Coupling target** — keep both endpoints and remove the selected edge.
      A coordinate change derived from that isolated pair acts on the entire
      Hamiltonian, including parallel and spectator interactions. The exact
      route retains a full unitary transformation; SW retains terms through
      second order. The correction preserves per-level shifts without folding
      them into endpoint frequencies or a uniform cross-Kerr coefficient.

    Parameters
    ----------
    chip
        Source chip (never mutated).
    target
        The device or coupling to eliminate — label string or object.
    method
        ``"sw"`` (default) retains second-order Schrieffer-Wolff terms using
        the chip's approximation. ``"exact"`` uses the unapproximated static
        Hamiltonian. For device targets it diagonalizes the full chip and
        selects a retained subspace, rejecting ambiguous computational labels.
        For edge targets it diagonalizes the selected isolated pair and
        transforms the full chip through that unitary. Surviving noise operators
        follow the captured map while components retain rate ownership.
        Control operators are not transformed yet.

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
    raise KeyError(f"'{resolve_label(target)}' is neither a device nor a coupling on chip '{chip.label}'.")
