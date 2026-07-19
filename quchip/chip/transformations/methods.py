"""Reduction-method registry for device elimination (extension seam).

:func:`~quchip.chip.transformations.dispatch.eliminate`'s device path is
method-agnostic up to a handful of decision points: how the surviving pair
parameters are extracted, how the eliminated mode's jump operator is carried
into the reduced frame, whether a residual ZZ and a pathway attribution are
available, and what higher-order physics the reduction drops. This module
factors those decision points into a :class:`ReductionMethod` strategy keyed
in :data:`_REDUCTION_METHODS`, so a new route (a higher-order Schrieffer-Wolff,
a numeric fit) registers a strategy with :func:`register_reduction_method`
rather than threading another ``method ==`` branch through the fold — the
same registration pattern as :mod:`quchip.chip.retarget`'s rule registry.

Dispatch keys on the *static* ``method`` string, never a traced value:
the two shipped strategies, :class:`SchriefferWolffMethod`
and :class:`ExactReduction`, are selected once by name and then operate on the
:class:`DeviceReductionContext` the caller computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np

from quchip.chip.sw import (
    bare_index,
    basis_row,
    exact_reduction,
    exact_transform_collapse,
    extract_pair_parameters,
    h_effective_second_order,
    pathway_attribution,
    transform_collapse,
)


@dataclass(frozen=True)
class DeviceReductionContext:
    """Method-agnostic inputs of one device elimination.

    Computed once by the device path of
    :func:`~quchip.chip.transformations.dispatch.eliminate` and handed to the
    chosen :class:`ReductionMethod`. Every field is shared by both shipped
    routes: the reduction differs only in how it *reads* this context, not in
    what the context contains.

    Attributes
    ----------
    chip
        The source chip being reduced (never mutated).
    mode
        The device being eliminated.
    mode_label
        Its label.
    survivor_labels
        The touching survivors in bare-label order — the same ordering the
        pair extraction and the fold loop key on, so every ``("J", a, b)``
        lookup agrees regardless of coupling-scan order.
    labels
        Device labels in the bare product-basis order.
    dims
        Per-device Hilbert-space dimensions, aligned with ``labels``.
    h
        The bare chip Hamiltonian as a dense array.
    s
        The Sylvester generator rotating the P/Q partition of ``h``.
    p_mask
        Boolean mask selecting the kept (P) block of the product basis.
    """

    chip: Any
    mode: Any
    mode_label: str
    survivor_labels: list[str]
    labels: list[str]
    dims: tuple[int, ...]
    h: Any
    s: Any
    p_mask: Any


class ReductionMethod:
    """One route from a bare chip to its reduced effective parameters.

    A concrete strategy declares its :attr:`name` (the ``method`` string
    :func:`eliminate` dispatches on) and implements the five hooks below. Each
    receives the :class:`DeviceReductionContext` the caller assembled; a hook
    returns the same shape its ``method == name`` branch produced before this
    seam existed, so the fold loop is identical across routes.

    Every hook body runs on ``jax.grad``/``jit`` paths and must stay traceable:
    no ``float()``/``int()``/``bool()`` or Python branching on a traced value.
    """

    name: ClassVar[str]

    def pair_parameters(self, ctx: DeviceReductionContext) -> dict:
        """Reduced per-survivor and per-pair parameters.

        Returns a mapping matching :func:`~quchip.chip.sw.extract_pair_parameters`
        /:func:`~quchip.chip.sw.exact_reduction`: ``{survivor: {"freq_after":
        ...}}`` for each survivor, plus a ``("J", a, b)`` entry per survivor
        pair (and, for a route that resolves it, a ``("zz", a, b)`` entry).
        """
        raise NotImplementedError

    def survivor_amplitudes(self, ctx: DeviceReductionContext) -> dict[str, Any]:
        """Survivor-lowering amplitude of the mode's transformed unit jump operator.

        One entry per ``ctx.survivor_labels``: the amplitude with which the
        eliminated mode's own (unit, dimensionless) lowering operator, carried
        into the reduced frame with the route's rotation, drives that survivor.
        The caller folds each amplitude into a Purcell rate with
        :func:`~quchip.chip.sw.purcell_rate_from`. Only called when the mode
        dissipates.
        """
        raise NotImplementedError

    def residual_zz(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> Any | None:
        """Residual ZZ between survivor pair ``(a, b)``, or ``None`` if the route cannot resolve it."""
        raise NotImplementedError

    def pathways(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> list | None:
        """Top virtual-state attribution of the pair's mediated exchange, or ``None``."""
        raise NotImplementedError

    def dropped_suffix(self) -> str:
        """Trailing clause of the ``"Dropped: ..."`` note — the physics this route omits."""
        raise NotImplementedError


class SchriefferWolffMethod(ReductionMethod):
    """2nd-order Schrieffer-Wolff reduction (``method="sw"``).

    Perturbative and differentiable: pair parameters come from the projected
    2nd-order effective Hamiltonian, and the mediated exchange carries a
    virtual-state pathway attribution. Residual ZZ is a higher-order
    correction this route does not represent
    (Bravyi, DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011)).
    """

    name: ClassVar[str] = "sw"

    def pair_parameters(self, ctx: DeviceReductionContext) -> dict:
        h_eff = h_effective_second_order(ctx.h, ctx.s, ctx.p_mask)
        p_index = np.flatnonzero(ctx.p_mask)
        return extract_pair_parameters(h_eff, p_index, ctx.labels, ctx.dims, ctx.mode_label)

    def survivor_amplitudes(self, ctx: DeviceReductionContext) -> dict[str, Any]:
        mode_index = ctx.labels.index(ctx.mode_label)
        c_full = jnp.asarray(
            ctx.chip.backend.to_array(ctx.chip.backend.embed(ctx.mode.lowering_operator(), mode_index, ctx.dims)),
            dtype=complex,
        )
        c_eff = transform_collapse(c_full, ctx.s, ctx.p_mask)
        p_index = np.flatnonzero(ctx.p_mask)
        ground_row = basis_row(p_index, ctx.labels, ctx.dims)
        return {surv: c_eff[ground_row, basis_row(p_index, ctx.labels, ctx.dims, surv)] for surv in ctx.survivor_labels}

    def residual_zz(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> Any | None:
        return None

    def pathways(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> list | None:
        i_idx = bare_index(ctx.labels, ctx.dims, a)
        j_idx = bare_index(ctx.labels, ctx.dims, b)
        return pathway_attribution(ctx.h, ctx.s, ctx.p_mask, i_idx, j_idx)

    def dropped_suffix(self) -> str:
        return ", higher-order (>2) corrections."


class ExactReduction(ReductionMethod):
    """Exact-from-dressing reduction (``method="exact"``).

    Reads the reduced parameters off the chip's exact dressed spectrum: exact
    kept-block energies (what residual ZZ needs) at the cost of a full
    diagonalization. It has no perturbative generator, so no pathway
    attribution is available (:func:`~quchip.chip.sw.exact_reduction`).
    """

    name: ClassVar[str] = "exact"

    def pair_parameters(self, ctx: DeviceReductionContext) -> dict:
        return exact_reduction(ctx.chip, ctx.mode_label, ctx.survivor_labels)

    def survivor_amplitudes(self, ctx: DeviceReductionContext) -> dict[str, Any]:
        mode_index = ctx.labels.index(ctx.mode_label)
        c_full = jnp.asarray(
            ctx.chip.backend.to_array(ctx.chip.backend.embed(ctx.mode.lowering_operator(), mode_index, ctx.dims)),
            dtype=complex,
        )
        analysis = ctx.chip._analysis
        _, evecs, _, labeling = analysis._compute_array_labeled()
        kept = [int(labeling.indices[bare_index(ctx.labels, ctx.dims)])]
        kept += [int(labeling.indices[bare_index(ctx.labels, ctx.dims, surv)]) for surv in ctx.survivor_labels]
        c_eff = exact_transform_collapse(c_full, evecs, jnp.array(kept))
        return {surv: c_eff[0, i + 1] for i, surv in enumerate(ctx.survivor_labels)}

    def residual_zz(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> Any | None:
        return jnp.real(pair_params[("zz", a, b)])

    def pathways(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> list | None:
        return None

    def dropped_suffix(self) -> str:
        return "."


_REDUCTION_METHODS: dict[str, ReductionMethod] = {}


def register_reduction_method(method: ReductionMethod) -> None:
    """Register a reduction strategy under its :attr:`ReductionMethod.name`."""
    _REDUCTION_METHODS[method.name] = method


def lookup_reduction_method(name: str) -> ReductionMethod | None:
    """Return the strategy registered under ``name``, or ``None`` if none is."""
    return _REDUCTION_METHODS.get(name)


def reduction_method_names() -> tuple[str, ...]:
    """The registered method names, in registration order."""
    return tuple(_REDUCTION_METHODS)


register_reduction_method(SchriefferWolffMethod())
register_reduction_method(ExactReduction())
