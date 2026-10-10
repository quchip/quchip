"""Reduction-method registry for device elimination (extension seam).

:func:`~quchip.chip.transformations.dispatch.eliminate`'s device path is
method-agnostic except at a few decision points. They are how to extract the
surviving pair parameters and how to carry the eliminated mode's jump operator
into the reduced frame. They also include whether a residual ZZ and a pathway
attribution are available, and what higher-order physics the reduction drops.
This module puts those decision points into a :class:`ReductionMethod`
strategy. New routes, such as a higher-order Schrieffer-Wolff method or numeric
fit, register with :func:`register_reduction_method`, following the rule
registry in :mod:`quchip.chip.retarget`.

Dispatch keys on the *static* ``method`` string, never on a traced value. The
two shipped strategies, :class:`SchriefferWolffMethod` and
:class:`ExactReduction`, are selected once by name and then operate on the
:class:`DeviceReductionContext` that the caller computed.
"""
# Reduction method strategies are keyed in `_REDUCTION_METHODS`.

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

from jax.scipy.linalg import expm
import numpy as np
import scipy.linalg

from quchip.chip.sw import (
    bare_index,
    exact_mode_subspace,
    exact_pair_parameters,
    extract_pair_parameters,
    h_effective_second_order,
    pathway_attribution,
    sylvester_generator,
)
from quchip.utils.jax_utils import concrete_array_module, contains_tracer


@dataclass(frozen=True)
class DeviceReductionContext:
    """Method-agnostic inputs of one device elimination.

    The device path of :func:`~quchip.chip.transformations.dispatch.eliminate`
    computes this context once and passes it to the chosen
    :class:`ReductionMethod`. The perturbative generator is computed only when
    a method requests it, and exact reduction does not use it.

    Attributes
    ----------
    mode_label
        Its label.
    survivor_labels
        The touching survivors in bare-label order. The retained Hamiltonian
        and the pair diagnostics use the same order, so every ``("J", a, b)``
        lookup agrees for all coupling-scan orders.
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
    sectors
        Total excitation number of each product state when ``h`` conserves
        it, else ``None``.
    """

    mode_label: str
    survivor_labels: list[str]
    labels: list[str]
    dims: tuple[int, ...]
    h: Any
    p_mask: Any
    sectors: np.ndarray | None = None

    @cached_property
    def s(self) -> Any:
        """Return the SW generator, computed once per reduction on request."""
        return sylvester_generator(self.h, self.p_mask)[0]

    @cached_property
    def sw_embedding(self) -> Any:
        rotation = expm(-self.s) if contains_tracer(self.s) else scipy.linalg.expm(-self.s)
        return rotation[:, np.flatnonzero(self.p_mask)]

    @cached_property
    def sw_hamiltonian(self) -> Any:
        return h_effective_second_order(self.h, self.s, self.p_mask)

    @cached_property
    def exact(self) -> Any:
        return exact_mode_subspace(self.h, self.labels, self.dims, self.mode_label, self.survivor_labels,
                                   self.sectors)


class ReductionMethod:
    """Compute a retained Hamiltonian, embedding and reduction diagnostics.

    A concrete strategy declares its :attr:`name` (the ``method`` string that
    :func:`eliminate` dispatches on) and implements the hooks below. Each hook
    receives the :class:`DeviceReductionContext` the caller assembled and uses
    its fixed product energy coordinates.

    Every hook body runs on ``jax.grad``/``jit`` paths and must stay traceable.
    Do not use ``float()``/``int()``/``bool()`` or Python branching on a traced
    value.
    """

    name: ClassVar[str]

    def source_approximation(self, chip: Any) -> Any:
        """Approximation strategy of the static model this route reads."""
        return chip.approximation

    def retained_hamiltonian(self, ctx: DeviceReductionContext) -> Any:
        """Return the complete Hamiltonian on the retained product coordinates."""
        raise NotImplementedError

    def embedding(self, ctx: DeviceReductionContext) -> Any:
        """Map retained energy coordinates into the captured full energy basis."""
        raise NotImplementedError

    def pair_parameters(self, ctx: DeviceReductionContext) -> dict:
        """Reduced per-survivor and per-pair parameters.

        Returns a mapping that matches
        :func:`~quchip.chip.sw.extract_pair_parameters`
        /:func:`~quchip.chip.sw.exact_pair_parameters`, with
        ``{survivor: {"freq_after": ...}}`` for each survivor and a ``("J", a, b)``
        entry for each survivor pair. A route that resolves it also adds a
        ``("zz", a, b)`` entry.
        """
        raise NotImplementedError

    def transform_operator(self, ctx: DeviceReductionContext, operator: Any) -> Any:
        """Carry a full operator into the kept mode-ground manifold."""
        embedding = self.embedding(ctx)
        return embedding.conj().T @ concrete_array_module(embedding, operator).asarray(operator) @ embedding

    def residual_zz(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> Any | None:
        """Residual ZZ between survivor pair ``(a, b)``, or ``None`` if the route cannot resolve it."""
        raise NotImplementedError

    def pathways(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> list | None:
        """Top virtual-state attribution of the pair's mediated exchange, or ``None``."""
        raise NotImplementedError

    def dropped_suffix(self) -> str:
        """Trailing clause of the ``"Dropped: ..."`` note: the physics this route omits."""
        raise NotImplementedError


class SchriefferWolffMethod(ReductionMethod):
    """2nd-order Schrieffer-Wolff reduction (``method="sw"``).

    This reduction is perturbative and differentiable, and the pair parameters
    come from the projected 2nd-order effective Hamiltonian. The mediated
    exchange carries a virtual-state pathway attribution. Residual ZZ is a
    higher-order correction that this route does not represent (Bravyi,
    DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011)).
    """

    name: ClassVar[str] = "sw"

    def retained_hamiltonian(self, ctx: DeviceReductionContext) -> Any:
        return ctx.sw_hamiltonian

    def embedding(self, ctx: DeviceReductionContext) -> Any:
        return ctx.sw_embedding

    def pair_parameters(self, ctx: DeviceReductionContext) -> dict:
        h_eff = self.retained_hamiltonian(ctx)
        p_index = np.flatnonzero(ctx.p_mask)
        return extract_pair_parameters(h_eff, p_index, ctx.labels, ctx.dims, ctx.mode_label)

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

    This reduction reads reduced parameters from an exact diagonalization of
    the same engine-consumed static model as the SW route. It gives exact
    kept-block energies (which residual ZZ needs), but requires a full
    diagonalization. It has no perturbative generator, so no pathway
    attribution is available (:func:`~quchip.chip.sw.exact_pair_parameters`).
    """

    name: ClassVar[str] = "exact"

    def retained_hamiltonian(self, ctx: DeviceReductionContext) -> Any:
        return ctx.exact.hamiltonian

    def embedding(self, ctx: DeviceReductionContext) -> Any:
        return ctx.exact.embedding

    def pair_parameters(self, ctx: DeviceReductionContext) -> dict:
        return exact_pair_parameters(ctx.exact, ctx.labels, ctx.dims, ctx.mode_label, ctx.survivor_labels)

    def residual_zz(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> Any | None:
        zz = pair_params[("zz", a, b)]
        return concrete_array_module(zz).real(zz)

    def pathways(self, ctx: DeviceReductionContext, pair_params: dict, a: str, b: str) -> list | None:
        return None

    def dropped_suffix(self) -> str:
        return "."


_REDUCTION_METHODS: dict[str, ReductionMethod] = {}


def register_reduction_method(method: ReductionMethod) -> None:
    """Register a reduction strategy under its name.

    Parameters
    ----------
    method : ReductionMethod
        Strategy added under its unique ``name``.
    """
    _REDUCTION_METHODS[method.name] = method


def lookup_reduction_method(name: str) -> ReductionMethod | None:
    """Return the strategy registered under ``name``, or ``None`` if none is."""
    return _REDUCTION_METHODS.get(name)


def reduction_method_names() -> tuple[str, ...]:
    """The registered method names, in registration order."""
    return tuple(_REDUCTION_METHODS)


register_reduction_method(SchriefferWolffMethod())
register_reduction_method(ExactReduction())
