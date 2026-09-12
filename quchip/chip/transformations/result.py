"""Result and typing layer for :mod:`quchip.chip.transformations`.

:class:`ChipTransform` is the structural protocol every transformation result
satisfies; :class:`EliminationResult` is :func:`~quchip.chip.transformations.dispatch.eliminate`'s
return type, and :class:`LazyEffectiveParams` is the deferred-``chi``
dict it stores its per-survivor entries in. This module holds no dispatch
logic. Reduction maps capture numerical coordinates independently of the source chip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax.numpy as jnp


from quchip.utils.values import DeferredValue


if TYPE_CHECKING:
    from quchip.chip.chip import Chip


@runtime_checkable
class ChipTransform(Protocol):
    """Structural protocol for any transformation result that yields a chip."""

    @property
    def chip(self) -> "Chip": ...


@dataclass(frozen=True)
class ReductionMap:
    """A captured subspace map in the source and target lab-frame solver bases.

    Inputs are full numerical operators, kets or density matrices, including
    backend-native objects. Outputs use the captured backend and dimensions.
    Projection does not renormalize: its lost norm or trace is the weight
    outside the retained subspace. Lifting preserves norm and trace.

    Exact mode maps use the retained Löwdin embedding; edge maps use the
    full pair eigenbasis rotation. SW maps exponentiate the
    first-order generator; their interpretation is perturbative, even though
    the map is isometric. Jumps follow that map; the SW Hamiltonian retains
    its separately stated second-order truncation.

    Parameters
    ----------
    source_labels, target_labels : tuple[str, ...]
        Device labels defining the source and retained tensor-product orders.
    source_dims, target_dims : tuple[int, ...]
        Hilbert-space dimensions aligned with the corresponding labels.
    """

    source_labels: tuple[str, ...]
    source_dims: tuple[int, ...]
    target_labels: tuple[str, ...]
    target_dims: tuple[int, ...]
    _backend: Any = field(repr=False, compare=False)
    _embedding: DeferredValue = field(repr=False, compare=False)

    @property
    def embedding(self) -> Any:
        """Dense matrix mapping target lab coordinates into source lab coordinates."""
        return self._embedding()

    def project_operator(self, operator: Any) -> Any:
        """Restrict a source operator to the retained subspace.

        Parameters
        ----------
        operator : array-like or backend operator
            Full source-space operator.
        """
        array = jnp.asarray(self._backend.to_array(operator))
        size = prod(self.source_dims)
        if array.shape != (size, size):
            raise ValueError(f"Source operator must have shape {(size, size)}; got {array.shape}.")
        embedding = self.embedding
        reduced = embedding.conj().T @ array @ embedding
        return self._backend.from_array(reduced, dims=[list(self.target_dims), list(self.target_dims)])

    def project_state(self, state: Any) -> Any:
        """Project a source state without renormalizing it.

        Parameters
        ----------
        state : array-like or backend state
            Source-space ket or density matrix.
        """
        return self._map_state(state, self.embedding.conj().T, self.source_dims, self.target_dims)

    def lift_state(self, state: Any) -> Any:
        """Lift a retained state into the captured source space.

        Parameters
        ----------
        state : array-like or backend state
            Retained-space ket or density matrix.
        """
        return self._map_state(state, self.embedding, self.target_dims, self.source_dims)

    def _map_state(self, state: Any, transform: Any, source: tuple[int, ...], target: tuple[int, ...]) -> Any:
        array = jnp.asarray(self._backend.to_array(state))
        size = prod(source)
        if array.shape == (size,):
            array = array[:, None]
        if array.shape == (size, 1):
            value, columns = transform @ array, [1] * len(target)
        elif array.shape == (size, size):
            value, columns = transform @ array @ transform.conj().T, list(target)
        else:
            raise ValueError(f"State must have shape {(size,)}, {(size, 1)} or {(size, size)}; got {array.shape}.")
        return self._backend.from_array(value, dims=[list(target), columns])


@dataclass(frozen=True)
class EliminationResult:
    """Store a reduced chip, its numerical maps and scientific diagnostics.

    Attributes
    ----------
    chip
        Reduced model with complete retained Hamiltonian corrections and
        transformed removed-component channels in ``effective_terms``.
        Survivor noise remains separately owned. The source is not mutated.
    effective_params
        Per-survivor ``lamb_shift``, ``freq_after``, ``chi`` (GHz), and
        ``purcell_rate``, ``kappa`` (1/ns). The Purcell rate summarizes the
        first lowering transition; it does not replace the retained jumps.
        ``chi`` is the full conditional mode-frequency difference, twice
        the sigma-Z half-pull convention. It is zero for a bus touching
        multiple survivors; otherwise it is evaluated from the captured
        source spectrum on demand and cached only when concrete.

        For two touching survivors, ``exchange`` holds one dict with
        ``j_eff``, ``dJ_domega_c``, ``between``, ``coupling``, ``zz`` and
        ``pathways``. For more survivors it is keyed by survivor pairs.
        ``coupling`` names the emitted mediated edge. The flux-retargeting
        derivative remains second-order even with exact reduction.
        ``zz`` is available for the exact route; ``pathways`` for SW.
    validity
        Per-coupling ``g_over_delta``, ``is_valid`` and ``min_block_gap``.
        The validity flag uses ``g_over_delta < 0.1`` and remains a native
        boolean under JAX tracing. It is a perturbative diagnostic.
    notes
        Approximation order, omitted physics and control retargeting.
    mapping
        Captured :class:`ReductionMap` in lab-frame solver coordinates,
        or ``None`` when a transformation supplies no map.

    Diagnostic mappings accept device/coupling objects or labels. Exchange
    pair keys accept either order. Scalar summaries describe the reduction;
    the retained matrices determine the model's calculations."""

    chip: "Chip"
    effective_params: dict[str, Any]
    validity: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    mapping: ReductionMap | None = None

    def describe(self) -> str:
        """Describe frequency shifts, transformed loss, exchange and validity.

        Intrinsic T1 stays separate from collective or multilevel inherited
        loss. Retargeting and approximation notes follow the scalar summaries.
        Traced values render as ``<traced>`` without host conversion."""
        from quchip.chip.describe import describe_elimination

        return describe_elimination(self)


class LazyEffectiveParams(dict):
    """Expose deferred values alongside ordinary diagnostic entries.

    DeferredValue entries evaluate captured inputs and cache concrete results.
    Reading a value under JAX tracing leaves the stored evaluator intact.
    Representation does not trigger deferred calculations."""

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        return value() if callable(value) else value

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self) -> Any:
        return [(key, self[key]) for key in self]

    def values(self) -> Any:
        return [self[key] for key in self]

    def __repr__(self) -> str:
        shown = {key: ("<deferred>" if callable(value) else value) for key, value in dict.items(self)}
        return repr(shown)
