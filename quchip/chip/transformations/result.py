"""Result and typing layer for :mod:`quchip.chip.transformations`.

:class:`ChipTransform` is the structural protocol that every transformation result satisfies.
:class:`EliminationResult` is :func:`~quchip.chip.transformations.dispatch.eliminate`'s return
type. :class:`LazyEffectiveParams` is the deferred-``chi`` dict in which the result stores its
per-survivor entries. Reduction maps capture numerical coordinates independently of the source
chip.
"""
# This module holds no dispatch logic.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from quchip.utils.jax_utils import concrete_array_module
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
    Projection does not renormalize, so the norm or trace it loses is the
    weight outside the retained subspace. Lifting keeps norm and trace.

    Exact mode maps use the retained Löwdin embedding. Edge maps use the full
    pair eigenbasis rotation. SW maps exponentiate the first-order generator.
    Their interpretation is perturbative, but the map is isometric. Jumps
    follow that map. The SW Hamiltonian keeps its separately stated
    second-order truncation.

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
        array = self._backend.to_array(operator)
        array = concrete_array_module(array).asarray(array)
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
        array = self._backend.to_array(state)
        array = concrete_array_module(array).asarray(array)
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
        ``purcell_rate``, ``kappa`` (1/ns). ``lamb_shift`` is ``freq_after``
        minus the survivor's 0-to-1 transition on the diagonal of the incoming
        Hamiltonian. With ``method="sw"``, ``freq_after`` is that transition on
        the diagonal of the retained Hamiltonian, so it holds only the shift
        through the removed mode. With ``method="exact"``, it is the labeled
        dressed transition, so it also holds the dressing by couplings that
        stay on the chip. The two methods report different quantities, not
        two estimates of one shift. The Purcell rate summarizes the
        first downward transition, summed over the removed device's channels.
        ``kappa`` is its isolated 1-to-0 rate. These diagnostics do not replace
        the retained jumps. ``chi`` is the full conditional mode-frequency
        difference, twice the sigma-Z half-pull convention. It is zero for a
        bus that touches multiple survivors. Otherwise, it is evaluated from
        the captured source spectrum on demand and cached only when concrete.

        For two touching survivors, ``exchange`` holds one dict with ``j_eff``,
        ``dJ_domega_c``, ``between``, ``coupling``, ``zz`` and ``pathways``.
        For more survivors it is keyed by survivor pairs. ``coupling`` names
        the emitted mediated edge. ``j_eff`` and ``dJ_domega_c`` are in the
        units of that edge. Its real strength carries the representable part
        of the mediated exchange; the retained correction keeps the remainder.
        The flux-retargeting derivative remains second-order even with exact
        reduction. ``zz`` is available for the exact route and ``pathways``
        for SW. ``pathways`` lists ``(occupation, amount)`` pairs: each
        occupation is an intermediate bare product state of the source chip
        in its device order, and the amounts in GHz sum to the second-order
        part of the exchange element. :meth:`pathways` attributes any other
        retained element.

        An effective-terms target reports ``freq_after``, ``lamb_shift``,
        ``anharmonicity`` (with three or more levels) and ``cross_kerr`` for
        each device that the terms act on. ``cross_kerr`` is a mapping from
        each other such device to the full-pull shift ``E11 - E10 - E01 + E00``
        (GHz). All values come from the exactly diagonalized isolated terms.
    validity
        Per-coupling ``g_over_delta``, ``is_valid`` and ``min_block_gap``.
        ``g_over_delta`` divides the resolved exchange element between the
        coupled devices by their bare detuning. The validity flag uses
        ``g_over_delta < 0.1`` and remains a native boolean under JAX
        tracing. It is a perturbative diagnostic. The exact effective-terms
        route has no such diagnostic and leaves this mapping empty. A
        one-pass section on a shared feedline adds ``kappa_over_delta`` and
        ``is_valid`` under its label. ``kappa_over_delta`` estimates the
        dispersive field-error scale at incoming kept-port frequencies beyond
        the section. Dressed shifts can exceed this scale. ``is_valid`` uses
        ``kappa_over_delta < 0.1`` as a diagnostic, not an accuracy guarantee.
        A kept coupled response adds ``self_energy_over_width`` and
        ``is_valid`` under ``"<mode>_response"``. For each kept target of the
        other ports, the value divides the self-energy of the return path
        through the omitted pole, at the target's frequency, by the target's
        amplitude damping. The entry reports the largest value, and
        ``is_valid`` uses ``self_energy_over_width < 0.1``. Both are ``None``
        when the reduced chip has no vacuum one-excitation model, for example
        with a thermal bath. The exact route of a device adds
        ``"<mode>_labeling"``: the ``labels`` mapping gives each diagnostic
        label's occupation its largest dressed ``weight``, its ``runner_up``
        and its ``margin`` to ``threshold`` ``0.5 + 1e-6``. The entry also
        gives the smallest ``margin``, the ``gram_min_eigenvalue`` of the kept
        overlap matrix, and the orthonormalization ``residual`` with its
        ``residual_margin`` to ``residual_threshold`` ``1e-9`` (PHYSICS.md
        §10.4). A failed check raises
        :class:`~quchip.chip.sw.EliminationError` instead.
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
    _element_pathways: Any = field(default=None, repr=False, compare=False)

    def pathways(self, final: Mapping[Any, int],
                 initial: Mapping[Any, int] | None = None) -> list[tuple[tuple[int, ...], Any]]:
        """Attribute one element of the retained Hamiltonian to virtual states.

        The element is ``⟨final|H|initial⟩`` between bare product states of the
        reduced chip, given as level mappings such as ``{"qa": 1}``. Omitted
        devices are in their ground state, and ``initial`` defaults to
        ``final``, the diagonal element. Each pathway pairs an intermediate
        bare product state of the source chip, as an occupation tuple in its
        device order, with its contribution in GHz (PHYSICS.md §10.3). The
        contributions sum to the second-order part of the element. Under
        tracing, every intermediate state is listed, with exact zeros.

        Parameters
        ----------
        final : mapping
            Levels of the final state, keyed by device label or object.
        initial : mapping, optional
            Levels of the initial state. Defaults to ``final``.

        Raises
        ------
        ValueError
            For a result without a perturbative generator, such as
            ``method="exact"``, or for a state outside the retained space.
        """
        if self._element_pathways is None:
            raise ValueError("pathways() needs a device elimination with method='sw'. "
                             "Other routes have no perturbative generator.")
        return self._element_pathways(final, final if initial is None else initial)

    def describe(self) -> str:
        """Describe frequency shifts, transformed loss, exchange and validity.

        Intrinsic T1 stays separate from collective or multilevel inherited
        loss. Retargeting and approximation notes follow the scalar summaries.
        Traced values render as ``<traced>`` without host conversion."""
        from quchip.chip.describe import describe_elimination

        return describe_elimination(self)


class LazyEffectiveParams(dict):
    """Expose deferred values together with ordinary diagnostic entries.

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
