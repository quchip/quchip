"""Invariant excitation spaces for deterministic evolution.

Coordinates are product energy labels. Support proofs ignore scalar values;
unknown support never supplies an exact reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import prod
from typing import Any

import numpy as np

from quchip.utils.memory import dense_bytes, level_table_bytes, require_memory
from quchip.utils.jax_utils import concrete_array_module, contains_tracer


# Largest relative weight of a supplied state outside an excitation cutoff that
# counts as basis-conversion round-off. Larger weight raises instead of being dropped.
_CUTOFF_RESIDUE = 1e-20


_OPERATOR_REMEDY = "Reduce the model, the initial excitation support, or an excitation cutoff's max_total."


def _require_inside(cutoff, outside, total, source):
    """Raise when a supplied state has more than round-off weight outside the cutoff."""
    if cutoff is not None and outside > _CUTOFF_RESIDUE * total:
        raise ValueError(
            f"The {source} has weight {outside / total:.3g} on labels with N > {cutoff}, outside the "
            f"excitation cutoff. quchip treats weight below {_CUTOFF_RESIDUE:g} as basis-conversion round-off, "
            "drops it and does not renormalize. "
            "Prepare the state inside the cutoff, for example with chip.compact_state(), or raise max_total."
        )


@dataclass(frozen=True, eq=False)
class PulseSpace:
    dims: tuple[int, ...]
    labels: tuple[str, ...]
    totals: tuple[int, ...]
    levels: np.ndarray
    bases: Any

    @classmethod
    def create(cls, dims, labels, totals, bases):
        from quchip.engine.excitation import sector_levels, sector_size

        totals = tuple(sorted(set(totals)))
        size = sum(sector_size(dims, total) for total in totals)
        require_memory(level_table_bytes(size, len(dims)), task=f"Exact pulse space (dimension {size})",
                       remedy=_OPERATOR_REMEDY)
        levels = np.concatenate([sector_levels(tuple(dims), n) for n in totals])
        levels.flags.writeable = False
        return cls(tuple(dims), tuple(labels), totals, levels, dict(bases))

    @property
    def size(self):
        return len(self.levels)

    @property
    def solver_dims(self):
        return (self.size,)

    @cached_property
    def rows(self):
        return {tuple(label): index for index, label in enumerate(self.levels)}

    def entries(self, backend, operator, support):
        """Return the rows, columns and values of a resolved local operator on this space."""
        from quchip.engine.assembly import _energy_operator
        from quchip.engine.sectors import _level_pairs

        support = tuple(support) or tuple(range(len(self.dims)))
        matrix = _energy_operator(operator, tuple(self.labels[axis] for axis in support), self.bases, backend)
        rows, columns, local_rows, local_columns = _level_pairs(self.dims, support, self.levels, self.levels)
        return rows, columns, matrix[local_rows, local_columns]

    def embed(self, backend, operator, support):
        """Gather a resolved local operator without a product-space allocation.

        The backend chooses the storage of the result.
        """
        return backend.compact_operator(*self.entries(backend, operator, support), self.size)

    def authored(self, backend, operator, support):
        """Lower a factored authored sum directly into energy coordinates."""
        from quchip.declarative.expr import PhysicsExpr
        from quchip.engine.assembly import _local_products

        if not isinstance(operator, PhysicsExpr):
            return None
        labels = tuple(self.labels[i] for i in support) if support else self.labels
        terms = _local_products(operator, labels, self.bases)
        if terms is None:
            return None
        return backend.compact_operator(*self._authored_entries(terms, labels), self.size)

    def _authored_entries(self, terms, labels):
        """Return the rows, columns and values of a sum of local products, with repeated entries."""
        from quchip.engine.sectors import _level_pairs

        xp = concrete_array_module(terms, tuple(self.bases[label].energy_vectors for label in labels))
        rows, columns, values = [np.zeros(0, dtype=int)], [np.zeros(0, dtype=int)], [xp.zeros(0, dtype=complex)]
        for coefficient, factors in terms:
            axes = tuple(sorted(self.labels.index(label) for label in factors))
            if not axes:
                diagonal = np.arange(self.size)
                rows.append(diagonal)
                columns.append(diagonal)
                values.append(xp.full(self.size, coefficient, dtype=complex))
                continue
            local = xp.ones((1, 1), dtype=complex)
            for axis in axes:
                vectors = xp.asarray(self.bases[self.labels[axis]].energy_vectors)
                local = xp.kron(local, xp.conj(vectors.T) @ xp.asarray(factors[self.labels[axis]]) @ vectors)
            pair_rows, pair_columns, local_rows, local_columns = _level_pairs(self.dims, axes, self.levels, self.levels)
            rows.append(pair_rows)
            columns.append(pair_columns)
            values.append(coefficient * local[local_rows, local_columns])
        return np.concatenate(rows), np.concatenate(columns), xp.concatenate(values)

    def lift(self, state, backend):
        """Expand a state into captured solver coordinates, after a memory check."""
        matrix = backend.to_array(state)
        dimension = prod(self.dims)
        ket = matrix.shape[-1] == 1
        require_memory(
            dense_bytes(*((dimension,) if ket else (dimension, dimension)), copies=4),
            task="Full pulse state export",
            remedy="Use compact_state(), population(), or reduced_state() instead.",
        )
        xp = backend.array_module
        # Build only the isometry columns present in the compact space.
        require_memory(
            dense_bytes(dimension, self.size, copies=3),
            task="Pulse state coordinate map",
            remedy="Use compact_state() or reduced_state() instead.",
        )
        columns = []
        for levels in self.levels:
            column = xp.ones(1, dtype=complex)
            for label, level in zip(self.labels, levels):
                column = xp.kron(column, xp.asarray(self.bases[label].energy_state(int(level))))
            columns.append(column)
        transform = xp.stack(columns, axis=1)
        result = transform @ xp.asarray(matrix)
        if not ket:
            result = result @ xp.conj(transform.T)
        return backend.from_array(result, dims=[list(self.dims), [1] * len(self.dims) if ket else list(self.dims)])

    def project(self, state, backend):
        """Project an already allocated native solver state into these coordinates."""
        xp = backend.array_module
        matrix = xp.asarray(backend.to_array(state))
        tensor = matrix.reshape((*self.dims, *((1,) if matrix.shape[-1] == 1 else self.dims)))
        for axis, label in enumerate(self.labels):
            vectors = self.bases[label].energy_to_solver()
            if vectors is not None:
                tensor = xp.moveaxis(xp.tensordot(xp.conj(xp.asarray(vectors).T), tensor, axes=([1], [axis])), 0, axis)
        if matrix.shape[-1] != 1:
            for axis, label in enumerate(self.labels, len(self.dims)):
                vectors = self.bases[label].energy_to_solver()
                if vectors is not None:
                    tensor = xp.moveaxis(xp.tensordot(xp.asarray(vectors).T, tensor, axes=([1], [axis])), 0, axis)
        indices = np.ravel_multi_index(tuple(self.levels.T), self.dims)
        energy = tensor.reshape(matrix.shape)
        result = energy[indices]
        if matrix.shape[-1] != 1:
            result = result[:, indices]
        return backend.from_array(result, dims=[[self.size], [1] if matrix.shape[-1] == 1 else [self.size]])

    def reduced(self, state, backend, axes):
        """Trace out spectators directly, preserving inter-sector coherence."""
        axes = tuple(axes)
        xp = backend.array_module
        matrix = xp.asarray(backend.to_array(state))
        density = matrix @ xp.conj(matrix.T) if matrix.shape[-1] == 1 else matrix
        outside = [i for i in range(len(self.dims)) if i not in axes]
        rows, columns = np.nonzero(np.all(self.levels[:, None, outside] == self.levels[None, :, outside], axis=-1))
        dims = tuple(self.dims[i] for i in axes)
        indices = np.ravel_multi_index(tuple(self.levels[:, axes].T), dims)
        result = xp.zeros((prod(dims), prod(dims)), dtype=complex)
        if xp is np:
            np.add.at(result, (indices[rows], indices[columns]), density[rows, columns])
        else:
            result = result.at[indices[rows], indices[columns]].add(density[rows, columns])
        transform = xp.ones((1, 1), dtype=complex)
        for axis in axes:
            vectors = self.bases[self.labels[axis]].energy_to_solver()
            transform = xp.kron(transform, xp.eye(self.dims[axis]) if vectors is None else xp.asarray(vectors))
        result = transform @ result @ xp.conj(transform.T)
        return backend.from_array(result, dims=[list(dims), list(dims)])


def initial_totals(chip, state, resolution, cutoff=None):
    """Return the excitation totals that an initial state occupies, or ``None`` when unknown.

    With ``cutoff``, a state with more than round-off weight on totals above
    it raises, and only totals up to the cutoff are returned.
    """
    from collections.abc import Mapping
    from quchip.chip.states import normalize_device_state_mapping
    from quchip.engine.states import CompactState, bare_local_kets, materialize_state_spec

    if state is None:
        return {0}
    if isinstance(state, CompactState):
        if state.space.labels != tuple(d.label for d in chip.devices) or state.space.dims != resolution.dims:
            raise ValueError("Compact state devices and dimensions do not match this chip.")
        if cutoff is not None and max(state.space.totals) > cutoff:
            raise ValueError(
                f"The compact state holds totals {list(state.space.totals)}, outside the excitation cutoff "
                f"N <= {cutoff}. Prepare it for this chip with chip.compact_state()."
            )
        return set(state.space.totals)
    if isinstance(state, (Mapping, str)):
        prepared = normalize_device_state_mapping(chip, state, {})
        if all(isinstance(n, int) and not isinstance(n, bool) for n in prepared.values()):
            for label, n in prepared.items():
                if label not in resolution.bases or not 0 <= n < resolution.bases[label].resolved_dim:
                    raise ValueError(f"Invalid initial level {n!r} for {label!r}.")
            total = sum(prepared.values())
            if cutoff is not None and total > cutoff:
                raise ValueError(
                    f"The initial label has total excitation {total}, outside the excitation cutoff N <= {cutoff}."
                )
            return {total}
        kets = bare_local_kets(chip, prepared, resolution.bases)
        totals = {0}
        weights = np.ones(1)
        for device, ket in zip(chip.devices, kets):
            vector = chip.backend.to_array(ket).reshape(-1)
            transform = resolution.bases[device.label].energy_to_solver()
            if contains_tracer((vector, transform)):
                return None
            if transform is not None:
                vector = np.asarray(transform).conj().T @ np.asarray(vector)
            local = np.flatnonzero(np.asarray(vector) != 0)
            totals = {n + int(level) for n in totals for level in local}
            weights = np.convolve(weights, np.abs(np.asarray(vector)) ** 2)
        if cutoff is not None:
            _require_inside(cutoff, float(np.sum(weights[cutoff + 1:])), float(np.sum(weights)), "initial state")
            totals = {n for n in totals if n <= cutoff}
        return totals
    if contains_tracer(state):
        return None
    # Native state support is known only for concrete captured arrays.
    native = materialize_state_spec(chip, state, resolution.bases)
    if contains_tracer(native):
        return None
    dims = resolution.dims
    matrix = np.asarray(chip.backend.to_array(native))
    ket = matrix.shape[-1] == 1
    tensor = matrix.reshape((*dims, *((1,) if ket else dims)))
    for axis, device in enumerate(chip.devices):
        vectors = resolution.bases[device.label].energy_to_solver()
        if vectors is not None:
            tensor = np.moveaxis(np.tensordot(np.asarray(vectors).conj().T, tensor, axes=([1], [axis])), 0, axis)
    if not ket:
        for axis, device in enumerate(chip.devices, len(dims)):
            vectors = resolution.bases[device.label].energy_to_solver()
            if vectors is not None:
                tensor = np.moveaxis(np.tensordot(np.asarray(vectors).T, tensor, axes=([1], [axis])), 0, axis)
    energy = tensor.reshape(matrix.shape)
    occupied = np.any(energy != 0, axis=1)
    if not ket:
        occupied |= np.any(energy != 0, axis=0)
    coordinates = np.asarray(np.unravel_index(np.flatnonzero(occupied), dims))
    totals = set(coordinates.sum(axis=0).tolist())
    if cutoff is not None:
        weights = np.abs(energy[:, 0]) ** 2 if ket else np.abs(np.real(np.diagonal(energy)))
        level_sums = np.asarray(np.unravel_index(np.arange(weights.size), dims)).sum(axis=0)
        _require_inside(cutoff, float(np.sum(weights[level_sums > cutoff])), float(np.sum(weights)), "initial state")
        totals = {n for n in totals if n <= cutoff}
    return totals


def plan_pulse_space(chip, operations, states, resolution, approximation, dissipation, *, whole=False):
    """Prove a union of total-index sectors invariant under the complete generator.

    Returns the space, the reason, and the boundary totals. Under an active
    excitation cutoff ``K``, every step of the closure stays inside
    ``[0, K]``, unknown support uses every label with ``N <= K``, and the
    product space is never used. ``whole`` selects that cutoff space after
    the initial states are checked. The boundary holds the retained totals
    that a generator change can move above ``K``, or is ``None`` when the
    changes are unknown or no cutoff is active.
    """
    from quchip.chip.ports import Port
    from quchip.cutoff import active_cutoff
    from quchip.engine.assembly import _build_delivered_signals, _component_authored_terms
    from quchip.engine.excitation import (
        authored_excitation_changes, drive_excitation_changes, excitation_structure, keeps_zero_total_only, sector_size,
    )
    from quchip.engine.ir import CoherentOp

    bases, dims = resolution.bases, resolution.dims
    labels = tuple(d.label for d in chip.devices)
    policy = active_cutoff(chip)
    cutoff = None if policy is None else policy.max_total

    structure = excitation_structure(chip, approximation, resolution=resolution)

    def fallback(reason):
        if cutoff is None:
            return None, reason, None
        space = PulseSpace.create(dims, labels, range(cutoff + 1), bases)
        return space, f"{reason} The solve keeps every label with N <= {cutoff}.", None

    occupied = set()
    if any(state is None for state in states):
        if cutoff is not None:
            # The default state is the vacuum-assigned eigenvector of the truncated static model.
            occupied.update(next(group for group in structure.classes(cutoff) if 0 in group))
        elif not structure.conserves:
            return None, "The default dressed state has no proven finite excitation support.", None
        else:
            occupied.add(0)
    for state in states:
        if state is None:
            continue
        totals = initial_totals(chip, state, resolution, cutoff)
        if totals is None and cutoff is not None:
            raise ValueError(
                "The support of the traced initial state is unknown, so quchip cannot prove that it lies inside "
                f"the excitation cutoff N <= {cutoff}. Pass chip.compact_state(...) or a label mapping instead."
            )
        if totals is None or not totals:
            return None, "Initial excitation support is unknown.", None
        occupied.update(totals)
    changes = {0}
    for support in structure.interaction_changes():
        if support is None:
            return fallback("Static interaction support is unknown.")
        changes.update(support)
        changes.update(-n for n in support)
    for operator, _coefficient, support, _owner, _origin, _tag in _component_authored_terms(
        chip, resolution, chip.backend
    ):
        if len(support) == 2 and keeps_zero_total_only(approximation):
            continue
        support_changes = authored_excitation_changes(operator, tuple(labels[i] for i in support), chip.backend, bases)
        if support_changes is None:
            return fallback("Intrinsic time-dependent support is unknown.")
        changes.update(support_changes)
        changes.update(-n for n in support_changes)
    for delivered in _build_delivered_signals(chip, list(operations)).values():
        weights = drive_excitation_changes(chip, delivered.drive, delivered.target, bases, approximation)
        if weights is None:
            return fallback("Delivered drive support is unknown.")
        changes.update(weights)
        changes.update(-n for n in weights)
    # A jump and its no-jump product have separate support obligations. Network
    # scattering and coherent inputs combine port channels only.
    coherent = any(isinstance(op, CoherentOp) for op in operations)
    ports = set()
    for operator, _rate, support, *_, owner in chip._collapse_contributions_with_owners(bases):
        port = isinstance(owner, Port)
        if not (dissipation or port and (coherent or resolution.network is not None)):
            continue
        weights = authored_excitation_changes(
            operator, tuple(labels[i] for i in support) or labels, chip.backend, bases
        )
        if weights is None:
            return fallback("Collapse operator support is unknown.")
        if dissipation:
            changes.update(weights)
            changes.update(a - b for a in weights for b in weights)
        if port:
            ports.update(weights)
    if resolution.network is not None:
        # Mixed port channels put their cross products in H and L†L.
        changes.update(a - b for a in ports for b in ports)
        if dissipation and any(
            field.input_occupation is not None for field in chip.port_network._field_channels(resolution.network)
        ):
            changes.update(-n for n in ports)
    if coherent:
        changes.update(ports)
        changes.update(-n for n in ports)
    maximum = sum(d - 1 for d in dims) if cutoff is None else cutoff
    if whole:
        occupied = set(range(maximum + 1))
    while True:
        reached = occupied | {n - change for n in occupied for change in changes if 0 <= n - change <= maximum}
        if reached == occupied:
            # Native solver algebras treat 1x1 QArrays/Qobjs as scalars. Add
            # another physical sector and prove closure again, never a dummy level.
            if sum(sector_size(dims, n) for n in reached) == 1 and maximum > 0:
                occupied.add(next(n for n in range(maximum + 1) if n not in occupied))
                continue
            break
        occupied = reached
    size = sum(sector_size(dims, n) for n in occupied)
    if cutoff is not None:
        # Changes are column minus row, so a negative change raises the total.
        boundary = tuple(sorted(n for n in occupied if any(n - change > cutoff for change in changes)))
        reason = (f"Every label with N <= {cutoff} is kept." if whole else
                  "The initial support and the truncated generator close inside the excitation cutoff.")
        return PulseSpace.create(dims, labels, occupied, bases), reason, boundary
    if size == prod(dims):
        return None, "The proven generator closure reaches the full product space.", None
    if not chip.backend.compact_operators_fit(size, 8):
        return None, f"The backend's compact operators on {size} states do not fit in available memory.", None
    reason = "The initial support and complete generator close exactly."
    return PulseSpace.create(dims, labels, occupied, bases), reason, None
