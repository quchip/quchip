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

from quchip.backend._memory import fits_memory, require_memory
from quchip.utils.jax_utils import concrete_array_module, contains_tracer


def _dense_bytes(size, devices):
    """Peak bytes of dense compact operators and coordinate labels on ``size`` states."""
    return 16 * 8 * size**2 + 8 * devices * size


@dataclass(frozen=True, eq=False)
class PulseSpace:
    dims: tuple[int, ...]
    labels: tuple[str, ...]
    totals: tuple[int, ...]
    levels: np.ndarray
    bases: Any

    @classmethod
    def create(cls, dims, labels, totals, bases):
        from quchip.engine.sectors import _sector_levels, sector_size

        totals = tuple(sorted(set(totals)))
        size = sum(sector_size(dims, total) for total in totals)
        require_memory(
            _dense_bytes(size, len(dims)),
            task=f"Exact pulse space (dimension {size})",
            remedy="Reduce the model or initial excitation support.",
        )
        levels = np.concatenate([_sector_levels(tuple(dims), n) for n in totals])
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

    def embed(self, backend, operator, support):
        """Gather a resolved local operator without a product-space allocation."""
        from quchip.engine.assembly import _energy_operator
        from quchip.engine.sectors import _level_pairs

        support = tuple(support) or tuple(range(len(self.dims)))
        matrix = _energy_operator(operator, tuple(self.labels[axis] for axis in support), self.bases, backend)
        xp = concrete_array_module(matrix)
        rows, columns, local_rows, local_columns = _level_pairs(self.dims, support, self.levels, self.levels)
        values = matrix[local_rows, local_columns]
        result = xp.zeros((self.size, self.size), dtype=complex)
        if xp is np:
            result[rows, columns] = values
        else:
            result = result.at[rows, columns].set(values)
        return backend.from_array(result, dims=[[self.size], [self.size]])

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
        xp = concrete_array_module(terms, tuple(self.bases[label].energy_vectors for label in labels))
        result = xp.zeros((self.size, self.size), dtype=complex)
        for coefficient, factors in terms:
            value = xp.ones((self.size, self.size), dtype=complex) * coefficient
            for axis, label in enumerate(self.labels):
                levels = self.levels[:, axis]
                if label not in factors:
                    value = value * (levels[:, None] == levels[None, :])
                else:
                    vectors = xp.asarray(self.bases[label].energy_vectors)
                    local = xp.conj(vectors.T) @ xp.asarray(factors[label]) @ vectors
                    value = value * local[levels[:, None], levels[None, :]]
            result = result + value
        return backend.from_array(result, dims=[[self.size], [self.size]])

    def lift(self, state, backend):
        """Expand a state into captured solver coordinates, after a memory check."""
        matrix = backend.to_array(state)
        dimension = prod(self.dims)
        ket = matrix.shape[-1] == 1
        require_memory(
            16 * 4 * (dimension if ket else dimension**2),
            task="Full pulse state export",
            remedy="Use compact_state(), population(), or reduced_state() instead.",
        )
        xp = backend.array_module
        # Build only the isometry columns present in the compact space.
        require_memory(
            16 * 3 * dimension * self.size,
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


def initial_totals(chip, state, resolution):
    from collections.abc import Mapping
    from quchip.chip.states import normalize_device_state_mapping, materialize_state_spec
    from quchip.results.compact import CompactState

    if state is None:
        return {0}
    if isinstance(state, CompactState):
        if state.space.labels != tuple(d.label for d in chip.devices) or state.space.dims != resolution.dims:
            raise ValueError("Compact state devices and dimensions do not match this chip.")
        return set(state.space.totals)
    if isinstance(state, (Mapping, str)):
        prepared = normalize_device_state_mapping(chip, state, {})
        if all(isinstance(n, int) and not isinstance(n, bool) for n in prepared.values()):
            for label, n in prepared.items():
                if label not in resolution.bases or not 0 <= n < resolution.bases[label].resolved_dim:
                    raise ValueError(f"Invalid initial level {n!r} for {label!r}.")
            return {sum(prepared.values())}
        from quchip.chip.states import _bare_local_kets

        kets = _bare_local_kets(chip, prepared, resolution.bases)
        totals = {0}
        for device, ket in zip(chip.devices, kets):
            vector = chip.backend.to_array(ket).reshape(-1)
            transform = resolution.bases[device.label].energy_to_solver()
            if contains_tracer((vector, transform)):
                return None
            if transform is not None:
                vector = np.asarray(transform).conj().T @ np.asarray(vector)
            local = np.flatnonzero(np.asarray(vector) != 0)
            totals = {n + int(level) for n in totals for level in local}
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
    return set(coordinates.sum(axis=0).tolist())


def plan_pulse_space(chip, operations, states, resolution, approximation, dissipation):
    """Prove a union of total-index sectors invariant under the complete generator."""
    from quchip.chip.effective import authored_excitation_changes, conserves_excitation_number
    from quchip.chip.ports import Port
    from quchip.engine.assembly import (
        _interaction_contributions,
        _build_delivered_signals,
        drive_excitation_changes,
        _component_authored_terms,
    )
    from quchip.engine.ir import CoherentOp

    bases, dims = resolution.bases, resolution.dims
    labels = tuple(d.label for d in chip.devices)
    # The implicit dressed vacuum requires the static ground-sector proof.
    if any(state is None for state in states) and not conserves_excitation_number(
        chip, approximation, resolution=resolution
    ):
        return None, "The default dressed state has no proven finite excitation support."
    occupied = set()
    for state in states:
        totals = initial_totals(chip, state, resolution)
        if totals is None or not totals:
            return None, "Initial excitation support is unknown."
        occupied.update(totals)
    changes = {0}
    for term in _interaction_contributions(chip):
        if not term.retained and approximation.filters_terms and approximation.conserves_excitation_number():
            continue
        support = authored_excitation_changes(term.expression(), term.labels, chip.backend, bases)
        if support is None:
            return None, "Static interaction support is unknown."
        changes.update(support)
        changes.update(-n for n in support)
    for operator, _coefficient, support, _owner, _origin, _tag in _component_authored_terms(
        chip, resolution, chip.backend
    ):
        if len(support) == 2 and approximation.filters_terms and approximation.conserves_excitation_number():
            continue
        support_changes = authored_excitation_changes(operator, tuple(labels[i] for i in support), chip.backend, bases)
        if support_changes is None:
            return None, "Intrinsic time-dependent support is unknown."
        changes.update(support_changes)
        changes.update(-n for n in support_changes)
    for delivered in _build_delivered_signals(chip, list(operations)).values():
        weights = drive_excitation_changes(chip, delivered.drive, delivered.target, bases, approximation)
        if weights is None:
            return None, "Delivered drive support is unknown."
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
            return None, "Collapse operator support is unknown."
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
    maximum = sum(d - 1 for d in dims)
    while True:
        reached = occupied | {n - change for n in occupied for change in changes if 0 <= n - change <= maximum}
        if reached == occupied:
            from quchip.engine.sectors import sector_size

            # Native solver algebras treat 1x1 QArrays/Qobjs as scalars. Add
            # another physical sector and prove closure again, never a dummy level.
            if sum(sector_size(dims, n) for n in reached) == 1 and maximum > 0:
                occupied.add(next(n for n in range(maximum + 1) if n not in occupied))
                continue
            break
        occupied = reached
    from quchip.engine.sectors import sector_size

    size = sum(sector_size(dims, n) for n in occupied)
    if size == prod(dims):
        return None, "The proven generator closure reaches the full product space."
    if not fits_memory(_dense_bytes(size, len(dims))):
        return None, f"Dense compact operators on {size} states do not fit in available memory."
    return PulseSpace.create(dims, labels, occupied, bases), "The initial support and complete generator close exactly."
