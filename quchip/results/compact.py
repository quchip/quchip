"""Reusable states in captured excitation coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from quchip.engine.pulse_space import PulseSpace


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CompactState:
    """Keep a ket or density matrix without expanding its product coordinates.

    Create this value with ``chip.compact_state`` or ``result.compact_state``.
    Pass it as ``initial_state`` to a later simulation of the same device bases.
    Numerical arrays are JAX pytree leaves.

    Parameters
    ----------
    values : array_like
        Ket columns or density matrix in the captured excitation space.
    space : object
        Captured coordinate map, supplied by the state factory.
    """

    values: Any
    space: PulseSpace

    def tree_flatten(self):
        """Return numerical leaves and immutable coordinate metadata for JAX."""
        records = tuple(self.space.bases[label] for label in self.space.labels)
        leaves = (self.values, *((r.vectors, r.energies, r.energy_vectors) for r in records))
        metadata = (
            self.space.dims,
            self.space.labels,
            self.space.totals,
            tuple((r.kind, r.native_dim, r.resolved_dim) for r in records),
        )
        return leaves, metadata

    @classmethod
    def tree_unflatten(cls, metadata, leaves):
        """Restore a state from JAX's coordinate metadata and numerical leaves.

        Parameters
        ----------
        metadata : tuple
            Captured dimensions, labels, totals, and basis kinds.
        leaves : tuple
            State values and numerical basis arrays.
        """
        from quchip.engine.basis import BasisRecord
        from quchip.engine.sectors import _sector_levels

        dims, labels, totals, records = metadata
        bases = {
            label: BasisRecord(kind, arrays[0], arrays[1], arrays[2], native, resolved)
            for label, (kind, native, resolved), arrays in zip(labels, records, leaves[1:])
        }
        levels = np.concatenate([_sector_levels(dims, total) for total in totals])
        return cls(leaves[0], PulseSpace(dims, labels, totals, levels, bases))


def compact_state(chip, *components, dressed=False):
    from quchip.chip.states import normalize_device_state_mapping
    from quchip.engine.assembly import _resolve_system
    from quchip.engine.sectors import SectorModel
    from quchip.chip.effective import conserves_excitation_number
    from quchip.utils.jax_utils import maybe_concrete_scalar

    resolution = _resolve_system(chip, chip.backend)
    labels = tuple(d.label for d in chip.devices)
    entries = []
    for component in components or ({},):
        amplitude, spec = component if isinstance(component, tuple) and len(component) == 2 else (1, component)
        prepared = normalize_device_state_mapping(chip, spec, {})
        if set(prepared) - set(labels):
            raise ValueError("Compact state contains an unknown device.")
        levels = tuple(prepared.get(label, 0) for label in labels)
        if any(
            not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < d for n, d in zip(levels, resolution.dims)
        ):
            raise ValueError("Compact state specifications require valid integer energy levels.")
        entries.append((amplitude, levels))
    space = PulseSpace.create(resolution.dims, labels, {sum(levels) for _, levels in entries}, resolution.bases)
    xp = chip.backend.array_module
    vector = xp.zeros(space.size, dtype=complex)
    if dressed and not conserves_excitation_number(chip, chip.approximation, resolution=resolution):
        raise ValueError("Compact dressed preparation requires proven static excitation conservation.")
    model = SectorModel(chip, resolution=resolution) if dressed else None
    for amplitude, levels in entries:
        if model is None:
            contribution = xp.asarray(np.arange(space.size) == space.rows[levels], dtype=complex)
        else:
            system = model.eigensystem(sum(levels))
            row = system.space.row(levels)
            eigenvector = xp.asarray(system.eigenvectors)[:, system.labeling.indices[row]]
            overlap = eigenvector[row]
            phase = xp.where(xp.abs(overlap) > 0, overlap / xp.where(xp.abs(overlap) > 0, xp.abs(overlap), 1), 1)
            indices = np.asarray([space.rows[label] for label in system.space.labels])
            contribution = xp.asarray(np.eye(space.size)[:, indices]) @ (eigenvector / phase)
        vector = vector + amplitude * contribution
    norm = xp.linalg.norm(vector)
    if maybe_concrete_scalar(norm) == 0:
        raise ValueError("Compact state superposition has zero norm.")
    return CompactState((vector / xp.where(norm > 0, norm, 1)).reshape(-1, 1), space)



def _compatible_values(state, labels, dims, bases, backend):
    """Validate captured coordinates before interpreting a saved energy state."""
    from quchip.results._time import require_valid

    if state.space.labels != tuple(labels) or state.space.dims != tuple(dims):
        raise ValueError("Compact state coordinates do not match this calculation.")
    xp = backend.array_module
    values = xp.asarray(state.values)
    for label in labels:
        old, new = state.space.bases[label], bases[label]
        values = require_valid(
            values,
            xp.any(xp.asarray(old.energy_vectors) != xp.asarray(new.energy_vectors)),
            "Compact state energy bases changed; prepare the state for this chip.",
        )
    return values


def materialize_compact(chip, spec, space):
    from quchip.chip.states import materialize_state_spec, normalize_device_state_mapping
    from collections.abc import Mapping
    backend = chip.backend
    xp = backend.array_module
    if spec is None or isinstance(spec, (Mapping, str)):
        prepared = normalize_device_state_mapping(chip, spec, {})
        label = tuple(prepared.get(key, 0) for key in space.labels)
        if all(isinstance(n, int) for n in label):
            vector = xp.asarray(np.arange(space.size) == space.rows[label], dtype=complex).reshape(-1, 1)
            return backend.from_array(vector, dims=[[space.size], [1]])
        from quchip.chip.states import _bare_local_kets

        vector = xp.ones(space.size, dtype=complex)
        for axis, (key, ket) in enumerate(zip(space.labels, _bare_local_kets(chip, prepared, space.bases))):
            local = xp.asarray(backend.to_array(ket)).reshape(-1)
            transform = space.bases[key].energy_to_solver()
            if transform is not None:
                local = xp.conj(xp.asarray(transform).T) @ local
            vector = vector * local[space.levels[:, axis]]
        return backend.from_array(vector.reshape(-1, 1), dims=[[space.size], [1]])
    if isinstance(spec, CompactState):
        values = _compatible_values(spec, space.labels, space.dims, space.bases, backend)
        indices = np.asarray([space.rows[tuple(label)] for label in spec.space.levels])
        transform = xp.asarray(np.eye(space.size)[:, indices])
        values = transform @ values
        if spec.values.shape[-1] != 1:
            values = values @ xp.conj(transform.T)
        return backend.from_array(values, dims=[[space.size], [1] if values.shape[-1] == 1 else [space.size]])
    return space.project(materialize_state_spec(chip, spec, space.bases), backend)
