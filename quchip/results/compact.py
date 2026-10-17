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
    """Prepare a normalized state on the excitation totals of its components.

    Parameters
    ----------
    chip : Chip
        Chip whose resolved energy bases define the coordinates.
    *components : mapping, str, or (amplitude, specification) pairs
        Bare product energy labels, as in ``Chip.compact_state``.
    dressed : bool, default=False
        Use the static dressed eigenstate assigned to each label.

    Returns
    -------
    CompactState
        Captured state accepted as a pulse simulation's initial_state.
    """
    from quchip.chip.states import normalize_device_state_mapping
    from quchip.cutoff import active_cutoff
    from quchip.engine.assembly import _resolve_system
    from quchip.engine.sectors import SectorModel
    from quchip.chip.effective import conserves_excitation_number
    from quchip.utils.jax_utils import maybe_concrete_scalar

    resolution = _resolve_system(chip, chip.backend)
    labels = tuple(d.label for d in chip.devices)
    policy = active_cutoff(chip)
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
        if policy is not None and sum(levels) > policy.max_total:
            raise ValueError(
                f"Compact state label {levels} has total excitation {sum(levels)}, outside the excitation "
                f"cutoff N <= {policy.max_total}."
            )
        entries.append((amplitude, levels))
    if dressed and policy is None and not conserves_excitation_number(chip, chip.approximation, resolution=resolution):
        raise ValueError("Compact dressed preparation requires proven static excitation conservation.")
    model = None
    totals = {sum(levels) for _, levels in entries}
    if dressed:
        # Under a cutoff, a dressed state spans the totals of its class.
        model = chip._analysis._sector_model() if policy is not None else SectorModel(chip, resolution=resolution)
        totals = {total for n in totals for total in model.class_of(n)}
    space = PulseSpace.create(resolution.dims, labels, totals, resolution.bases)
    xp = chip.backend.array_module
    vector = xp.zeros(space.size, dtype=complex)
    for amplitude, levels in entries:
        if model is None:
            contribution = xp.asarray(np.arange(space.size) == space.rows[levels], dtype=complex)
        else:
            system = model.eigensystem(sum(levels))
            row = system.space.row(levels)
            eigenvector = xp.asarray(system.eigenvectors)[:, system.labeling.indices[row]]
            indices = np.asarray([space.rows[label] for label in system.space.labels])
            contribution = xp.asarray(np.eye(space.size)[:, indices]) @ eigenvector
        vector = vector + amplitude * contribution
    norm = xp.linalg.norm(vector)
    if maybe_concrete_scalar(norm) == 0:
        raise ValueError("Compact state superposition has zero norm.")
    return CompactState((vector / xp.where(norm > 0, norm, 1)).reshape(-1, 1), space)


def truncated_vacuum(chip, result, start):
    """Prepare the default state of a solve under an excitation cutoff.

    The state is the eigenvector of the truncated static lab-frame model
    ``P H P`` that is assigned to the bare vacuum, under the solve's
    approximation. Its overlap with the bare vacuum is real and nonnegative.
    Each compact basis state ``|n>`` then takes the phase
    ``exp(i 2π t0 Σ_i f_i n_i)`` of the solve frame at ``start``.

    Parameters
    ----------
    chip : Chip
        Chip whose truncated static model defines the state.
    result : EngineResult
        Compact solve contract with the frame, approximation, and space.
    start : scalar
        Start time of the solve, in ns.

    Returns
    -------
    state
        Native backend ket in the coordinates of ``result.excitation_space``.
    """
    import jax.numpy as jnp

    from quchip.engine.basis import _differentiable_eigenvector
    from quchip.engine.sectors import SectorModel
    from quchip.utils.constants import TWO_PI
    from quchip.utils.jax_utils import contains_tracer

    backend, space = chip.backend, result.excitation_space
    model = (chip._analysis._sector_model(_advise=False) if result.approximation == chip.approximation
             else SectorModel(chip, approximation=result.approximation, advisories=[]))
    system = model.eigensystem(0)
    row = system.space.row((0,) * len(space.labels))
    index = system.labeling.indices[row]
    traced = contains_tracer((system.eigenvalues, system.eigenvectors, index))
    xp = jnp if traced else backend.array_module
    column = xp.asarray(system.eigenvectors)[:, index]
    if contains_tracer(system.eigenvectors):
        # JAX's eigh derivative can be NaN at an unrelated exact tie.
        matrix = jnp.asarray(model.hamiltonian(system.space.totals))
        column = _differentiable_eigenvector(matrix, system.eigenvalues[index], system.eigenvalues,
                                             system.eigenvectors, column)
    frequencies = result.resolved_frame.frequencies
    angle = sum(xp.asarray(frequencies.get(label, 0.0)) * system.space.levels[:, axis]
                for axis, label in enumerate(space.labels))
    column = column * xp.exp(1j * TWO_PI * xp.asarray(start) * angle)
    indices = np.asarray([space.rows[label] for label in system.space.labels])
    vector = xp.zeros(space.size, dtype=complex)
    vector = vector.at[indices].set(column) if xp is jnp else _scatter(vector, indices, column)
    return backend.from_array(vector.reshape(-1, 1), dims=[[space.size], [1]])


def _scatter(vector, indices, values):
    """Return ``vector`` with ``values`` written at ``indices``."""
    vector[indices] = values
    return vector


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
            "Compact state energy bases changed. Prepare the state again for this chip.",
        )
    return values


def materialize_compact(chip, spec, space):
    """Prepare an initial state in the coordinates of an exact pulse space.

    Parameters
    ----------
    chip : Chip
        Chip whose resolved energy bases define the state.
    spec : state_like, mapping, str, CompactState, or None
        Initial-state specification. ``None`` prepares the bare vacuum.
    space : PulseSpace
        Exact excitation space of the solve.

    Returns
    -------
    state
        Native backend ket or density matrix in ``space`` coordinates.
    """
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
