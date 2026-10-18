"""State preparation in the solver basis.

Product states from energy labels or authored local kets, authored states of
the full chip, the default initial state of a solve, and reusable states in
captured excitation coordinates. The chip's state queries and the solve
problems call these functions.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from quchip.backend.protocol import State
from quchip.engine.pulse_space import PulseSpace
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import array_namespace, maybe_concrete_scalar

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


def bare_state_from_bases(chip: "Chip", resolved: Mapping[str, Any], bases: Mapping[str, Any]) -> State:
    """Build a product state from the shared local state preparation."""
    kets = bare_local_kets(chip, resolved, bases)
    return kets[0] if len(kets) == 1 else chip.backend.tensor_states(*kets)


def bare_local_kets(
    chip: "Chip",
    resolved: Mapping[str, Any],
    bases: Mapping[str, Any],
) -> list[State]:
    """Resolve each local ket without allocating its tensor product."""
    backend = chip.backend
    available = list(chip._device_map.keys())
    prepared = dict(resolved)

    for label in resolved:
        if label not in chip._device_map:
            raise ValueError(f"Unknown device label '{label}'. Available labels: {available}")

    for label, val in resolved.items():
        if isinstance(val, bool):
            raise ValueError(f"Level index for '{label}' must be an integer, got {type(val).__name__}: {val!r}")
        basis = bases[label]
        if isinstance(val, int):
            if val < 0:
                raise ValueError(f"Level index for '{label}' must be >= 0, got {val}")
            if val >= basis.resolved_dim:
                raise ValueError(
                    f"Level index {val} for '{label}' exceeds the resolved "
                    f"dimension {basis.resolved_dim}."
                )
        else:
            from quchip.declarative.expr import (
                as_state_expr,
                materialize_expr,
            )

            device = chip[label]
            expression = as_state_expr(
                val,
                labels=(device.label,),
                dims=(basis.native_dim,),
                name=rf"\lvert\psi_{{{device.label}}}\rangle",
                owner=device,
                scope=device.label,
            )
            val = materialize_expr(expression, backend)
            prepared[label] = val
            shape = getattr(val, "shape", None)
            is_array_ket = shape is not None and (
                len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)
            )
            if not is_array_ket and not backend.is_ket(val):
                raise ValueError(f"State for '{label}' must be a ket vector, got a non-ket state")
            if shape is not None and shape[0] != basis.native_dim:
                raise ValueError(
                    f"Authored state dimension for '{label}' is {val.shape[0]}, "
                    f"expected {basis.native_dim}."
                )

    kets: list[State] = []
    for dev in chip.devices:
        val = prepared.get(dev.label)
        basis = bases[dev.label]
        if val is None:
            level = 0
        elif isinstance(val, int):
            level = val
        else:
            authored = backend.to_array(val).reshape(basis.native_dim, -1)
            projected = basis.vectors.conj().T @ authored
            authored_norm = backend.array_module.linalg.norm(authored)
            projected_norm = backend.array_module.linalg.norm(projected)
            lost = maybe_concrete_scalar(authored_norm**2 - projected_norm**2)
            if lost is not None and lost > 1e-10:
                warnings.warn(
                    f"Projection discarded {lost:.3g} of the state norm on {dev.label!r}.",
                    stacklevel=2,
                )
            kets.append(
                backend.from_array(
                    projected,
                    dims=[[basis.resolved_dim], [1]],
                )
            )
            continue

        vector = basis.energy_state(level).reshape(basis.resolved_dim, 1)
        kets.append(backend.from_array(vector, dims=[[basis.resolved_dim], [1]]))

    return kets


def default_initial_state(chip: "Chip", engine_result: Any, start_time: Any) -> State:
    """Return the default initial state for a solve.

    Use the eigenstate assigned to the all-ground label, taken from the
    undriven static lab-frame Hamiltonian that ``engine_result.approximation``
    keeps. Make its overlap with the bare product real and nonnegative, then
    express it in the solve frame at ``start_time``. If the bare product is
    already an eigenstate, return the bare product directly.
    """
    from quchip.engine.assembly import _may_raise_ground

    approximation = engine_result.approximation
    moves = engine_result.slh.has_network_hamiltonian or _may_raise_ground(chip, approximation, engine_result.bases)
    ground = chip.analysis._ground_ket(approximation) if moves else None
    if ground is None:
        return bare_state_from_bases(chip, {}, engine_result.bases)
    ket = _to_solve_frame(ground, chip, engine_result, start_time)
    dims = list(engine_result.dims)
    return chip.backend.from_array(ket.reshape(-1, 1), dims=[dims, [1] * len(dims)])


def _to_solve_frame(ket: Any, chip: "Chip", engine_result: Any, start_time: Any) -> Any:
    """Express a lab-frame ket in the solve's frame at ``start_time``.

    The solver evolves ``U(t)† psi`` with ``U(t) = exp(-i 2π t Σ_i f_i n_i)``
    in absolute time, where ``n_i`` is each device's energy-level index.
    """
    xp = chip.backend.array_module
    frequencies = engine_result.resolved_frame.frequencies
    tensor = ket.reshape(tuple(engine_result.dims))
    for axis, device in enumerate(chip.devices):
        angle = TWO_PI * start_time * frequencies.get(device.label, 0.0)
        if maybe_concrete_scalar(angle) == 0.0:
            continue
        record = engine_result.bases[device.label]
        phases = xp.exp(1j * xp.asarray(angle) * xp.arange(record.resolved_dim))
        transform = record.energy_to_solver()
        if transform is None:
            local = xp.diag(phases)
        else:
            transform = xp.asarray(transform)
            local = (transform * phases) @ xp.conj(transform).T
        tensor = xp.moveaxis(xp.tensordot(local, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def materialize_state_spec(
    chip: "Chip",
    state_spec: Any,
    bases: Mapping[str, Any],
) -> State:
    """Materialize one authored state specification in the engine's solver basis.

    An omitted state is :func:`default_initial_state`, which needs the solve's engine result.
    """
    if isinstance(state_spec, (Mapping, str)):
        from quchip.chip.states import normalize_device_state_mapping

        resolved = normalize_device_state_mapping(chip, state_spec, {})
        return bare_state_from_bases(chip, resolved, bases)

    from quchip.declarative.expr import as_state_expr, materialize_expr

    backend = chip.backend
    ordered = [bases[device.label] for device in chip.devices]
    authored_dims = tuple(record.native_dim for record in ordered)
    resolved_dims = tuple(record.resolved_dim for record in ordered)
    authored_dim = prod(authored_dims)
    resolved_dim = prod(resolved_dims)
    # QuTiP ``Qobj`` (``full``) and dynamiqs ``QArray`` (``to_jax``) states are in
    # solver coordinates whichever backend solves; plain arrays and symbolic
    # specifications are authored.
    resolved_native = (
        backend.is_native_state(state_spec) or hasattr(state_spec, "full") or hasattr(state_spec, "to_jax")
    )

    shape = getattr(state_spec, "shape", None)
    if shape is None:
        state_spec = as_state_expr(
            state_spec,
            labels=tuple(device.label for device in chip.devices),
            dims=authored_dims,
            name=r"\lvert\psi\rangle",
            owner=chip,
            scope="chip",
        )
    state = materialize_expr(state_spec, backend)
    shape = getattr(state, "shape", None)
    if shape is None:
        raise TypeError("Initial state must be a semantic specification, callable, or ket.")
    if resolved_native:
        if shape[0] != resolved_dim:
            raise ValueError(
                f"A solver-native initial state must have dimension {resolved_dim}, got {shape}."
            )
        return state
    if shape[0] != authored_dim or len(shape) not in (1, 2) or (len(shape) == 2 and shape[1] != 1):
        raise ValueError(
            "Initial-state dimension must match the authored or resolved chip space; "
            f"got {shape}, authored {authored_dim}, resolved {resolved_dim}."
        )

    authored = backend.to_array(state).reshape(authored_dim, 1)
    xp = array_namespace(ordered[0].vectors)
    transform = ordered[0].vectors
    for record in ordered[1:]:
        transform = xp.kron(transform, record.vectors)
    projected = transform.conj().T @ authored
    authored_norm = xp.linalg.norm(authored)
    projected_norm = xp.linalg.norm(projected)
    lost = maybe_concrete_scalar(authored_norm**2 - projected_norm**2)
    if lost is not None and lost > 1e-10:
        warnings.warn(
            f"Projection discarded {lost:.3g} of the initial-state norm.",
            stacklevel=2,
        )
    return backend.from_array(projected, dims=[list(resolved_dims), [1]])


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
    from quchip.engine.excitation import conserves_excitation_number
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


def compatible_values(state, labels, dims, bases, backend):
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
    from quchip.chip.states import normalize_device_state_mapping

    backend = chip.backend
    xp = backend.array_module
    if spec is None or isinstance(spec, (Mapping, str)):
        prepared = normalize_device_state_mapping(chip, spec, {})
        label = tuple(prepared.get(key, 0) for key in space.labels)
        if all(isinstance(n, int) for n in label):
            vector = xp.asarray(np.arange(space.size) == space.rows[label], dtype=complex).reshape(-1, 1)
            return backend.from_array(vector, dims=[[space.size], [1]])
        vector = xp.ones(space.size, dtype=complex)
        for axis, (key, ket) in enumerate(zip(space.labels, bare_local_kets(chip, prepared, space.bases))):
            local = xp.asarray(backend.to_array(ket)).reshape(-1)
            transform = space.bases[key].energy_to_solver()
            if transform is not None:
                local = xp.conj(xp.asarray(transform).T) @ local
            vector = vector * local[space.levels[:, axis]]
        return backend.from_array(vector.reshape(-1, 1), dims=[[space.size], [1]])
    if isinstance(spec, CompactState):
        values = compatible_values(spec, space.labels, space.dims, space.bases, backend)
        indices = np.asarray([space.rows[tuple(label)] for label in spec.space.levels])
        transform = xp.asarray(np.eye(space.size)[:, indices])
        values = transform @ values
        if spec.values.shape[-1] != 1:
            values = values @ xp.conj(transform.T)
        return backend.from_array(values, dims=[[space.size], [1] if values.shape[-1] == 1 else [space.size]])
    return space.project(materialize_state_spec(chip, spec, space.bases), backend)
