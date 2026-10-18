"""State preparation in the solver basis.

Product states from energy labels or authored local kets, authored states of
the full chip, the dressed initial state of a label, and reusable states in
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


def label_levels(chip: "Chip", spec: Any, dims: Any, *, role: str = "An initial label") -> tuple[int, ...]:
    """Return the product label of a level mapping or state shorthand, in device order.

    Devices without a level are in level 0. A mapping that holds a local ket
    is not a label, and ``chip.bare_state`` takes it.

    Parameters
    ----------
    chip : Chip
        Chip whose devices and state order read the label.
    spec : mapping or str
        Energy level of each named device, or a state shorthand.
    dims : sequence of int
        Level count of each device.
    role : str, optional
        Name of the label in error messages.
    """
    from quchip.chip.states import normalize_device_state_mapping

    labels = tuple(device.label for device in chip.devices)
    prepared = normalize_device_state_mapping(chip, spec, {})
    if set(prepared) - set(labels):
        raise ValueError(f"{role} names a device that the chip does not have: {sorted(set(prepared) - set(labels))}.")
    levels = tuple(prepared.get(label, 0) for label in labels)
    if any(not isinstance(n, int) or isinstance(n, bool) for n in levels):
        raise TypeError(
            f"{role} takes integer energy levels. Use chip.bare_state(...) for a product of local kets. "
            "On a large chip, chip.compact_state(..., dressed=False) keeps compact coordinates."
        )
    if any(not 0 <= n < d for n, d in zip(levels, dims)):
        raise ValueError(f"{role} needs integer energy levels below {tuple(dims)}; got {levels}.")
    return levels


def label_state(chip: "Chip", spec: Any, result: Any, start: Any) -> State:
    """Return the dressed state of a label at the start of a solve, in the solve's coordinates and frame.

    The state is the eigenstate ``|d_k⟩`` assigned to label k of the undriven
    static lab-frame model that ``result.approximation`` keeps, with real,
    nonnegative overlap on the bare product ``|k⟩``, as ``chip.state()``
    assigns it. Under an excitation cutoff the model is the truncated ``P H P``.
    The frame map places the state at ``start`` as the label target of
    ``overlap()``: ``U(start)†|d_k⟩ exp(−i 2π start Σᵢ fᵢ kᵢ)``. The labels of
    an uncoupled chip start in ``|k⟩`` in every frame.

    Parameters
    ----------
    chip : Chip
        Chip whose static model defines the state.
    spec : mapping, str, or None
        Level label. None is the all-ground label.
    result : EngineResult
        Solve contract with the frame, approximation, bases, and excitation space.
    start : scalar
        Start time of the solve, in ns.

    Returns
    -------
    state
        Native backend ket in the coordinates of the solve.
    """
    from quchip.engine.assembly import _may_raise_ground
    from quchip.engine.frames import FrameMap
    from quchip.utils.constants import TWO_PI

    space = result.excitation_space
    # A compact solve has one coordinate axis, and its space keeps the device dimensions.
    dims = result.dims if space is None else space.dims
    levels = (0,) * len(chip.devices) if spec is None else label_levels(chip, spec, dims)
    targets = LabelTargets(chip, result.approximation)
    vector = None
    # A ground that no retained term can move is the bare vacuum, without an eigensystem. Every other label
    # takes its assigned eigenvector, because a residual test cannot prove an eigenstate near a degeneracy.
    if any(levels) or space is not None or result.slh.has_network_hamiltonian or _may_raise_ground(
            chip, result.approximation, result.bases):
        vector = targets.dressed(levels, result.bases, space)
    if vector is None:
        vector = targets.bare(levels, result.bases, space)
    else:
        labels = tuple(device.label for device in chip.devices)
        bases = result.bases if space is None else space.bases
        xp = array_namespace(vector)
        frame_map = FrameMap.create(result.resolved_frame, labels, bases, xp, space=space)
        vector = frame_map.kets(vector, start) * xp.exp(-1j * TWO_PI * start * frame_map.label_energy(levels))
    if space is not None:
        return chip.backend.from_array(vector.reshape(-1, 1), dims=[[space.size], [1]])
    dims = list(result.dims)
    return chip.backend.from_array(vector.reshape(-1, 1), dims=[dims, [1] * len(dims)])


def materialize_state_spec(
    chip: "Chip",
    state_spec: Any,
    bases: Mapping[str, Any],
) -> State:
    """Materialize one authored state specification in the engine's solver basis.

    Labels and an omitted state are dressed states, which :func:`label_state` prepares from the solve's engine result.
    """
    if isinstance(state_spec, (Mapping, str)):
        raise TypeError("A level label is a dressed state of a solve. Use chip.bare_state(...) for a product state.")

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
        from quchip.engine.excitation import sector_levels

        dims, labels, totals, records = metadata
        bases = {
            label: BasisRecord(kind, arrays[0], arrays[1], arrays[2], native, resolved)
            for label, (kind, native, resolved), arrays in zip(labels, records, leaves[1:])
        }
        levels = np.concatenate([sector_levels(dims, total) for total in totals])
        return cls(leaves[0], PulseSpace(dims, labels, totals, levels, bases))


def compact_state(chip, *components, dressed=True):
    """Prepare a normalized state on the excitation totals of its components.

    Parameters
    ----------
    chip : Chip
        Chip whose resolved energy bases define the coordinates.
    *components : mapping, str, or (amplitude, specification) pairs
        Energy-level labels, as in ``Chip.compact_state``. With
        ``dressed=False``, a mapping can also give a device a local ket.
    dressed : bool, default=True
        Use the static dressed eigenstate assigned to each label. False uses
        the bare product of each label.

    Returns
    -------
    CompactState
        Captured state accepted as a pulse simulation's initial_state.
    """
    from quchip.chip.states import normalize_device_state_mapping
    from quchip.cutoff import active_cutoff
    from quchip.engine.assembly import _resolve_system
    from quchip.engine.dressing import warn_weak_assignment
    from quchip.engine.excitation import excitation_structure
    from quchip.engine.sectors import SectorModel
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
        if not dressed and not all(isinstance(n, int) and not isinstance(n, bool) for n in prepared.values()):
            entries.append((amplitude, *_bare_factors(chip, prepared, resolution, policy)))
            continue
        levels = tuple(prepared.get(label, 0) for label in labels)
        if any(not isinstance(n, int) or isinstance(n, bool) for n in levels):
            raise TypeError("A dressed compact state takes integer energy levels. Use dressed=False for local kets.")
        if any(not 0 <= n < d for n, d in zip(levels, resolution.dims)):
            raise ValueError("Compact state specifications require valid integer energy levels.")
        if policy is not None and sum(levels) > policy.max_total:
            raise ValueError(
                f"Compact state label {levels} has total excitation {sum(levels)}, outside the excitation "
                f"cutoff N <= {policy.max_total}."
            )
        entries.append((amplitude, levels, {sum(levels)}))
    if dressed and policy is None and not excitation_structure(
            chip, chip.approximation, resolution=resolution).conserves:
        raise ValueError(
            "Compact dressed preparation requires proven static excitation conservation or an excitation cutoff. "
            "Use chip.compact_state(..., dressed=False) for bare products."
        )
    model = None
    totals = set().union(*(occupied for *_, occupied in entries))
    if dressed:
        # Under a cutoff, a dressed state spans the totals of its class.
        model = chip._analysis._sector_model() if policy is not None else SectorModel(chip, resolution=resolution)
        totals = {total for n in totals for total in model.class_of(n)}
    space = PulseSpace.create(resolution.dims, labels, totals, resolution.bases)
    xp = chip.backend.array_module
    vector = xp.zeros(space.size, dtype=complex)
    for amplitude, levels, _ in entries:
        if model is None and all(isinstance(n, int) for n in levels):
            contribution = xp.asarray(np.arange(space.size) == space.rows[levels], dtype=complex)
        elif model is None:
            # A product of local kets, read on the retained labels.
            contribution = xp.ones(space.size, dtype=complex)
            for axis, local in enumerate(levels):
                contribution = contribution * local[space.levels[:, axis]]
        else:
            system = model.eigensystem(sum(levels))
            row = system.space.row(levels)
            warn_weak_assignment(levels, system.labeling.overlaps[row])
            eigenvector = xp.asarray(system.eigenvectors)[:, system.labeling.indices[row]]
            indices = np.asarray([space.rows[label] for label in system.space.labels])
            contribution = xp.asarray(np.eye(space.size)[:, indices]) @ eigenvector
        vector = vector + amplitude * contribution
    norm = xp.linalg.norm(vector)
    if maybe_concrete_scalar(norm) == 0:
        raise ValueError("Compact state superposition has zero norm.")
    return CompactState((vector / xp.where(norm > 0, norm, 1)).reshape(-1, 1), space)


def _bare_factors(chip, prepared, resolution, policy):
    """Return the energy-basis factors of a bare product and the totals that it occupies.

    Under a cutoff, a product with more than round-off weight above it
    raises, and only the totals up to the cutoff are kept.
    """
    from quchip.engine.pulse_space import _require_inside
    from quchip.utils.jax_utils import contains_tracer

    xp = chip.backend.array_module
    factors, totals, weights = [], {0}, np.ones(1)
    for device, ket, dim in zip(chip.devices, bare_local_kets(chip, prepared, resolution.bases), resolution.dims):
        level = prepared.get(device.label, 0)
        if isinstance(level, int):
            # A level is an exact unit vector, so a basis conversion or a trace adds no support.
            weight = (np.arange(dim) == level).astype(float)
            local, support = xp.asarray(weight, dtype=complex), (level,)
        else:
            local = xp.asarray(chip.backend.to_array(ket)).reshape(-1)
            transform = resolution.bases[device.label].energy_to_solver()
            if transform is not None:
                local = xp.conj(xp.asarray(transform).T) @ local
            # A traced ket can occupy every level.
            weight = None if contains_tracer(local) else np.abs(np.asarray(local)) ** 2
            support = range(dim) if weight is None else np.flatnonzero(weight != 0)
        factors.append(local)
        totals = {n + int(index) for n in totals for index in support}
        weights = None if weight is None or weights is None else np.convolve(weights, weight)
    if policy is not None:
        if weights is None and max(totals) > policy.max_total:
            raise ValueError(
                f"A traced local ket can occupy totals above the excitation cutoff N <= {policy.max_total}, so "
                "quchip cannot prove that the product lies inside it. Pass integer energy levels."
            )
        if weights is not None:
            outside = float(np.sum(weights[policy.max_total + 1:]))
            _require_inside(policy.max_total, outside, float(np.sum(weights)), "bare product")
        totals = {n for n in totals if n <= policy.max_total}
    return tuple(factors), totals


def block_model(chip, approximation):
    """Return the static block model of ``approximation``, or None when dressing needs the full eigensystem.

    Under an excitation cutoff, every chip has a block model. Without one, the
    static model must conserve the total energy-level index.

    Parameters
    ----------
    chip : Chip
        Chip whose static model is split.
    approximation : Approximation
        Approximation of the static model.
    """
    from quchip.cutoff import active_cutoff
    from quchip.engine.excitation import excitation_structure
    from quchip.engine.sectors import SectorModel

    if approximation == chip.approximation:
        return chip._analysis._sector_model(_advise=False)
    if active_cutoff(chip) is None and not excitation_structure(chip, approximation).conserves:
        return None
    return SectorModel(chip, approximation=approximation, advisories=[])


def sector_label_ket(chip, model, space, levels):
    """Return the dressed eigenstate assigned to a label in the rows of an excitation space.

    The eigenvector comes from the block of ``model`` that holds the label.
    Rows of the block outside ``space`` are dropped, because the states of
    the solve have no amplitude there.

    Parameters
    ----------
    chip : Chip
        Chip whose backend sets the array module.
    model : SectorModel
        Static block model of the solve's approximation.
    space : PulseSpace
        Excitation space of the solve.
    levels : tuple of int
        Product label of local energy levels.
    """
    import jax.numpy as jnp

    from quchip.engine.basis import _differentiable_eigenvector
    from quchip.engine.dressing import gauge_phases, warn_weak_assignment
    from quchip.utils.jax_utils import contains_tracer

    system = model.eigensystem(sum(levels))
    row = system.space.row(tuple(levels))
    index = system.labeling.indices[row]
    warn_weak_assignment(levels, system.labeling.overlaps[row])
    traced = contains_tracer((system.eigenvalues, system.eigenvectors, index))
    xp = jnp if traced else chip.backend.array_module
    column = xp.asarray(system.eigenvectors)[:, index]
    if contains_tracer(system.eigenvectors):
        # JAX's eigh derivative can be NaN at an unrelated exact tie.
        matrix = jnp.asarray(model.hamiltonian(system.space.totals))
        column = _differentiable_eigenvector(matrix, system.eigenvalues[index], system.eigenvalues,
                                             system.eigenvectors, column)
        # That derivative transports the phase. The gauge restores the derivative of the anchor's phase.
        # Rolling the rows puts the label's own amplitude first, where the gauge anchors.
        column = column * gauge_phases(jnp.roll(column[:, None], -row, axis=0), jnp.zeros(1, dtype=int))[0]
    kept = np.asarray([i for i, label in enumerate(system.space.labels) if label in space.rows], dtype=int)
    indices = np.asarray([space.rows[system.space.labels[i]] for i in kept], dtype=int)
    vector = xp.zeros(space.size, dtype=complex)
    return vector.at[indices].set(column[kept]) if xp is jnp else _scatter(vector, indices, column[kept])


@dataclass(frozen=True, eq=False)
class LabelTargets:
    """Prepare the label targets of one solve from a snapshot of its chip.

    :meth:`capture` takes an isolated clone when the solve finishes, so later
    changes to the source chip do not change the targets. The clone runs its
    dressed analysis only when a dressed target is requested.

    Parameters
    ----------
    chip : Chip
        Isolated clone of the solve's chip.
    approximation : Approximation
        Approximation of the solve's static model.
    """

    chip: Any
    approximation: Any

    @classmethod
    def capture(cls, chip, approximation):
        """Record an isolated clone of a chip.

        Parameters
        ----------
        chip : Chip
            Source chip of the solve.
        approximation : Approximation
            Approximation of the solve's static model.
        """
        return cls(chip.clone(), approximation)

    def levels(self, spec, dims):
        """Return the product label of a mapping or state shorthand, in device order.

        Parameters
        ----------
        spec : mapping or str
            Local energy level of each named device. Devices without a level are in level 0.
        dims : sequence of int
            Level count of each device.
        """
        return label_levels(self.chip, spec, dims, role="A label target")

    def bare(self, levels, bases, space=None):
        """Return the bare product state of a label, in solver coordinates or in the rows of ``space``.

        Parameters
        ----------
        levels : tuple of int
            Product label of local energy levels.
        bases : mapping
            Resolved basis record of each device.
        space : PulseSpace or None, optional
            Excitation space of the solve. A label outside it gives a zero vector.
        """
        backend = self.chip.backend
        if space is not None:
            return backend.array_module.asarray(np.arange(space.size) == space.rows.get(levels, -1), dtype=complex)
        labels = tuple(device.label for device in self.chip.devices)
        return backend.to_array(bare_state_from_bases(self.chip, dict(zip(labels, levels)), bases)).reshape(-1)

    def dressed(self, levels, bases, space=None):
        """Return the lab-frame dressed eigenstate assigned to a label, as :meth:`bare` places it.

        The static model is the one that the solve's approximation keeps.
        None means the all-ground label of a conserving model, whose bare
        product is the only state with total level 0.

        Parameters
        ----------
        levels : tuple of int
            Product label of local energy levels.
        bases : mapping
            Resolved basis record of each device.
        space : PulseSpace or None, optional
            Excitation space of the solve.
        """
        backend = self.chip.backend
        if space is not None:
            model = block_model(self.chip, self.approximation)
            if model is not None:
                return sector_label_ket(self.chip, model, space, levels)
        ket = self.chip.analysis._label_ket(levels, self.approximation)
        if ket is None:
            return None
        if space is not None:
            dims = list(space.dims)
            native = backend.from_array(ket.reshape(-1, 1), dims=[dims, [1] * len(dims)])
            ket = backend.to_array(space.project(native, backend)).reshape(-1)
        return ket


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
    spec : state_like or CompactState
        Initial state. :func:`label_state` prepares labels and an omitted state.
    space : PulseSpace
        Exact excitation space of the solve.

    Returns
    -------
    state
        Native backend ket or density matrix in ``space`` coordinates.
    """
    backend = chip.backend
    xp = backend.array_module
    if isinstance(spec, CompactState):
        values = compatible_values(spec, space.labels, space.dims, space.bases, backend)
        indices = np.asarray([space.rows[tuple(label)] for label in spec.space.levels])
        transform = xp.asarray(np.eye(space.size)[:, indices])
        values = transform @ values
        if spec.values.shape[-1] != 1:
            values = values @ xp.conj(transform.T)
        return backend.from_array(values, dims=[[space.size], [1] if values.shape[-1] == 1 else [space.size]])
    return space.project(materialize_state_spec(chip, spec, space.bases), backend)
