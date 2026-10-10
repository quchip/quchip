"""Captured effective Hamiltonian and Lindblad terms on a retained product space."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.expr import PhysicsExpr, declared_excitation_changes, materialize_expr
from quchip.utils.jax_utils import concrete_array_module, contains_tracer
from quchip.utils.values import copy_value, value_fingerprint


def authored_excitation_changes(
    operator: Any,
    labels: tuple[str, ...],
    backend: Any,
    bases: Mapping[str, Any] | None,
) -> frozenset[int] | None:
    """Return the total energy-level changes an authored operator carries.

    A declared matrix contribution returns its declaration. Otherwise the
    operator is evaluated at compile time in the captured energy bases of
    ``labels``, so a constant operator stays concrete inside ``jax.jit``.
    Missing bases, traced bases, or parameter-dependent operators return
    ``None``.

    Parameters
    ----------
    operator : PhysicsExpr or backend operator
        Operator on ``labels`` in their authored coordinates.
    labels : tuple[str, ...]
        Ordered device labels supporting ``operator``.
    backend : Backend
        Backend used to materialize ``operator``.
    bases : mapping or None
        Captured basis records keyed by device label.
    """
    from quchip.engine.bands import concrete_excitation_changes

    declared = declared_excitation_changes(operator)
    if declared is not None:
        return declared
    if bases is None or any(label not in bases for label in labels):
        return None
    records = [bases[label] for label in labels]
    with jax.ensure_compile_time_eval(), backend.eager_operators():
        matrix = backend.to_array(materialize_expr(operator, backend, local_bases=bases))
        vectors = records[0].energy_vectors
        for record in records[1:]:
            vectors = jnp.kron(vectors, record.energy_vectors)
        dims = tuple(int(record.energy_vectors.shape[1]) for record in records)
        return concrete_excitation_changes(matrix, dims, vectors)


def conserves_excitation_number(chip: Any, approximation: Any) -> bool:
    """Return whether a chip's static model structurally conserves the total energy-level index.

    Device Hamiltonians are diagonal in their energy bases. The approximation
    must keep only bands of zero total weight, every retained term must declare
    conservation, and no port pair can generate a cascade Hamiltonian.

    Parameters
    ----------
    chip : Chip
        Chip whose static model is checked.
    approximation : Approximation
        Approximation the static model is resolved with.
    """
    if not approximation.conserves_excitation_number():
        return False
    if any(terms.excitation_changes is None for terms in chip.effective_terms):
        return False
    return chip.port_network is None or not chip.port_network._active_generated_pairs()


@dataclass(frozen=True, eq=False)
class OperatorProjection:
    """Captured authored coordinates for operators of surviving components.

    The rectangular embedding maps target authored coordinates into the source
    product space. ``parents`` are earlier maps in the order they apply. An
    operator passes through each map whose source devices it acts on, and its
    other devices keep their coordinates. A chain of local reductions therefore
    never forms a full-space embedding. Rates and local operators remain owned
    by the components. This record carries only their change of coordinates.
    """

    source_labels: tuple[str, ...]
    source_dims: tuple[int, ...]
    target_labels: tuple[str, ...]
    target_dims: tuple[int, ...]
    embedding: Any
    overrides: tuple[tuple[str, OperatorProjection], ...] = ()
    parents: tuple[OperatorProjection, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_labels", "source_dims", "target_labels", "target_dims"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for labels, dims in ((self.source_labels, self.source_dims), (self.target_labels, self.target_dims)):
            if len(labels) != len(dims) or len(set(labels)) != len(labels) or not labels:
                raise ValueError("OperatorProjection needs unique labels and matching dimensions.")
            if any(not isinstance(label, str) or not label for label in labels):
                raise ValueError("OperatorProjection labels must be nonempty strings.")
            if any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in dims):
                raise ValueError("OperatorProjection dimensions must be positive integers.")
        if not set(self.target_labels) <= set(self.source_labels):
            raise ValueError("Projected devices must belong to the source space.")
        matrix = copy_value(self.embedding, readonly=True)
        if getattr(matrix, "shape", None) != (prod(self.source_dims), prod(self.target_dims)):
            raise ValueError("OperatorProjection embedding has incompatible dimensions.")
        if not contains_tracer(matrix) and not np.all(np.isfinite(matrix)):
            raise ValueError("OperatorProjection embedding must be finite.")
        object.__setattr__(self, "embedding", matrix)
        overrides = tuple(self.overrides)
        if len({key for key, _ in overrides}) != len(overrides):
            raise ValueError("An operator owner cannot have two coordinate overrides.")
        if any(p.target_labels != self.target_labels or p.target_dims != self.target_dims for _, p in overrides):
            raise ValueError("Operator coordinate overrides must use the same target space.")
        object.__setattr__(self, "overrides", overrides)
        parents = tuple(self.parents)
        if any(not isinstance(parent, OperatorProjection) for parent in parents):
            raise TypeError("OperatorProjection parents must be OperatorProjection values.")
        object.__setattr__(self, "parents", parents)

    @classmethod
    def capture(cls, chip: Any, target_labels: tuple[str, ...], target_dims: tuple[int, ...],
                embedding: Any, *, compose: bool = True) -> OperatorProjection:
        """Compose a reduction with the source's already captured operator coordinates.

        ``embedding`` maps the targets into the product space of all of
        ``chip``'s devices. With ``compose``, one earlier map over all of them
        is multiplied in. Otherwise the earlier maps become parents, in the
        order they apply, and keep their own embeddings.
        """
        labels, dims = tuple(d.label for d in chip.devices), tuple(chip.authored_dims)
        previous = tuple(t.projection for t in chip.effective_terms if t.projection is not None)
        if not (compose and len(previous) == 1 and previous[0].target_labels == labels):
            return cls(labels, dims, target_labels, target_dims, embedding, parents=lineage(previous))
        parent = previous[0]

        def advance(previous: OperatorProjection) -> OperatorProjection:
            return replace(previous, target_labels=target_labels, target_dims=target_dims,
                           embedding=previous.embedding @ embedding, overrides=())

        return replace(advance(parent), overrides=tuple((key, advance(p)) for key, p in parent.overrides))

    def with_current_operators(self, owner_keys: tuple[str, ...]) -> OperatorProjection:
        """Mark operators already authored in this retained space, preserving their lineage."""
        if not owner_keys:
            return self
        current = type(self)(self.target_labels, self.target_dims, self.target_labels,
                             self.target_dims, jnp.eye(prod(self.target_dims), dtype=complex))
        overrides = {**dict(self.overrides), **dict.fromkeys(owner_keys, current)}
        return replace(self, overrides=tuple(overrides.items()))

    def override_keys(self) -> frozenset[str]:
        """Return the owners whose operators a map in this chain already holds in retained coordinates."""
        return frozenset(key for key, _ in self.overrides).union(*(p.override_keys() for p in self.parents))

    def _steps(self) -> tuple[OperatorProjection, ...]:
        """Return the maps of this lineage in the order they apply. Each applies its own embedding only."""
        return (*(step for parent in self.parents for step in parent._steps()), self)

    def transport(self, operator: Any, labels: tuple[str, ...], dims: tuple[int, ...],
                  owner_key: str | None = None) -> tuple[Any, tuple[str, ...], tuple[int, ...]]:
        """Carry a dense operator into the target coordinates through every map of the lineage.

        ``labels`` and ``dims`` give the operator's tensor order in authored
        coordinates. Each map acts when the operator acts on one of its source
        devices. Other labels are spectators. They keep their coordinates and
        follow the map's targets in the returned order. An operator whose owner
        a map holds in retained coordinates starts at that map's override.
        """
        operator, labels, dims, _acted = _walk(self._steps(), operator, tuple(labels), tuple(dims), owner_key)
        return operator, labels, dims

    def _embed(self, operator: Any, labels: tuple[str, ...],
               dims: tuple[int, ...]) -> tuple[Any, tuple[str, ...], tuple[int, ...]]:
        """Apply this map's own embedding. Labels outside its source keep their coordinates."""
        inside = tuple(index for index, label in enumerate(labels) if label in self.source_labels)
        outside = tuple(index for index in range(len(labels)) if index not in inside)
        support = tuple(self.source_labels.index(labels[index]) for index in inside)
        rest = tuple(index for index in range(len(self.source_dims)) if index not in support)
        local_size = prod(self.source_dims[index] for index in support)
        rest_size = prod(self.source_dims[index] for index in rest)
        target_size = prod(self.target_dims)
        spectator_dims = tuple(dims[index] for index in outside)
        spectator_size = prod(spectator_dims)
        xp = concrete_array_module(operator, self.embedding)
        matrix = xp.asarray(operator)
        if (matrix.shape != (prod(dims), prod(dims))
                or any(dims[index] != self.source_dims[source] for index, source in zip(inside, support))):
            raise ValueError("Operator dimensions do not match its captured source support.")
        tensor = xp.asarray(self.embedding).reshape(self.source_dims + (target_size,))
        tensor = xp.transpose(tensor, support + rest + (len(self.source_dims),))
        tensor = tensor.reshape(local_size, rest_size, target_size)
        # Greedy pairwise contraction makes each step a matrix product. NumPy
        # otherwise limits intermediates to the largest operand and can fall
        # back to one unoptimized loop.
        path = ("greedy", 2**62) if xp is np else "greedy"
        if not outside:
            projected = xp.einsum("ari,ab,brj->ij", tensor.conj(), matrix, tensor, optimize=path)
            return projected, self.target_labels, self.target_dims
        # Spectators keep their coordinates: put them after the mapped devices.
        order = inside + outside
        matrix = xp.transpose(matrix.reshape(dims + dims), order + tuple(len(dims) + index for index in order))
        matrix = matrix.reshape(local_size, spectator_size, local_size, spectator_size)
        projected = xp.einsum("ari,asbt,brj->isjt", tensor.conj(), matrix, tensor, optimize=path)
        size = target_size * spectator_size
        return (projected.reshape(size, size), self.target_labels + tuple(labels[index] for index in outside),
                self.target_dims + spectator_dims)

    def apply(self, operator: Any, labels: tuple[str, ...], owner_key: str | None = None, *,
              excitation_changes: frozenset[int] | None = None, dims: tuple[int, ...] | None = None) -> PhysicsExpr:
        """Project a live local array without allocating its full-space identity embedding.

        ``excitation_changes`` declares the source operator's total energy-level
        changes. Pass it only when the captured map conserves the total level
        index, so that the projected operator carries the same changes.
        ``dims`` is needed only for spectator labels outside this map.
        """
        if dims is None:
            known = {label: dim for projection in self._steps()
                     for label, dim in zip(projection.source_labels, projection.source_dims)}
            if any(label not in known for label in labels):
                raise ValueError("Pass dims for operator labels outside the captured source space.")
            dims = tuple(known[label] for label in labels)
        matrix, labels, dims = self.transport(operator, tuple(labels), tuple(dims), owner_key)
        return PhysicsExpr.from_matrix(matrix, labels=labels, dims=dims,
                                       name="projected_operator", excitation_changes=excitation_changes)

    def fingerprint(self) -> Any:
        own = value_fingerprint((self.source_labels, self.source_dims, self.target_labels,
                                 self.target_dims, self.embedding,
                                 tuple((key, p.fingerprint()) for key, p in self.overrides)))
        # Parent keys are already fingerprints. Hashing them again would double
        # their nesting depth at every step of a chain.
        return own if not self.parents else (own, tuple(p.fingerprint() for p in self.parents))

    def to_dict(self) -> dict[str, Any]:
        matrix = np.asarray(self.embedding)
        data = dict(source_labels=list(self.source_labels), source_dims=list(self.source_dims),
                    target_labels=list(self.target_labels), target_dims=list(self.target_dims),
                    real=matrix.real.tolist(), imag=matrix.imag.tolist(),
                    overrides={key: p.to_dict() for key, p in self.overrides})
        if self.parents:
            data["parents"] = [p.to_dict() for p in self.parents]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorProjection:
        required = {"source_labels", "source_dims", "target_labels", "target_dims", "real", "imag", "overrides"}
        if not required <= set(data) <= required | {"parents"}:
            raise ValueError("Invalid serialized OperatorProjection fields.")
        return cls(tuple(data["source_labels"]), tuple(data["source_dims"]), tuple(data["target_labels"]),
                   tuple(data["target_dims"]), np.asarray(data["real"]) + 1j * np.asarray(data["imag"]),
                   tuple((key, cls.from_dict(p)) for key, p in data["overrides"].items()),
                   tuple(cls.from_dict(p) for p in data.get("parents", ())))


def lineage(projections: tuple[OperatorProjection, ...]) -> tuple[OperatorProjection, ...]:
    """Return the maps of ``projections`` and of their parents, each without parents, in the order they apply."""
    return tuple(replace(step, parents=()) if step.parents else step
                 for projection in projections for step in projection._steps())


def _entry(steps: tuple[OperatorProjection, ...],
           owner_key: str | None) -> tuple[int, OperatorProjection | None]:
    """Return where an owner's operator enters ``steps``, and the override it enters through.

    An owner whose operator a map already holds in retained coordinates skips
    that map and every earlier one. The latest such map decides.
    """
    start, override = 0, None
    if owner_key is not None:
        for index, step in enumerate(steps):
            for key, candidate in step.overrides:
                if key == owner_key:
                    start, override = index + 1, candidate
    return start, override


def _walk(steps: tuple[OperatorProjection, ...], operator: Any, labels: tuple[str, ...], dims: tuple[int, ...],
          owner_key: str | None) -> tuple[Any, tuple[str, ...], tuple[int, ...], bool]:
    """Apply each map that the operator meets, in order, and report whether any map acted."""
    start, override = _entry(steps, owner_key)
    acted = False
    for step in (() if override is None else override._steps()) + steps[start:]:
        if not set(step.source_labels).isdisjoint(labels):
            operator, labels, dims = step._embed(operator, labels, dims)
            acted = True
    return operator, labels, dims, acted


def _reach(steps: tuple[OperatorProjection, ...], labels: tuple[str, ...],
           owner_key: str | None) -> tuple[str, ...] | None:
    """Return the devices that an operator on ``labels`` acts on after ``steps``, or ``None`` if no map acts."""
    start, override = _entry(steps, owner_key)
    acted = False
    for step in (() if override is None else override._steps()) + steps[start:]:
        if not set(step.source_labels).isdisjoint(labels):
            labels = step.target_labels + tuple(label for label in labels if label not in step.source_labels)
            acted = True
    return labels if acted else None


def _in_device_order(chip: Any, operator: Any, labels: tuple[str, ...],
                     dims: tuple[int, ...]) -> tuple[Any, tuple[str, ...], tuple[int, ...]]:
    order = tuple(sorted(range(len(labels)), key=lambda index: chip.device_index(labels[index])))
    if order == tuple(range(len(labels))):
        return operator, labels, dims
    xp = concrete_array_module(operator)
    matrix = xp.asarray(operator).reshape(dims + dims)
    matrix = xp.transpose(matrix, order + tuple(len(dims) + index for index in order))
    return (matrix.reshape(prod(dims), prod(dims)), tuple(labels[index] for index in order),
            tuple(dims[index] for index in order))


def _retained_owners(chip: Any, labels: tuple[str, ...], owner_key: str | None) -> tuple[list[Any], tuple[str, ...]]:
    """Return the effective terms whose maps act on an operator on ``labels``, and the operator's final devices."""
    owners = []
    for terms in chip.effective_terms:
        if terms.projection is not None:
            reached = _reach(terms.projection._steps(), labels, owner_key)
            if reached is not None:
                owners.append(terms)
                labels = reached
    return owners, labels


def retained_support(chip: Any, labels: tuple[str, ...], owner_key: str | None = None) -> tuple[str, ...]:
    """Return, in device order, the devices that a surviving operator on ``labels`` acts on in retained coordinates.

    Parameters
    ----------
    chip : Chip
        Chip whose captured maps the operator follows.
    labels : tuple[str, ...]
        Devices the operator acts on in authored coordinates.
    owner_key : str or None, default=None
        Owner whose operator a map can already hold in retained coordinates.
    """
    _owners, reached = _retained_owners(chip, tuple(labels), owner_key)
    return tuple(sorted(reached, key=chip.device_index))


def transport_retained_operator(chip: Any, operator: Any, labels: tuple[str, ...], dims: tuple[int, ...],
                                owner_key: str | None = None) -> tuple[Any, tuple[str, ...], tuple[int, ...]] | None:
    """Carry a dense operator through every captured map that it meets.

    Returns the operator, labels and dimensions in ``chip``'s device order, or
    ``None`` when no captured map acts on ``labels``. Each map applies in its
    lineage order, and the maps of different effective terms in chip order.
    """
    acted = False
    labels, dims = tuple(labels), tuple(dims)
    for terms in chip.effective_terms:
        if terms.projection is not None:
            operator, labels, dims, moved = _walk(terms.projection._steps(), operator, labels, dims, owner_key)
            acted = acted or moved
    if not acted:
        return None
    return _in_device_order(chip, operator, labels, dims)


def retained_operator(chip: Any, operator: Any, labels: tuple[str, ...], backend: Any,
                      owner_key: str | None = None, *, bases: Mapping[str, Any] | None = None,
                      local_bases: Mapping[str, Any] | None = None) -> PhysicsExpr | None:
    """Return a surviving component's operator in the retained coordinates of ``chip``.

    The operator acts on ``labels`` in their authored coordinates. The result
    acts on the targets of every captured map it meets and on its other
    devices, in device order. It is ``None`` when no captured map acts on the
    operator. The result keeps the operator's declared level changes when
    every map it crosses conserves the total level index.
    """
    owners, _reached = _retained_owners(chip, tuple(labels), owner_key)
    if not owners:
        return None
    changes = None if any(t.excitation_changes is None for t in owners) else authored_excitation_changes(
        operator, labels, backend, bases,
    )
    matrix = backend.to_array(materialize_expr(operator, backend, local_bases=local_bases))
    dims = tuple(chip[label].local_space().dimension for label in labels)
    transported = transport_retained_operator(chip, matrix, tuple(labels), dims, owner_key)
    assert transported is not None
    matrix, labels, dims = transported
    return PhysicsExpr.from_matrix(matrix, labels=labels, dims=dims, name="projected_operator",
                                   excitation_changes=changes)


@dataclass(frozen=True, eq=False)
class EffectiveTerms:
    """Retained matrix terms in the named devices' authored coordinates.

    ``hamiltonian`` is in GHz. Channels carry unscaled operators and rates in
    1/ns. These terms already express the reduction's selected approximation,
    so assembly changes their frame without dropping further operator bands.
    Values are captured at construction. If you edit a surviving device, its
    authored terms change, but this captured correction is not recalculated.
    ``projection`` carries the source coordinates of surviving operators.
    Channels already stored here are in retained coordinates and bypass
    ``projection``.

    Parameters
    ----------
    labels : tuple[str, ...]
        Device labels defining the retained tensor-product order.
    dims : tuple[int, ...]
        Authored dimensions aligned with ``labels``.
    hamiltonian : array-like
        Hermitian retained Hamiltonian in GHz.
    channels : tuple[CollapseChannel, ...], default=()
        Retained jump operators and Lindblad rates in 1/ns.
    label : str, default="effective"
        Name used for diagnostics and the generated expression.
    projection : OperatorProjection or None, default=None
        Captured source-to-retained operator coordinates.
    notes : tuple[str, ...], default=()
        Approximations stated by the producer of these terms, reported by
        :meth:`physics_notes` after the notes derived from the terms.
    excitation_changes : mapping of str to iterable of int, or None, default=None
        Structure declared by the producer. A mapping states that the
        Hamiltonian and ``projection`` conserve the retained devices' total
        energy-level index. It also gives the total level changes, column minus
        row, that each named channel can carry. Band decomposition then treats
        every other change as zero, even for traced values. ``None`` declares
        no structure.
    """

    labels: tuple[str, ...]
    dims: tuple[int, ...]
    hamiltonian: Any
    channels: tuple[CollapseChannel, ...] = ()
    label: str = "effective"
    projection: OperatorProjection | None = None
    notes: tuple[str, ...] = ()
    excitation_changes: Mapping[str, frozenset[int]] | None = None

    def __post_init__(self) -> None:
        labels, dims = tuple(self.labels), tuple(self.dims)
        if not labels or len(set(labels)) != len(labels) or len(labels) != len(dims):
            raise ValueError("EffectiveTerms requires unique labels and matching dimensions.")
        if any(not isinstance(label, str) or not label for label in labels) or not self.label:
            raise ValueError("EffectiveTerms labels must be nonempty strings.")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 for dim in dims):
            raise ValueError("EffectiveTerms dimensions must be positive integers.")
        if self.projection is not None and (
            self.projection.target_labels != labels or self.projection.target_dims != dims
        ):
            raise ValueError("EffectiveTerms projection must use the retained contribution coordinates.")
        shape = (prod(dims), prod(dims))
        h = copy_value(self.hamiltonian, readonly=True)
        if getattr(h, "shape", None) != shape:
            raise ValueError(f"Effective Hamiltonian must have shape {shape}.")
        if not contains_tracer(h):
            concrete = np.asarray(h)
            # A zero matrix, as in terms that hold only channels, needs no further check.
            if np.any(concrete) and (not np.all(np.isfinite(concrete)) or not np.allclose(
                concrete, concrete.conj().T, atol=1e-12, rtol=1e-12
            )):
                raise ValueError("Effective Hamiltonian must be finite and Hermitian.")
        if any(not isinstance(channel, CollapseChannel) for channel in self.channels):
            raise TypeError("EffectiveTerms channels must be CollapseChannel values.")
        channels = tuple(
            CollapseChannel(copy_value(c.operator, readonly=True), copy_value(c.rate, readonly=True), c.name)
            for c in self.channels
        )
        if any(getattr(c.operator, "shape", None) != shape for c in channels):
            raise ValueError(f"Effective jump operators must have shape {shape}.")
        for channel in channels:
            if np.shape(channel.rate) != ():
                raise ValueError("Effective channel rates must be scalar.")
            if not contains_tracer((channel.operator, channel.rate)):
                if not np.all(np.isfinite(np.asarray(channel.operator))) or not np.isfinite(channel.rate):
                    raise ValueError("Effective channel operators and rates must be finite.")
        if len({c.name for c in channels}) != len(channels):
            raise ValueError("Effective channel names must be unique within one contribution.")
        notes = (self.notes,) if isinstance(self.notes, str) else tuple(self.notes)
        if any(not isinstance(note, str) or not note for note in notes):
            raise ValueError("EffectiveTerms notes must be nonempty strings.")
        if self.excitation_changes is not None:
            names = [channel.name for channel in channels]
            unknown = set(self.excitation_changes) - set(names)
            if unknown:
                raise ValueError(f"Excitation changes name unknown effective channels {sorted(unknown)}.")
            changes = {
                name: frozenset(int(change) for change in self.excitation_changes[name])
                for name in names if name in self.excitation_changes
            }
            object.__setattr__(self, "excitation_changes", changes)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "hamiltonian", h)
        object.__setattr__(self, "channels", channels)

    def expression(self, matrix: Any = None) -> PhysicsExpr:
        """Return a captured matrix through the authored-expression path.

        Parameters
        ----------
        matrix : array-like or None, default=None
            Matrix in GHz. ``None`` uses :attr:`hamiltonian`.
        """
        conserving = matrix is None and self.excitation_changes is not None
        return PhysicsExpr.from_matrix(
            self.hamiltonian if matrix is None else matrix, labels=self.labels, dims=self.dims, name=self.label,
            excitation_changes=(0,) if conserving else None,
        )

    def channel_expression(self, channel: CollapseChannel) -> PhysicsExpr:
        """Return one retained jump operator with its declared level changes.

        Parameters
        ----------
        channel : CollapseChannel
            One of :attr:`channels`.
        """
        changes = None if self.excitation_changes is None else self.excitation_changes.get(channel.name)
        return PhysicsExpr.from_matrix(
            channel.operator, labels=self.labels, dims=self.dims, name=self.label, excitation_changes=changes,
        )

    def physics_notes(self) -> list[str]:
        """Return the terms' support, assembly rule, channels and coordinate map, then the producer's notes."""
        support = ", ".join(f"{label} ({dim} levels)" for label, dim in zip(self.labels, self.dims))
        notes = [
            f"Captured matrix terms on {support}; editing a device does not recompute them.",
            "Assembly moves them into the chip frame without dropping operator bands, "
            "whatever the chip approximation.",
        ]
        if self.channels:
            notes.append("Retained Lindblad channels: " + ", ".join(c.name for c in self.channels) + ".")
        if self.projection is not None:
            notes.append("Operators of surviving components follow the captured coordinate map.")
        if self.excitation_changes is not None:
            notes.append("The retained terms and coordinate map conserve the total excitation number.")
        return notes + list(self.notes)

    def validate_for(self, chip: Any) -> None:
        """Validate retained labels and dimensions against a chip.

        Parameters
        ----------
        chip : Chip
            Chip that will receive these terms.
        """
        unknown = set(self.labels) - chip.device_map.keys()
        if unknown:
            raise ValueError(f"Effective terms {self.label!r} target unknown devices {sorted(unknown)}.")
        actual = tuple(chip[label].local_space().dimension for label in self.labels)
        if actual != self.dims:
            raise ValueError(f"Effective terms {self.label!r} require authored dimensions {self.dims}; got {actual}.")

    def fingerprint(self) -> Any:
        return value_fingerprint(
            (
                self.label,
                self.labels,
                self.dims,
                self.hamiltonian,
                tuple((c.name, c.operator, c.rate) for c in self.channels),
                None if self.projection is None else self.projection.fingerprint(),
                None if self.excitation_changes is None else tuple(
                    (name, tuple(sorted(changes))) for name, changes in self.excitation_changes.items()
                ),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Persist concrete retained terms without serializing a live calculation."""

        def matrix(value: Any) -> dict[str, Any]:
            array = np.asarray(value, dtype=complex)
            return {"real": array.real.tolist(), "imag": array.imag.tolist()}

        data = dict(
            projection=None if self.projection is None else self.projection.to_dict(),
            label=self.label,
            labels=list(self.labels),
            dims=list(self.dims),
            hamiltonian=matrix(self.hamiltonian),
            channels=[dict(name=c.name, operator=matrix(c.operator), rate=float(c.rate)) for c in self.channels],
        )
        if self.notes:
            data["notes"] = list(self.notes)
        if self.excitation_changes is not None:
            data["excitation_changes"] = {
                name: sorted(changes) for name, changes in self.excitation_changes.items()
            }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EffectiveTerms:
        """Recreate validated retained terms from their numerical payload.

        Parameters
        ----------
        data : dict[str, Any]
            Payload produced by :meth:`to_dict`.
        """
        required = {"label", "labels", "dims", "hamiltonian", "channels", "projection"}
        if not required <= set(data) <= required | {"notes", "excitation_changes"}:
            raise ValueError("Invalid serialized EffectiveTerms fields.")

        def matrix(value: dict[str, Any]) -> Any:
            return np.asarray(value["real"]) + 1j * np.asarray(value["imag"])

        return cls(
            tuple(data["labels"]),
            tuple(data["dims"]),
            matrix(data["hamiltonian"]),
            tuple(CollapseChannel(matrix(c["operator"]), c["rate"], c["name"]) for c in data["channels"]),
            data["label"],
            None if data["projection"] is None else OperatorProjection.from_dict(data["projection"]),
            tuple(data.get("notes", ())),
            data.get("excitation_changes"),
        )
