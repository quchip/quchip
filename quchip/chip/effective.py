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
from quchip.utils.jax_utils import contains_tracer
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
    conservation, and no port pair may generate a cascade Hamiltonian.

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
    product space. Rates and local operators remain owned by the components;
    this record carries only their change of coordinates.
    """

    source_labels: tuple[str, ...]
    source_dims: tuple[int, ...]
    target_labels: tuple[str, ...]
    target_dims: tuple[int, ...]
    embedding: Any
    overrides: tuple[tuple[str, OperatorProjection], ...] = ()

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

    @classmethod
    def capture(cls, chip: Any, target_labels: tuple[str, ...], target_dims: tuple[int, ...],
                embedding: Any) -> OperatorProjection:
        """Compose a reduction with the source's already captured operator coordinates."""
        previous = [t.projection for t in chip.effective_terms if t.projection is not None]
        if not previous:
            return cls(tuple(d.label for d in chip.devices), tuple(chip.authored_dims),
                       target_labels, target_dims, embedding)
        if len(previous) != 1 or previous[0].target_labels != tuple(d.label for d in chip.devices):
            raise NotImplementedError("Combining partial operator projections requires a common source space.")
        parent = previous[0]
        def advance(previous: OperatorProjection) -> OperatorProjection:
            return cls(previous.source_labels, previous.source_dims, target_labels, target_dims,
                       previous.embedding @ embedding)

        return replace(advance(parent), overrides=tuple((key, advance(p)) for key, p in parent.overrides))

    def with_current_operators(self, owner_keys: tuple[str, ...]) -> OperatorProjection:
        """Mark operators already authored in this retained space, preserving their lineage."""
        if not owner_keys:
            return self
        current = type(self)(self.target_labels, self.target_dims, self.target_labels,
                             self.target_dims, jnp.eye(prod(self.target_dims), dtype=complex))
        overrides = {**dict(self.overrides), **dict.fromkeys(owner_keys, current)}
        return replace(self, overrides=tuple(overrides.items()))

    def apply(self, operator: Any, labels: tuple[str, ...], owner_key: str | None = None, *,
              excitation_changes: frozenset[int] | None = None) -> PhysicsExpr:
        """Project a live local array without allocating its full-space identity embedding.

        ``excitation_changes`` declares the source operator's total energy-level
        changes. Pass it only when the captured map conserves the total level
        index, so that the projected operator carries the same changes.
        """
        for key, projection in self.overrides:
            if key == owner_key:
                return projection.apply(operator, labels, excitation_changes=excitation_changes)
        support = tuple(self.source_labels.index(label) for label in labels)
        rest = tuple(index for index in range(len(self.source_dims)) if index not in support)
        local_size = prod(self.source_dims[index] for index in support)
        rest_size = prod(self.source_dims[index] for index in rest)
        target_size = prod(self.target_dims)
        matrix = jnp.asarray(operator)
        if matrix.shape != (local_size, local_size):
            raise ValueError("Operator dimensions do not match its captured source support.")
        tensor = jnp.asarray(self.embedding).reshape(self.source_dims + (target_size,))
        tensor = jnp.transpose(tensor, support + rest + (len(self.source_dims),))
        tensor = tensor.reshape(local_size, rest_size, target_size)
        acted = jnp.einsum("ab,brj->arj", matrix, tensor)
        projected = jnp.einsum("ari,arj->ij", tensor.conj(), acted)
        return PhysicsExpr.from_matrix(projected, labels=self.target_labels, dims=self.target_dims,
                                       name="projected_operator", excitation_changes=excitation_changes)

    def fingerprint(self) -> Any:
        return value_fingerprint((self.source_labels, self.source_dims, self.target_labels,
                                  self.target_dims, self.embedding,
                                  tuple((key, p.fingerprint()) for key, p in self.overrides)))

    def to_dict(self) -> dict[str, Any]:
        matrix = np.asarray(self.embedding)
        return dict(source_labels=list(self.source_labels), source_dims=list(self.source_dims),
                    target_labels=list(self.target_labels), target_dims=list(self.target_dims),
                    real=matrix.real.tolist(), imag=matrix.imag.tolist(),
                    overrides={key: p.to_dict() for key, p in self.overrides})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorProjection:
        if set(data) != {"source_labels", "source_dims", "target_labels", "target_dims", "real", "imag", "overrides"}:
            raise ValueError("Invalid serialized OperatorProjection fields.")
        return cls(tuple(data["source_labels"]), tuple(data["source_dims"]), tuple(data["target_labels"]),
                   tuple(data["target_dims"]), np.asarray(data["real"]) + 1j * np.asarray(data["imag"]),
                   tuple((key, cls.from_dict(p)) for key, p in data["overrides"].items()))


@dataclass(frozen=True, eq=False)
class EffectiveTerms:
    """Retained matrix terms in the named devices' authored coordinates.

    ``hamiltonian`` is in GHz; channels carry unscaled operators and rates in
    1/ns. These terms already express the reduction's selected approximation,
    so assembly changes their frame without dropping further operator bands.
    Values are captured at construction. Editing a surviving device changes
    its authored terms; it does not recompute this captured correction.
    ``projection`` carries the source coordinates of surviving operators;
    channels already stored here are in retained coordinates and bypass it.

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
        Hamiltonian and ``projection`` conserve the total energy-level index
        of the retained devices, and gives the total level changes, column
        minus row, that each named channel can carry. Band decomposition then
        treats every other change as zero, also for traced values. ``None``
        declares no structure.
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
            if not np.all(np.isfinite(concrete)) or not np.allclose(
                concrete, concrete.conj().T, atol=1e-12, rtol=1e-12
            ):
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
