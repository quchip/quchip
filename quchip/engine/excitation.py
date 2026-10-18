"""Excitation-number structure of resolved chips.

``N`` is the total isolated energy-level index of a product state. This module
finds the changes of ``N`` that authored operators carry, proves whether the
retained static model of a chip conserves ``N``, and indexes product states by
``N``. :class:`ExcitationStructure` holds these facts for one resolved system
and one approximation, and the engine's routes read them from it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from quchip.declarative.expr import (
    PhysicsExpr, _bound_values, declared_excitation_changes, is_energy_diagonal, materialize_expr,
)
from quchip.utils.jax_utils import contains_tracer


def excitation_sectors(dims: tuple[int, ...]) -> np.ndarray:
    """Total energy-level index of every C-order product state."""
    return np.indices(tuple(dims)).reshape(len(dims), -1).sum(axis=0)


@lru_cache(maxsize=1024)
def sector_levels(dims: tuple[int, ...], total: int) -> np.ndarray:
    """Return every level product below ``dims`` that sums to ``total``, in C order."""
    if not dims:
        return np.zeros((1 if total == 0 else 0, 0), dtype=np.int64)
    blocks = []
    for level in range(min(dims[0] - 1, total) + 1):
        rest = sector_levels(dims[1:], total - level)
        if rest.shape[0]:
            blocks.append(np.column_stack((np.full(rest.shape[0], level, dtype=np.int64), rest)))
    levels = np.concatenate(blocks) if blocks else np.zeros((0, len(dims)), dtype=np.int64)
    levels.flags.writeable = False
    return levels


def sector_size(dims: Sequence[int], total: int) -> int:
    """Count the product states of ``dims`` whose levels sum to ``total``.

    Parameters
    ----------
    dims : sequence of int
        Level count of each device.
    total : int
        Total energy-level index.
    """
    counts = [1] + [0] * total
    for dimension in dims:
        counts = [sum(counts[n - level] for level in range(min(dimension - 1, n) + 1)) for n in range(total + 1)]
    return counts[total]


def excitation_classes(changes: frozenset[int] | None, maximum: int) -> tuple[tuple[int, ...], ...]:
    """Partition the totals ``0..maximum`` into the classes that ``changes`` connect.

    A class holds the totals that steps ``n -> n +- c`` with every
    intermediate total in ``[0, maximum]`` connect. Unknown changes give one
    class.

    Parameters
    ----------
    changes : frozenset of int or None
        Changes of the total energy-level index, or ``None`` when unknown.
    maximum : int
        Largest retained total.
    """
    if changes is None:
        return (tuple(range(maximum + 1)),)
    steps = sorted({abs(change) for change in changes} - {0})
    seen: set[int] = set()
    classes: list[tuple[int, ...]] = []
    for start in range(maximum + 1):
        if start in seen:
            continue
        members, frontier = {start}, [start]
        while frontier:
            total = frontier.pop()
            for step in steps:
                for reached in (total - step, total + step):
                    if 0 <= reached <= maximum and reached not in members:
                        members.add(reached)
                        frontier.append(reached)
        seen |= members
        classes.append(tuple(sorted(members)))
    return tuple(classes)


def sector_eigh(h: Any, sectors: np.ndarray) -> tuple[Any, Any]:
    """Diagonalize each total-excitation block; return ascending eigenpairs.

    Eigenvectors have exact zeros outside their own sector, so round-off can
    neither mix sectors nor make the result depend on near-degeneracies
    between them. A concrete Hamiltonian is diagonalized on the host, where
    the differently sized blocks compile no device programs.
    """
    if not contains_tracer(h):
        h = np.asarray(h)
        eigenvalues: Any = np.zeros(h.shape[0], dtype=h.real.dtype)
        eigenvectors: Any = np.zeros(h.shape, dtype=h.dtype)
        start = 0
        for sector in np.unique(sectors):
            rows = np.flatnonzero(sectors == sector)
            columns = np.arange(start, start + rows.size)
            eigenvalues[columns], eigenvectors[np.ix_(rows, columns)] = np.linalg.eigh(h[np.ix_(rows, rows)])
            start += rows.size
        order: Any = np.argsort(eigenvalues, kind="stable")
        return eigenvalues[order], eigenvectors[:, order]
    eigenvalues = jnp.zeros(h.shape[0], dtype=h.real.dtype)
    eigenvectors = jnp.zeros(h.shape, dtype=h.dtype)
    start = 0
    for sector in np.unique(sectors):
        rows = np.flatnonzero(sectors == sector)
        columns = np.arange(start, start + rows.size)
        values, vectors = jnp.linalg.eigh(h[np.ix_(rows, rows)])
        eigenvalues = eigenvalues.at[columns].set(values)
        eigenvectors = eigenvectors.at[np.ix_(rows, columns)].set(vectors)
        start += rows.size
    order = jnp.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def concrete_excitation_changes(
    matrix: Any,
    dims: tuple[int, ...],
    energy_vectors: Any | None = None,
) -> frozenset[int] | None:
    """Return the total level changes carried by a concrete operator.

    ``matrix`` acts on a product basis whose energy-ordered states are the
    columns of ``energy_vectors``; ``None`` means the product basis is already
    energy ordered. ``dims`` counts the energy levels per subsystem. Exact
    nonzero comparison can add round-off weights but never omits a band.
    Traced payloads return ``None``: their bands are not statically known.
    """
    if contains_tracer((matrix, energy_vectors)):
        return None
    values = np.asarray(matrix, dtype=complex)
    if energy_vectors is not None:
        vectors = np.asarray(energy_vectors, dtype=complex)
        values = vectors.conj().T @ values @ vectors
    levels = excitation_sectors(dims)
    rows, cols = np.nonzero(values)
    return frozenset(int(change) for change in np.unique(levels[cols] - levels[rows]))


def authored_excitation_changes(
    operator: Any,
    labels: tuple[str, ...],
    backend: Any,
    bases: Mapping[str, Any] | None,
) -> frozenset[int] | None:
    """Return the total energy-level changes an authored operator carries.

    Declarations and factored algebra supply structural support. Scalar
    coefficients do not change it, including at zero. Constant leaves are
    checked in the captured energy bases at compile time. Unknown matrix
    support or traced energy vectors return ``None``. No small-entry
    threshold supplies a conservation proof. A product on a truncated energy
    basis is checked as one complete matrix.

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
    declared = declared_excitation_changes(operator)
    if declared is not None:
        return declared
    if bases is None or any(label not in bases for label in labels):
        return None
    bindings = _bound_values(operator) if isinstance(operator, PhysicsExpr) else None

    def changes(node: Any, support: tuple[str, ...]) -> frozenset[int] | None:
        declared = declared_excitation_changes(node)
        if declared is not None:
            return declared
        if isinstance(node, PhysicsExpr):
            if not node.labels or is_energy_diagonal(node, bases):
                return frozenset({0})
            if node.kind in ("scale", "embed"):
                operand = node.args[1] if node.kind == "scale" else node.args[0]
                return changes(operand, operand.labels)
            # A product on a truncated energy basis also passes through discarded
            # levels, so only its complete matrix gives its retained changes.
            truncated = node.kind == "matmul" and any(
                bases[label].resolved_dim < bases[label].native_dim for label in node.labels
            )
            if node.kind in ("add", "sub", "tensor", "matmul") and not truncated:
                left, right = (changes(arg, arg.labels) for arg in node.args)
                if left is None or right is None:
                    return None
                if node.kind in ("add", "sub"):
                    return left | right
                return frozenset(a + b for a in left for b in right)
            # An opaque parameter-dependent function has no fixed support proof.
            if node.parameter_paths():
                return None
            if node.kind == "matrix" and contains_tracer(node.args[0]):
                return None
            if node.kind == "function" and contains_tracer(node.numeric_values()):
                return None
        elif contains_tracer(node):
            return None
        energy = _energy_matrix(node, support, backend, bases, bindings)
        return None if energy is None else concrete_excitation_changes(*energy)

    return changes(operator, labels)


def largest_changing_entry(
    operator: Any,
    labels: tuple[str, ...],
    backend: Any,
    bases: Mapping[str, Any] | None,
) -> float | None:
    """Return the largest magnitude of an entry of ``operator`` that changes ``N``.

    The operator is read as one complete matrix in the captured energy bases,
    in its own units. ``None`` means that a basis is missing, that the matrix
    would not fit in memory, or that the operator, its parameters or the
    energy vectors are traced.

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
    if bases is None or any(label not in bases for label in labels):
        return None
    bindings = _bound_values(operator) if isinstance(operator, PhysicsExpr) else None
    if contains_tracer((operator, bindings)):
        return None
    from quchip.utils.memory import dense_bytes, fits_memory

    dimension = prod(bases[label].native_dim for label in labels)
    if not fits_memory(dense_bytes(dimension, dimension, copies=_ENERGY_MATRIX_COPIES)):
        return None
    energy = _energy_matrix(operator, labels, backend, bases, bindings)
    if energy is None:
        return None
    values, dims = energy
    levels = excitation_sectors(dims)
    return float(np.abs(values[levels[:, None] != levels[None, :]]).max(initial=0.0))


# The local operator, the energy vectors and the products of the basis change.
_ENERGY_MATRIX_COPIES = 5


def _energy_matrix(node: Any, support: tuple[str, ...], backend: Any, bases: Mapping[str, Any],
                   bindings: Any) -> tuple[np.ndarray, tuple[int, ...]] | None:
    """Return ``node`` on ``support`` in the product energy basis and its level counts, or ``None`` when traced."""
    records = [bases[label] for label in support]
    if contains_tracer(tuple(record.energy_vectors for record in records)):
        return None
    from quchip.utils.memory import dense_bytes, require_memory

    dimension = prod(record.native_dim for record in records)
    require_memory(dense_bytes(dimension, dimension, copies=_ENERGY_MATRIX_COPIES),
                   task="Excitation-number proof on local support",
                   remedy="Use factored operators or declare their excitation_changes.")
    with jax.ensure_compile_time_eval(), backend.eager_operators():
        matrix = backend.to_array(materialize_expr(node, backend, bindings=bindings, local_bases=bases))
        vectors = records[0].energy_vectors
        for record in records[1:]:
            vectors = jnp.kron(vectors, record.energy_vectors)
    if contains_tracer((matrix, vectors)):
        return None
    vectors = np.asarray(vectors, dtype=complex)
    values = vectors.conj().T @ np.asarray(matrix, dtype=complex) @ vectors
    return values, tuple(int(record.energy_vectors.shape[1]) for record in records)


def format_changes(changes: frozenset[int] | None) -> str:
    """Return ``changes`` as text, with each symmetric pair written ``±c``.

    Parameters
    ----------
    changes : frozenset of int or None
        Changes of ``N``, or ``None`` when unknown.
    """
    if changes is None:
        return "unknown"
    if not changes:
        return "none"
    if all(-change in changes for change in changes):
        return ", ".join("0" if step == 0 else f"±{step}" for step in sorted({abs(change) for change in changes}))
    return ", ".join(f"{change:+d}" if change else "0" for change in sorted(changes))


def keeps_zero_total_only(approximation: Any) -> bool:
    """Return whether ``approximation`` keeps only the zero-total bands of ordinary interactions.

    Such an approximation removes every change of ``N`` from a coupling or an
    intrinsic two-device term. Retained terms bypass the filter.

    Parameters
    ----------
    approximation : Approximation
        Approximation the model is resolved with.
    """
    return bool(approximation.filters_terms and approximation.conserves_excitation_number())


def lowers_by_one(changes: frozenset[int] | None) -> bool:
    """Return whether ``changes`` proves that an operator lowers ``N`` by one and does nothing else.

    Parameters
    ----------
    changes : frozenset of int or None
        Changes of ``N`` that the operator carries, or ``None`` when unknown.
    """
    return changes is not None and changes <= {1}


@dataclass(frozen=True)
class ExcitationTerm:
    """The changes of the total excitation number ``N`` that one term carries.

    A change ``c`` maps a product state with total ``N`` to total ``N - c``,
    so a lowering operator has the change ``1``.

    Attributes
    ----------
    label : str
        Label of the coupling, retained term, port, drive or collapse channel.
        A port pair is labeled ``"upstream -> downstream"`` and a collapse
        channel ``"owner.name"``.
    kind : str
        ``"coupling"``, ``"retained"``, ``"cascade"`` (the Hamiltonian that a
        port network generates for one port pair), ``"port"``, ``"drive"`` or
        ``"channel"``.
    changes : frozenset of int or None
        Changes of ``N`` that quchip finds for the term under the
        approximation, or ``None`` when it cannot prove them.
    largest_entry : float or None
        For a coupling or retained term that changes ``N``, the largest
        magnitude in GHz of a static entry that changes ``N``. ``None`` for
        other terms and when the entry is traced.
    """

    label: str
    kind: str
    changes: frozenset[int] | None
    largest_entry: float | None = None


class ExcitationStructure:
    """Excitation-number structure of one resolved chip under one approximation.

    ``N`` is the total isolated energy-level index of a product state. The
    static model conserves ``N`` when every coupling, retained term and
    generated port pair keeps only zero changes of ``N``, and every port of a
    generated pair lowers ``N`` by one. quchip reads these changes from exact
    operator structure: a nonzero entry that changes ``N`` breaks conservation
    however small it is, and unknown structure is not a proof. A conserving
    static model is block diagonal with one block for each ``N``, and dressed
    queries diagonalize only the blocks of their labels. Under an active
    :class:`~quchip.cutoff.ExcitationCutoff`, each block is one class of
    totals that the static changes connect.

    :meth:`Chip.excitation_structure() <quchip.chip.chip.Chip.excitation_structure>`
    returns a complete record. The engine builds one record for each resolved
    system and approximation, and reads each fact the first time a route asks
    for it.

    Parameters
    ----------
    chip : Chip
        Chip whose static model is read.
    approximation : Approximation
        Approximation that selects the static terms.
    resolution : object
        Resolved local bases of ``chip``.

    Attributes
    ----------
    approximation : Approximation
        Approximation that selects the static terms.
    dims : tuple of int
        Resolved level count of each device, in chip order.
    cutoff : int or None
        Largest total of the chip's active excitation cutoff, or ``None``.
    """

    def __init__(self, chip: Any, approximation: Any, resolution: Any) -> None:
        from quchip.cutoff import active_cutoff
        from quchip.engine.assembly import _interaction_contributions

        policy = active_cutoff(chip)
        self.approximation = approximation
        self.dims = tuple(int(dimension) for dimension in resolution.dims)
        self.cutoff = None if policy is None else int(policy.max_total)
        self._chip, self._bases = chip, resolution.bases
        self._interactions = tuple(_interaction_contributions(chip))
        self._pairs = () if resolution.network is None else tuple(resolution.network.generated_pairs)
        self._pair_ports = tuple(dict.fromkeys(label for downstream, upstream, _ in self._pairs
                                               for label in (downstream, upstream)))
        self._interaction_changes: list[frozenset[int] | None] = []
        self._authored_ports: dict[str, frozenset[int] | None] = {}
        self._retained_ports: dict[str, frozenset[int] | None] = {}
        self._entries: dict[int, float | None] = {}

    @cached_property
    def changes(self) -> frozenset[int] | None:
        """Changes of ``N`` of the static model, closed under negation and holding ``0``, or ``None`` when unknown."""
        found = {0}
        for changes in self.interaction_changes():
            if changes is None:
                return None
            found.update(changes)
        weights: dict[str, frozenset[int]] = {}
        for label in self._pair_ports:
            changes = self._authored_port(label)
            if changes is None:
                return None
            weights[label] = changes
        # A generated pair L_d^dagger L_u changes N by u - d.
        for downstream, upstream, _ in self._pairs:
            found.update(up - down for up in weights[upstream] for down in weights[downstream])
        return frozenset(found | {-change for change in found})

    @cached_property
    def conserves(self) -> bool:
        """Whether the static model conserves ``N``."""
        return self.changes == {0} and all(lowers_by_one(self._retained_port(label)) for label in self._pair_ports)

    def interaction_changes(self) -> Iterator[frozenset[int] | None]:
        """Yield the changes of ``N`` of each coupling and retained term, in chip order.

        Each term is read once. A coupling under an approximation that keeps
        only zero-total bands yields ``{0}``.
        """
        for index, term in enumerate(self._interactions):
            if index == len(self._interaction_changes):
                filtered = not term.retained and keeps_zero_total_only(self.approximation)
                self._interaction_changes.append(frozenset({0}) if filtered else authored_excitation_changes(
                    term.expression(), term.labels, self._chip.backend, self._bases))
            yield self._interaction_changes[index]

    @cached_property
    def terms(self) -> tuple[ExcitationTerm, ...]:
        """Changes of ``N`` of every coupling, retained term and generated port pair."""
        return tuple(ExcitationTerm(label, kind, changes, None if index is None or not _breaks(changes)
                                    else self._largest_entry(index))
                     for index, label, kind, changes in self._term_changes)

    @cached_property
    def _term_changes(self) -> tuple[tuple[int | None, str, str, frozenset[int] | None], ...]:
        """Index, label, kind and changes of every term; only couplings and retained terms have an index."""
        found: list[tuple[int | None, str, str, frozenset[int] | None]] = [
            (index, term.label, "retained" if term.retained else "coupling", changes)
            for index, (term, changes) in enumerate(zip(self._interactions, self.interaction_changes()))
        ]
        for downstream, upstream, _ in self._pairs:
            up, down = self._authored_port(upstream), self._authored_port(downstream)
            changes = None if up is None or down is None else frozenset(
                sign * (u - d) for u in up for d in down for sign in (1, -1))
            found.append((None, f"{upstream} -> {downstream}", "cascade", changes))
        return tuple(found)

    @cached_property
    def ports(self) -> tuple[ExcitationTerm, ...]:
        """Changes of ``N`` of each port of a generated pair, in the chip's retained coordinates."""
        return tuple(ExcitationTerm(label, "port", self._retained_port(label)) for label in self._pair_ports)

    @cached_property
    def breaking(self) -> tuple[ExcitationTerm, ...]:
        """Terms and ports that prevent a proof of conservation, empty when the static model conserves ``N``."""
        return (*(term for term in self.terms if _breaks(term.changes)),
                *(port for port in self.ports if not lowers_by_one(port.changes)))

    @cached_property
    def drives(self) -> tuple[ExcitationTerm, ...]:
        """Changes of ``N`` of each wired control line under the approximation."""
        equipment = self._chip.control_equipment
        found = []
        for line in () if equipment is None else equipment.lines:
            if line._target is not None:
                changes = drive_excitation_changes(self._chip, line, line._target, self._bases, self.approximation)
                found.append(ExcitationTerm(line.label, "drive", None if changes is None else frozenset(changes)))
        return tuple(found)

    @cached_property
    def channels(self) -> tuple[ExcitationTerm, ...]:
        """Changes of ``N`` of each collapse channel, ports included."""
        chip, labels = self._chip, tuple(device.label for device in self._chip.devices)
        found = []
        for operator, _rate, support, source, name, *_ in chip._collapse_contributions_with_owners(self._bases):
            support_labels = tuple(labels[index] for index in support) or labels
            changes = authored_excitation_changes(operator, support_labels, chip.backend, self._bases)
            found.append(ExcitationTerm(f"{source}.{name}", "channel", changes))
        return tuple(found)

    @cached_property
    def level_span(self) -> float | None:
        """Largest spread of one device's resolved energies in GHz, or ``None`` when they are traced."""
        energies = tuple(self._bases[device.label].energies for device in self._chip.devices)
        if contains_tracer(energies):
            return None
        return max((float(np.ptp(np.asarray(values))) for values in energies), default=0.0)

    def classes(self, max_total: int) -> tuple[tuple[int, ...], ...]:
        """Return the classes of totals ``0..max_total`` that the static changes connect.

        Parameters
        ----------
        max_total : int
            Largest retained total excitation number.
        """
        return excitation_classes(self.changes, max_total)

    def sector_sizes(self, max_total: int) -> tuple[int, ...]:
        """Return the number of product states with each total ``0..max_total``.

        Parameters
        ----------
        max_total : int
            Largest total excitation number.
        """
        return tuple(sector_size(self.dims, total) for total in range(max_total + 1))

    def largest_block(self, max_total: int) -> int:
        """Return the size of the largest block that dressed queries diagonalize for labels up to ``max_total``.

        A conserving chip diagonalizes one sector for each total, a chip with
        an active excitation cutoff one class, and any other chip its complete
        static model.

        Parameters
        ----------
        max_total : int
            Largest total excitation number of the queried labels.
        """
        if self.cutoff is not None:
            return max(sum(sector_size(self.dims, total) for total in group)
                       for group in self.classes(self.cutoff) if group[0] <= max_total)
        if self.conserves:
            return max(self.sector_sizes(max_total))
        return prod(self.dims)

    def _authored_port(self, label: str) -> frozenset[int] | None:
        if label not in self._authored_ports:
            port = self._chip.port(label)
            ((channel, _paths),) = port._collapse_channels_with_paths(self._chip)
            self._authored_ports[label] = authored_excitation_changes(
                channel.operator, port.resolve_targets(self._chip), self._chip.backend, self._bases)
        return self._authored_ports[label]

    def _retained_port(self, label: str) -> frozenset[int] | None:
        """Read a port's operator in the chip's retained coordinates, as assembly does."""
        if label not in self._retained_ports:
            from quchip.chip.effective import retained_operator

            chip, port = self._chip, self._chip.port(label)
            ((channel, _paths),) = port._collapse_channels_with_paths(chip)
            operator, labels = channel.operator, tuple(port.resolve_targets(chip))
            retained = retained_operator(chip, operator, labels, chip.backend, f"port:{port.label}",
                                         bases=self._bases, local_bases=self._bases)
            if retained is not None:
                operator, labels = retained, retained.labels
            self._retained_ports[label] = authored_excitation_changes(operator, labels, chip.backend, self._bases)
        return self._retained_ports[label]

    def _largest_entry(self, index: int) -> float | None:
        if index not in self._entries:
            term = self._interactions[index]
            self._entries[index] = largest_changing_entry(term.expression(), term.labels, self._chip.backend,
                                                          self._bases)
        return self._entries[index]

    def _evaluated(self) -> ExcitationStructure:
        """Evaluate every fact, then release the chip so that the record no longer reads it."""
        for name in ("changes", "conserves", "terms", "ports", "breaking", "drives", "channels", "level_span"):
            getattr(self, name)
        self._chip = self._bases = None
        return self


def round_off_advisory(structure: ExcitationStructure) -> str | None:
    """Return a warning when only entries far below the level spacings break conservation.

    Every term that breaks conservation must be a coupling or retained term
    whose largest entry that changes ``N`` is below ``1e-12`` times the
    largest level span of one device. The heuristic does not change the
    verdict, and ``None`` means that it does not apply. The search stops at
    the first term that rules it out.

    Parameters
    ----------
    structure : ExcitationStructure
        Record of a chip whose static model does not conserve ``N``.
    """
    span = structure.level_span
    breaking = [(index, label, changes) for index, label, _kind, changes in structure._term_changes
                if _breaks(changes)]
    if span is None or not breaking or any(index is None or changes is None for index, _, changes in breaking):
        return None
    if not all(lowers_by_one(port.changes) for port in structure.ports):
        return None
    largest = 0.0
    for index, _label, _changes in breaking:
        assert index is not None
        entry = structure._largest_entry(index)
        if entry is None or entry >= 1e-12 * span:
            return None
        largest = max(largest, entry)
    names = ", ".join(f"{label!r} ({format_changes(changes)})" for _index, label, changes in breaking)
    return (
        f"The total excitation number N changes only through entries of at most {largest:.1e} GHz in {names}, "
        f"far below the largest local level span of {span:.3g} GHz. quchip decides conservation from exact "
        f"operator structure, so dressed queries diagonalize the complete {prod(structure.dims)}-state static "
        "model instead of N sectors. Remove these entries if they are round-off, or the terms if they are "
        "switched off."
    )


def _breaks(changes: frozenset[int] | None) -> bool:
    """Return whether ``changes`` fails to prove that a term keeps ``N``."""
    return changes is None or not changes <= {0}


def excitation_structure(chip: Any, approximation: Any, *, resolution: Any = None,
                         complete: bool = False) -> ExcitationStructure:
    """Return the excitation structure of ``chip`` under ``approximation``.

    The record is built once for each resolved system and approximation.

    Parameters
    ----------
    chip : Chip
        Chip whose static model is read.
    approximation : Approximation
        Approximation that selects the static terms.
    resolution : optional
        Resolved local bases to reuse. ``None`` resolves the chip.
    complete : bool, default=False
        Evaluate every fact now and keep no reference to ``chip``. Such a
        record is not cached.
    """
    from quchip.engine.assembly import _resolve_system

    resolution = _resolve_system(chip, chip.backend) if resolution is None else resolution
    if complete:
        return ExcitationStructure(chip, approximation, resolution)._evaluated()
    cache = resolution.excitation
    if approximation not in cache:
        cache[approximation] = ExcitationStructure(chip, approximation, resolution)
    return cache[approximation]


def drive_excitation_changes(chip: Any, drive: Any, target: Any, bases: Mapping[str, Any],
                             approximation: Any) -> set[int] | None:
    """Return the changes of ``N`` that a delivered drive keeps under ``approximation``, or ``None``.

    The operator-band filter applies to a coupling drive under every
    approximation, so an approximation whose retained bands conserve ``N``
    removes every change of a coupling drive that no captured map retains.

    Parameters
    ----------
    chip : Chip
        Chip that owns the drive.
    drive : BaseDrive
        Drive whose authored terms are read.
    target : object
        Device or coupling that receives the drive.
    bases : mapping
        Resolved basis records keyed by device label.
    approximation : Approximation
        Approximation the model is resolved with.
    """
    from quchip.control.drive import CouplingDrive
    from quchip.engine.assembly import _drive_authored_terms, _keeps_drive_band, _retained_operator

    changes: set[int] = set()
    for _, authored in _drive_authored_terms(chip, drive, target, chip.backend):
        support = tuple(chip.device_index(label) for label in authored.labels)
        retained = any(term.projection is not None for term in chip.effective_terms)
        if retained:
            authored, support = _retained_operator(chip, authored, support, chip.backend,
                                                   f"drive:{drive.label}", bases=bases)
        weights: set[int] | frozenset[int] | None
        if not retained and isinstance(drive, CouplingDrive) and approximation.conserves_excitation_number():
            weights = {0}
        else:
            weights = authored_excitation_changes(authored, tuple(chip.devices[i].label for i in support),
                                                  chip.backend, bases)
            if weights is None:
                return None
            if len(support) == 1:
                weights = {n for n in weights if _keeps_drive_band(drive, approximation, (n,))}
        changes.update(weights)
    return changes
