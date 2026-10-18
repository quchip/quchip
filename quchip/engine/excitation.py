"""Excitation-number structure of resolved chips.

``N`` is the total isolated energy-level index of a product state. This module
finds the changes of ``N`` that authored operators carry, proves whether the
retained static model of a chip conserves ``N``, and indexes product states by
``N``.
"""

from __future__ import annotations

from math import prod
from typing import Any, Mapping

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
        records = [bases[label] for label in support]
        if contains_tracer(tuple(record.energy_vectors for record in records)):
            return None
        from quchip.backend._memory import require_memory

        dimension = prod(record.native_dim for record in records)
        require_memory(16 * 5 * dimension**2, task="Excitation-number proof on local support",
                       remedy="Use factored operators or declare their excitation_changes.")
        with jax.ensure_compile_time_eval(), backend.eager_operators():
            matrix = backend.to_array(materialize_expr(node, backend, bindings=bindings, local_bases=bases))
            vectors = records[0].energy_vectors
            for record in records[1:]:
                vectors = jnp.kron(vectors, record.energy_vectors)
            dims = tuple(int(record.energy_vectors.shape[1]) for record in records)
            return concrete_excitation_changes(matrix, dims, vectors)

    return changes(operator, labels)


def static_excitation_changes(chip: Any, approximation: Any, *, resolution: Any = None) -> frozenset[int] | None:
    """Return the total energy-level changes of the retained static model, or ``None`` when unknown.

    The set is symmetric and holds ``0``. A zero-total band filter removes
    every change of an ordinary interaction. Retained terms bypass that
    filter. A generated port pair ``L_d^dagger L_u`` changes the total by
    ``u - d`` for each change ``u`` of the upstream port and ``d`` of the
    downstream port.

    Parameters
    ----------
    chip : Chip
        Chip whose static model is checked.
    approximation : Approximation
        Approximation the static model is resolved with.
    resolution : optional
        Local basis resolution to reuse.
    """
    from quchip.engine.assembly import _interaction_contributions, _resolve_system

    resolution = _resolve_system(chip, chip.backend) if resolution is None else resolution
    bases = resolution.bases
    found = {0}
    for term in _interaction_contributions(chip):
        if not term.retained and approximation.filters_terms and approximation.conserves_excitation_number():
            continue
        changes = authored_excitation_changes(term.expression(), term.labels, chip.backend, bases)
        if changes is None:
            return None
        found.update(changes)
    pairs = () if resolution.network is None else resolution.network.generated_pairs
    weights: dict[str, frozenset[int]] = {}
    for label in dict.fromkeys(label for downstream, upstream, _ in pairs for label in (downstream, upstream)):
        port = chip.port(label)
        ((channel, _paths),) = port._collapse_channels_with_paths(chip)
        changes = authored_excitation_changes(channel.operator, port.resolve_targets(chip), chip.backend, bases)
        if changes is None:
            return None
        weights[label] = changes
    for downstream, upstream, _ in pairs:
        found.update(up - down for up in weights[upstream] for down in weights[downstream])
    return frozenset(found | {-change for change in found})


def conserves_excitation_number(chip: Any, approximation: Any, *, resolution: Any = None) -> bool:
    """Prove conservation of the total isolated energy-level index from retained operators.

    A zero-total band filter proves conservation of ordinary interactions.
    Otherwise their authored algebra or exact local matrix support must prove
    it. Retained terms bypass that filter. Unknown support is not a proof.
    Generated port pairs additionally require single-excitation lowering at
    both ends.

    Parameters
    ----------
    chip : Chip
        Chip whose static model is checked.
    approximation : Approximation
        Approximation the static model is resolved with.
    resolution : optional
        Local basis resolution to reuse.
    """
    from quchip.engine.assembly import _resolve_system

    resolution = _resolve_system(chip, chip.backend) if resolution is None else resolution
    if static_excitation_changes(chip, approximation, resolution=resolution) != {0}:
        return False
    pairs = () if resolution.network is None else resolution.network.generated_pairs
    labels = dict.fromkeys(label for downstream, upstream, _ in pairs for label in (downstream, upstream))
    return all(_port_lowers_level_index(chip, chip.port(label), resolution.bases) for label in labels)


def _port_lowers_level_index(chip: Any, port: Any, bases: Mapping[str, Any]) -> bool:
    """Return whether a port's operator lowers the total energy-level index by one and does nothing else.

    The check reads the operator in the chip's retained coordinates, as assembly does.
    """
    from quchip.chip.effective import retained_operator

    ((channel, _paths),) = port._collapse_channels_with_paths(chip)
    operator, labels = channel.operator, tuple(port.resolve_targets(chip))
    retained = retained_operator(chip, operator, labels, chip.backend, f"port:{port.label}", bases=bases,
                                 local_bases=bases)
    if retained is not None:
        operator, labels = retained, retained.labels
    changes = authored_excitation_changes(operator, labels, chip.backend, bases)
    return changes is not None and changes <= {1}
