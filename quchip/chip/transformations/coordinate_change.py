"""Keep a captured change of coordinates on a rebuilt chip.

Coupling and effective-term reductions rotate the whole product space by a
unitary from one isolated interaction. Devices and kept edges keep their
authored terms. One :class:`~quchip.chip.effective.EffectiveTerms` contribution
holds the remaining part of the rotated Hamiltonian, which includes earlier
effective terms and carries their channels and notes through the rotation.
"""

from __future__ import annotations

from functools import reduce
from typing import Any

import jax.numpy as jnp

from quchip.chip.dressing import BareProductReference, label_eigensystem
from quchip.chip.effective import EffectiveTerms, OperatorProjection
from quchip.chip.sw import bare_hamiltonian
from quchip.chip.transformations.plumbing import inherited_notes, rebuild_chip
from quchip.chip.transformations.result import ReductionMap
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.expr import materialize_expr
from quchip.engine.bands import embed_on_support
from quchip.engine.basis import _lowest_eigenpairs
from quchip.utils.values import DeferredValue


def product_energies(bases: Any, labels: Any) -> Any:
    """Return the local energies of ``labels`` summed over their product basis, in label order."""
    return reduce(lambda total, energies: (total[:, None] + energies[None, :]).reshape(-1),
                  (bases[label].energies for label in labels))


def isolated_hamiltonian(chip: Any, devices: Any, approximation: Any, *, couplings: Any = (),
                         effective_terms: Any = ()) -> tuple[Any, Any, tuple[int, ...]]:
    """Return ``devices`` with the selected interaction alone: matrix, local product energies and dims.

    The diagonal holds the local product energies. The selected ``couplings``
    or ``effective_terms`` supply all other elements. Uses ``chip``'s basis and
    backend.
    """
    from quchip.chip.chip import Chip

    settings = dict(basis=chip.basis, backend=chip.backend, approximation=approximation)
    group = Chip(devices, list(couplings), effective_terms=effective_terms, **settings)
    matrix, labels, dims = bare_hamiltonian(group)
    energies = product_energies(group.resolve(frame="lab").bases, labels)
    # Local energies define H0. Subtract their assembled matrix before adding
    # that diagonal: basis roundoff must not masquerade as a selected coupling
    # between exactly degenerate isolated levels.
    local = bare_hamiltonian(Chip(devices, **settings))[0]
    return jnp.diag(energies) + (matrix - local), energies, dims


def labeled_eigenvectors(matrix: Any, dims: tuple[int, ...]) -> Any:
    """Return the eigenvectors of ``matrix`` ordered by the bare product label of largest overlap."""
    _, vectors = _lowest_eigenpairs(matrix, matrix.shape[0])
    return vectors[:, label_eigensystem(vectors, BareProductReference(dims)).indices]


def embed_on_labels(backend: Any, labels: Any, dims: tuple[int, ...], operator: Any, support: tuple[str, ...]) -> Any:
    """Embed a dense ``operator`` on ``support`` in the product space ordered by ``labels``."""
    indices = tuple(labels.index(label) for label in support)
    local_dims = [dims[index] for index in indices]
    native = backend.from_array(operator, dims=[local_dims, local_dims])
    return jnp.asarray(backend.to_array(embed_on_support(backend, native, indices, dims)))


def retain_coordinate_change(
    chip: Any,
    *,
    source: Any,
    labels: Any,
    dims: tuple[int, ...],
    rotation: Any,
    retained_h: Any,
    approximation: Any,
    devices: Any,
    couplings: Any,
    label: str,
    notes: tuple[str, ...],
    removed_owners: tuple[Any, ...] = (),
) -> tuple[Any, ReductionMap]:
    """Rebuild ``chip`` and keep in one contribution everything that its authored terms miss.

    ``rotation`` and ``retained_h`` use the source's local energy product basis
    in ``labels`` order. ``retained_h`` is the source Hamiltonian in the
    rotated coordinates. The rebuilt chip keeps ``devices`` and ``couplings``
    with their authored parameters. ``retained_h`` minus their Hamiltonian,
    assembled at ``approximation``, becomes one :class:`EffectiveTerms` named
    ``label``. This contribution absorbs the source's effective terms, so their
    channels follow the rotation and their notes come before ``notes``. The
    channels of ``removed_owners`` also follow the rotation. Returns the
    rebuilt chip and its captured :class:`ReductionMap`.
    """
    labels, dims = tuple(labels), tuple(dims)
    final = rebuild_chip(chip, devices=devices, couplings=couplings, effective_terms=())
    retained_dims = tuple(final.authored_dims)
    resolved = final.resolve(frame="lab", approximation=approximation)
    lift = reduce(jnp.kron, (source.bases[name].energy_vectors for name in labels))
    final_lift = reduce(jnp.kron, (resolved.bases[name].vectors for name in labels))
    remaining_h = jnp.asarray(resolved.hamiltonian().matrix(backend=chip.backend))
    correction = lift @ retained_h @ lift.conj().T - final_lift @ remaining_h @ final_lift.conj().T

    channels = []
    contributions = chip._collapse_contributions_with_owners(source.bases)
    for operator, rate, support, owner_label, name, _paths, owner in contributions:
        if not (isinstance(owner, EffectiveTerms) or any(owner is removed for removed in removed_owners)):
            continue
        support_labels = tuple(labels[index] for index in support) if support else labels
        local = jnp.asarray(chip.backend.to_array(materialize_expr(operator, chip.backend)))
        basis = reduce(jnp.kron, (source.bases[support_label].energy_vectors for support_label in support_labels))
        embedded = embed_on_labels(chip.backend, labels, dims, basis.conj().T @ local @ basis, support_labels)
        transformed = rotation.conj().T @ embedded @ rotation
        channels.append(CollapseChannel(lift @ transformed @ lift.conj().T,
                                        materialize_expr(rate, chip.backend), f"{owner_label}.{name}"))

    projection = OperatorProjection.capture(chip, labels, retained_dims, lift @ rotation @ lift.conj().T)
    terms = EffectiveTerms(labels, retained_dims, correction, tuple(channels), label=label,
                           projection=projection, notes=(*inherited_notes(chip), *notes))
    terms.validate_for(final)
    final._effective_terms = (terms,)
    source_to_solver = reduce(jnp.kron, (source.bases[name].vectors.conj().T
                                         @ source.bases[name].energy_vectors for name in labels))
    target_to_solver = final_lift.conj().T @ lift
    embedding = source_to_solver @ rotation @ target_to_solver.conj().T
    mapping = ReductionMap(labels, dims, labels, dims, chip.backend, DeferredValue(lambda: embedding))
    return final, mapping
