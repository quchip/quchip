"""Diagonalize selected effective terms through a captured change of coordinates.

Every device and edge survives with its authored parameters. An exact rotation
diagonalizes the selected terms together with the local Hamiltonians of the
devices they act on. The kept matrix holds the resulting level-dependent shifts
and the effect of the rotation on every other interaction in the chip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from quchip.chip.sw import bare_hamiltonian
from quchip.chip.transformations.coordinate_change import (
    embed_on_labels, isolated_hamiltonian, labeled_eigenvectors, retain_coordinate_change,
)
from quchip.chip.transformations.dispatch import EliminationTarget, register_elimination_target
from quchip.chip.transformations.plumbing import reattach_equipment
from quchip.chip.transformations.result import EliminationResult
from quchip.utils.labeling import LabelKeyedDict, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


def reduce_effective_terms(chip: "Chip", target: Any, method: str) -> EliminationResult:
    """Diagonalize one effective contribution exactly and keep its complete correction.

    The rotation uses only the selected terms and the isolated local
    Hamiltonians of the devices they act on. Dressed states take the bare label
    of largest overlap. Its action on the full chip, including couplings and
    other effective terms, is kept, so dressed queries on the reduced chip
    return the source spectrum at the chip's approximation. Only
    ``method='exact'`` is implemented, because a second-order expansion is not
    reliable for a strong nonlinearity such as a junction cosine. Exact
    derivatives require a locally stable dressed assignment and nondegenerate
    eigenpairs.
    """
    if method != "exact":
        raise NotImplementedError(
            "Effective-term reduction implements method='exact' only; a second-order expansion "
            "is not reliable for a strong nonlinearity such as a junction cosine."
        )
    terms_label = resolve_label(target)
    selected = next(terms for terms in chip.effective_terms if terms.label == terms_label)
    approximation = chip.approximation
    source = chip.resolve(frame="lab", approximation=approximation)
    h, labels, dims = bare_hamiltonian(chip, approximation=approximation)
    group_labels = tuple(label for label in labels if label in selected.labels)
    cloned = chip.clone()
    group_devices = [cloned[label] for label in group_labels]
    group_h, group_e, group_dims = isolated_hamiltonian(chip, group_devices, approximation,
                                                        effective_terms=[selected])
    group_rotation = labeled_eigenvectors(group_h, group_dims)
    rotation = embed_on_labels(chip.backend, labels, dims, group_rotation, group_labels)
    group_after = group_rotation.conj().T @ group_h @ group_rotation

    notes = [
        f"Diagonalized effective terms '{terms_label}' exactly with the local Hamiltonians of "
        f"{', '.join(group_labels)} and retained the complete correction; device parameters and "
        "edges stay authored.",
        "Dressed states take the bare label of largest overlap; levels near the Fock truncation "
        "carry its error.",
        "The full-space coordinate map includes the effect on couplings and other effective terms.",
        "Surviving channel operators follow the captured map; control operators are not transformed yet.",
    ]
    final, mapping = retain_coordinate_change(
        chip, source=source, labels=labels, dims=dims, rotation=rotation,
        retained_h=rotation.conj().T @ h @ rotation, approximation=approximation,
        devices=cloned.devices, couplings=cloned.couplings, label=f"retained_{terms_label}", notes=tuple(notes),
    )
    effective_params = _level_diagnostics(group_labels, group_dims, jnp.real(jnp.diagonal(group_after)), group_e)
    equipment = chip.control_equipment
    lines = [] if equipment is None else list(equipment.lines)
    reattach_equipment(chip, final, equipment, lines, [], mode_label=terms_label, result_kind="effective",
                       edges={}, notes=notes)
    return EliminationResult(chip=final, effective_params=effective_params, validity=LabelKeyedDict(),
                             notes=notes, mapping=mapping)


def _level_diagnostics(labels: tuple[str, ...], dims: tuple[int, ...], energies: Any, bare: Any) -> dict[str, Any]:
    """Return each device's dressed transition, Lamb shift, anharmonicity and full-pull cross-Kerr shifts."""

    def level(values: Any, excitations: dict[str, int]) -> Any:
        return values[int(np.ravel_multi_index(tuple(excitations.get(name, 0) for name in labels), dims))]

    params: dict[str, Any] = LabelKeyedDict()
    for label, dim in zip(labels, dims):
        freq_after = level(energies, {label: 1}) - energies[0]
        entry: dict[str, Any] = {
            "freq_after": freq_after,
            "lamb_shift": freq_after - (level(bare, {label: 1}) - bare[0]),
            "cross_kerr": LabelKeyedDict({
                other: level(energies, {label: 1, other: 1}) - freq_after - level(energies, {other: 1})
                for other in labels if other != label
            }),
        }
        if dim >= 3:
            entry["anharmonicity"] = level(energies, {label: 2}) - 2 * freq_after - energies[0]
        params[label] = entry
    return params


register_elimination_target(EliminationTarget(
    kind="effective terms",
    claims=lambda chip, target: resolve_label(target) in {terms.label for terms in chip.effective_terms},
    reduce=reduce_effective_terms,
))
