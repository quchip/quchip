"""Remove a selected edge through a captured change of coordinates.

Both endpoint devices survive with their authored parameters. The kept matrix
includes the selected interaction's level-dependent shifts and its
transformation of every other interaction in the chip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax.scipy.linalg import expm

from quchip.chip.sw import bare_hamiltonian, interaction_generator
from quchip.chip.transformations.coordinate_change import (
    embed_on_labels, isolated_hamiltonian, labeled_eigenvectors, product_energies, retain_coordinate_change,
)
from quchip.chip.transformations.dispatch import EliminationTarget, register_elimination_target
from quchip.chip.transformations.plumbing import StrandedLine, plan_stranded_lines, reattach_equipment
from quchip.chip.transformations.result import EliminationResult
from quchip.utils.labeling import LabelKeyedDict, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip

def reduce_coupling(chip: "Chip", target: Any, method: str) -> EliminationResult:
    """Diagonalize an edge exactly or remove its exchange to second order.

    The pair rotation uses only its isolated devices and selected edge. Its
    action on the full chip, including parallel and spectator interactions, is
    kept. ``method='sw'`` truncates the Hamiltonian at second order in
    interactions. ``method='exact'`` applies a full unitary transformation. The
    coordinate map keeps the full Hilbert space. Exact derivatives require a
    locally stable dressed assignment and nondegenerate eigenpairs.

    Channels follow the captured operator projection, and surviving components
    continue to own their rates. SW channels use the exponentiated first-order
    coordinate map. The Hamiltonian keeps its second-order truncation.
    Surviving control operators are not transformed yet.
    """
    if method not in {"sw", "exact"}:
        raise NotImplementedError(f"Coupling-target reduction does not implement method {method!r}.")
    coupling_label = resolve_label(target)
    coupling = chip.coupling_map[coupling_label]
    if coupling._time_terms():
        raise NotImplementedError(
            f"Coupling {coupling_label!r} has intrinsic time-dependent terms. "
            "Their retained coordinate transformation is not implemented; keep this edge."
        )
    pair_labels = (coupling.device_a_label, coupling.device_b_label)
    result_kind = "coupling"
    equipment = chip.control_equipment

    def classify(line: Any) -> StrandedLine | None:
        from quchip.control.drive import CouplingDrive

        if not isinstance(line, CouplingDrive) or line.target_label != coupling_label:
            return None
        return StrandedLine(
            rule_target=coupling,
            missing_rule_message=(
                f"Control line '{line.label}' ({type(line).__name__}) pumps the eliminated "
                f"coupling '{coupling_label}'. No retarget rule converts it "
                f"(register_retarget_rule({type(line).__name__}, {type(coupling).__name__}, "
                f"'{result_kind}', ...)); unwire the line first (chip.unwire('{line.label}')), "
                "or keep the coupling."
            ),
        )

    survivor_lines, retarget_plan = plan_stranded_lines(equipment, classify, result_kind)
    approximation = chip.approximation
    source = chip.resolve(frame="lab", approximation=approximation)
    h, labels, dims = bare_hamiltonian(chip, approximation=approximation)
    cloned = chip.clone()
    pair_devices = [cloned[label] for label in pair_labels]
    pair_h, pair_e, pair_dims = isolated_hamiltonian(chip, pair_devices, approximation,
                                                     couplings=[cloned.coupling_map[coupling_label]])

    if method == "exact":
        pair_rotation = labeled_eigenvectors(pair_h, pair_dims)
        rotation = embed_on_labels(chip.backend, labels, dims, pair_rotation, pair_labels)
        pair_after = pair_rotation.conj().T @ pair_h @ pair_rotation
        retained_h = rotation.conj().T @ h @ rotation
    else:
        pair_s = interaction_generator(pair_e, pair_h)
        s = embed_on_labels(chip.backend, labels, dims, pair_s, pair_labels)
        rotation = expm(-s)
        energies = product_energies(source.bases, labels)

        def second_order(matrix: Any, energies: Any, generator: Any) -> Any:
            h0 = jnp.diag(energies)
            first = generator @ h0 - h0 @ generator
            interaction = matrix - h0
            return (matrix + first + generator @ interaction - interaction @ generator
                    + .5 * (generator @ first - first @ generator))

        retained_h = second_order(h, energies, s)
        pair_after = second_order(pair_h, pair_e, pair_s)

    notes = [f"Removed '{coupling_label}' and retained its complete {method} Hamiltonian correction; "
             "device parameters and remaining edges stay authored.",
             "The full-space coordinate map includes the effect on parallel and spectator interactions.",
             "Surviving channel operators follow the captured map; control operators are not transformed yet."]
    final, mapping = retain_coordinate_change(
        chip, source=source, labels=labels, dims=dims, rotation=rotation, retained_h=retained_h,
        approximation=approximation, devices=cloned.devices,
        couplings=[c for c in cloned.couplings if c.label != coupling_label],
        label=f"retained_{coupling_label}", notes=tuple(notes), removed_owners=(coupling,),
    )
    # These diagnostics describe the isolated selected pair, not a refolding
    # of all the remaining chip interactions into two frequencies.
    diagonal = jnp.real(jnp.diagonal(pair_after))
    indices = (pair_dims[1], 1)
    chi = diagonal[pair_dims[1] + 1] - diagonal[indices[0]] - diagonal[indices[1]] + diagonal[0]
    effective_params: dict[str, Any] = LabelKeyedDict({
        label: {"lamb_shift": diagonal[index] - diagonal[0] - (pair_e[index] - pair_e[0]),
                "freq_after": diagonal[index] - diagonal[0], "chi": chi}
        for label, index in zip(pair_labels, indices)
    })
    delta = pair_e[indices[0]] - pair_e[indices[1]]
    ratio = jnp.abs(coupling.coupling_strength / delta)
    validity: dict[str, Any] = LabelKeyedDict({coupling_label: {"g_over_delta": ratio, "is_valid": ratio < .1}})
    reattach_equipment(chip, final, equipment, survivor_lines, retarget_plan,
                       mode_label=coupling_label, result_kind=result_kind, edges={}, notes=notes)
    return EliminationResult(chip=final, effective_params=effective_params, validity=validity,
                             notes=notes, mapping=mapping)


register_elimination_target(EliminationTarget(
    kind="coupling",
    claims=lambda chip, target: resolve_label(target) in chip.coupling_map,
    reduce=reduce_coupling,
))
