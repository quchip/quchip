"""Remove a selected edge through a captured change of coordinates.

Both endpoint devices survive with their authored parameters. The retained
matrix includes the selected interaction's level-dependent shifts and its
transformation of every other interaction in the chip.
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax.scipy.linalg import expm

from quchip.approximations import Exact
from quchip.chip.dressing import BareProductReference, label_eigensystem
from quchip.chip.effective import EffectiveTerms, OperatorProjection
from quchip.chip.sw import bare_hamiltonian, interaction_generator
from quchip.chip.transformations.dispatch import EliminationTarget, register_elimination_target
from quchip.chip.transformations.plumbing import (
    StrandedLine, plan_stranded_lines, reattach_equipment, rebuild_chip,
)
from quchip.chip.transformations.result import EliminationResult, ReductionMap
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.expr import materialize_expr
from quchip.engine.bands import embed_on_support
from quchip.engine.basis import _lowest_eigenpairs
from quchip.utils.labeling import LabelKeyedDict, resolve_label
from quchip.utils.values import DeferredValue

if TYPE_CHECKING:
    from quchip.chip.chip import Chip

def reduce_coupling(chip: "Chip", target: Any, method: str) -> EliminationResult:
    """Diagonalize an edge exactly or remove its exchange to second order.

    The pair rotation uses only its isolated devices and selected edge. Its
    action on the full chip is retained, including parallel and spectator
    interactions. ``method='sw'`` truncates the Hamiltonian at second order
    in interactions; ``method='exact'`` applies a full unitary transformation.
    The coordinate map preserves the full Hilbert space. Exact derivatives
    require a locally stable dressed assignment and nondegenerate eigenpairs.

    Channels follow the captured operator projection while surviving components
    continue to own their rates. SW channels use the exponentiated first-order
    coordinate map; the Hamiltonian keeps its second-order truncation.
    Surviving control operators are not transformed yet.
    """
    from quchip.chip.chip import Chip

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
    approximation = Exact() if method == "exact" else chip.approximation
    source = chip.resolve(frame="lab", approximation=approximation)
    h, labels, dims = bare_hamiltonian(chip, approximation=approximation)
    cloned = chip.clone()
    pair_devices = [cloned[label] for label in pair_labels]
    pair = Chip(pair_devices, [cloned.coupling_map[coupling_label]],
                basis=chip.basis, backend=chip.backend, approximation=approximation)
    pair_h, _, pair_dims = bare_hamiltonian(pair)
    pair_bases = pair.resolve(frame="lab").bases
    pair_e = (pair_bases[pair_labels[0]].energies[:, None]
              + pair_bases[pair_labels[1]].energies[None, :]).reshape(-1)
    isolated = Chip(pair_devices, basis=chip.basis, backend=chip.backend, approximation=approximation)
    local_h = bare_hamiltonian(isolated)[0]
    # Local energies define H0. Subtract their assembled matrix before adding
    # that diagonal: basis roundoff must not masquerade as a selected coupling
    # between exactly degenerate isolated levels.
    pair_h = jnp.diag(pair_e) + (pair_h - local_h)

    def embed(operator: Any, support_labels: tuple[str, ...]) -> Any:
        support = tuple(labels.index(label) for label in support_labels)
        local_dims = [dims[index] for index in support]
        native = chip.backend.from_array(operator, dims=[local_dims, local_dims])
        return jnp.asarray(chip.backend.to_array(embed_on_support(chip.backend, native, support, dims)))

    if method == "exact":
        _, vectors = _lowest_eigenpairs(pair_h, pair_h.shape[0])
        assignment = label_eigensystem(vectors, BareProductReference(pair_dims))
        pair_rotation = vectors[:, assignment.indices]
        rotation = embed(pair_rotation, pair_labels)
        pair_after = pair_rotation.conj().T @ pair_h @ pair_rotation
        retained_h = rotation.conj().T @ h @ rotation
    else:
        pair_s = interaction_generator(pair_e, pair_h)
        s = embed(pair_s, pair_labels)
        rotation = expm(-s)
        energies = source.bases[labels[0]].energies
        for label in labels[1:]:
            energies = (energies[:, None] + source.bases[label].energies[None, :]).reshape(-1)

        def second_order(matrix: Any, energies: Any, generator: Any) -> Any:
            h0 = jnp.diag(energies)
            first = generator @ h0 - h0 @ generator
            interaction = matrix - h0
            return (matrix + first + generator @ interaction - interaction @ generator
                    + .5 * (generator @ first - first @ generator))

        retained_h = second_order(h, energies, s)
        pair_after = second_order(pair_h, pair_e, pair_s)

    def transform(operator: Any) -> Any:
        return rotation.conj().T @ operator @ rotation

    final = rebuild_chip(chip, devices=cloned.devices,
                         couplings=[c for c in cloned.couplings if c.label != coupling_label],
                         effective_terms=())
    resolved = final.resolve(frame="lab", approximation=approximation)
    lift = reduce(jnp.kron, (source.bases[label].energy_vectors for label in labels))
    final_lift = reduce(jnp.kron, (resolved.bases[label].vectors for label in labels))
    remaining_h = jnp.asarray(resolved.hamiltonian().matrix(backend=chip.backend))
    correction = lift @ retained_h @ lift.conj().T - final_lift @ remaining_h @ final_lift.conj().T
    channels = []

    def inherit(channel: CollapseChannel, support: tuple[str, ...], name: str) -> None:
        local = jnp.asarray(chip.backend.to_array(materialize_expr(channel.operator, chip.backend)))
        basis = reduce(jnp.kron, (source.bases[label].energy_vectors for label in support))
        transformed = transform(embed(basis.conj().T @ local @ basis, support))
        channels.append(CollapseChannel(lift @ transformed @ lift.conj().T,
                                       materialize_expr(channel.rate, chip.backend), name))

    contributions = chip._collapse_contributions_with_owners(source.bases)
    for operator, rate, support, owner_label, name, _paths, owner in contributions:
        if owner is coupling or isinstance(owner, EffectiveTerms):
            support_labels = tuple(labels[index] for index in support) if support else tuple(labels)
            inherit(CollapseChannel(operator, rate, name), support_labels, f"{owner_label}.{name}")
    projection = OperatorProjection.capture(
        chip, tuple(labels), tuple(final.authored_dims), lift @ rotation @ lift.conj().T,
    )
    terms = EffectiveTerms(tuple(labels), tuple(final.authored_dims), correction,
                           tuple(channels), label=f"retained_{coupling_label}", projection=projection)
    terms.validate_for(final)
    final._effective_terms = (terms,)
    source_to_solver = reduce(jnp.kron, (source.bases[label].vectors.conj().T
                                       @ source.bases[label].energy_vectors for label in labels))
    target_to_solver = final_lift.conj().T @ lift
    embedding = source_to_solver @ rotation @ target_to_solver.conj().T
    mapping = ReductionMap(tuple(labels), tuple(dims), tuple(labels), tuple(dims),
                           chip.backend, DeferredValue(lambda: embedding))
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
    notes = [f"Removed '{coupling_label}' and retained its complete {method} Hamiltonian correction; "
             "device parameters and remaining edges stay authored.",
             "The full-space coordinate map includes the effect on parallel and spectator interactions.",
             "Surviving channel operators follow the captured map; control operators are not transformed yet."]
    reattach_equipment(chip, final, equipment, survivor_lines, retarget_plan,
                       mode_label=coupling_label, result_kind=result_kind, edges={}, notes=notes)
    return EliminationResult(chip=final, effective_params=effective_params, validity=validity,
                             notes=notes, mapping=mapping)


register_elimination_target(EliminationTarget(
    kind="coupling",
    claims=lambda chip, target: resolve_label(target) in chip.coupling_map,
    reduce=reduce_coupling,
))
