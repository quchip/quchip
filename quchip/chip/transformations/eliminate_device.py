"""Device-target elimination: adiabatic reduction of a far-detuned mode.

A mode touching **one** survivor (leaf) contributes a retained Hamiltonian
correction and inherited channels when the removed components dissipate.
Surviving devices keep their authored parameters and local bases. A mode touching **two or more**
survivors — bus / tunable-coupler (bridge) or several at once — additionally
induces a mediated exchange ``J = g_a g_b / 2 · (1/Δ_a + 1/Δ_b)`` between every
survivor pair, represented by its own edge. Authored direct couplings keep
their parameters, channels and controls. For capacitive legs, a fixed eliminated mode emits a
:class:`~quchip.chip.couplings.Capacitive`; a frequency-controlled mode (or an
already-modulable direct edge) emits a
:class:`~quchip.chip.couplings.TunableCapacitive`. Other interactions emit a
first-transition exchange edge; the retained correction holds the remaining elements.

The reduction route (``method="sw"`` / ``method="exact"``) is a
:class:`~quchip.chip.transformations.methods.ReductionMethod` strategy; the
generic P/Q partitioning kernels live in :mod:`quchip.chip.sw`. This module
owns the fold — reading a route's reduced parameters into a rebuilt chip and
retargeting stranded control lines — and registers a device-kind
:class:`~quchip.chip.transformations.dispatch.EliminationTarget` at import time,
so :func:`~quchip.chip.transformations.dispatch.eliminate` dispatches any device
label here without importing this module directly.
"""

from __future__ import annotations

from functools import reduce
from itertools import combinations
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from quchip.utils.values import DeferredValue
from quchip.chip.couplings import Capacitive, TunableCapacitive
from quchip.chip.effective import (
    EffectiveTerms,
    OperatorProjection,
    authored_excitation_changes,
    conserves_excitation_number,
)
from quchip.declarative.dissipation import CollapseChannel
from quchip.chip.ports import Port
from quchip.engine.bands import embed_on_support
from quchip.chip.sw import (
    _exact_eigensystem,
    bare_hamiltonian,
    bare_index,
    basis_row,
    mode_blocks,
    cross_block_gap,
    excitation_sectors,
)
from quchip.chip.transformations.dispatch import EliminationTarget, register_elimination_target
from quchip.chip.transformations.methods import DeviceReductionContext, lookup_reduction_method
from quchip.chip.transformations.plumbing import (
    StrandedLine,
    inherited_notes,
    plan_stranded_lines,
    reattach_equipment,
    rebuild_chip,
)
from quchip.chip.transformations.result import EliminationResult, LazyEffectiveParams, ReductionMap
from quchip.control.drive import FluxDrive
from quchip.declarative.expr import PhysicsExpr, materialize_expr
from quchip.declarative.models import CouplingModel
from quchip.declarative.parameters import Scalar, parameter
from quchip.devices.protocols import FrequencyControlled
from quchip.devices.spaces import FockSpace
from quchip.utils.labeling import LabelKeyedDict, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


class _MediatedExchange(CouplingModel):
    """First-transition exchange; all other retained elements stay in EffectiveTerms."""

    g: Scalar = parameter(unit="GHz")

    def interaction(self, a: Any, b: Any, p: Any) -> Any:
        return p.g * self.parametric_interaction(a, b, p)

    def parametric_interaction(self, a: Any, b: Any, p: Any) -> Any:
        from quchip.backend import get_default_backend
        from quchip.declarative.expr import as_operator_expr

        backend = get_default_backend()
        operator = (backend.tensor(a.device.sigma_plus, b.device.sigma_minus)
                    + backend.tensor(a.device.sigma_minus, b.device.sigma_plus))
        return as_operator_expr(operator, labels=(a.label, b.label),
                                dims=(a.space.dimension, b.space.dimension), name="exchange")


def _retained_port_labels(chip: "Chip") -> set[str]:
    """Labels of ports whose operators an earlier reduction put in retained coordinates."""
    return {
        key.removeprefix("port:")
        for terms in chip.effective_terms if terms.projection is not None
        for key, _ in terms.projection.overrides if key.startswith("port:")
    }


def reduce_device(chip: "Chip", target: Any, method: str) -> EliminationResult:
    """Adiabatically eliminate a far-detuned device, folding its effect into the survivors.

    The registry guarantees ``target`` names a device on ``chip``, and
    :func:`eliminate` has already validated ``method``. See
    :func:`~quchip.chip.transformations.dispatch.eliminate` for the full
    physics contract and the ``method`` semantics.

    Parameters
    ----------
    chip
        Source chip (never mutated).
    target
        The device to eliminate — label string or object.
    method
        The reduction route (``"sw"`` or ``"exact"``), already validated.

    Returns
    -------
    EliminationResult
    """
    mode_label = resolve_label(target)
    touching = [c for c in chip.couplings if mode_label in (c.device_a_label, c.device_b_label)]
    survivors = []
    for c in touching:
        other = c.device_b_label if c.device_a_label == mode_label else c.device_a_label
        survivors.append((other, c))
    is_multi = len({label for label, _ in survivors}) >= 2
    mode_device: Any = chip[mode_label]
    affected_ports = [
        port
        for port in chip.ports
        if mode_label in port.resolve_targets(chip)
    ]
    # A port that an earlier reduction transformed already acts on every
    # survivor of that reduction and reaches this mode only through its
    # dressing; it is transformed again like an inherited channel. The mode's
    # own ports form the reduced boundary and keep its reflection.
    retained_ports = _retained_port_labels(chip)
    carried_ports = [port for port in affected_ports if port.label in retained_ports]
    boundary_ports = [port for port in affected_ports if port.label not in retained_ports]
    reflection_plane: str | None = None
    if affected_ports:
        from quchip.engine.linear_response import is_linear_mode

        if boundary_ports and not is_linear_mode(mode_device, chip.backend):
            raise NotImplementedError(
                "Network-connected elimination currently supports a linear Fock-mode target; "
                f"{mode_label!r} is {type(mode_device).__name__}. Keep the boundary mode."
            )
        if not survivors:
            raise NotImplementedError(
                f"Cannot eliminate port-coupled mode {mode_label!r} without a coupled survivor."
            )
        if any(
            device.resolved_dimension(chip.basis) != device.local_space().dimension
            for device in chip.devices if device.label != mode_label
        ):
            raise NotImplementedError(
                "Network-connected elimination does not yet lift a transformed port from a "
                "projected survivor basis back into its authored local space."
            )
        affected_labels = {port.label for port in affected_ports}
        assert chip.port_network is not None
        if any(
            affected_labels.intersection((downstream, upstream))
            for downstream, upstream, _coefficient in chip.port_network._active_generated_pairs()
        ):
            raise NotImplementedError(
                "Cannot eliminate a port that participates in a cascade-generated Hamiltonian; "
                "joint SLH adiabatic elimination is not implemented."
            )
        unsupported = [
            port.label
            for port in boundary_ports
            if port.resolve_targets(chip) != (mode_label,) or port.operator is not None
        ]
        if unsupported:
            raise NotImplementedError(
                "Network-connected linear-mode elimination currently transforms the mode's "
                "default lowering ports and ports an earlier reduction transformed; "
                f"unsupported ports: {unsupported}."
            )
        if len(boundary_ports) > 1:
            raise NotImplementedError(
                f"Mode {mode_label!r} couples to ports {[port.label for port in boundary_ports]}; its "
                "frequency-dependent transmission between them has no reduced boundary yet. Keep the mode."
            )
    if boundary_ports:
        assert chip.port_network is not None
        reflection_plane = chip.port_network._exclusive_exposure(boundary_ports[0].label)
        non_fock = [
            label
            for label, _ in survivors
            if not isinstance(chip[label].local_space(), FockSpace)
        ]
        if non_fock:
            raise NotImplementedError(
                "Network-connected linear-mode elimination requires Fock-space survivors so "
                f"the effective lowering channel is explicit; unsupported: {non_fock}."
            )

    # A control line wired to the eliminated mode (or to a coupling that
    # touches it) has no image in the reduced model unless a registered rule
    # converts it (chip/retarget.py). plan_stranded_lines() looks the rules up
    # *before* any fold work, so a missing rule fails fast — exactly like the
    # bath guard above.
    result_kind = "edge" if is_multi else "leaf-fold"
    equipment = chip.control_equipment
    def classify(line: Any) -> StrandedLine | None:
        from quchip.control.drive import CouplingDrive

        rule_target: Any
        if isinstance(line, CouplingDrive):
            edge_coupling: Any = line._target
            doomed = mode_label in (edge_coupling.device_a_label, edge_coupling.device_b_label)
            rule_target = edge_coupling
            what = f"coupling '{edge_coupling.label}' (touches mode '{mode_label}')"
        else:
            doomed = line._target is not None and line._target.label == mode_label
            rule_target = mode_device
            what = f"mode '{mode_label}'"
        if not doomed:
            return None
        return StrandedLine(
            rule_target=rule_target,
            missing_rule_message=(
                f"Control line '{line.label}' ({type(line).__name__}) targets the "
                f"eliminated {what}. No retarget rule converts it "
                f"(register_retarget_rule({type(line).__name__}, {type(rule_target).__name__}, "
                f"'{result_kind}', ...)); unwire the line first (chip.unwire('{line.label}')), "
                "or keep the target."
            ),
        )

    survivor_lines, retarget_plan = plan_stranded_lines(equipment, classify, result_kind)

    reduction = lookup_reduction_method(method)
    assert reduction is not None  # method membership checked above; keeps mypy's Optional narrow

    effective_params: dict[str, Any] = LabelKeyedDict()
    validity: dict[str, Any] = LabelKeyedDict()
    # The route reads one static model; the remainder below is assembled at
    # the same approximation so the retained correction is exactly what the
    # route resolved (PHYSICS.md §10.6).
    approximation = reduction.source_approximation(chip)
    dropped_items = ["ring-up transients"]
    if approximation.filters_terms and survivors:
        dropped_items.insert(0, "counter-rotating terms")
    notes = [
        f"Adiabatic elimination (method='{method}'): steady-state (vacuum) reduction.",
        "Dropped: " + ", ".join(dropped_items) + reduction.dropped_suffix(),
    ]

    # Build the reduced chip from cloned survivors (mode + its couplings removed).
    survivor_labels = [d.label for d in chip.devices if d.label != mode_label]
    reduced = chip.clone()
    kept_couplings = [c for c in reduced.couplings if mode_label not in (c.device_a_label, c.device_b_label)]
    reduced_devices = [reduced[lbl] for lbl in survivor_labels]

    mode = chip[mode_label]
    mode_is_frequency_controlled = isinstance(mode, FrequencyControlled) or any(
        isinstance(line, FluxDrive) for line, _ in retarget_plan
    )
    h, labels, dims = bare_hamiltonian(chip, approximation=approximation)
    # A structurally conserving model has no matrix elements between
    # total-excitation sectors; removing their round-off keeps both routes,
    # and every captured map, inside the sectors.
    sectors = excitation_sectors(dims) if conserves_excitation_number(chip, approximation) else None
    if sectors is not None:
        h = jnp.where(sectors[:, None] == sectors[None, :], h, 0.0)
    # Survivor pairs are keyed in the chip's device order everywhere — the
    # pair extraction, the exact route, and the fold loop below — so the two
    # sides of every ("J", a, b) lookup agree no matter what order the legs
    # were scanned in. Coupling-scan order and device order genuinely differ
    # on real chips (a center mode declared before its outer neighbors).
    scanned = {lbl for lbl, _ in survivors}
    touching_labels = [lbl for lbl in labels if lbl in scanned]

    p_mask, _ = mode_blocks(dims, labels, mode_label)
    min_gap = cross_block_gap(h, p_mask)

    ctx = DeviceReductionContext(
        mode_label=mode_label,
        survivor_labels=touching_labels,
        labels=labels,
        dims=dims,
        h=h,
        p_mask=p_mask,
        sectors=sectors,
    )
    pair_params = reduction.pair_parameters(ctx)
    incoming_frequencies = {
        label: jnp.real(h[bare_index(labels, dims, label), bare_index(labels, dims, label)] - h[0, 0])
        for label in labels
    }

    source_bases = chip.resolve(frame="lab").bases

    def transform_operator(operator: Any, support_labels: tuple[str, ...]) -> tuple[Any, Any]:
        local = jnp.asarray(chip.backend.to_array(materialize_expr(operator, chip.backend)))
        transform = reduce(jnp.kron, (source_bases[label].energy_vectors for label in support_labels))
        local = transform.conj().T @ local @ transform
        support = tuple(labels.index(label) for label in support_labels)
        local_dims = [dims[index] for index in support]
        native = chip.backend.from_array(local, dims=[local_dims, local_dims])
        embedded = embed_on_support(chip.backend, native, support, dims)
        return local, reduction.transform_operator(ctx, jnp.asarray(chip.backend.to_array(embedded)))

    # Every collapse operator as the source chip resolves it, including the
    # captured coordinates of an earlier reduction.
    contributions = chip._collapse_contributions_with_owners(source_bases)
    port_sources: dict[str, tuple[Any, tuple[str, ...]]] = {}
    for operator, _rate, support, _source, _name, _paths, owner in contributions:
        if any(owner is port for port in affected_ports):
            port_sources[owner.label] = (operator, tuple(labels[index] for index in support) or tuple(labels))
    transformed_ports = {
        label: transform_operator(operator, support_labels)[1]
        for label, (operator, support_labels) in port_sources.items()
    }
    mode_lowering: Any | None = None
    if boundary_ports:
        mode_lowering, _ = transform_operator(boundary_ports[0]._authored_operator(chip), (mode_label,))

    for survivor_label in touching_labels:
        freq_after = jnp.real(pair_params[survivor_label]["freq_after"])
        lamb_shift = freq_after - incoming_frequencies[survivor_label]

        chi_value: Any
        if is_multi:
            # A mode coupling several survivors is a bus/coupler, not a readout mode.
            chi_value = 0.0
        else:
            mode_index = labels.index(mode_label)
            survivor_index = labels.index(survivor_label)

            def _chi() -> Any:
                from quchip.chip.analysis import kerr_entry

                values, _, labeling = _exact_eigensystem(h, dims, sectors)
                return kerr_entry(
                    mode_index, survivor_index, dims=dims, eigenvalues=values, labeling=labeling,
                )

            chi_value = DeferredValue(_chi)
        effective_params[survivor_label] = LazyEffectiveParams({
            "lamb_shift": lamb_shift,
            "purcell_rate": 0.0,
            "freq_after": freq_after,
            "chi": chi_value,
            "kappa": 0.0,
        })
    for survivor_label, coupling in survivors:
        delta = incoming_frequencies[survivor_label] - incoming_frequencies[mode_label]
        g_over_delta = abs(coupling.coupling_strength / delta)
        validity[coupling.label] = {
            "g_over_delta": g_over_delta,
            "is_valid": g_over_delta < 0.1,
            "min_block_gap": min_gap,
        }

    exchange_by_pair: dict[tuple[str, str], dict[str, Any]] = LabelKeyedDict()
    if is_multi:
        # The full matrix remains authoritative; an explicit mediated edge
        # supplies its exchange component and a target for converted flux drives.
        # Authored direct edges retain their parameters, channels and controls.
        leg_g = {
            label: sum(c.coupling_strength for endpoint, c in survivors if endpoint == label)
            for label in touching_labels
        }
        mode_freq = incoming_frequencies[mode_label]
        leg_delta = {lbl: incoming_frequencies[lbl] - mode_freq for lbl in touching_labels}

        capacitive_legs = all(isinstance(edge, (Capacitive, TunableCapacitive)) for _, edge in survivors)
        pairs = list(combinations(touching_labels, 2))
        single_pair = len(pairs) == 1
        used_labels = set(survivor_labels) | {edge.label for edge in kept_couplings}
        for label_a, label_b in pairs:
            before = ctx.h[bare_index(labels, dims, label_a), bare_index(labels, dims, label_b)]
            mediated_strength = jnp.real(pair_params[("J", label_a, label_b)] - before)
            dj_domega_c = (
                leg_g[label_a] * leg_g[label_b] / 2.0
                * (1.0 / leg_delta[label_a] ** 2 + 1.0 / leg_delta[label_b] ** 2)
            )
            zz = reduction.residual_zz(ctx, pair_params, label_a, label_b)
            pathways = reduction.pathways(ctx, pair_params, label_a, label_b)

            fresh_label = f"elim_{mode_label}" if single_pair else f"elim_{mode_label}_{label_a}_{label_b}"
            edge_label = fresh_label
            suffix = 1
            while edge_label in used_labels:
                edge_label = f"{fresh_label}_{suffix}"
                suffix += 1
            mediated: CouplingModel
            if not capacitive_legs:
                mediated = _MediatedExchange(
                    reduced[label_a], reduced[label_b], g=mediated_strength, label=edge_label,
                )
            elif mode_is_frequency_controlled:
                mediated = TunableCapacitive(
                    reduced[label_a], reduced[label_b], g_0=mediated_strength, label=edge_label,
                )
            else:
                mediated = Capacitive(
                    reduced[label_a], reduced[label_b], g=mediated_strength, label=edge_label,
                )
            kept_couplings.append(mediated)
            used_labels.add(edge_label)

            exchange_by_pair[(label_a, label_b)] = {
                "j_eff": mediated_strength,
                "dJ_domega_c": dj_domega_c,
                "between": (label_a, label_b),
                "coupling": edge_label,
                "zz": zz,
                "pathways": pathways,
            }

        notes.append(
            "Mediated exchange J = g_a*g_b/2*(1/Δ_a + 1/Δ_b) per survivor pair; mediated terms "
            "beyond exchange (e.g. coupler-induced ZZ) are a higher-order correction under "
            "method='sw' (available exactly as 'zz' under method='exact')."
        )
        effective_params["exchange"] = (
            next(iter(exchange_by_pair.values())) if single_pair else exchange_by_pair
        )

    # Retained energy coordinates lift to the survivors' authored coordinates.
    transforms = [source_bases[label].energy_vectors for label in survivor_labels]
    lift = transforms[0]
    for transform in transforms[1:]:
        lift = jnp.kron(lift, transform)
    port_replacements: dict[str, Port] = {}
    if affected_ports:
        # The transformed boundary acts on every survivor: the reduction mixes
        # the eliminated mode into all retained coordinates it couples to.
        port_targets = tuple(survivor_labels) if len(survivor_labels) > 1 else survivor_labels[0]
        survivor_dims = tuple(chip[label].local_space().dimension for label in survivor_labels)
        for port in affected_ports:
            port_operator: Any = lift @ transformed_ports[port.label] @ lift.conj().T
            source_operator, source_labels = port_sources[port.label]
            port_changes = None if sectors is None else authored_excitation_changes(
                source_operator, source_labels, chip.backend, source_bases,
            )
            if port_changes is not None:
                # Declared so the frame check keeps one band when the payload is traced.
                port_operator = PhysicsExpr.from_matrix(
                    port_operator, labels=tuple(survivor_labels), dims=survivor_dims,
                    name="transformed_port", excitation_changes=port_changes,
                )
            port_replacements[port.label] = Port(
                port_targets,
                rate=port.rate_value(chip),
                operator=port_operator,
                phase=port.phase,
                label=port.label,
            )
        notes.append(
            f"Transformed port(s) {sorted(port_replacements)} through the {method} reduction; "
            "the PortNetwork scattering and existing reference sections are retained."
        )
        if carried_ports:
            notes.append(
                f"Port(s) {sorted(port.label for port in carried_ports)} from an earlier reduction reach "
                f"{mode_label!r} only through its dressing; like inherited channels, they drop their direct "
                f"scattering through {mode_label!r}, of order rate·|<0|L|1>|²/Δ."
            )

    final = rebuild_chip(
        chip,
        devices=reduced_devices,
        couplings=kept_couplings,
        port_replacements=port_replacements,
        effective_terms=(),
        baths=(),
    )
    # Keep every computed correction in the retained product coordinates.
    # Scalar summaries and convenient exchange edges are only a decomposition
    # of this matrix; they must not determine which elements survive.
    retained_h = reduction.retained_hamiltonian(ctx)
    final_resolved = final.resolve(frame="lab", approximation=approximation)
    final_matrix = jnp.asarray(final_resolved.hamiltonian().matrix(backend=final.backend), dtype=complex)
    final_lift = final_resolved.bases[survivor_labels[0]].vectors
    for label in survivor_labels[1:]:
        final_lift = jnp.kron(final_lift, final_resolved.bases[label].vectors)
    correction = lift @ retained_h @ lift.conj().T - final_lift @ final_matrix @ final_lift.conj().T
    inherited_channels = []
    channel_changes: dict[str, frozenset[int]] = {}
    # Internal loss of the eliminated mode as the damping of <a> near vacuum,
    # -2 Re <0|D†[L](a)|1> per channel: lowering counts +rate, raising -rate,
    # pure dephasing +rate.
    internal_rate: Any = 0.0

    def inherit_channel(channel: CollapseChannel, support_labels: tuple[str, ...], name: str) -> None:
        local, transformed = transform_operator(channel.operator, support_labels)
        rate = materialize_expr(channel.rate, chip.backend)
        inherited_channels.append(CollapseChannel(
            lift @ transformed @ lift.conj().T,
            rate, name,
        ))
        if sectors is not None:
            changes = authored_excitation_changes(channel.operator, support_labels, chip.backend, source_bases)
            if changes is not None:
                channel_changes[name] = changes
        if support_labels == (mode_label,):
            nonlocal internal_rate
            p_index = np.flatnonzero(ctx.p_mask)
            ground = basis_row(p_index, labels, dims)
            for survivor in touching_labels:
                row = basis_row(p_index, labels, dims, survivor)
                effective_params[survivor]["purcell_rate"] += rate * jnp.abs(transformed[ground, row]) ** 2
                effective_params[survivor]["kappa"] += rate * jnp.abs(local[0, 1]) ** 2
            if mode_lowering is not None:
                jump = jnp.asarray(local)
                number = jump.conj().T @ jump
                adjoint = jump.conj().T @ mode_lowering @ jump - 0.5 * (number @ mode_lowering + mode_lowering @ number)
                internal_rate = internal_rate - 2.0 * rate * jnp.real(adjoint[0, 1])

    removed_owners = {id(mode), *(id(coupling) for coupling in touching)}
    for operator, rate, support, owner_label, name, _paths, owner in contributions:
        if id(owner) in removed_owners or isinstance(owner, EffectiveTerms):
            support_labels = tuple(labels[index] for index in support) if support else tuple(labels)
            inherit_channel(CollapseChannel(operator, rate, name), support_labels,
                            name if owner is mode else f"{owner_label}.{name}")
    source_lift = reduce(jnp.kron, (source_bases[label].energy_vectors for label in labels))
    step_embedding = source_lift @ reduction.embedding(ctx) @ lift.conj().T
    projected_baths = []
    for bath in chip.baths:
        copied = bath.copy()
        copied._retained = {}
        for label, frequency, lowering, number in bath._target_operators(chip, source_bases):
            matrices = [jnp.asarray(chip.backend.to_array(materialize_expr(op, chip.backend)))
                        for op in (lowering, number)]
            if bath._retained is None:
                for terms in chip.effective_terms:
                    if terms.projection is not None:
                        matrices = [jnp.asarray(chip.backend.to_array(materialize_expr(
                            terms.projection.apply(op, tuple(labels)), chip.backend))) for op in matrices]
            copied._retained[label] = (frequency, *[step_embedding.conj().T @ op @ step_embedding
                                                   for op in matrices])
        projected_baths.append(copied)
    projection = OperatorProjection.capture(
        chip, tuple(survivor_labels), tuple(final.authored_dims),
        step_embedding,
    ).with_current_operators(tuple(f"port:{label}" for label in port_replacements))
    order = "second-order" if method == "sw" else "exact projected"
    notes.append(f"Retained the full {order} Hamiltonian correction and each transformed channel "
                 "from the removed components; inherited decay remains collective and separate "
                 "from intrinsic survivor noise.")
    terms = EffectiveTerms(tuple(survivor_labels), tuple(final.authored_dims), correction,
                           tuple(inherited_channels), label=f"retained_{mode_label}", projection=projection,
                           notes=(*inherited_notes(chip), *notes),
                           excitation_changes=None if sectors is None else channel_changes)
    final = rebuild_chip(chip, devices=final.devices, couplings=final.couplings,
                         port_replacements=port_replacements,
                         effective_terms=(*final.effective_terms, terms), baths=projected_baths)
    if boundary_ports:
        # The transformed port carries the survivors' emission; the eliminated
        # mode's own reflection, S_r(f), stays on the external plane.
        network = final.port_network
        assert network is not None and reflection_plane is not None
        section_label = f"{mode_label}_reflection"
        suffix = 1
        while section_label in {component.label for component in network.components}:
            section_label = f"{mode_label}_reflection_{suffix}"
            suffix += 1
        section = network.mode_reflection(
            section_label,
            freq=incoming_frequencies[mode_label],
            external_rate=boundary_ports[0].rate_value(chip),
            internal_rate=internal_rate,
            reference_freq=incoming_frequencies[touching_labels[0]],
        )
        network._insert_reference_section(reflection_plane, section)
        notes.append(
            f"Kept the reflection of {mode_label!r} on plane {reflection_plane!r} as reference section "
            f"{section_label!r}. The reduced scattering misses only the frequency dependence of the "
            "Purcell coupling across a sweep, of order (g/Δ)²κ/Δ; the section's internal-loss bath is vacuum."
        )
    source_factors = tuple(
        source_bases[label].vectors.conj().T @ source_bases[label].energy_vectors for label in labels
    )
    target_to_solver = final_lift.conj().T @ lift
    mapping = ReductionMap(
        source_labels=tuple(labels), source_dims=tuple(dims),
        target_labels=tuple(survivor_labels), target_dims=tuple(final_resolved.dims),
        _backend=chip.backend,
        _embedding=DeferredValue(
            lambda: reduce(jnp.kron, source_factors) @ reduction.embedding(ctx) @ target_to_solver.conj().T
        ),
    )
    reattach_equipment(
        chip,
        final,
        equipment,
        survivor_lines,
        retarget_plan,
        mode_label=mode_label,
        result_kind=result_kind,
        edges=exchange_by_pair if is_multi else None,
        notes=notes,
    )
    return EliminationResult(chip=final, effective_params=effective_params, validity=validity,
                             notes=notes, mapping=mapping)


register_elimination_target(EliminationTarget(
    kind="device",
    claims=lambda chip, target: resolve_label(target) in chip.device_map,
    reduce=reduce_device,
))
