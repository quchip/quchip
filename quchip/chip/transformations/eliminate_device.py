"""Device-target elimination: adiabatic reduction of a far-detuned mode.

A mode touching **one** survivor (leaf) contributes a retained Hamiltonian
correction and inherited channels when the removed components dissipate.
Surviving devices keep their authored parameters and local bases. A mode touching **two or more**
survivors — bus / tunable-coupler (bridge) or several at once — additionally
induces a mediated exchange ``J = g_a g_b / 2 · (1/Δ_a + 1/Δ_b)`` between every
survivor pair, represented by its own edge. Authored direct couplings keep
their parameters, channels and controls. A fixed eliminated mode emits a
:class:`~quchip.chip.couplings.Capacitive`; a frequency-controlled mode (or an
already-modulable direct edge) emits a
:class:`~quchip.chip.couplings.TunableCapacitive`.

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
from quchip.approximations import Exact
from quchip.chip.couplings import Capacitive, TunableCapacitive
from quchip.chip.effective import EffectiveTerms, OperatorProjection
from quchip.declarative.dissipation import CollapseChannel
from quchip.chip.ports import Port
from quchip.engine.bands import embed_on_support
from quchip.chip.sw import (
    _exact_eigensystem,
    bare_hamiltonian,
    bare_index,
    basis_row,
    mode_blocks,
    purcell_rate_from,
    cross_block_gap,
)
from quchip.chip.transformations.dispatch import EliminationTarget, register_elimination_target
from quchip.chip.transformations.methods import DeviceReductionContext, lookup_reduction_method
from quchip.chip.transformations.plumbing import (
    StrandedLine,
    plan_stranded_lines,
    reattach_equipment,
    rebuild_chip,
)
from quchip.chip.transformations.result import EliminationResult, LazyEffectiveParams, ReductionMap
from quchip.control.drive import FluxDrive
from quchip.declarative.expr import materialize_expr
from quchip.devices.protocols import FrequencyControlled
from quchip.devices.resonator import Resonator
from quchip.devices.spaces import FockSpace
from quchip.utils.labeling import LabelKeyedDict, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


def mode_decay_rate(mode: Any) -> tuple[Any, bool]:
    """``(kappa, has_purcell)``: the eliminated mode's own decay rate, and whether it decays at all.

    Reads :meth:`~quchip.devices.base.BaseDevice.intrinsic_decay_rate`, which
    each device class owns — e.g. :class:`~quchip.devices.resonator.Resonator`
    combines its Q-derived photon loss with any ``T1``, matching its actual
    :meth:`~quchip.devices.resonator.Resonator.collapse_operators`. Whether a
    channel exists is a *static* decision (does the hook return ``None``?),
    never a traced-zero comparison on the resulting rate, which would
    concretize a traced value and break differentiability.
    """
    rate = mode.intrinsic_decay_rate()
    if rate is None:
        return 0.0, False
    return rate, True


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
    if affected_ports:
        if not isinstance(mode_device, Resonator):
            raise NotImplementedError(
                "Network-connected elimination currently supports a linear Resonator target; "
                f"{mode_label!r} is {type(mode_device).__name__}. Keep the boundary mode."
            )
        if not survivors:
            raise NotImplementedError(
                f"Cannot eliminate port-coupled Resonator {mode_label!r} without a coupled survivor."
            )
        if len(chip.devices) != 2:
            raise NotImplementedError(
                "Network-connected elimination currently requires exactly one survivor; "
                "multi-device effective boundary support is deferred."
            )
        survivor_device = chip[survivors[0][0]]
        if survivor_device.resolved_dimension(chip.basis) != survivor_device.local_space().dimension:
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
            for port in affected_ports
            if port.resolve_targets(chip) != (mode_label,) or port.operator is not None
        ]
        if unsupported:
            raise NotImplementedError(
                "Network-connected linear-mode elimination currently transforms the mode's "
                f"default lowering ports only; unsupported ports: {unsupported}."
            )
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
    # Survivor pairs are keyed in the chip's device order everywhere — the
    # pair extraction, the exact route, and the fold loop below — so the two
    # sides of every ("J", a, b) lookup agree no matter what order the legs
    # were scanned in. Coupling-scan order and device order genuinely differ
    # on real chips (a center mode declared before its outer neighbors).
    scanned = {lbl for lbl, _ in survivors}
    touching_labels = [lbl for lbl in labels if lbl in scanned]

    # Capture the full lab-frame matrix now; diagonalize only if chi is read.
    report_h = None
    if not is_multi and survivors:
        report_h = h if not approximation.filters_terms else bare_hamiltonian(chip, approximation=Exact())[0]

    p_mask, _ = mode_blocks(dims, labels, mode_label)
    min_gap = cross_block_gap(h, p_mask)

    ctx = DeviceReductionContext(
        mode_label=mode_label,
        survivor_labels=touching_labels,
        labels=labels,
        dims=dims,
        h=h,
        p_mask=p_mask,
    )
    pair_params = reduction.pair_parameters(ctx)
    incoming_frequencies = {
        label: jnp.real(h[bare_index(labels, dims, label), bare_index(labels, dims, label)] - h[0, 0])
        for label in labels
    }

    kappa, has_purcell = mode_decay_rate(mode)
    transformed_mode_operator: Any | None = None
    if has_purcell or affected_ports:
        mode_index = labels.index(mode_label)
        mode_operator = jnp.asarray(
            chip.backend.to_array(
                chip.backend.embed(mode.lowering_operator(), mode_index, dims)
            ),
            dtype=complex,
        )
        transformed_mode_operator = reduction.transform_operator(ctx, mode_operator)
    amplitudes: dict[str, Any] = {}
    if has_purcell:
        assert transformed_mode_operator is not None
        p_index = np.flatnonzero(ctx.p_mask)
        ground_row = basis_row(p_index, ctx.labels, ctx.dims)
        amplitudes = {
            survivor: transformed_mode_operator[
                ground_row,
                basis_row(p_index, ctx.labels, ctx.dims, survivor),
            ]
            for survivor in ctx.survivor_labels
        }

    for survivor_label in touching_labels:
        freq_after = jnp.real(pair_params[survivor_label]["freq_after"])
        lamb_shift = freq_after - incoming_frequencies[survivor_label]
        purcell_rate = purcell_rate_from(amplitudes[survivor_label], kappa) if has_purcell else 0.0

        chi_value: Any
        if is_multi:
            # A mode coupling several survivors is a bus/coupler, not a readout mode.
            chi_value = 0.0
        else:
            mode_index = labels.index(mode_label)
            survivor_index = labels.index(survivor_label)

            def _chi() -> Any:
                from quchip.chip.analysis import kerr_entry

                values, _, labeling = _exact_eigensystem(report_h, dims)
                return kerr_entry(
                    mode_index, survivor_index, dims=dims, eigenvalues=values, labeling=labeling,
                )

            chi_value = DeferredValue(_chi)
        effective_params[survivor_label] = LazyEffectiveParams({
            "lamb_shift": lamb_shift,
            "purcell_rate": purcell_rate,
            "freq_after": freq_after,
            "chi": chi_value,
            "kappa": kappa,
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
            mediated: Capacitive | TunableCapacitive
            if mode_is_frequency_controlled:
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

    port_replacements: dict[str, Port] = {}
    if affected_ports:
        assert transformed_mode_operator is not None
        port_target = survivor_labels[0]
        for port in affected_ports:
            port_replacements[port.label] = Port(
                port_target,
                rate=port.rate_value(chip),
                operator=transformed_mode_operator,
                phase=port.phase,
                label=port.label,
            )
        notes.append(
            f"Transformed port(s) {sorted(port_replacements)} through the {method} reduction; "
            "the PortNetwork scattering and exposure reference planes are retained."
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
    source_bases = chip.resolve(frame="lab").bases
    transforms = [source_bases[label].energy_vectors for label in survivor_labels]
    lift = transforms[0]
    for transform in transforms[1:]:
        lift = jnp.kron(lift, transform)
    final_resolved = final.resolve(frame="lab", approximation=approximation)
    final_matrix = jnp.asarray(final_resolved.hamiltonian().matrix(backend=final.backend), dtype=complex)
    final_lift = final_resolved.bases[survivor_labels[0]].vectors
    for label in survivor_labels[1:]:
        final_lift = jnp.kron(final_lift, final_resolved.bases[label].vectors)
    correction = lift @ retained_h @ lift.conj().T - final_lift @ final_matrix @ final_lift.conj().T
    inherited_channels = []

    def inherit_channel(channel: CollapseChannel, support_labels: tuple[str, ...], name: str) -> None:
        local = jnp.asarray(chip.backend.to_array(materialize_expr(channel.operator, chip.backend)))
        transform = source_bases[support_labels[0]].energy_vectors
        for label in support_labels[1:]:
            transform = jnp.kron(transform, source_bases[label].energy_vectors)
        local = transform.conj().T @ local @ transform
        support = tuple(labels.index(label) for label in support_labels)
        local_dims = [dims[index] for index in support]
        native = chip.backend.from_array(local, dims=[local_dims, local_dims])
        embedded = embed_on_support(chip.backend, native, support, dims)
        transformed = reduction.transform_operator(ctx, jnp.asarray(chip.backend.to_array(embedded)))
        inherited_channels.append(CollapseChannel(
            lift @ transformed @ lift.conj().T,
            materialize_expr(channel.rate, chip.backend), name,
        ))

    removed_owners = {id(mode), *(id(coupling) for coupling in touching)}
    contributions = chip._collapse_contributions_with_owners(source_bases)
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
    terms = EffectiveTerms(tuple(survivor_labels), tuple(final.authored_dims), correction,
                           tuple(inherited_channels), label=f"retained_{mode_label}", projection=projection)
    final = rebuild_chip(chip, devices=final.devices, couplings=final.couplings,
                         port_replacements=port_replacements,
                         effective_terms=(*final.effective_terms, terms), baths=projected_baths)
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
    order = "second-order" if method == "sw" else "exact projected"
    notes.append(f"Retained the full {order} Hamiltonian correction and each transformed channel "
                 "from the removed components; inherited decay remains collective and separate "
                 "from intrinsic survivor noise.")

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
