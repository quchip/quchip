"""Device-target elimination: adiabatic reduction of a far-detuned mode.

A mode that touches **one** survivor (leaf) contributes a kept Hamiltonian correction, plus
inherited channels when the removed components dissipate. Surviving devices keep their
authored parameters and local bases. A mode can touch **two or more** survivors: a bus /
tunable-coupler (bridge), or several survivors at the same time. Such a mode also induces a
mediated exchange ``J = g_a g_b / 2 · (1/Δ_a + 1/Δ_b)`` between every survivor pair, which its
own edge represents. Authored direct couplings keep their parameters, channels and controls.

For capacitive legs, a fixed eliminated mode emits a :class:`~quchip.chip.couplings.Capacitive`.
A frequency-controlled mode (or a direct edge that is already modulable) emits a
:class:`~quchip.chip.couplings.TunableCapacitive`. Other interactions emit a first-transition
exchange edge, and the kept correction holds the remaining elements.

The reduction route (``method="sw"`` / ``method="exact"``) is a
:class:`~quchip.chip.transformations.methods.ReductionMethod` strategy. This
module registers a device-kind
:class:`~quchip.chip.transformations.dispatch.EliminationTarget` at import time,
so :func:`~quchip.chip.transformations.dispatch.eliminate` dispatches any device
label here and you need not import this module directly.
"""
# The generic P/Q partitioning kernels are in `quchip.chip.sw`. This module owns
# the fold, which reads a route's reduced parameters into a rebuilt chip and
# retargets stranded control lines.

from __future__ import annotations

from dataclasses import replace
from functools import reduce
from itertools import combinations
from math import prod
from typing import TYPE_CHECKING, Any, Mapping

import jax
import numpy as np

from quchip.utils.jax_utils import concrete_array_module, contains_tracer
from quchip.utils.values import DeferredValue
from quchip.chip.couplings import Capacitive, TunableCapacitive
from quchip.chip.effective import (
    EffectiveTerms,
    OperatorProjection,
    _in_device_order,
    authored_excitation_changes,
    conserves_excitation_number,
    lineage,
    lineage_labels,
    retained_support,
    transport_retained_operator,
)
from quchip.declarative.dissipation import CollapseChannel
from quchip.chip.ports import Port
from quchip.engine.bands import embed_on_support
from quchip.engine.assembly import _analysis_matrix_ghz
from quchip.chip.sw import (
    _WORKING_PRECISION,
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
    from quchip.chip.coupling_base import BaseCoupling


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
        for key in terms.projection.override_keys() if key.startswith("port:")
    }


def _structure_probe(paths: tuple[str, ...]) -> dict[str, float]:
    """Return a different generic value for each coupling parameter, drawn from a fixed seed.

    A diagonal entry that depends on the parameters vanishes at such a point
    only by coincidence. Equal values could cancel, as in ``(x - y) n_a n_b``.
    The patch then follows which entries can be nonzero, not their values, so
    traced and concrete reductions read the same patch.
    """
    values = np.random.default_rng(0).uniform(0.5, 1.5, len(paths))
    return dict(zip(paths, values.tolist()))


def _level_dependent(chip: "Chip", operator: Any, labels: tuple[str, ...], label: str, bases: Any,
                     bindings: Mapping[str, Any] | None = None) -> bool:
    """Return whether the energy-diagonal part of ``operator`` on ``labels`` can depend on the level of ``label``.

    Such a part shifts the energies of ``label`` by the state of the other
    devices, so it changes the Schrieffer-Wolff denominators. A diagonal that
    stays traced counts as dependent.
    """
    from quchip.engine.assembly import _project_on_support, _support_semantic_transform

    support = tuple(chip.device_index(name) for name in labels)
    backend = chip.backend
    with jax.ensure_compile_time_eval():
        local = materialize_expr(operator, backend, bindings=dict(bindings or {}), local_bases=bases)
        matrix = _array(backend.to_array(_project_on_support(chip, local, support, bases, backend)))
        transform = _support_semantic_transform(chip, support, bases)
        if transform is not None:
            matrix = transform.conj().T @ matrix @ transform
        dims = tuple(bases[name].resolved_dim for name in labels)
        xp = concrete_array_module(matrix)
        diagonal = xp.real(xp.diagonal(matrix)).reshape(dims)
    if contains_tracer(diagonal):
        return True
    diagonal = np.moveaxis(np.asarray(diagonal), labels.index(label), 0)
    return bool(not np.all(np.isfinite(diagonal)) or np.max(np.abs(diagonal - diagonal[:1])) > _WORKING_PRECISION)


def _coupling_level_dependent(chip: "Chip", coupling: "BaseCoupling", label: str, bases: Any) -> bool:
    """Return whether a coupling's energy-diagonal part can depend on the level of ``label``.

    The coupling is evaluated with a different generic value for each parameter.
    """
    operator = coupling.interaction_hamiltonian()
    probe = _structure_probe(tuple(operator.parameter_paths())) if isinstance(operator, PhysicsExpr) else {}
    return _level_dependent(chip, operator, (coupling.device_a_label, coupling.device_b_label), label, bases, probe)


def _holds_hamiltonian(terms: EffectiveTerms) -> bool:
    """Return whether effective terms carry a Hamiltonian that can be nonzero."""
    return contains_tracer(terms.hamiltonian) or bool(np.any(np.asarray(terms.hamiltonian) != 0.0))


def _local_patch(chip: "Chip", mode_label: str,
                 bases: Any) -> tuple[tuple[str, ...], tuple[EffectiveTerms, ...]]:
    """Return the devices that a local reduction of ``mode_label`` reads, and the effective Hamiltonians it absorbs.

    The core is the mode and every device that shares a coupling, an effective
    Hamiltonian or a port with it. The generator acts only on the core. The
    patch adds the far device of every coupling, and the full support of every
    effective Hamiltonian, whose energy-diagonal part depends on a core level.
    A port pair whose series composition generates a Hamiltonian on a core
    device joins whole. With that, the patch generator equals the full-chip
    generator. The patch also holds every device that the operators of the
    mode's ports act on. A mode without a neighbour adds the first other
    device, which carries the projection onto the mode's ground state. The
    patch absorbs the single-device effective Hamiltonians on its devices, and
    the others that touch the core inside the part that the generator needs.
    Devices are returned in chip order.
    """
    statics = [terms for terms in chip.effective_terms if _holds_hamiltonian(terms)]
    core = {mode_label}
    for coupling in chip.couplings:
        if mode_label in (coupling.device_a_label, coupling.device_b_label):
            core.update((coupling.device_a_label, coupling.device_b_label))
    for terms in statics:
        if mode_label in terms.labels:
            core.update(terms.labels)
    ports = [port for port in chip.ports if mode_label in port.resolve_targets(chip)]
    for port in ports:
        core.update(port.resolve_targets(chip))
    patch = set(core)
    if core == {mode_label}:
        patch.update([device.label for device in chip.devices if device.label != mode_label][:1])
    if chip.port_network is not None:
        for support in chip.port_network.dynamical_supports(chip):
            if not core.isdisjoint(support):
                patch.update(support)
    for coupling in chip.couplings:
        ends = (coupling.device_a_label, coupling.device_b_label)
        inner = [label for label in ends if label in core]
        outer = [label for label in ends if label not in patch]
        if len(inner) == 1 and outer and _coupling_level_dependent(chip, coupling, inner[0], bases):
            patch.update(outer)
    for terms in statics:
        inner = [label for label in terms.labels if label in core]
        if inner and not patch.issuperset(terms.labels) and any(
            _level_dependent(chip, terms.expression(), terms.labels, label, bases) for label in inner
        ):
            patch.update(terms.labels)
    required = set(patch)
    for port in ports:
        patch.update(retained_support(chip, tuple(port.resolve_targets(chip)), f"port:{port.label}"))
    absorbed = tuple(terms for terms in statics if patch.issuperset(terms.labels) and (
        len(terms.labels) == 1 or (not core.isdisjoint(terms.labels) and required.issuperset(terms.labels))))
    return tuple(device.label for device in chip.devices if device.label in patch), absorbed


def _patch_chip(chip: "Chip", clone: "Chip", labels: tuple[str, ...],
                absorbed: tuple[EffectiveTerms, ...]) -> "Chip":
    """Build the patch as a chip with its internal couplings and ports and the Hamiltonians it absorbs.

    Channels and coordinate maps stay with the full chip, so the patch chip
    holds only the absorbed Hamiltonians.
    """
    from quchip.chip.partition import _build_component_chip

    network = None
    if chip.port_network is not None:
        members = set(labels)
        ports = [port.label for port in chip.port_network.ports if members.issuperset(port.resolve_targets(chip))]
        if ports:
            try:
                network = chip.port_network.restrict(ports)
            except ValueError as error:
                raise NotImplementedError(
                    f"Local elimination needs a port network that the patch {list(labels)} owns alone. {error}"
                ) from error
    patch = _build_component_chip(chip, clone, 0, list(labels), [], network)
    patch._effective_terms = tuple(
        replace(terms, channels=(), projection=None,
                excitation_changes=None if terms.excitation_changes is None else {})
        for terms in absorbed
    )
    return patch


def _single_device_parts(matrix: Any, dims: tuple[int, ...]) -> tuple[Any, list[Any], Any]:
    """Split a product-space matrix into its mean, its traceless single-device parts and the rest.

    The mean is the normalized trace. The part on a device is the normalized
    partial trace onto that device minus the mean. The rest has no
    single-device content, so each of its partial traces onto one device is
    zero.
    """
    xp = concrete_array_module(matrix)
    total = prod(dims)
    mean = xp.trace(matrix) / total
    parts = []
    rest = matrix - mean * xp.eye(total)
    for axis, dim in enumerate(dims):
        before, after = prod(dims[:axis]), prod(dims[axis + 1:])
        reduced = xp.einsum("iajibj->ab", matrix.reshape(before, dim, after, before, dim, after)) * (dim / total)
        part = reduced - mean * xp.eye(dim)
        parts.append(part)
        rest = rest - xp.kron(xp.kron(xp.eye(before), part), xp.eye(after))
    return mean, parts, rest


def _unique_label(base: str, taken: set[str]) -> str:
    label, suffix = base, 1
    while label in taken:
        label = f"{base}_{suffix}"
        suffix += 1
    taken.add(label)
    return label


def _magnitude(matrix: Any) -> float | None:
    """Return the largest entry magnitude of a concrete matrix, or ``None`` for a traced one."""
    return None if contains_tracer(matrix) else float(np.max(np.abs(np.asarray(matrix))))


# A concrete part of a local correction whose entries stay below this
# multiple of the patch Hamiltonian's largest entry is round-off.
_ROUND_OFF = 1e3 * float(np.finfo(float).eps)


def _local_effective_terms(chip: "Chip", mode_label: str, taken: set[str], *, absorbed: tuple[EffectiveTerms, ...],
                           joined: tuple[EffectiveTerms, ...],
                           patch: tuple[str, ...], survivors: tuple[str, ...], survivor_dims: tuple[int, ...],
                           correction: Any, energy_dims: tuple[int, ...], vectors: tuple[Any, ...],
                           scale: float | None, carried: list[Any], projection: OperatorProjection,
                           notes: tuple[str, ...], conserving: bool) -> tuple[EffectiveTerms, ...]:
    """Return the chip's effective terms after a local reduction.

    ``correction`` is in the survivors' energy coordinates. Its single-device
    parts are stored on their own devices, and the rest on the patch
    survivors. A concrete part at round-off level is dropped. Each carried
    channel joins the terms of its support. The new map is stored once, on the
    terms of the patch survivors. Its parents are the maps of the ``joined``
    terms, which reach the patch. Earlier terms keep their Hamiltonians unless
    the patch absorbs them. They lose the channels that this step carries and
    the maps that become parents. Maps of other regions stay in place.
    """
    absorbed_ids, joined_ids = {id(terms) for terms in absorbed}, {id(terms) for terms in joined}
    taken = taken | {terms.label for terms in chip.effective_terms}
    head_label = _unique_label(f"retained_{mode_label}", taken)
    kept: list[EffectiveTerms] = []
    dropped_notes: list[str] = []
    for terms in chip.effective_terms:
        holds = id(terms) not in absorbed_ids and _holds_hamiltonian(terms)
        channels = () if not set(patch).isdisjoint(terms.labels) else terms.channels
        own_map = None if id(terms) in joined_ids else terms.projection
        if not holds and not channels and own_map is None:
            dropped_notes.extend(f"{terms.label}: {note}" for note in terms.notes)
            continue
        if own_map is not terms.projection or len(channels) != len(terms.channels):
            declared, names = terms.excitation_changes, {channel.name for channel in channels}
            terms = replace(terms, channels=channels, projection=own_map, excitation_changes=None
                            if declared is None else
                            {name: changes for name, changes in declared.items() if name in names})
        kept.append(terms)
    xp = concrete_array_module(correction)
    tolerance = None if scale is None or contains_tracer(correction) else _ROUND_OFF * scale
    mean, parts, rest = _single_device_parts(correction, energy_dims)
    added = []
    for index, (label, dim, part, vector) in enumerate(zip(survivors, survivor_dims, parts, vectors)):
        if index == 0:
            part = part + mean * xp.eye(energy_dims[0])
        if tolerance is not None and np.max(np.abs(part)) <= tolerance:
            continue
        added.append(EffectiveTerms((label,), (dim,), vector @ part @ vector.conj().T,
                                    label=_unique_label(f"retained_{mode_label}_{label}", taken),
                                    excitation_changes={} if conserving else None))
    if len(survivors) == 1 or (tolerance is not None and np.max(np.abs(rest)) <= tolerance):
        remainder = np.zeros((prod(survivor_dims),) * 2, dtype=complex)
    else:
        lift = _kron(*vectors)
        remainder = lift @ rest @ lift.conj().T
    # Carried channels can cross any earlier map, so their level changes hold
    # only when every map conserves the total level index.
    lineage_conserves = conserving and all(
        t.excitation_changes is not None for t in chip.effective_terms if t.projection is not None
    )
    groups: dict[tuple[str, ...], list[Any]] = {}
    for channel, labels, dims, changes in carried:
        groups.setdefault(labels, []).append((channel, dims, changes))
    own = groups.pop(survivors, [])
    for labels, members in groups.items():
        dims = members[0][1]
        added.append(EffectiveTerms(
            labels, dims, np.zeros((prod(dims),) * 2, dtype=complex), tuple(channel for channel, _, _ in members),
            label=_unique_label(f"retained_{mode_label}_{'_'.join(labels)}", taken),
            excitation_changes={channel.name: changes for channel, _, changes in members if changes is not None}
            if lineage_conserves else None,
        ))
    head = EffectiveTerms(
        survivors, survivor_dims, remainder, tuple(channel for channel, _, _ in own), label=head_label,
        projection=projection, notes=(*dropped_notes, *notes),
        excitation_changes={channel.name: changes for channel, _, changes in own if changes is not None}
        if lineage_conserves else None,
    )
    return (*kept, *added, head)


def _embed_patch_map(embedding: Any, labels: tuple[str, ...], dims: tuple[int, ...],
                     target_labels: tuple[str, ...], target_dims: tuple[int, ...],
                     source: tuple[tuple[str, int], ...], target: tuple[tuple[str, int], ...]) -> Any:
    """Extend a patch map by the identity on the other devices, in chip order.

    ``source`` and ``target`` list every device of the full source and
    reduced chips with its dimension.
    """
    xp = concrete_array_module(embedding)
    spectators = tuple((label, dim) for label, dim in source if label not in labels)
    size = prod(dim for _, dim in spectators)
    full = xp.kron(xp.asarray(embedding), xp.eye(size, dtype=complex))
    rows = labels + tuple(label for label, _ in spectators)
    columns = target_labels + tuple(label for label, _ in spectators)
    shape = dims + tuple(dim for _, dim in spectators) + target_dims + tuple(dim for _, dim in spectators)
    order = tuple(rows.index(label) for label, _ in source)
    order += tuple(len(rows) + columns.index(label) for label, _ in target)
    full = xp.transpose(full.reshape(shape), order)
    return full.reshape(prod(dim for _, dim in source), prod(dim for _, dim in target))


# A concrete reduction runs on the host, where it compiles no device programs;
# a traced one stays in JAX.
def _array(value: Any) -> Any:
    return concrete_array_module(value).asarray(value, dtype=complex)


def _kron(*factors: Any) -> Any:
    xp = concrete_array_module(factors)
    return reduce(xp.kron, (xp.asarray(factor) for factor in factors))


def _real(value: Any) -> Any:
    return concrete_array_module(value).real(value)


def _unit_exchange_element(edge: CouplingModel, bases: Any, backend: Any) -> Any:
    """Return the edge's ``<1_a 0_b|H|0_a 1_b>`` per unit strength, in its endpoints' energy coordinates.

    A capacitive edge, ``strength * Q_a Q_b``, gives ``conj(<0|Q_a|1>) <0|Q_b|1>``.
    Each factor comes from the charge operator that the edge's interaction
    receives, one endpoint at a time, so no two-device product space is formed.
    The factor is 1 for Duffing and resonator endpoints. The first-transition
    exchange edge acts between the lowest energy levels and gives 1.
    """
    if isinstance(edge, _MediatedExchange):
        return 1.0
    elements = []
    for ops in edge._endpoint_ops():
        charge = _array(backend.to_array(materialize_expr(ops.charge, backend)))
        vectors = bases[ops.label].energy_vectors
        elements.append(vectors[:, 0].conj() @ charge @ vectors[:, 1])
    return elements[0].conj() * elements[1]


def _in_edge_units(element: Any, unit: Any) -> Any:
    """Return the real edge strength that reproduces ``element`` from the edge's ``unit`` element.

    A survivor whose charge operator has no ``0-1`` element gives a zero
    ``unit``. The edge then carries no strength, and the retained correction
    keeps the whole element. The inner guard keeps the gradient finite.
    """
    xp = concrete_array_module(element, unit)
    nonzero = unit != 0.0
    return xp.where(nonzero, xp.real(element / xp.where(nonzero, unit, 1.0)), 0.0)[()]


def reduce_device(chip: "Chip", target: Any, method: str, *, local: bool = False) -> EliminationResult:
    """Adiabatically eliminate a far-detuned device and fold its effect into the survivors.

    See :func:`~quchip.chip.transformations.dispatch.eliminate` for the full
    physics contract and the ``method`` semantics.

    Parameters
    ----------
    chip
        Source chip (never mutated).
    target
        The device to eliminate, as a label string or object.
    method
        The reduction route (``"sw"`` or ``"exact"``), already validated.
    local
        Read only the patch around the device. Every numerical step then runs
        on the patch, and the rest of the chip stays unchanged.

    Returns
    -------
    EliminationResult
    """
    # The registry guarantees that `target` names a device on `chip`, and
    # `eliminate` already validated `method`.
    mode_label = resolve_label(target)
    if local and method != "sw":
        raise NotImplementedError(
            "local=True implements method='sw' only. The exact route diagonalizes the whole chip, so its "
            "dressed states are not local."
        )
    if local and chip.baths:
        raise NotImplementedError("local=True does not yet transform baths. Use local=False for a chip with baths.")
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
    upstream_ports: tuple[str, ...] = ()
    downstream_ports: tuple[str, ...] = ()
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
        reflection_plane, upstream_ports, downstream_ports = chip.port_network._line_exposure(boundary_ports[0].label)
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
    # Every numerical step reads `numeric`: the whole chip, or the patch whose
    # generator equals the full-chip generator (PHYSICS.md §10.7).
    numeric, chip_bases = chip, None
    absorbed: tuple[EffectiveTerms, ...] = ()
    if local:
        from quchip.engine.assembly import _resolve_system

        chip_bases = _resolve_system(chip, chip.backend).bases
        patch, absorbed = _local_patch(chip, mode_label, chip_bases)
        numeric = _patch_chip(chip, reduced, patch, absorbed)
        notes.append(
            f"Reduced locally from the patch {list(patch)}. Devices, couplings and effective Hamiltonians "
            "outside it are unchanged, and chi describes the patch alone. Retained channels that act on the "
            "patch follow its map."
        )
    numeric_mode = numeric[mode_label]
    numeric_touching = [c for c in numeric.couplings if mode_label in (c.device_a_label, c.device_b_label)]
    h, labels, dims = bare_hamiltonian(numeric, approximation=approximation, include_network=False)
    # A structurally conserving model has no matrix elements between
    # total-excitation sectors; removing their round-off keeps both routes,
    # and every captured map, inside the sectors.
    sectors = excitation_sectors(dims) if conserves_excitation_number(numeric, approximation) else None
    if sectors is not None:
        h = concrete_array_module(h).where(sectors[:, None] == sectors[None, :], h, 0.0)
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
        label: _real(h[bare_index(labels, dims, label), bare_index(labels, dims, label)] - h[0, 0])
        for label in labels
    }

    source_bases = numeric.resolve(frame="lab").bases

    def transform_operator(operator: Any, support_labels: tuple[str, ...]) -> tuple[Any, Any]:
        local = _array(chip.backend.to_array(materialize_expr(operator, chip.backend)))
        transform = _kron(*(source_bases[label].energy_vectors for label in support_labels))
        local = transform.conj().T @ local @ transform
        support = tuple(labels.index(label) for label in support_labels)
        local_dims = [dims[index] for index in support]
        native = chip.backend.from_array(local, dims=[local_dims, local_dims])
        embedded = embed_on_support(chip.backend, native, support, dims)
        return local, reduction.transform_operator(ctx, _array(chip.backend.to_array(embedded)))

    # Every collapse operator as the source chip resolves it, including the
    # captured coordinates of an earlier reduction. A local reduction reads
    # only the removed components and the mode's ports, from the full chip.
    if local:
        contributions = chip._collapse_contributions_with_owners(
            chip_bases, owners=(chip[mode_label], *touching, *affected_ports),
        )
        contribution_labels = tuple(device.label for device in chip.devices)
        removed_owners = {id(chip[mode_label]), *(id(coupling) for coupling in touching)}
    else:
        contributions = numeric._collapse_contributions_with_owners(source_bases)
        contribution_labels = tuple(labels)
        removed_owners = {id(numeric_mode), *(id(coupling) for coupling in numeric_touching)}
    affected_port_labels = {port.label for port in affected_ports}
    port_sources: dict[str, tuple[Any, tuple[str, ...]]] = {}
    for operator, _rate, support, _source, _name, _paths, owner in contributions:
        if isinstance(owner, Port) and owner.label in affected_port_labels:
            port_sources[owner.label] = (operator, tuple(contribution_labels[index] for index in support)
                                         or contribution_labels)
    transformed_ports = {
        label: transform_operator(operator, support_labels)[1]
        for label, (operator, support_labels) in port_sources.items()
    }
    mode_lowering: Any | None = None
    if boundary_ports:
        mode_lowering, _ = transform_operator(boundary_ports[0]._authored_operator(chip), (mode_label,))

    for survivor_label in touching_labels:
        freq_after = _real(pair_params[survivor_label]["freq_after"])
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
    # Resolved exchange element <1_s|H|1_mode> of each touching survivor, in
    # the energy coordinates the route reads. It differs from the authored
    # strength when a charge operator has a non-unit or complex 0-1 element.
    mode_row = bare_index(labels, dims, mode_label)
    leg_element = {label: ctx.h[bare_index(labels, dims, label), mode_row] for label in touching_labels}
    for survivor_label, coupling in survivors:
        delta = incoming_frequencies[survivor_label] - incoming_frequencies[mode_label]
        g_over_delta = abs(leg_element[survivor_label] / delta)
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
        mode_freq = incoming_frequencies[mode_label]
        leg_delta = {lbl: incoming_frequencies[lbl] - mode_freq for lbl in touching_labels}

        capacitive_legs = all(isinstance(edge, (Capacitive, TunableCapacitive)) for _, edge in survivors)
        edge_type: type[CouplingModel]
        if not capacitive_legs:
            edge_type, strength_name = _MediatedExchange, "g"
        elif mode_is_frequency_controlled:
            edge_type, strength_name = TunableCapacitive, "g_0"
        else:
            edge_type, strength_name = Capacitive, "g"
        pairs = list(combinations(touching_labels, 2))
        single_pair = len(pairs) == 1
        used_labels = set(survivor_labels) | {edge.label for edge in kept_couplings}
        for label_a, label_b in pairs:
            fresh_label = f"elim_{mode_label}" if single_pair else f"elim_{mode_label}_{label_a}_{label_b}"
            edge_label = fresh_label
            suffix = 1
            while edge_label in used_labels:
                edge_label = f"{fresh_label}_{suffix}"
                suffix += 1

            # Strengths use the units of an edge authored between the
            # survivors: each resolved element is divided by the emitted
            # edge's own element per unit strength.
            unit = _unit_exchange_element(
                edge_type(reduced[label_a], reduced[label_b], **{strength_name: 1.0}, label=edge_label),
                source_bases, chip.backend,
            )
            before = ctx.h[bare_index(labels, dims, label_a), bare_index(labels, dims, label_b)]
            mediated_strength = _in_edge_units(pair_params[("J", label_a, label_b)] - before, unit)
            dj_domega_c = (
                _in_edge_units(leg_element[label_a] * leg_element[label_b].conj(), unit) / 2.0
                * (1.0 / leg_delta[label_a] ** 2 + 1.0 / leg_delta[label_b] ** 2)
            )
            zz = reduction.residual_zz(ctx, pair_params, label_a, label_b)
            pathways = reduction.pathways(ctx, pair_params, label_a, label_b)

            mediated = edge_type(reduced[label_a], reduced[label_b], **{strength_name: mediated_strength},
                                 label=edge_label)
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
            "Mediated exchange J = g_a*conj(g_b)/2*(1/Δ_a + 1/Δ_b) per survivor pair, with resolved leg "
            "elements g_s = <1_s|H|1_mode>. Each emitted edge carries J in its own units. Mediated terms "
            "beyond exchange, e.g. coupler-induced ZZ, are a higher-order correction under method='sw'. "
            "method='exact' reports them exactly as 'zz'."
        )
        effective_params["exchange"] = (
            next(iter(exchange_by_pair.values())) if single_pair else exchange_by_pair
        )

    # Retained energy coordinates lift to the survivors' authored coordinates.
    numeric_survivors = [label for label in labels if label != mode_label]
    lift = _kron(*(source_bases[label].energy_vectors for label in numeric_survivors))
    port_replacements: dict[str, Port] = {}
    if affected_ports:
        # The transformed boundary acts on every survivor that the reduction
        # reads: it mixes the eliminated mode into all retained coordinates it couples to.
        port_targets = tuple(numeric_survivors) if len(numeric_survivors) > 1 else numeric_survivors[0]
        survivor_dims = tuple(chip[label].local_space().dimension for label in numeric_survivors)
        for port in affected_ports:
            port_operator: Any = lift @ transformed_ports[port.label] @ lift.conj().T
            source_operator, source_labels = port_sources[port.label]
            port_changes = None if sectors is None else authored_excitation_changes(
                source_operator, source_labels, chip.backend, source_bases if chip_bases is None else chip_bases,
            )
            if port_changes is not None:
                # Declared so the frame check keeps one band when the payload is traced.
                port_operator = PhysicsExpr.from_matrix(
                    port_operator, labels=tuple(numeric_survivors), dims=survivor_dims,
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

    kept_numeric = set(numeric_survivors)
    numeric_final = rebuild_chip(
        numeric,
        devices=[reduced[label] for label in numeric_survivors],
        couplings=[c for c in kept_couplings if kept_numeric.issuperset((c.device_a_label, c.device_b_label))],
        port_replacements=port_replacements,
        effective_terms=(),
        baths=(),
    )
    # Keep every computed correction in the retained product coordinates.
    # Scalar summaries and convenient exchange edges are only a decomposition
    # of this matrix; they must not determine which elements survive.
    retained_h = reduction.retained_hamiltonian(ctx)
    final_resolved = numeric_final.resolve(frame="lab", approximation=approximation)
    final_matrix = _array(_analysis_matrix_ghz(final_resolved, include_network=False))
    final_lift = _kron(*(final_resolved.bases[label].vectors for label in numeric_survivors))
    correction = lift @ retained_h @ lift.conj().T - final_lift @ final_matrix @ final_lift.conj().T
    inherited_channels = []
    channel_changes: dict[str, frozenset[int]] = {}
    source_lift = _kron(*(source_bases[label].energy_vectors for label in labels))
    step_embedding = source_lift @ reduction.embedding(ctx) @ lift.conj().T
    # A local map keeps each earlier map that reaches the patch as a parent,
    # in the order they apply. Maps of other regions act on other devices, so
    # they stay on their own effective terms and the partition keeps them with
    # their devices. The map carries each channel that acts on the patch.
    # Devices outside the patch keep their coordinates, so the channel keeps
    # its own support.
    joined = tuple(t for t in chip.effective_terms
                   if t.projection is not None and not lineage_labels(t.projection).isdisjoint(labels))
    parents = lineage(tuple(t.projection for t in joined if t.projection is not None))
    step = OperatorProjection(tuple(labels), tuple(numeric.authored_dims), tuple(numeric_survivors),
                              tuple(numeric_final.authored_dims), step_embedding, parents=parents) if local else None
    carried: list[tuple[CollapseChannel, tuple[str, ...], tuple[int, ...], frozenset[int] | None]] = []

    def carry(operator: Any, support_labels: tuple[str, ...], support_dims: tuple[int, ...], rate: Any,
              name: str, changes: frozenset[int] | None) -> None:
        assert step is not None
        matrix, carried_labels, carried_dims = _in_device_order(
            chip, *step._embed(_array(operator), support_labels, support_dims),
        )
        carried.append((CollapseChannel(matrix, rate, name), carried_labels, carried_dims, changes))

    # Internal loss of the eliminated mode as the damping of <a> near vacuum,
    # -2 Re <0|D†[L](a)|1> per channel: lowering counts +rate, raising -rate,
    # pure dephasing +rate.
    internal_rate: Any = 0.0

    def add_internal_loss(operator: Any, support_labels: tuple[str, ...], rate: Any, *, bases: Any = None) -> None:
        nonlocal internal_rate
        if mode_lowering is None or mode_label not in support_labels:
            return
        if bases is None:
            jump = operator
            support_dims = tuple(source_bases[label].resolved_dim for label in support_labels)
        else:
            transform = _kron(*(bases[label].energy_vectors for label in support_labels))
            jump = _array(chip.backend.to_array(materialize_expr(operator, chip.backend)))
            jump = transform.conj().T @ jump @ transform
            support_dims = tuple(bases[label].resolved_dim for label in support_labels)
        lowering = _kron(*(mode_lowering if label == mode_label else np.eye(dim)
                           for label, dim in zip(support_labels, support_dims)))
        number = jump.conj().T @ jump
        adjoint = jump.conj().T @ lowering @ jump - 0.5 * (number @ lowering + lowering @ number)
        row = bare_index(list(support_labels), support_dims, mode_label)
        internal_rate = internal_rate - 2.0 * rate * _real(adjoint[0, row])

    def inherit_channel(channel: CollapseChannel, support_labels: tuple[str, ...], name: str) -> None:
        rate = materialize_expr(channel.rate, chip.backend)
        changes = None
        if sectors is not None:
            changes = authored_excitation_changes(channel.operator, support_labels, chip.backend,
                                                  source_bases if chip_bases is None else chip_bases)
        if local:
            carry(chip.backend.to_array(materialize_expr(channel.operator, chip.backend)), support_labels,
                  tuple(chip[label].local_space().dimension for label in support_labels), rate, name, changes)
            add_internal_loss(channel.operator, support_labels, rate, bases=chip_bases)
        if local and support_labels != (mode_label,):
            return
        in_energy, transformed = transform_operator(channel.operator, support_labels)
        if not local:
            add_internal_loss(in_energy, support_labels, rate)
            inherited_channels.append(CollapseChannel(
                lift @ transformed @ lift.conj().T,
                rate, name,
            ))
            if changes is not None:
                channel_changes[name] = changes
        if support_labels == (mode_label,):
            p_index = np.flatnonzero(ctx.p_mask)
            ground = basis_row(p_index, labels, dims)
            for survivor in touching_labels:
                row = basis_row(p_index, labels, dims, survivor)
                effective_params[survivor]["purcell_rate"] += rate * abs(transformed[ground, row]) ** 2
                effective_params[survivor]["kappa"] += rate * abs(in_energy[0, 1]) ** 2

    mode_owner = chip[mode_label] if local else numeric_mode
    for operator, rate, support, owner_label, name, _paths, owner in contributions:
        if id(owner) in removed_owners or isinstance(owner, EffectiveTerms):
            support_labels = (tuple(contribution_labels[index] for index in support) if support
                              else contribution_labels)
            inherit_channel(CollapseChannel(operator, rate, name), support_labels,
                            name if owner is mode_owner else f"{owner_label}.{name}")
    if local:
        # Retained channels are stored in the coordinates before this step.
        for terms in chip.effective_terms:
            if terms.channels and not set(labels).isdisjoint(terms.labels):
                declared = None if sectors is None else terms.excitation_changes
                for channel in terms.channels:
                    add_internal_loss(terms.channel_expression(channel), terms.labels,
                                      materialize_expr(channel.rate, chip.backend), bases=chip_bases)
                    carry(channel.operator, terms.labels, terms.dims, channel.rate, f"{terms.label}.{channel.name}",
                          None if declared is None else declared.get(channel.name))
    projected_baths = []
    for bath in chip.baths:
        copied = bath.copy()
        copied._retained = {}
        for label, frequency, lowering, number in bath._target_operators(chip, source_bases):
            matrices = [_array(chip.backend.to_array(materialize_expr(op, chip.backend)))
                        for op in (lowering, number)]
            if bath._retained is None:
                transported = [transport_retained_operator(chip, op, tuple(labels), tuple(chip.authored_dims))
                               for op in matrices]
                matrices = [op if moved is None else _array(moved[0]) for op, moved in zip(matrices, transported)]
            copied._retained[label] = (frequency, *[step_embedding.conj().T @ op @ step_embedding
                                                   for op in matrices])
        projected_baths.append(copied)
    projection = (step if step is not None else OperatorProjection.capture(
        numeric, tuple(numeric_survivors), tuple(numeric_final.authored_dims), step_embedding,
    )).with_current_operators(tuple(f"port:{label}" for label in port_replacements))
    order = "second-order" if method == "sw" else "exact projected"
    notes.append(f"Retained the full {order} Hamiltonian correction and each transformed channel "
                 "from the removed components; inherited decay remains collective and separate "
                 "from intrinsic survivor noise.")
    taken = {d.label for d in reduced_devices} | {c.label for c in kept_couplings}
    if local:
        effective_terms = _local_effective_terms(
            chip, mode_label, taken, absorbed=absorbed, joined=joined, patch=tuple(labels),
            survivors=tuple(numeric_survivors),
            survivor_dims=tuple(numeric_final.authored_dims),
            correction=lift.conj().T @ correction @ lift, energy_dims=tuple(dims[labels.index(label)]
                                                                            for label in numeric_survivors),
            vectors=tuple(source_bases[label].energy_vectors for label in numeric_survivors),
            scale=_magnitude(h), carried=carried, projection=projection, notes=tuple(notes),
            conserving=sectors is not None,
        )
    else:
        terms = EffectiveTerms(tuple(numeric_survivors), tuple(numeric_final.authored_dims), correction,
                               tuple(inherited_channels), label=_unique_label(f"retained_{mode_label}", taken),
                               projection=projection, notes=(*inherited_notes(numeric), *notes),
                               excitation_changes=None if sectors is None else channel_changes)
        effective_terms = (terms,)
    final = rebuild_chip(chip, devices=reduced_devices, couplings=kept_couplings,
                         port_replacements=port_replacements,
                         effective_terms=effective_terms, baths=projected_baths)
    if boundary_ports:
        # The transformed port carries survivor emission. The section keeps
        # the removed mode's scattering on the external plane.
        network = final.port_network
        assert network is not None and reflection_plane is not None
        shared_line = bool(upstream_ports or downstream_ports)
        kept_upstream = tuple(label for label in upstream_ports if label not in retained_ports)
        kept_downstream = tuple(label for label in downstream_ports if label not in retained_ports)
        inbound = bool(kept_downstream) and not kept_upstream
        section_kind = "transmission" if shared_line else "reflection"
        section_label = f"{mode_label}_{section_kind}"
        suffix = 1
        while section_label in {component.label for component in network.components}:
            section_label = f"{mode_label}_{section_kind}_{suffix}"
            suffix += 1
        lowering = _kron(*(mode_lowering if label == mode_label else np.eye(dim)
                           for label, dim in zip(labels, dims)))
        frequency, weight = reduction.dressed_transition(ctx, lowering)
        external_rate = weight * boundary_ports[0].rate_value(chip)
        if shared_line:
            section = network.mode_transmission(
                section_label, freq=frequency, external_rate=external_rate, internal_rate=weight * internal_rate,
            )
            beyond = kept_upstream if inbound else kept_downstream
            targets = [target for port in chip.ports if port.label in beyond for target in port.resolve_targets(chip)]
            ratio: Any = 0.0
            if targets:
                xp = concrete_array_module(external_rate, tuple(incoming_frequencies.values()))
                detunings = xp.asarray([incoming_frequencies[target] - incoming_frequencies[mode_label]
                                        for target in targets])
                ratio = external_rate / (2.0 * xp.pi * xp.min(xp.abs(detunings)))
            validity[section_label] = {"kappa_over_delta": ratio, "is_valid": ratio < 0.1}
        else:
            section = network.mode_reflection(
                section_label, freq=frequency, external_rate=external_rate, internal_rate=weight * internal_rate,
                reference_freq=incoming_frequencies[touching_labels[0]],
            )
        network._insert_reference_section(reflection_plane, section, reverse=inbound)
        frequency_error = ""
        if method == "sw":
            frequency_error = ("SW misses frequency shifts beyond second order, of order g⁴/Δ³ with one survivor "
                               "and 2g_1g_2J/(Δ_1Δ_2) when a coupling J joins two survivors. ")
        if shared_line:
            leg = "inbound" if inbound else "outbound"
            if method == "sw":
                frequency_error = (
                    "SW misses frequency shifts beyond second order. With one survivor, their scale is g⁴/Δ³. "
                    "A coupling J between two survivors adds shifts of order 2g_1g_2J/(Δ_1Δ_2). "
                )
            notes.append(
                f"Kept the transmission of {mode_label!r} as reference section {section_label!r} "
                f"on the {leg} leg of plane {reflection_plane!r}. "
                "It uses the dressed transition and mode-weighted external and internal rates. "
                "The section drops damping from survivor channels, of order (g/Δ)²γ_s. "
                "Purcell dispersion leaves a scattering residual of order (g/Δ)²κ_e/Δ. "
                f"{frequency_error}The section's internal-loss bath is vacuum. "
                "The leg does not change the weak-probe response from the line input. "
                "Near kept-port frequencies beyond the section, emitted or received fields err by up to "
                "the reported kappa_over_delta."
            )
        else:
            notes.append(
                f"Kept the reflection of {mode_label!r} on plane {reflection_plane!r} as reference section "
                f"{section_label!r}, using its dressed transition and mode-weighted external and internal rates. "
                "The section drops damping from survivor channels, of order (g/Δ)²γ_s. "
                "Purcell dispersion leaves a scattering residual of order (g/Δ)²κ_e/Δ. "
                f"{frequency_error}The section's internal-loss bath is vacuum."
            )
    source_factors = tuple(
        _array(source_bases[label].vectors).conj().T @ _array(source_bases[label].energy_vectors) for label in labels
    )
    target_to_solver = final_lift.conj().T @ lift

    def numeric_map() -> Any:
        return _kron(*source_factors) @ reduction.embedding(ctx) @ target_to_solver.conj().T

    if chip_bases is None:
        mapping = ReductionMap(
            source_labels=tuple(labels), source_dims=tuple(dims),
            target_labels=tuple(survivor_labels), target_dims=tuple(final_resolved.dims),
            _backend=chip.backend,
            _embedding=DeferredValue(numeric_map),
        )
    else:
        # The patch map acts as the identity on the other devices. The full
        # map is formed only when it is read.
        source_dims = tuple((d.label, chip_bases[d.label].resolved_dim) for d in chip.devices)
        target_dims = tuple((label, dim) for label, dim in source_dims if label != mode_label)
        mapping = ReductionMap(
            source_labels=tuple(label for label, _ in source_dims), source_dims=tuple(d for _, d in source_dims),
            target_labels=tuple(label for label, _ in target_dims), target_dims=tuple(d for _, d in target_dims),
            _backend=chip.backend,
            _embedding=DeferredValue(lambda: _embed_patch_map(
                numeric_map(), tuple(labels), tuple(dims), tuple(numeric_survivors), tuple(final_resolved.dims),
                source_dims, target_dims,
            )),
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
    reduce_local=lambda chip, target, method: reduce_device(chip, target, method, local=True),
))
