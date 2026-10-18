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

import warnings
from collections.abc import Iterator
from dataclasses import replace
from functools import reduce
from itertools import combinations, product
from math import prod
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from quchip.backend import _backend_context
from quchip.utils.jax_utils import concrete_array_module, contains_tracer
from quchip.utils.values import DeferredValue
from quchip.chip.couplings import Capacitive, TunableCapacitive
from quchip.chip.effective import (
    EffectiveTerms,
    OperatorProjection,
    _in_device_order,
    _retained_owners,
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
    _exact_eigensystem,
    bare_hamiltonian,
    bare_index,
    basis_row,
    mode_blocks,
    cross_block_gap,
    excitation_sectors,
)
from quchip.chip.transformations.dispatch import EliminationTarget, register_elimination_target
from quchip.chip.transformations.methods import (
    DeviceReductionContext,
    _sw_dressed_transition,
    lookup_reduction_method,
)
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


def _line_has_coupled_response(chip: "Chip", labels: tuple[str, ...], retained_ports: set[str]) -> bool:
    """Return whether a line meets a mode that also emits through other ports."""
    network = chip.port_network
    assert network is not None
    line = set(labels)
    ports_by_target: dict[str, set[str]] = {}
    for port in chip.ports:
        if port.label not in retained_ports:
            for target in port.resolve_targets(chip):
                ports_by_target.setdefault(target, set()).add(port.label)
    groups = [*ports_by_target.values(), *(set(mode.ports) for mode in network._mode_responses)]
    return any(len(ports) > 1 and not line.isdisjoint(ports) for ports in groups)


def _response_validity(chip: "Chip", mode_label: str, ports: set[str]) -> dict[str, Any]:
    """Compare the return path through a retained pole with the damping of the kept targets of other ports.

    In the vacuum drift ``A`` of the reduced chip, the omitted pole ``p``
    changes ``D_k = h_k + i2π(f_k - f)`` of a kept mode ``k`` by
    ``-A_kp A_pk / D_p``. The value is the largest ratio of this self-energy
    to ``h_k = -Re A_kk`` at ``2π f_k = -Im A_kk``. It is ``None`` when the
    chip has no vacuum one-excitation model.
    """
    from quchip.engine.linear_response import vacuum_drift

    model = vacuum_drift(chip)
    if model is None:
        return {"self_energy_over_width": None, "is_valid": None}
    labels, drift = model
    xp = concrete_array_module(drift)
    pole = labels.index(mode_label)
    targets = {target for port in chip.ports if port.label not in ports for target in port.resolve_targets(chip)}
    ratios = []
    for k, label in enumerate(labels):
        if label not in targets:
            continue
        coupling = xp.abs(drift[k, pole] * drift[pole, k])
        denominator = xp.abs(1j * xp.imag(drift[k, k]) - drift[pole, pole]) * -xp.real(drift[k, k])
        safe = xp.where(denominator > 0, denominator, 1.)
        ratios.append(xp.where(coupling > 0, xp.where(denominator > 0, coupling / safe, xp.inf), 0.))
    ratio: Any = xp.max(xp.stack(ratios)) if ratios else 0.0
    return {"self_energy_over_width": ratio, "is_valid": ratio < 0.1}


def _operator_terms(expr: PhysicsExpr) -> Iterator[PhysicsExpr]:
    """Yield the operator products that ``expr`` sums, without their scalar coefficients."""
    if expr.kind in ("add", "sub"):
        for arg in expr.args:
            yield from _operator_terms(arg)
    elif expr.kind == "scale":
        yield from _operator_terms(expr.args[1])
    elif expr.kind in ("tensor", "matmul"):
        for left, right in product(_operator_terms(expr.args[0]), _operator_terms(expr.args[1])):
            yield PhysicsExpr(expr.kind, (left, right), expr.labels)
    elif expr.kind == "embed":
        for inner in _operator_terms(expr.args[0]):
            yield PhysicsExpr("embed", (inner, *expr.args[1:]), expr.labels)
    else:
        yield expr


def _holds_parameter(expr: PhysicsExpr) -> bool:
    """Return whether a parameter enters ``expr`` other than through a device's energy levels."""
    if expr.kind == "level":
        return False
    return expr.kind == "parameter" or any(
        _holds_parameter(arg) for arg in expr.args if isinstance(arg, PhysicsExpr)
    )


def _level_dependent(chip: "Chip", operator: Any, labels: tuple[str, ...], label: str, bases: Any) -> bool:
    """Return whether the energy-diagonal part of ``operator`` on ``labels`` can depend on the level of ``label``.

    Such a part shifts the energies of ``label`` by the state of the other
    devices, so it changes the Schrieffer-Wolff denominators. A diagonal that
    stays traced counts as dependent.
    """
    from quchip.engine.assembly import _project_on_support, _support_semantic_transform

    support = tuple(chip.device_index(name) for name in labels)
    backend = chip.backend
    with jax.ensure_compile_time_eval():
        local = materialize_expr(operator, backend, local_bases=bases)
        matrix = _array(backend.to_array(_project_on_support(chip, local, support, bases, backend)))
        transform = _support_semantic_transform(labels, bases)
        if transform is not None:
            matrix = transform.conj().T @ matrix @ transform
        dims = tuple(bases[name].resolved_dim for name in labels)
        xp = concrete_array_module(matrix)
        diagonal = xp.diagonal(matrix).reshape(dims)
    if contains_tracer(diagonal):
        return True
    diagonal = np.moveaxis(np.asarray(diagonal), labels.index(label), 0)
    return bool(not np.all(np.isfinite(diagonal)) or np.any(diagonal != diagonal[:1]))


def _coupling_level_dependent(chip: "Chip", coupling: "BaseCoupling", label: str, bases: Any) -> bool:
    """Check each operator product without scalar coefficients, including cancelled couplings."""
    with _backend_context(chip.backend):
        operator = coupling.interaction_hamiltonian()
    terms = tuple(_operator_terms(operator)) if isinstance(operator, PhysicsExpr) else (operator,)
    labels = (coupling.device_a_label, coupling.device_b_label)
    return any((isinstance(term, PhysicsExpr) and _holds_parameter(term))
               or _level_dependent(chip, term, labels, label, bases) for term in terms)


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
    With that, the patch generator equals the full-chip generator. The patch
    also holds every device that the operators of the mode's ports act on.
    It absorbs the single-device effective Hamiltonians on its devices, and
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
    # The reduction excludes network-generated Hamiltonians. The full network
    # regenerates them from transformed ports, so line-only neighbours cannot
    # change this generator or its energy denominators.
    patch = set(core)
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

    Channels, coordinate maps and field wiring stay with the full chip. The
    numerical patch reads independent copies of its local ports, because its
    Hamiltonian excludes the terms generated by the field network.
    """
    from quchip.chip.partition import _build_component_chip

    network = None
    if chip.port_network is not None:
        from quchip.chip.port_network import PortNetwork

        members = set(labels)
        ports = [port.copy() for port in chip.ports if members.issuperset(port.resolve_targets(chip))]
        if ports:
            network = PortNetwork.from_ports(ports, label=chip.port_network.label)
    carrier = [device.label for device in chip.devices if device.label != labels[0]][:1] if len(labels) == 1 else []
    group = [device.label for device in chip.devices if device.label in (*labels, *carrier)]
    patch = _build_component_chip(chip, clone, 0, group, [], network)
    patch._effective_terms = tuple(
        replace(terms, channels=(), projection=None,
                excitation_changes=None if terms.excitation_changes is None else {}, omitted_responses={})
        for terms in absorbed
    )
    return patch


def _hamiltonian_interactions(chip: "Chip", resolution: Any, approximation: Any) -> Iterator[tuple[Any, Any]]:
    """Yield the kept static terms in energy coordinates, skipping effective terms without a Hamiltonian.

    An effective term can hold only carried channels, on many devices. It adds
    no energy or coupling, so projecting it would only cost memory.
    """
    from quchip.engine.assembly import _resolved_energy_interactions

    static = tuple(terms for terms in chip.effective_terms if _holds_hamiltonian(terms))
    return _resolved_energy_interactions(chip, resolution, approximation, effective_terms=static)


def _one_excitation_elements(chip: "Chip", resolution: Any, labels: tuple[str, ...], approximation: Any,
                             *, pairs: bool = False) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    """Read single-excitation gaps and exchange elements from local terms, without network terms or a full-space matrix.

    The gap of a device is ``E(1) - E(0)`` on the diagonal of the Hamiltonian,
    as the SW denominators read it. With ``pairs``, the element of two devices
    ``(a, b)`` in chip order is ``<1_a|H|1_b>``, summed over the terms on both.
    Values are in GHz.
    """
    frequencies = {label: resolution.bases[label].energies[1] - resolution.bases[label].energies[0]
                   for label in labels}
    elements: dict[tuple[str, str], Any] = {}
    indices = {chip.device_index(label): label for label in labels}
    for operator, support in _hamiltonian_interactions(chip, resolution, approximation):
        dims = tuple(resolution.dims[index] for index in support)
        rows = {index: prod(dims[axis + 1:]) for axis, index in enumerate(support)}
        for index in support:
            if index in indices:
                row = rows[index]
                label = indices[index]
                frequencies[label] = frequencies[label] + _real(operator[row, row] - operator[0, 0])
        if pairs:
            for first, second in combinations(sorted(index for index in support if index in indices), 2):
                key = (indices[first], indices[second])
                elements[key] = elements.get(key, 0.0) + operator[rows[first], rows[second]]
    return frequencies, elements


def _bare_transition_frequencies(chip: "Chip", resolution: Any, labels: tuple[str, ...],
                                 approximation: Any) -> dict[str, Any]:
    """Read single-excitation diagonal gaps from local terms, without network terms or a full-space matrix."""
    return _one_excitation_elements(chip, resolution, labels, approximation)[0]


# A transported operator whose dense matrix needs this many bytes or more makes a local step warn.
_TRANSPORT_WARNING_BYTES = 2**28


def _transport_note(chip: "Chip", mode_label: str) -> str:
    """Describe the widest operator support after a local step, and warn when its dense matrix is large.

    An operator on a device passes through every captured map that it meets
    (PHYSICS.md §10.7), so this support sets the size of the drives,
    observables and channels on that device.
    """
    dims = dict(zip((d.label for d in chip.devices), chip.authored_dims))
    supports = {d.label: retained_support(chip, (d.label,)) for d in chip.devices}
    label = max(supports, key=lambda name: prod(dims[x] for x in supports[name]))
    support = supports[label]
    states = prod(dims[x] for x in support)
    if 16 * states**2 >= _TRANSPORT_WARNING_BYTES:
        # Hops: _transport_note -> reduce_device -> reduce_local -> eliminate -> caller.
        warnings.warn(
            f"eliminate({mode_label!r}, local=True): an operator on {label!r} now acts on {len(support)} "
            f"devices in retained coordinates ({states:,} states, {16 * states**2 / 2**20:,.0f} MiB as a dense "
            "matrix). Drives, observables and channels on it pass through these maps. Another elimination "
            "order can keep it smaller; eliminate_spectators() chooses one.",
            UserWarning, stacklevel=5,
        )
    return (f"An operator on {label!r} acts on {list(support)} in retained coordinates ({states} states), "
            "the widest of any device.")


def _local_dressed_transition(chip: "Chip", resolution: Any, mode_label: str, lowering: Any,
                              approximation: Any) -> tuple[Any, Any]:
    """Read the SW section from local terms on the states connected to vacuum and one mode excitation.

    These states need not lie in the elimination generator's patch. Each
    interaction supplies its possible local outcomes with all other levels
    fixed. Selection uses dimensions and proved conservation, never parameter
    values. All local terms supply the intermediate states' diagonal energies.
    """
    from quchip.backend._memory import require_memory

    dims = resolution.dims
    mode_axis = chip.device_index(mode_label)
    terms = tuple(_hamiltonian_interactions(chip, resolution, approximation))
    conserving = conserves_excitation_number(chip, approximation, resolution=resolution)
    ground = (0,) * len(dims)
    excited = tuple(int(index == mode_axis) for index in range(len(dims)))
    states = dict.fromkeys((ground, excited))
    for _operator, support in terms:
        for reference in (ground, excited):
            total = sum(reference[index] for index in support)
            for local_levels in product(*(range(dims[index]) for index in support)):
                if conserving and sum(local_levels) != total:
                    continue
                state = list(reference)
                for index, level in zip(support, local_levels):
                    state[index] = level
                states[tuple(state)] = None
    size = len(states)
    require_memory(16 * 12 * size**2 + 8 * len(dims) * size,
                   task=f"Local SW transition (dimension {size})",
                   remedy="Use interactions on smaller local supports or reduce the model.")
    levels = np.asarray(tuple(states), dtype=np.int64)
    energies = tuple(resolution.bases[device.label].energies for device in chip.devices)
    xp = concrete_array_module(energies, tuple(operator for operator, _ in terms), lowering)

    def block(operator: Any, support: tuple[int, ...]) -> Any:
        local_dims = tuple(dims[index] for index in support)
        rows = np.ravel_multi_index(tuple(levels[:, support].T), local_dims)
        outside = [index for index in range(len(dims)) if index not in support]
        other = levels[:, outside]
        equal = np.all(other[:, None, :] == other[None, :, :], axis=2)
        return xp.asarray(operator)[rows[:, None], rows[None, :]] * equal

    diagonal = xp.zeros(size)
    for index, values in enumerate(energies):
        diagonal = diagonal + xp.asarray(values)[levels[:, index]]
    h = xp.diag(diagonal).astype(complex)
    for operator, support in terms:
        h = h + block(operator, support)
    if conserving:
        totals = levels.sum(axis=1)
        h = xp.where(totals[:, None] == totals[None, :], h, 0.0)
    return _sw_dressed_transition(h, block(lowering, (mode_axis,)), np.array([0, 1]))


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


def _local_effective_terms(chip: "Chip", mode_label: str, taken: set[str], *, absorbed: tuple[EffectiveTerms, ...],
                           joined: tuple[EffectiveTerms, ...],
                           patch: tuple[str, ...], survivors: tuple[str, ...], survivor_dims: tuple[int, ...],
                           correction: Any, energy_dims: tuple[int, ...], vectors: tuple[Any, ...],
                           scale: float | None, carried: list[Any], projection: OperatorProjection | None,
                           notes: tuple[str, ...], conserving: bool) -> tuple[EffectiveTerms, ...]:
    """Return the chip's effective terms after a local reduction.

    ``correction`` is in the survivors' energy coordinates. Its single-device
    parts are stored on their own devices, and the rest on the patch
    survivors. Small single-device shifts are retained. Each carried
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
                            {name: changes for name, changes in declared.items() if name in names},
                            omitted_responses={name: responses for name, responses
                                               in terms.omitted_responses.items() if name in names})
        kept.append(terms)
    xp = concrete_array_module(correction)
    tolerance = None if scale is None or contains_tracer(correction) else float(np.spacing(scale))
    mean, parts, rest = _single_device_parts(correction, energy_dims)
    added = []
    for index, (label, dim, part, vector) in enumerate(zip(survivors, survivor_dims, parts, vectors)):
        if index == 0:
            part = part + mean * xp.eye(energy_dims[0])
        added.append(EffectiveTerms((label,), (dim,), vector @ part @ vector.conj().T,
                                    label=_unique_label(f"retained_{mode_label}_{label}", taken),
                                    excitation_changes={} if conserving else None))
    if len(survivors) == 1:
        remainder = np.zeros((prod(survivor_dims),) * 2, dtype=complex)
    else:
        if tolerance is not None:
            omitted = np.where(np.maximum(np.abs(rest), np.abs(rest.T)) <= tolerance, rest, 0.0)
            rest = rest - omitted
            error = float(np.linalg.norm(omitted, ord="fro"))
            if error:
                dropped_notes.append(
                    f"Local Hamiltonian entries at or below the patch's {tolerance:.6g} GHz floating-point "
                    f"spacing were omitted; their Frobenius norm is {error:.6g} GHz."
                )
        lift = _kron(*vectors)
        remainder = lift @ rest @ lift.conj().T
    # Carried channels can cross any earlier map, so their level changes hold
    # only when every map conserves the total level index.
    lineage_conserves = conserving and all(
        t.excitation_changes is not None for t in chip.effective_terms if t.projection is not None
    )
    groups: dict[tuple[str, ...], list[Any]] = {}
    for channel, labels, dims, changes, omitted in carried:
        groups.setdefault(labels, []).append((channel, dims, changes, omitted))
    own = groups.pop(survivors, [])
    for labels, members in groups.items():
        dims = members[0][1]
        added.append(EffectiveTerms(
            labels, dims, np.zeros((prod(dims),) * 2, dtype=complex), tuple(channel for channel, *_ in members),
            label=_unique_label(f"retained_{mode_label}_{'_'.join(labels)}", taken),
            excitation_changes={channel.name: changes for channel, _, changes, _ in members if changes is not None}
            if lineage_conserves else None,
            omitted_responses={channel.name: omitted for channel, _, _, omitted in members if omitted},
        ))
    head = EffectiveTerms(
        survivors, survivor_dims, remainder, tuple(channel for channel, *_ in own), label=head_label,
        projection=projection, notes=(*dropped_notes, *notes),
        excitation_changes={channel.name: changes for channel, _, changes, _ in own if changes is not None}
        if lineage_conserves else None,
        omitted_responses={channel.name: omitted for channel, _, _, omitted in own if omitted},
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
        Compute the generator on the patch around the device. The removed
        mode's reference section also reads spectator transitions from local
        Hamiltonian terms, without assembling the full product space.

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
    if boundary_ports:
        assert chip.port_network is not None
        lines = [chip.port_network._line_exposure(port.label) for port in boundary_ports]
        planes = [plane for plane, _, _ in lines]
        shared = [port.label for port, plane in zip(boundary_ports, planes, strict=True) if planes.count(plane) > 1]
        if shared:
            raise NotImplementedError(f"Ports {shared} of {mode_label!r} share a directional line. Keep the mode.")
        reflection_plane, upstream_ports, downstream_ports = lines[0]
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
    # The retained model reads `numeric`: the whole chip, or the patch whose
    # generator equals the full-chip generator (PHYSICS.md §10.7).
    numeric, chip_bases, lone = chip, None, False
    absorbed: tuple[EffectiveTerms, ...] = ()
    if local:
        from quchip.engine.assembly import _resolve_system

        resolution = _resolve_system(chip, chip.backend)
        chip_bases = resolution.bases
        patch, absorbed = _local_patch(chip, mode_label, chip_bases)
        numeric = _patch_chip(chip, reduced, patch, absorbed)
        lone = len(patch) == 1
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
    conserving = sectors is not None
    if local and conserving and boundary_ports:
        # A retained pole serves the vacuum response of the whole reduced chip.
        conserving = conserves_excitation_number(chip, approximation)
    if len(boundary_ports) > 1 and not conserving:
        raise NotImplementedError(
            f"Eliminating {mode_label!r} with ports {[port.label for port in boundary_ports]} requires an "
            "excitation-conserving model such as RWA(). Keep the mode."
        )
    coupled_response = len(boundary_ports) > 1 or (
        len(boundary_ports) == 1 and conserving
        and _line_has_coupled_response(chip, upstream_ports + downstream_ports, retained_ports)
    )
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
    # The SW generator has no support outside the same patch used by local
    # reduction. Operators on those spectators commute with the rotation.
    unchanged_labels: tuple[str, ...] = ()
    if method == "sw" and not local:
        generator_labels, _ = _local_patch(numeric, mode_label, source_bases)
        unchanged_labels = tuple(label for label in labels if label not in generator_labels
                                 and source_bases[label].native_dim == source_bases[label].resolved_dim)

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
            replacement = Port(
                port_targets,
                rate=port.rate_value(chip),
                operator=port_operator,
                phase=port.phase,
                label=port.label,
            )
            # The transformed field of a boundary port lacks the response that the reduction keeps.
            replacement._omitted_responses = (tuple(sorted({*port._omitted_responses, mode_label}))
                                              if port.label not in retained_ports else port._omitted_responses)
            port_replacements[port.label] = replacement
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
    map_support = patch if local else labels
    joined = tuple(t for t in chip.effective_terms
                   if t.projection is not None and not lineage_labels(t.projection).isdisjoint(map_support))
    parents = lineage(tuple(t.projection for t in joined if t.projection is not None))
    step = OperatorProjection(tuple(labels), tuple(numeric.authored_dims), tuple(numeric_survivors),
                              tuple(numeric_final.authored_dims), step_embedding, parents=parents) if local else None
    carried: list[tuple[CollapseChannel, tuple[str, ...], tuple[int, ...], frozenset[int] | None,
                        tuple[str, ...]]] = []
    channel_responses: dict[str, tuple[str, ...]] = {}

    def omitted_by(support_labels: tuple[str, ...], omitted: tuple[str, ...]) -> tuple[str, ...]:
        # A channel on the removed mode keeps only its adiabatic part, so its
        # time-domain field lacks the kept pole or section.
        return tuple(sorted({*omitted, mode_label})) if boundary_ports and mode_label in support_labels else omitted

    def carry(operator: Any, support_labels: tuple[str, ...], support_dims: tuple[int, ...], rate: Any,
              name: str, changes: frozenset[int] | None, omitted: tuple[str, ...]) -> None:
        assert step is not None
        matrix, carried_labels, carried_dims = _in_device_order(
            chip, *step._embed(_array(operator), support_labels, support_dims),
        )
        carried.append((CollapseChannel(matrix, rate, name), carried_labels, carried_dims, changes,
                        omitted_by(support_labels, omitted)))

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

    def inherit_channel(channel: CollapseChannel, support_labels: tuple[str, ...], name: str,
                        omitted: tuple[str, ...]) -> None:
        rate = materialize_expr(channel.rate, chip.backend)
        changes = None
        if sectors is not None:
            changes = authored_excitation_changes(channel.operator, support_labels, chip.backend,
                                                  source_bases if chip_bases is None else chip_bases)
        if local:
            carry(chip.backend.to_array(materialize_expr(channel.operator, chip.backend)), support_labels,
                  tuple(chip[label].local_space().dimension for label in support_labels), rate, name, changes,
                  omitted)
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
            responses = omitted_by(support_labels, omitted)
            if responses:
                channel_responses[name] = responses
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
                            name if owner is mode_owner else f"{owner_label}.{name}",
                            owner.omitted_responses.get(name, ()) if isinstance(owner, EffectiveTerms) else ())
    if local:
        # Retained channels are stored in the coordinates before this step.
        for terms in chip.effective_terms:
            if terms.channels and not set(patch).isdisjoint(terms.labels):
                declared = None if sectors is None else terms.excitation_changes
                for channel in terms.channels:
                    add_internal_loss(terms.channel_expression(channel), terms.labels,
                                      materialize_expr(channel.rate, chip.backend), bases=chip_bases)
                    carry(channel.operator, terms.labels, terms.dims, channel.rate, f"{terms.label}.{channel.name}",
                          None if declared is None else declared.get(channel.name),
                          terms.omitted_responses.get(channel.name, ()))
    projected_baths = []
    for bath in chip.baths:
        copied = bath.copy()
        copied._retained = {}
        # Authored bath operators first cross the maps of earlier reductions.
        crossed = _retained_owners(chip, tuple(labels), None)[0] if bath._retained is None else []
        for label, frequency, lowering, number in bath._target_operators(chip, source_bases):
            # A conserving reduction keeps each operator's total level change, as for channels, when
            # every earlier map that the operator crosses also conserves the total level index.
            changes = None if sectors is None or any(t.excitation_changes is None for t in crossed) else tuple(
                authored_excitation_changes(op, tuple(labels), chip.backend,
                                            source_bases if chip_bases is None else chip_bases)
                for op in (lowering, number)
            )
            matrices = [_array(chip.backend.to_array(materialize_expr(op, chip.backend)))
                        for op in (lowering, number)]
            if bath._retained is None:
                transported = [transport_retained_operator(chip, op, tuple(labels), tuple(chip.authored_dims))
                               for op in matrices]
                matrices = [op if moved is None else _array(moved[0]) for op, moved in zip(matrices, transported)]
            copied._retained[label] = (frequency, *[step_embedding.conj().T @ op @ step_embedding
                                                   for op in matrices], changes)
        if boundary_ports and mode_label in copied._retained:
            copied._omitted_responses = tuple(sorted({*bath._omitted_responses, mode_label}))
        projected_baths.append(copied)
    projection = None if lone and not joined else (step if step is not None else OperatorProjection.capture(
        numeric, tuple(numeric_survivors), tuple(numeric_final.authored_dims), step_embedding,
        unchanged_labels=unchanged_labels,
    )).with_current_operators(tuple(f"port:{label}" for label in port_replacements))
    order = "second-order" if method == "sw" else "exact projected"
    notes.append(f"Retained the full {order} Hamiltonian correction and each transformed channel "
                 "from the removed components; inherited decay remains collective and separate "
                 "from intrinsic survivor noise.")
    taken = {d.label for d in reduced_devices} | {c.label for c in kept_couplings}
    if local:
        effective_terms = _local_effective_terms(
            chip, mode_label, taken, absorbed=absorbed, joined=joined, patch=tuple(patch),
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
                               excitation_changes=None if sectors is None else channel_changes,
                               omitted_responses=channel_responses)
        effective_terms = (terms,)
    final = rebuild_chip(chip, devices=reduced_devices, couplings=kept_couplings,
                         port_replacements=port_replacements,
                         effective_terms=effective_terms, baths=projected_baths)
    if boundary_ports:
        if local:
            frequency, weight = _local_dressed_transition(chip, resolution, mode_label, mode_lowering, approximation)
        else:
            lowering = _kron(*(mode_lowering if label == mode_label else np.eye(dim)
                               for label, dim in zip(labels, dims)))
            frequency, weight = reduction.dressed_transition(ctx, lowering)
    if coupled_response:
        from quchip.engine.slh import ModeResponse

        network = final.port_network
        assert network is not None
        network._mode_responses += (ModeResponse(
            mode_label, tuple(port.label for port in boundary_ports), frequency, weight, weight * internal_rate,
        ),)
        notes.append(
            f"Kept the coupled response of {mode_label!r} at its original ports. "
            "Vacuum VNA sweeps compose its dressed pole with the network, including later scattering. "
            "Time-domain dynamics use only the transformed port operators. "
            "The pole omits survivor damping and correlations with the discarded internal-loss bath. "
            "Dissipation remains dispersive."
            + (" SW keeps second-order energies only." if method == "sw" else "")
            + " Pumped, finite-power, noisy and excited-reference field responses, time-domain "
            "coherent inputs at its ports, and time-domain output fields and measurement records "
            "that carry its emission require the full boundary mode."
        )
        validity[f"{mode_label}_response"] = _response_validity(
            final, mode_label, {port.label for port in boundary_ports})
    if len(boundary_ports) == 1 and not coupled_response:
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
        external_rate = weight * boundary_ports[0].rate_value(chip)
        if shared_line:
            section = network.mode_transmission(
                section_label, freq=frequency, external_rate=external_rate, internal_rate=weight * internal_rate,
            )
            beyond = kept_upstream if inbound else kept_downstream
            targets = [target for port in chip.ports if port.label in beyond for target in port.resolve_targets(chip)]
            ratio: Any = 0.0
            if targets:
                outside = tuple(dict.fromkeys(target for target in targets if target not in incoming_frequencies))
                if outside:
                    incoming_frequencies.update(_bare_transition_frequencies(chip, resolution, outside, approximation))
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
        location = f"on plane {reflection_plane!r}"
        if shared_line:
            leg = "inbound" if inbound else "outbound"
            location = f"on the {leg} leg of plane {reflection_plane!r}"
        note = (
            f"Kept the {section_kind} of {mode_label!r} as reference section {section_label!r} {location}. "
            "It uses the dressed transition and mode-weighted external and internal rates. "
            "The section drops damping from survivor channels, of order (g/Δ)²γ_s. "
            "Purcell dispersion leaves a scattering residual of order (g/Δ)²κ_e/(2π|Δ|). "
        )
        if method == "sw":
            note += ("SW misses frequency shifts beyond second order, of order g⁴/Δ³ with one survivor "
                     "and 2g_1g_2J/(Δ_1Δ_2) when a coupling J joins two survivors. ")
        note += ("The section's internal-loss bath is vacuum. Monitored records of the line through it and of "
                 "channels on the mode lack its response and require the full boundary mode.")
        if shared_line:
            note += (
                " Either leg preserves the weak-probe response from the line input. "
                "Near kept-port frequencies beyond the section, kappa_over_delta estimates the dispersive "
                "field-error scale from incoming frequencies. Dressed shifts can exceed this scale; "
                "is_valid is not an accuracy guarantee."
            )
        notes.append(note)
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
    if local:
        notes.append(_transport_note(final, mode_label))
    return EliminationResult(chip=final, effective_params=effective_params, validity=validity,
                             notes=notes, mapping=mapping)


register_elimination_target(EliminationTarget(
    kind="device",
    claims=lambda chip, target: resolve_label(target) in chip.device_map,
    reduce=reduce_device,
    reduce_local=lambda chip, target, method: reduce_device(chip, target, method, local=True),
))
