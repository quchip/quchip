"""Exact subsystem partitioning — connected components of a chip's independence graph.

Two device labels share a component when *any* structural object couples
them: a coupling edge, a non-separable bath whose target set contains both,
or classical drive crosstalk between lines targeting them. All decisions are
made from labels and object presence — never from parameter values, which
may be JAX tracers.

When splitting observables via split_e_ops, keys are resolved and validated
while local observables and cross-component factors are grouped. Factor indices
are assigned afterward, so collisions never depend on dict iteration order.
A user key whose value is already a list keeps index=None in its LocalEop — the wrapper passes the
user's own indices through unchanged, and any injected factor is appended
after them. A scalar local value that collides with one or more factors gets
re-indexed to 0, with each factor appended after it in encounter order. The
key plan built here is always keyed by the *resolved* label (or label pair),
matching how :meth:`PartitionedSimulationResult` normalizes lookups — never
by the raw user key, which may be an object rather than its label.

The independence graph only recognizes line-mixing as ``Crosstalk`` entries:
a user-authored :class:`~quchip.control.signal.SignalTransform` that mixes
drive lines any other way is invisible to it and must be expressed as
``Crosstalk`` for the corresponding devices to land in one component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from quchip.utils.labeling import resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.control.signal import SignalTransform


def _line_device_labels(chip: "Chip", line: Any) -> tuple[str, ...]:
    """Device labels a control line acts on (edge lines resolve to both endpoints)."""
    from quchip.control.drive import CouplingDrive

    target = line.target_label
    if target is None:
        return ()
    if isinstance(line, CouplingDrive):
        coupling = chip.coupling_map[target]
        return (coupling.device_a_label, coupling.device_b_label)
    return (target,)


def independence_edges(chip: "Chip", resolved: Any | None = None) -> list[tuple[str, str]]:
    """Label pairs that must share one solve.

    A clique over N labels is emitted as an (N-1)-edge star — identical
    connected components, fewer edges.
    """
    resolved = chip.resolve() if resolved is None else resolved
    edges: list[tuple[str, str]] = []
    for support in resolved.dynamical_supports:
        edges.extend((support[0], other) for other in support[1:])
    equipment = chip.control_equipment
    if equipment is not None:
        line_devices = {line.label: _line_device_labels(chip, line) for line in equipment.lines}
        for xt in chip.crosstalks:
            for a in line_devices.get(xt.source, ()):
                for b in line_devices.get(xt.victim, ()):
                    edges.append((a, b))
    return edges


def connected_components(labels: Sequence[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Union-find components, each in ``labels`` order, ordered by first member."""
    parent = {label: label for label in labels}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    groups: dict[str, list[str]] = {}
    for label in labels:
        groups.setdefault(find(label), []).append(label)
    return list(groups.values())


@dataclass(frozen=True)
class PartitionComponent:
    """One independent block: its device labels (chip order) and its solve-ready sub-chip."""

    labels: tuple[str, ...]
    chip: "Chip"


@dataclass(frozen=True)
class PartitionResult:
    """Connected components of a chip's independence graph, as solve-ready sub-chips.

    ``chip_order`` is the parent chip's original device-label order (its
    ``chip.devices`` order) — *not* the concatenation of ``components``'
    labels, which follows connected-component discovery order instead and
    can interleave differently whenever a chip's device order doesn't
    already group each component's members together. Anything that
    reconstructs a joint state from the per-component pieces (e.g.
    :class:`~quchip.results.partitioned.PartitionedSimulationResult`) must
    permute into ``chip_order`` to match the joint solve.
    """

    components: tuple[PartitionComponent, ...]
    chip_order: tuple[str, ...]
    notes: tuple[str, ...] = ()
    _owner: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        for i, comp in enumerate(self.components):
            for label in comp.labels:
                self._owner[label] = i

    @property
    def is_trivial(self) -> bool:
        return len(self.components) == 1

    def owner_of(self, device: Any) -> int:
        label = resolve_label(device)
        if label not in self._owner:
            raise ValueError(f"Device '{label}' is not in this partition. Known: {sorted(self._owner)}")
        return self._owner[label]

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)


def _component_baths(clone: "Chip", member_set: set[str]) -> list[Any]:
    """Baths restricted to one component; explicit target lists are filtered."""
    kept: list[Any] = []
    for bath in clone.baths:
        if bath._targets is None:
            kept.append(bath.copy())  # "all devices" re-scopes naturally to the sub-chip
            continue
        targets = [t for t in bath.resolve_targets(clone) if t in member_set]
        if not targets:
            continue
        data = bath.to_dict()
        data["targets"] = targets
        kept.append(type(bath).from_dict(data))
    return kept


def _split_network(chip: "Chip", groups: list[list[str]]) -> list[Any]:
    """Restrict the chip's field network to each device group, or explain why that changes the physics."""
    network = chip.port_network
    if network is None:
        return [None] * len(groups)
    per_group: list[Any] = []
    covered: set[str] = set()
    for group in groups:
        members = set(group)
        labels = [port.label for port in network.ports if set(port.resolve_targets(chip)).issubset(members)]
        if not labels:
            per_group.append(None)
            continue
        try:
            restricted = network.restrict(labels)
        except ValueError as error:
            raise ValueError(f"{error} Those ports belong to different device groups") from error
        covered.update(component.label for component in restricted.components)
        per_group.append(restricted)
    leftover = sorted({component.label for component in network.components} - covered)
    if leftover:
        raise ValueError(
            f"PortNetwork components {leftover} are not reachable from one device group's ports; "
            "restricting the field network would change the SLH dynamics"
        )
    return per_group


def _transform_drive_labels(transform: "SignalTransform") -> tuple[str, ...]:
    """Drive labels a signal transform references."""
    return transform.referenced_lines()


def _distribute_control_lines(
    clone: "Chip", groups: list[list[str]], owner: dict[str, int]
) -> tuple[list[list[Any]], list[str]]:
    """Bucket each control line by its target device's owning group; note lines with no target."""
    lines_per: list[list[Any]] = [[] for _ in groups]
    notes: list[str] = []
    equipment = clone.control_equipment
    if equipment is None:
        return lines_per, notes
    for line in equipment.lines:
        target_labels = _line_device_labels(clone, line)
        if not target_labels:
            notes.append(f"control line '{line.label}' has no target; dropped from the partition")
            continue
        lines_per[owner[target_labels[0]]].append(line)
    return lines_per, notes


def _build_component_chip(
    chip: "Chip", clone: "Chip", index: int, group: list[str], lines: list[Any], network: Any
) -> "Chip":
    """Assemble one component's solve-ready sub-chip: its devices, internal couplings, baths, frame, equipment."""
    from quchip.chip.chip import Chip as _Chip
    from quchip.control.equipment import ControlEquipment
    from quchip.utils.values import copy_value

    member_set = set(group)
    devices = [clone[label] for label in group]
    couplings = [
        c for c in clone.couplings
        if c.device_a_label in member_set and c.device_b_label in member_set
    ]
    frame: Any = (
        {k: v for k, v in clone.frame.items() if k in member_set}
        if isinstance(clone.frame, dict) else clone.frame
    )
    sub = _Chip(
        devices=devices,
        couplings=couplings or None,
        label=f"{chip.label}[{index}]" if chip.label else None,
        frame=copy_value(frame),
        approximation=clone.approximation,
        basis=clone.basis,
        backend=clone._backend,
        baths=_component_baths(clone, member_set) or None,
        port_network=network,
        effective_terms=[terms for terms in clone.effective_terms if set(terms.labels) <= member_set],
    )
    equipment = clone.control_equipment
    if equipment is not None and lines:
        line_labels = {ln.label for ln in lines}
        chain = [
            t for t in equipment.signal_chain
            if all(lbl in line_labels for lbl in _transform_drive_labels(t))
        ]
        sub.connect(ControlEquipment(lines=lines, signal_chain=chain).copy(sub.device_map, sub.coupling_map))
    from quchip.chip.states import copy_state_configuration

    copy_state_configuration(chip, sub)
    return sub


def partition_chip(
    chip: "Chip",
    *,
    resolved: Any | None = None,
    extra_supports: tuple[tuple[str, ...], ...] = (),
    clone_trivial: bool = True,
) -> PartitionResult:
    """Split a chip into independent sub-chips along its independence graph.

    Exact: the joint solve of the original chip factorizes as the tensor
    product of the component solves. Public results contain independent
    models even with one component. Internal dispatch can reuse a trivial
    component when it will immediately return to the joint solve.
    """

    labels = [d.label for d in chip.devices]
    chip_order = tuple(labels)
    resolved = chip.resolve() if resolved is None else resolved
    edges = independence_edges(chip, resolved)
    for support in extra_supports:
        edges.extend((support[0], other) for other in support[1:])
    groups = connected_components(labels, edges)
    if len(groups) == 1:
        return PartitionResult(
            components=(PartitionComponent(labels=chip_order, chip=chip.clone() if clone_trivial else chip),),
            chip_order=chip_order,
        )
    try:
        networks = _split_network(chip, groups)
    except ValueError as error:
        return PartitionResult(
            components=(PartitionComponent(labels=chip_order, chip=chip.clone() if clone_trivial else chip),),
            chip_order=chip_order,
            notes=(f"Kept a joint solve because {error}.",),
        )

    clone = chip.clone()
    owner: dict[str, int] = {}
    for i, group in enumerate(groups):
        for label in group:
            owner[label] = i

    lines_per, notes = _distribute_control_lines(clone, groups, owner)
    components = tuple(
        PartitionComponent(
            labels=tuple(group), chip=_build_component_chip(chip, clone, i, group, lines_per[i], networks[i])
        )
        for i, group in enumerate(groups)
    )

    return PartitionResult(components=components, chip_order=chip_order, notes=tuple(notes))


@dataclass(frozen=True)
class LocalEop:
    """Where one user e_ops key landed: component, component-dict key, optional list index."""

    component: int
    key: Any
    index: int | None = None


@dataclass(frozen=True)
class CrossEop:
    """A cross-component correlator: its two local factors, multiplied at recombine time."""

    a: LocalEop
    b: LocalEop


def split_drive_ops(part: PartitionResult, chip: "Chip", drive_ops: list) -> list[list]:
    """Route drive ops to their owning component (edge targets via an endpoint)."""
    per: list[list] = [[] for _ in part.components]
    for op in drive_ops:
        label = op.target_label
        if label in chip.coupling_map:
            label = chip.coupling_map[label].device_a_label
        per[part.owner_of(label)].append(op)
    return per


@dataclass
class _LabelGroup:
    """Local observables and correlator factors sharing one (component, label) slot.

    ``local`` is the plain (non-cross) entry for this slot, if any, as
    ``(resolved_key, value)``. ``factors`` are cross-component correlator
    halves that also land on this slot, each as ``(resolved_key, side, op)``
    with ``side`` one of ``"a"``/``"b"``.
    """

    local: tuple[Any, Any] | None = None
    factors: list[tuple[Any, str, Any]] = field(default_factory=list)


def split_e_ops(part: PartitionResult, e_ops: dict | None) -> tuple[list[dict], dict]:
    """Group local observables and cross-component factors before assigning indices."""
    per: list[dict] = [dict() for _ in part.components]
    plan: dict[Any, LocalEop | CrossEop] = {}
    if not e_ops:
        return per, plan

    groups: dict[tuple[int, str], _LabelGroup] = {}
    cross_owners: dict[tuple[str, str], tuple[int, int]] = {}
    seen: dict[Any, Any] = {}

    def group(component: int, label: str) -> _LabelGroup:
        return groups.setdefault((component, label), _LabelGroup())

    for key, value in e_ops.items():
        if isinstance(key, tuple):
            label_a, label_b = (resolve_label(k) for k in key)
            resolved: Any = (label_a, label_b)
        else:
            label_a = label_b = resolve_label(key)
            resolved = label_a
        if resolved in seen:
            raise ValueError(
                f"e_ops keys {seen[resolved]!r} and {key!r} both resolve to {resolved!r}; "
                "pass one merged entry instead of two keys for the same target."
            )
        seen[resolved] = key

        if not isinstance(key, tuple):
            group(part.owner_of(label_a), label_a).local = (resolved, value)
            continue
        comp_a, comp_b = part.owner_of(label_a), part.owner_of(label_b)
        if comp_a == comp_b:
            per[comp_a][resolved] = value
            plan[resolved] = LocalEop(component=comp_a, key=resolved)
            continue
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                f"Cross-component e_ops key {key!r} needs a 2-element (op_a, op_b) value; "
                f"got {value!r}"
            )
        op_a, op_b = value
        group(comp_a, label_a).factors.append((resolved, "a", op_a))
        group(comp_b, label_b).factors.append((resolved, "b", op_b))
        cross_owners[resolved] = (comp_a, comp_b)

    factor_index: dict[tuple[Any, str], int | None] = {}
    for (component, label), entries in groups.items():
        local, factors = entries.local, entries.factors
        if not factors:
            assert local is not None
            resolved, value = local
            per[component][label] = value
            plan[resolved] = LocalEop(component=component, key=label)
            continue
        if local is None and len(factors) == 1:
            key, side, op = factors[0]
            per[component][label] = op
            factor_index[(key, side)] = None
            continue

        values = []
        if local is not None:
            resolved, value = local
            values = list(value) if isinstance(value, list) else [value]
            plan[resolved] = LocalEop(
                component=component, key=label, index=None if isinstance(value, list) else 0,
            )
        for key, side, op in factors:
            factor_index[(key, side)] = len(values)
            values.append(op)
        per[component][label] = values

    for (label_a, label_b), (comp_a, comp_b) in cross_owners.items():
        key = (label_a, label_b)
        plan[key] = CrossEop(
            a=LocalEop(component=comp_a, key=label_a, index=factor_index[(key, "a")]),
            b=LocalEop(component=comp_b, key=label_b, index=factor_index[(key, "b")]),
        )
    return per, plan


def split_state_mapping(part: PartitionResult, mapping: Any) -> list[dict]:
    """Distribute a device-keyed state mapping over components."""
    per: list[dict] = [dict() for _ in part.components]
    for key, value in mapping.items():
        label = resolve_label(key)
        per[part.owner_of(label)][label] = value
    return per
