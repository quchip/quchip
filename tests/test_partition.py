"""Independence-graph partitioning of a Chip (spec: 2026-07-11-partition-active-patch)."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Bath, Capacitive, ChargeDrive, Chip, Crosstalk, DuffingTransmon, Gaussian
from quchip.chip.partition import (
    connected_components,
    independence_edges,
    split_drive_ops,
    split_e_ops,
    split_state_mapping,
)
from quchip.control.sequence import QuantumSequence


def test_bath_separable_flag():
    """A thermal bath is separable; collective-decay and correlated-dephasing baths are not."""
    thermal = Bath("thermal", temperature=20.0)
    collective = Bath("collective_decay", rate=0.01)
    dephasing = Bath("correlated_dephasing", rate=0.01)
    assert thermal.separable is True
    assert collective.separable is False
    assert dephasing.separable is False


def _four_qubits():
    return [
        DuffingTransmon(freq=5.0 + 0.1 * i, anharmonicity=-0.25, levels=3, label=f"q{i}")
        for i in range(4)
    ]


def test_components_from_couplings_only():
    """Capacitive coupling merges only the devices it directly links into one independence component."""
    q0, q1, q2, q3 = _four_qubits()
    chip = Chip([q0, q1, q2, q3], couplings=[Capacitive(q0, q1, g=0.005), Capacitive(q2, q3, g=0.005)])
    comps = connected_components([d.label for d in chip.devices], independence_edges(chip))
    assert comps == [["q0", "q1"], ["q2", "q3"]]


def test_isolated_device_is_its_own_component():
    """A device with no coupling edges is its own singleton independence component."""
    q0, q1, q2, _ = _four_qubits()
    chip = Chip([q0, q1, q2], couplings=[Capacitive(q0, q1, g=0.005)])
    comps = connected_components([d.label for d in chip.devices], independence_edges(chip))
    assert comps == [["q0", "q1"], ["q2"]]


def test_collective_bath_targets_merge_components():
    """A non-separable bath whose targets span two coupled components merges them into one independence component."""
    q0, q1, q2, q3 = _four_qubits()
    chip = Chip(
        [q0, q1, q2, q3],
        couplings=[Capacitive(q0, q1, g=0.005), Capacitive(q2, q3, g=0.005)],
        baths=[Bath("collective_decay", targets=[q0, q2], rate=0.001)],
    )
    comps = connected_components([d.label for d in chip.devices], independence_edges(chip))
    assert comps == [["q0", "q1", "q2", "q3"]]


def test_separable_thermal_bath_does_not_merge():
    """A separable thermal bath contributes no independence edges, leaving separately coupled components unmerged."""
    q0, q1, q2, q3 = _four_qubits()
    chip = Chip(
        [q0, q1, q2, q3],
        couplings=[Capacitive(q0, q1, g=0.005), Capacitive(q2, q3, g=0.005)],
        baths=[Bath("thermal", temperature=20.0, rate=0.001)],
    )
    comps = connected_components([d.label for d in chip.devices], independence_edges(chip))
    assert comps == [["q0", "q1"], ["q2", "q3"]]


def test_drive_crosstalk_merges_components():
    """Classical drive crosstalk between lines on different coupled components merges those components into one."""
    q0, q1, q2, q3 = _four_qubits()
    d1 = ChargeDrive(target=q1, label="d1")
    d3 = ChargeDrive(target=q3, label="d3")
    chip = Chip([q0, q1, q2, q3], couplings=[Capacitive(q0, q1, g=0.005), Capacitive(q2, q3, g=0.005)])
    chip.wire(d1, d3, signal_chain=[Crosstalk(d1, d3, beta=0.02)])
    comps = connected_components([d.label for d in chip.devices], independence_edges(chip))
    assert comps == [["q0", "q1", "q2", "q3"]]


def _disconnected_chip(with_bath: bool = False, with_lines: bool = False):
    q0, q1, q2, q3 = _four_qubits()
    baths = [Bath("thermal", targets=[q0, q2], temperature=20.0, rate=0.001)] if with_bath else None
    chip = Chip(
        [q0, q1, q2, q3],
        couplings=[Capacitive(q0, q1, g=0.005, label="c01"), Capacitive(q2, q3, g=0.005, label="c23")],
        baths=baths,
    )
    if with_lines:
        chip.wire(ChargeDrive(target=q0, label="d0"), ChargeDrive(target=q2, label="d2"))
    return chip


def _partitioned_disconnected_chip(with_bath: bool = False, with_lines: bool = False):
    chip = _disconnected_chip(with_bath=with_bath, with_lines=with_lines)
    return chip, chip.partition()


def test_partition_builds_two_subchips():
    """Each sub-chip holds only its own devices and couplings; owner_of() maps labels to component index."""
    _, part = _partitioned_disconnected_chip()
    assert len(part) == 2 and not part.is_trivial
    assert part.components[0].labels == ("q0", "q1")
    assert part.components[1].labels == ("q2", "q3")
    sub0 = part.components[0].chip
    assert [d.label for d in sub0.devices] == ["q0", "q1"]
    assert [c.label for c in sub0.couplings] == ["c01"]
    assert sub0.dims == (3, 3)
    assert part.owner_of("q3") == 1
    with pytest.raises(ValueError):
        part.owner_of("nope")


def test_partition_trivial_returns_original_chip():
    """A fully connected chip's partition is trivial and returns the original chip object without cloning."""
    q0, q1, _, _ = _four_qubits()
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.005)])
    part = chip.partition()
    assert part.is_trivial
    assert part.components[0].chip is chip


def test_partition_filters_bath_targets():
    """Partitioning filters each bath's target list down to the devices that landed in its own sub-chip."""
    _, part = _partitioned_disconnected_chip(with_bath=True)
    bath0 = part.components[0].chip.baths[0]
    bath1 = part.components[1].chip.baths[0]
    assert bath0.resolve_targets(part.components[0].chip) == ["q0"]
    assert bath1.resolve_targets(part.components[1].chip) == ["q2"]


def test_partition_routes_control_lines():
    """Partitioning routes each control line to the sub-chip that owns its target device."""
    _, part = _partitioned_disconnected_chip(with_lines=True)
    eq0 = part.components[0].chip.control_equipment
    eq1 = part.components[1].chip.control_equipment
    assert [ln.label for ln in eq0.lines] == ["d0"]
    assert [ln.label for ln in eq1.lines] == ["d2"]


def test_partition_does_not_mutate_source_chip():
    """Partitioning leaves the source chip's devices, baths, and control lines unchanged."""
    chip, part = _partitioned_disconnected_chip(with_bath=True, with_lines=True)
    assert len(chip.devices) == 4 and len(chip.baths) == 1
    assert [ln.label for ln in chip.control_equipment.lines] == ["d0", "d2"]
    assert part.components[0].chip is not chip


def test_partition_does_not_duplicate_connected_drives():
    """Partitioning connects each drive to its device exactly once, never duplicating the clone's own connection."""
    # Regression: chip.clone() (inside partition_chip) already connects the
    # full cloned equipment onto the clone's devices; partition_chip then
    # connects a per-component ControlEquipment(...).copy(...) onto the same
    # device objects. Each device must end up with exactly one drive per
    # label, not the original clone-connected drive plus a stale duplicate.
    _, part = _partitioned_disconnected_chip(with_lines=True)
    for comp in part.components:
        for label in comp.labels:
            device = comp.chip[label]
            drive_labels = [d.label for d in device.connected_drives]
            assert len(drive_labels) == len(set(drive_labels)), (
                f"device '{label}' has duplicate-label connected drives: {drive_labels}"
            )


def test_partition_subchip_schedules_without_duplicate_drive_error():
    """A partitioned sub-chip schedules its ChargeDrive without a spurious duplicate-drive error."""
    # Regression: a duplicate-label drive on the device made
    # QuantumSequence.charge() raise "Multiple ChargeDrive on 'q0'" even
    # though only one ChargeDrive was ever wired to q0.
    _, part = _partitioned_disconnected_chip(with_lines=True)
    sub0 = part.components[0].chip
    seq = QuantumSequence(sub0)
    handle = seq.charge("q0", envelope=Gaussian(amplitude=0.01, sigmas=3, duration=20.0))
    assert handle is not None


# ============================================================================
# Tests for split_drive_ops, split_e_ops, split_state_mapping
# ============================================================================


class _FakeOp:
    def __init__(self, target_label):
        self.target_label = target_label


def test_split_drive_ops_routes_by_target():
    """split_drive_ops() routes each op to the component owning its target, resolving edge targets to an endpoint."""
    chip, part = _partitioned_disconnected_chip()
    ops = [_FakeOp("q0"), _FakeOp("q3"), _FakeOp("c23")]
    per = split_drive_ops(part, chip, ops)
    assert [o.target_label for o in per[0]] == ["q0"]
    assert [o.target_label for o in per[1]] == ["q3", "c23"]


def test_split_e_ops_local_and_cross():
    """split_e_ops() places a local entry in its own component; a cross-component correlator becomes a CrossEop."""
    _, part = _partitioned_disconnected_chip()
    e_ops = {"q0": "Z", ("q1", "q2"): ("Z", "Z")}
    per, plan = split_e_ops(part, e_ops)
    assert per[0]["q0"] == "Z"
    assert per[0]["q1"] == "Z" and per[1]["q2"] == "Z"
    local = plan["q0"]
    assert (local.component, local.key, local.index) == (0, "q0", None)
    cross = plan[("q1", "q2")]
    assert (cross.a.component, cross.a.key) == (0, "q1")
    assert (cross.b.component, cross.b.key) == (1, "q2")


def test_split_e_ops_listifies_on_collision():
    """A local value colliding with an injected cross-component factor becomes a list, local value first."""
    _, part = _partitioned_disconnected_chip()
    e_ops = {"q1": "X", ("q1", "q2"): ("Z", "Z")}
    per, plan = split_e_ops(part, e_ops)
    assert per[0]["q1"] == ["X", "Z"]
    assert plan["q1"].index == 0
    assert plan[("q1", "q2")].a.index == 1


def test_split_e_ops_same_component_tuple_stays_verbatim():
    """A tuple key whose two labels resolve to the same component passes through unchanged, not split into factors."""
    _, part = _partitioned_disconnected_chip()
    per, plan = split_e_ops(part, {("q0", "q1"): ("Z", "Z")})
    assert per[0][("q0", "q1")] == ("Z", "Z")
    assert plan[("q0", "q1")].component == 0


def test_split_state_mapping():
    """split_state_mapping() distributes a device-keyed state mapping into per-component dicts by owning component."""
    _, part = _partitioned_disconnected_chip()
    per = split_state_mapping(part, {"q0": 1, "q3": 1})
    assert per == [{"q0": 1}, {"q3": 1}]


# ============================================================================
# Regression tests: order-independent factor collision handling (review findings)
# ============================================================================


def test_split_e_ops_hub_three_correlators_all_get_correct_indices():
    """Every correlator sharing a hub label gets its index re-pointed to the factor's final position, not the last."""
    # Finding 1 (Critical): a hub label shared by 3+ cross-component correlators
    # must have every CrossEop side re-point to its final list index — not just
    # the most recently injected one.
    q0, q1, q2, q3 = _four_qubits()
    chip = Chip([q0, q1, q2, q3])  # no couplings: four singleton components
    part = chip.partition()
    comp_q2 = part.owner_of("q2")
    e_ops = {
        ("q0", "q2"): ("A0", "B0"),
        ("q1", "q2"): ("A1", "B1"),
        ("q3", "q2"): ("A3", "B3"),
    }
    per, plan = split_e_ops(part, e_ops)
    assert per[comp_q2]["q2"] == ["B0", "B1", "B3"]
    for key, expected_index in (
        (("q0", "q2"), 0),
        (("q1", "q2"), 1),
        (("q3", "q2"), 2),
    ):
        cross = plan[key]
        assert cross.b.component == comp_q2 and cross.b.key == "q2"
        assert cross.b.index == expected_index


def test_split_e_ops_cross_before_local_preserves_both():
    """A cross-component correlator before a same-label local entry still keeps the local value first, factor after."""
    # Finding 2 (Critical): a local key processed after a cross tuple key
    # touching the same label must not silently overwrite the injected factor.
    _, part = _partitioned_disconnected_chip()
    e_ops = {("q1", "q2"): ("Z", "Z"), "q1": "X"}
    per, plan = split_e_ops(part, e_ops)
    assert per[0]["q1"] == ["X", "Z"]
    assert plan["q1"].index == 0
    assert plan[("q1", "q2")].a.index == 1


def test_split_e_ops_list_local_with_factor_appends_after_user_indices():
    """A list-valued local entry keeps the user's indices at the head; the injected factor appends after them."""
    # A list-valued local entry keeps the user's own indices at the head;
    # the injected factor is appended after them.
    _, part = _partitioned_disconnected_chip()
    e_ops = {"q1": ["X", "Y"], ("q1", "q2"): ("Z", "Z")}
    per, plan = split_e_ops(part, e_ops)
    assert per[0]["q1"] == ["X", "Y", "Z"]
    assert plan["q1"].index is None
    assert plan[("q1", "q2")].a.index == 2


def test_split_e_ops_duplicate_resolved_key_raises():
    """Two e_ops keys resolving to the same device label raise ValueError instead of silently overwriting."""
    # Finding 3 (Important): two user keys resolving to the same label
    # (device object + its label string) must raise, not silently overwrite.
    chip, part = _partitioned_disconnected_chip()
    q0_device = chip["q0"]
    with pytest.raises(ValueError, match="q0"):
        split_e_ops(part, {q0_device: "X", "q0": "Y"})


def test_split_e_ops_malformed_cross_value_raises_with_key_context():
    """A malformed cross-component correlator value raises ValueError naming the offending key pair."""
    # Finding 4 (Minor): a malformed correlator value must raise a clear
    # error naming the offending key, not a bare unpack ValueError.
    _, part = _partitioned_disconnected_chip()
    with pytest.raises(ValueError, match=r"q1.*q2|q2.*q1"):
        split_e_ops(part, {("q1", "q2"): "Z"})


# ============================================================================
# Tests for PartitionedSimulationResult
# ============================================================================


def _solved_components():
    # This solves single components directly, so partition dispatch must be
    # disabled explicitly — otherwise each singleton sub-chip would recurse
    # into its own (trivial) partition check for no benefit.
    from quchip.engine import simulate

    _, part = _partitioned_disconnected_chip(with_lines=True)
    tlist = np.linspace(0.0, 20.0, 21)
    results = []
    for comp in part:
        # Raw dict-form e_ops values must already be resolved operators, not
        # shorthand strings — build them through Chip.e_ops().
        e_ops = comp.chip.e_ops(**{comp.labels[0]: "Z"})
        results.append(simulate(comp.chip, [], tlist, e_ops=e_ops, partition=False))
    return part, results, tlist


def test_partitioned_result_local_access():
    """A local e_ops key or population lookup returns exactly the owning component's values."""
    from quchip.chip.partition import LocalEop
    from quchip.results.partitioned import PartitionedSimulationResult

    part, results, tlist = _solved_components()
    plan = {"q0": LocalEop(component=0, key="q0", index=None),
            "q2": LocalEop(component=1, key="q2", index=None)}
    combined = PartitionedSimulationResult(results, part, plan)
    assert np.allclose(combined.expect("q0"), results[0].expect("q0"))
    assert np.allclose(combined.population("q3", 0), results[1].population("q3", 0))
    assert combined.device_order == ("q0", "q1", "q2", "q3")
    assert "2 components" in repr(combined)


def test_partitioned_result_cross_product_and_states_warn():
    """A correlator's expect() is the product of its two factors; .states warns about the joint state."""
    from quchip.chip.partition import CrossEop, LocalEop
    from quchip.results.partitioned import PartitionedSimulationResult

    part, results, tlist = _solved_components()
    plan = {("q0", "q2"): CrossEop(a=LocalEop(0, "q0", None), b=LocalEop(1, "q2", None))}
    combined = PartitionedSimulationResult(results, part, plan)
    expected = np.asarray(results[0].expect("q0")) * np.asarray(results[1].expect("q2"))
    assert np.allclose(combined.expect(("q0", "q2")), expected)
    with pytest.warns(UserWarning, match="joint"):
        _ = combined.states


# ============================================================================
# Tests for engine.simulate(..., partition=True/False) dispatch
# ============================================================================


def _driven_disconnected_chip():
    q0, q1, q2, q3 = _four_qubits()
    chip = Chip(
        [q0, q1, q2, q3],
        couplings=[Capacitive(q0, q1, g=0.005, label="c01"), Capacitive(q2, q3, g=0.005, label="c23")],
        frame="rotating", rwa=True,
    )
    d0 = ChargeDrive(target=q0, label="d0")
    d2 = ChargeDrive(target=q2, label="d2")
    chip.wire(d0, d2)
    seq = QuantumSequence(chip)
    seq.schedule(d0, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q0))
    seq.schedule(d2, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q2))
    return chip, seq


def test_engine_partition_parity_with_joint_solve():
    """A partitioned solve reproduces the joint solve's expectation values for local and cross observables alike."""
    from quchip.engine import simulate
    from quchip.results.partitioned import PartitionedSimulationResult

    chip, seq = _driven_disconnected_chip()
    tlist = np.linspace(0.0, 20.0, 41)
    ops = list(seq.scheduled_ops)
    # Raw dict-form e_ops values must already be resolved operators, not
    # shorthand strings — build them through Chip.e_ops().
    e_ops = chip.e_ops(q0="Z", q2="Z", correlators={("q0", "q2"): ("Z", "Z")})

    joint = simulate(chip, ops, tlist, e_ops=e_ops, partition=False)
    split = simulate(chip, ops, tlist, e_ops=e_ops)

    assert isinstance(split, PartitionedSimulationResult)
    assert len(split.components) == 2
    for key in ("q0", "q2", ("q0", "q2")):
        assert np.allclose(np.asarray(split.expect(key)), np.asarray(joint.expect(key)), atol=1e-8)
    assert np.allclose(np.asarray(split.population("q1", 0)), np.asarray(joint.population("q1", 0)), atol=1e-8)


def test_engine_partition_declines_on_raw_state():
    """simulate() declines partition dispatch, returning a plain SimulationResult for a raw initial_state."""
    from quchip.engine import simulate
    from quchip.results.results import SimulationResult

    chip, seq = _driven_disconnected_chip()
    tlist = np.linspace(0.0, 20.0, 21)
    raw = chip.bare_state({"q0": 1})
    result = simulate(chip, list(seq.scheduled_ops), tlist, initial_state=raw)
    assert isinstance(result, SimulationResult)


def test_engine_partition_mapping_initial_state():
    """A label-keyed initial_state mapping partitions correctly, matching the joint solve's expectation values."""
    from quchip.engine import simulate
    from quchip.results.partitioned import PartitionedSimulationResult

    chip, seq = _driven_disconnected_chip()
    tlist = np.linspace(0.0, 20.0, 21)
    e_ops = chip.e_ops(q0="Z")
    joint = simulate(chip, [], tlist, e_ops=e_ops, initial_state={"q0": 1}, partition=False)
    split = simulate(chip, [], tlist, e_ops=e_ops, initial_state={"q0": 1})
    assert isinstance(split, PartitionedSimulationResult)
    assert np.allclose(np.asarray(split.expect("q0")), np.asarray(joint.expect("q0")), atol=1e-8)


# ============================================================================
# Tests for PartitionedSimulationResult.final_state with mixed ket/DM components
# ============================================================================


def _mixed_noise_disconnected_chip():
    # q0/q1 carry T1 -> that component auto-selects mesolve (DM final state).
    # q2/q3 carry no noise -> that component auto-selects sesolve (ket final state).
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0", T1=50.0)
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="q1", T1=50.0)
    q2 = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="q2")
    q3 = DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q3")
    return Chip(
        [q0, q1, q2, q3],
        couplings=[Capacitive(q0, q1, g=0.005, label="c01"), Capacitive(q2, q3, g=0.005, label="c23")],
    )


def test_final_state_mixed_ket_and_dm_components_promotes_to_joint_dm():
    """When components disagree on trajectory kind, final_state promotes each to a density matrix before tensoring."""
    from quchip.engine import simulate
    from quchip.results.partitioned import PartitionedSimulationResult

    chip = _mixed_noise_disconnected_chip()
    tlist = np.linspace(0.0, 5.0, 6)
    result = simulate(chip, [], tlist, solver=None)
    assert isinstance(result, PartitionedSimulationResult)

    # Pin the premise: components disagree on trajectory kind.
    backend = result.components[0]._backend
    assert backend.is_ket(result.components[1].final_state)
    assert not backend.is_ket(result.components[0].final_state)

    with pytest.warns(UserWarning, match="joint"):
        final = result.final_state

    assert not backend.is_ket(final)
    arr = np.asarray(backend.to_array(final), dtype=complex)
    assert arr.ndim == 2 and arr.shape[0] == arr.shape[1]
    assert np.isclose(np.trace(arr).real, 1.0, atol=1e-6)


def test_final_state_all_ket_components_stays_a_joint_ket():
    """When every component's solve stays pure, final_state reconstructs a joint ket rather than a density matrix."""
    from quchip.engine import simulate
    from quchip.results.partitioned import PartitionedSimulationResult

    chip, seq = _driven_disconnected_chip()
    tlist = np.linspace(0.0, 20.0, 21)
    result = simulate(chip, list(seq.scheduled_ops), tlist, solver=None)
    assert isinstance(result, PartitionedSimulationResult)

    backend = result.components[0]._backend
    for comp in result.components:
        assert backend.is_ket(comp.final_state)

    with pytest.warns(UserWarning, match="joint"):
        final = result.final_state

    assert backend.is_ket(final)


def test_partitioned_result_mismatched_length_raises():
    """PartitionedSimulationResult raises ValueError when component_results and partition.components misalign."""
    from quchip.chip.partition import LocalEop
    from quchip.results.partitioned import PartitionedSimulationResult

    part, results, tlist = _solved_components()
    plan = {"q0": LocalEop(component=0, key="q0", index=None)}
    with pytest.raises(ValueError, match="component_results"):
        PartitionedSimulationResult(results[:1], part, plan)


# ============================================================================
# Tests for QuantumSequence.simulate(..., partition=True/False) dispatch
# ============================================================================


def test_sequence_simulate_partitions_by_default():
    """QuantumSequence.simulate() partitions by default, matching an explicit joint (partition=False) solve."""
    from quchip.results.partitioned import PartitionedSimulationResult

    chip, seq = _driven_disconnected_chip()
    # Raw dict-form e_ops values must already be resolved operators, not
    # shorthand strings — build them through Chip.e_ops().
    e_ops = chip.e_ops(q0="Z", q2="Z")
    result = seq.simulate(e_ops=e_ops)
    assert isinstance(result, PartitionedSimulationResult)

    joint = seq.simulate(e_ops=e_ops, partition=False)
    assert np.allclose(np.asarray(result.expect("q0")), np.asarray(joint.expect("q0")), atol=1e-8)


# ============================================================================
# Regression tests: final-review findings (B1, B2, W4, S3)
# ============================================================================


def test_split_e_ops_object_key_matches_joint_solve_by_object_and_label():
    """Both an object-keyed and a label-keyed lookup on a partitioned result succeed and match the joint solve."""
    # B1: split_e_ops previously keyed `plan` by the raw user key in the
    # same-component and local (PASS 2) branches, while
    # PartitionedSimulationResult._normalize_key always resolves lookups to
    # labels. An object-keyed e_ops dict (the UX-favored form) landed a plan
    # entry under the *object*, so both expect(q0) and expect("q0") raised
    # KeyError. Both forms must now work and match the joint solve.
    from quchip.engine import simulate
    from quchip.results.partitioned import PartitionedSimulationResult

    chip = _disconnected_chip()
    q0 = chip["q0"]
    tlist = np.linspace(0.0, 5.0, 6)
    e_ops = {q0: q0.number_operator()}

    joint = simulate(chip, [], tlist, e_ops=e_ops, initial_state={"q0": 1}, partition=False)
    split = simulate(chip, [], tlist, e_ops=e_ops, initial_state={"q0": 1})
    assert isinstance(split, PartitionedSimulationResult)

    joint_vals = np.asarray(joint.expect(q0))
    assert np.allclose(np.asarray(split.expect(q0)), joint_vals, atol=1e-8)
    assert np.allclose(np.asarray(split.expect("q0")), joint_vals, atol=1e-8)


def test_partitioned_final_state_matches_joint_in_interleaved_device_order():
    """final_state matches the joint solve even when the chip's device order interleaves components differently."""
    # B2: states/final_state used to tensor components in connected-component
    # discovery order, silently permuting the joint state whenever the
    # chip's own device order interleaves components differently. Here the
    # chip lists devices [q0, q2, q1, q3] while the independence graph groups
    # them as (q0, q1) and (q2, q3) — discovery order is q0,q1,q2,q3, but the
    # chip's own order is q0,q2,q1,q3, so a naive concatenation misplaces the
    # excitation.
    from quchip.engine import simulate

    q0, q1, q2, q3 = _four_qubits()
    chip = Chip(
        [q0, q2, q1, q3],
        couplings=[Capacitive(q0, q1, g=0.005, label="c01"), Capacitive(q2, q3, g=0.005, label="c23")],
        frame="rotating", rwa=True,
    )
    tlist = np.linspace(0.0, 5.0, 6)

    joint = simulate(chip, [], tlist, initial_state={"q2": 1}, partition=False)
    split = simulate(chip, [], tlist, initial_state={"q2": 1})

    backend = joint._backend
    joint_arr = np.asarray(backend.to_array(joint.final_state))
    with pytest.warns(UserWarning, match="joint"):
        split_final = split.final_state
    split_arr = np.asarray(backend.to_array(split_final))
    # Both solves integrate the identical physics (the chip factorizes
    # exactly along its independence graph), but as two separate ODE
    # integrations — joint over the full 81-dim space, split over two
    # independent 9-dim subspaces — so a small step-size-dependent residual
    # (observed ~1e-6) is expected even though there is no global-phase
    # ambiguity to account for here.
    assert np.allclose(joint_arr, split_arr, atol=1e-5)


def test_partitioned_result_getattr_raises_directed_message():
    """An unimplemented attribute access raises AttributeError directing the caller to '.components[i]'."""
    from quchip.results.partitioned import PartitionedSimulationResult

    part, results, tlist = _solved_components()
    combined = PartitionedSimulationResult(results, part, {})
    with pytest.raises(AttributeError, match=r"components\[i\]"):
        combined.plot_populations


def test_expect_cross_eop_rejects_nonzero_index():
    """expect() rejects a nonzero index on a cross-component correlator key, since a correlator is a single trace."""
    from quchip.chip.partition import CrossEop, LocalEop
    from quchip.results.partitioned import PartitionedSimulationResult

    part, results, tlist = _solved_components()
    plan = {("q0", "q2"): CrossEop(a=LocalEop(0, "q0", None), b=LocalEop(1, "q2", None))}
    combined = PartitionedSimulationResult(results, part, plan)
    with pytest.raises(ValueError, match="index"):
        combined.expect(("q0", "q2"), index=0)
