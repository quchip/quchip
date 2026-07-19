"""ParametricDrive wiring surface (spec §4.3)."""

from __future__ import annotations

import pytest

from quchip import Capacitive, Chip, ControlEquipment, DuffingTransmon, ParametricDrive, TunableCapacitive


def _parts():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
    return q0, q1, tc


def test_wire_and_connect_edge_line():
    """A ParametricDrive from a coupling reports its label as target_label and appears in control-equipment lines."""
    q0, q1, tc = _parts()
    pump = ParametricDrive(tc, label="pump")
    chip = Chip([q0, q1], couplings=[tc])
    chip.connect(ControlEquipment([pump]))
    assert pump.target_label == "tc"
    assert chip.control_equipment.lines[0] is pump


def test_label_late_binding_via_chip():
    """A ParametricDrive from a coupling label rebinds to the chip's canonical coupling instance once connected."""
    q0, q1, tc = _parts()
    pump = ParametricDrive("tc", label="pump")
    chip = Chip([q0, q1], couplings=[tc])
    chip.connect(ControlEquipment([pump]))
    assert pump._target is tc  # rebound to the canonical instance


def test_static_coupling_is_rejected_with_teaching_error():
    """Wrapping a static coupling in ParametricDrive raises TypeError naming the missing parametric_interaction hook."""
    q0, q1, _ = _parts()
    c = Capacitive(q0, q1, g=0.005, label="c")
    with pytest.raises(TypeError, match="parametric_interaction"):
        ParametricDrive(c)


def test_unknown_coupling_label_raises_at_connect():
    """Connecting a ParametricDrive whose target label matches no chip coupling raises ValueError naming that label."""
    q0, q1, tc = _parts()
    pump = ParametricDrive("nope", label="pump")
    chip = Chip([q0, q1], couplings=[tc])
    with pytest.raises(ValueError, match="nope"):
        chip.connect(ControlEquipment([pump]))


def test_rwa_kwarg_rejected():
    """ParametricDrive rejects an explicit rwa keyword with ValueError; the coupling's own hook fixes the RWA policy."""
    _, _, tc = _parts()
    with pytest.raises(ValueError, match="RWA"):
        ParametricDrive(tc, rwa=True)  # type: ignore[call-arg]


def test_clone_rebinds_edge_lines():
    """Chip.clone() deep-copies an edge-targeting control line and rebinds it to the clone's own coupling instance."""
    q0, q1, tc = _parts()
    pump = ParametricDrive(tc, label="pump")
    chip = Chip([q0, q1], couplings=[tc], control_equipment=None)
    chip.connect(ControlEquipment([pump]))
    cloned = chip.clone()
    cloned_pump = cloned.control_equipment.lines[0]
    assert cloned_pump is not pump
    assert cloned_pump._target is cloned.coupling("tc")


def test_serialization_round_trip_resolves_edge_line_through_coupling_map():
    """Chip.from_dict(chip.to_dict()) rebinds an edge ParametricDrive to the restored chip's coupling, still usable."""
    import numpy as np

    from quchip import ChargeDrive, QuantumSequence, Square

    q0, q1, tc = _parts()
    pump = ParametricDrive(tc, label="pump")
    dq = ChargeDrive(q0, label="dq")
    chip = Chip([q0, q1], couplings=[tc], control_equipment=ControlEquipment([pump, dq]))

    restored = Chip.from_dict(chip.to_dict())

    restored_pump = restored.control_equipment.lines[0]
    assert restored_pump.target_kind == "edge"
    assert restored_pump._target is restored.coupling("tc")

    seq = QuantumSequence(restored)
    seq.pump("tc", envelope=Square(duration=50.0, amplitude=0.001))
    seq.build_problem(tlist=np.linspace(0.0, 50.0, 5))
