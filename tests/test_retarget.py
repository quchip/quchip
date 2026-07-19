"""Tests for the retarget registry: control lines stranded by eliminate() (spec §6.4)."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    FluxDrive,
    FluxTunableTransmon,
    ParametricDrive,
    QuantumSequence,
    Resonator,
    Square,
    TunableCapacitive,
    register_retarget_rule,
)
from quchip.chip.retarget import RetargetResult, _RETARGET_RULES
from quchip.chip.transformations import eliminate


def _flux_bridge_chip():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    fc = FluxTunableTransmon(freq=6.3, anharmonicity=-0.2, levels=3, label="fc")
    couplings = [Capacitive(q0, fc, g=0.08, label="leg0"), Capacitive(q1, fc, g=0.08, label="leg1")]
    flux = FluxDrive(fc, label="flux_fc")
    chip = Chip(
        [q0, q1, fc],
        couplings=couplings,
        control_equipment=ControlEquipment([flux]),
    )
    return chip, flux


def test_flux_drive_on_eliminated_bridge_converts_to_parametric_pump():
    """A FluxDrive on the bridge mode becomes a ParametricDrive on the emitted edge."""
    chip, flux = _flux_bridge_chip()
    res = eliminate(chip, "fc")
    reduced = res.chip

    ce = reduced.control_equipment
    assert ce is not None
    (line,) = ce.lines
    assert line.label == flux.label  # label preserved (spec §6.4)
    assert isinstance(line, ParametricDrive)
    assert line.target_label == "elim_fc"

    (gain,) = ce.signal_chain
    assert gain.line == flux.label
    assert np.isclose(complex(gain.factor).real, float(res.effective_params["exchange"]["dJ_domega_c"]))


def test_flux_drive_conversion_note_reports_the_swap():
    """The elimination's notes record the FluxDrive-to-ParametricDrive conversion by drive type and line label."""
    chip, flux = _flux_bridge_chip()
    res = eliminate(chip, "fc")
    assert any("FluxDrive" in note and "ParametricDrive" in note and flux.label in note for note in res.notes)


def test_retargeted_line_schedules_by_object_coupling_label_and_drive_label():
    """Scheduling by the drive's own label survives elimination; the retargeted line also schedules by object/label."""
    chip, flux = _flux_bridge_chip()

    # Portability: the drive label schedules on the FULL chip too.
    QuantumSequence(chip).schedule(flux.label, envelope=Square(duration=20.0, amplitude=0.01))

    reduced = eliminate(chip, "fc").chip
    (line,) = reduced.control_equipment.lines
    assert line.label == flux.label
    assert isinstance(line, ParametricDrive)

    # Object form.
    QuantumSequence(reduced).schedule(line, envelope=Square(duration=20.0, amplitude=0.01))
    # Coupling-label form — the standard ParametricDrive pump-line lookup.
    QuantumSequence(reduced).schedule("elim_fc", envelope=Square(duration=20.0, amplitude=0.01))
    # Drive-label form — schedule()'s string fallback onto the control-equipment
    # line directly; no freq needed (baseband).
    handle = QuantumSequence(reduced).schedule(flux.label, envelope=Square(duration=20.0, amplitude=0.01))
    assert handle is not None


def test_charge_drive_on_eliminated_leaf_still_raises_with_upgraded_message():
    """A ChargeDrive probe on an eliminated leaf raises ValueError, naming register_retarget_rule and chip.unwire()."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    probe = ChargeDrive(r, label="probe")
    chip = Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=0.05)],
        control_equipment=ControlEquipment([probe]),
    )
    with pytest.raises(ValueError) as exc_info:
        eliminate(chip, "r")
    message = str(exc_info.value)
    assert "register_retarget_rule" in message
    assert "chip.unwire('probe')" in message


def test_parametric_drive_on_doomed_leg_edge_raises():
    """A pump line on a leg coupling of the eliminated mode always raises (no edge->? rule ships)."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    leg = TunableCapacitive(q, r, g_0=0.05, label="leg")
    pump = ParametricDrive(leg, label="pump")
    chip = Chip(
        [q, r],
        couplings=[leg],
        control_equipment=ControlEquipment([pump]),
    )
    with pytest.raises(ValueError) as exc_info:
        eliminate(chip, "r")
    message = str(exc_info.value)
    assert "register_retarget_rule" in message
    assert "chip.unwire('pump')" in message


def test_custom_leaf_fold_rule_converts_and_reports_its_note():
    """A user-registered (ChargeDrive, Resonator, 'leaf-fold') rule converts and its note lands in res.notes."""

    def _swap_for_charge_on_q(line, ctx):
        replacement = ChargeDrive(ctx.reduced_chip["q"], label=line.label)
        return RetargetResult(lines=(replacement,), note=f"custom rule converted '{line.label}'")

    register_retarget_rule(ChargeDrive, Resonator, "leaf-fold", _swap_for_charge_on_q)
    try:
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        probe = ChargeDrive(r, label="probe")
        chip = Chip(
            [q, r],
            couplings=[Capacitive(q, r, g=0.05)],
            control_equipment=ControlEquipment([probe]),
        )
        res = eliminate(chip, "r")
        (line,) = res.chip.control_equipment.lines
        assert line.label == "probe"
        assert isinstance(line, ChargeDrive)
        assert line.device_label == "q"
        assert any("custom rule converted 'probe'" in note for note in res.notes)
    finally:
        del _RETARGET_RULES[(ChargeDrive, Resonator, "leaf-fold")]


def _three_survivor_chip():
    qs = [DuffingTransmon(freq=f, anharmonicity=-0.25, levels=3, label=f"q{i}") for i, f in enumerate([5.0, 5.2, 5.4])]
    fc = FluxTunableTransmon(freq=6.3, anharmonicity=-0.2, levels=3, label="fc")
    legs = [Capacitive(q, fc, g=0.08, label=f"leg{i}") for i, q in enumerate(qs)]
    return Chip(qs + [fc], couplings=legs, control_equipment=ControlEquipment([FluxDrive(fc, label="cflux")]))


def test_flux_drive_on_three_survivor_mode_converts_one_pump_per_edge():
    """One flux knob moves every pairwise J; the conversion emits one weighted pump per emitted edge."""
    from quchip.control.signal import Crosstalk, Gain

    res = eliminate(_three_survivor_chip(), "fc")
    ce = res.chip.control_equipment
    assert ce is not None

    exchange = res.effective_params["exchange"]
    assert set(exchange) == {("q0", "q1"), ("q0", "q2"), ("q1", "q2")}

    lines = {line.label: line for line in ce.lines}
    assert all(isinstance(line, ParametricDrive) for line in lines.values())
    # The first emitted pair's pump keeps the flux line's label (static,
    # emission-order choice); the rest carry derived labels.
    assert set(lines) == {"cflux", "cflux_q0_q2", "cflux_q1_q2"}
    pump_targets = {line.target_label for line in lines.values()}
    assert pump_targets == {entry["folded_into"] for entry in exchange.values()}

    copies = [t for t in ce.signal_chain if isinstance(t, Crosstalk)]
    gains = {t.line: t for t in ce.signal_chain if isinstance(t, Gain)}
    assert {(c.source, c.victim, c.beta) for c in copies} == {
        ("cflux", "cflux_q0_q2", 1.0),
        ("cflux", "cflux_q1_q2", 1.0),
    }
    # Every pump carries its own linearized weight — no ratios anywhere.
    expected_gain = {
        ("q0", "q1"): "cflux",
        ("q0", "q2"): "cflux_q0_q2",
        ("q1", "q2"): "cflux_q1_q2",
    }
    for pair, pump_label in expected_gain.items():
        assert np.isclose(complex(gains[pump_label].factor).real, float(exchange[pair]["dJ_domega_c"]))
    # Copies precede gains in the chain: a Gain on a copy-fed line is a
    # no-op until the copy has landed.
    chain_types = [type(t).__name__ for t in ce.signal_chain]
    assert chain_types.index("Gain") > max(i for i, t in enumerate(chain_types) if t == "Crosstalk")


def test_three_survivor_replay_compiles_through_stage_two():
    """The replayed schedule('cflux', ...) drives all three edges on the reduced chip."""
    res = eliminate(_three_survivor_chip(), "fc")
    seq = QuantumSequence(res.chip)
    seq.schedule("cflux", envelope=Square(duration=100.0, amplitude=0.02))
    problem = seq.build_problem(tlist=np.linspace(0.0, 100.0, 11))
    tags = [t.tag for t in problem.hamiltonian.dynamic_terms]
    # All three emitted edges receive drive terms: the scheduled pump compiles
    # its own edge (2 exchange bands), and the two copy-fed pumps compile
    # through the crosstalk path (2 bands each).
    assert tags.count("edge_pump") == 2
    assert tags.count("crosstalk") == 4

