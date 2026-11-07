"""Retarget control lines stranded by eliminate()."""

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
    Resonator,
    TunableCapacitive,
)
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
    assert line.label == flux.label  # retargeting preserves the line label
    assert isinstance(line, ParametricDrive)
    assert line.target_label == "elim_fc"

    (gain,) = ce.signal_chain
    assert gain.line == flux.label
    assert np.isclose(complex(gain.factor).real, float(res.effective_params["exchange"]["dJ_domega_c"]))


def test_charge_drive_on_eliminated_leaf_still_raises_with_upgraded_message():
    """A ChargeDrive probe on an eliminated leaf raises ValueError that says to unwire the line or keep the target."""
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
    assert "no retarget rule converts it" in message
    assert "chip.unwire('probe')" in message and "or keep the target" in message


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
    assert "no retarget rule converts it" in message
    assert "chip.unwire('pump')" in message and "or keep the target" in message


def _three_survivor_chip():
    qs = [DuffingTransmon(freq=f, anharmonicity=-0.25, levels=3, label=f"q{i}") for i, f in enumerate([5.0, 5.2, 5.4])]
    fc = FluxTunableTransmon(freq=6.3, anharmonicity=-0.2, levels=3, label="fc")
    legs = [Capacitive(q, fc, g=0.08, label=f"leg{i}") for i, q in enumerate(qs)]
    return Chip(qs + [fc], couplings=legs, control_equipment=ControlEquipment([FluxDrive(fc, label="cflux")]))


@pytest.mark.validation
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
    assert pump_targets == {entry["coupling"] for entry in exchange.values()}

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
