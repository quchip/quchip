"""Target-dispatched ``eliminate()`` preserves the bridge's control capability."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    CrossKerr,
    DuffingTransmon,
    FluxTunableTransmon,
    ParametricDrive,
    Resonator,
    TunableCapacitive,
)
from quchip.chip.transformations import eliminate


def _bridge_chip(direct_g=None):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=6.3, levels=4, label="bus")
    couplings = [Capacitive(q0, bus, g=0.08, label="leg0"), Capacitive(q1, bus, g=0.08, label="leg1")]
    if direct_g is not None:
        couplings.append(Capacitive(q0, q1, g=direct_g, label="direct"))
    return Chip([q0, q1, bus], couplings=couplings)


def test_fixed_bridge_emits_capacitive_edge():
    """Eliminating a fixed bus emits a Capacitive edge carrying the mediated exchange J."""
    res = eliminate(_bridge_chip(), "bus")
    edge = res.chip.coupling("elim_bus")
    assert type(edge) is Capacitive
    j = res.effective_params["exchange"]["j_eff"]
    # J = g1 g2 / 2 (1/Δ1 + 1/Δ2), Δ1 = 5.0-6.3, Δ2 = 5.2-6.3
    assert np.isclose(float(j), 0.08 * 0.08 / 2 * (1 / -1.3 + 1 / -1.1))
    assert np.isclose(float(edge.g), float(j))


def test_frequency_tunable_bridge_emits_tunable_capacitive_edge():
    """Eliminating a frequency-controlled bus emits a TunableCapacitive edge."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    coupler = FluxTunableTransmon(freq=6.3, anharmonicity=-0.3, levels=3, label="coupler")
    chip = Chip(
        [q0, q1, coupler],
        couplings=[
            Capacitive(q0, coupler, g=0.08, label="leg0"),
            Capacitive(q1, coupler, g=0.08, label="leg1"),
        ],
    )

    edge = eliminate(chip, coupler).chip.coupling("elim_coupler")

    assert isinstance(edge, TunableCapacitive)


def test_bridge_folds_into_existing_direct_edge_preserving_label():
    """The mediated exchange folds into an existing direct edge, keeping that edge's label."""
    res = eliminate(_bridge_chip(direct_g=0.004), "bus")
    edge = res.chip.coupling("direct")
    assert type(edge) is Capacitive
    j = res.effective_params["exchange"]["j_eff"]
    assert np.isclose(float(edge.g), 0.004 + float(j))
    assert res.effective_params["exchange"]["folded_into"] == "direct"


def test_exchange_entry_carries_dj_domega():
    """The exchange entry reports dJ/dωc, the mediated exchange's derivative with respect to the bus frequency."""
    res = eliminate(_bridge_chip(), "bus")
    dj = res.effective_params["exchange"]["dJ_domega_c"]
    assert np.isclose(float(dj), 0.08 * 0.08 / 2 * (1 / 1.3**2 + 1 / 1.1**2))


def test_bridge_chain_folds_second_exchange_into_first_edge():
    """A second fixed bus folds its exchange into the Capacitive the first left behind."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    b1 = Resonator(freq=6.3, levels=4, label="b1")
    b2 = Resonator(freq=6.8, levels=4, label="b2")
    couplings = [
        Capacitive(q0, b1, g=0.08, label="leg0b1"),
        Capacitive(q1, b1, g=0.08, label="leg1b1"),
        Capacitive(q0, b2, g=0.06, label="leg0b2"),
        Capacitive(q1, b2, g=0.06, label="leg1b2"),
    ]
    chip = Chip([q0, q1, b1, b2], couplings=couplings)

    step1 = eliminate(chip, "b1")
    step2 = eliminate(step1.chip, "b2")

    (edge,) = step2.chip.couplings
    assert edge.label == "elim_b1"
    assert type(edge) is Capacitive
    assert step2.effective_params["exchange"]["folded_into"] == "elim_b1"

    # J1: bare-frequency bridge exchange from eliminating b1 first.
    j1 = 0.08 * 0.08 / 2 * (1 / (5.0 - 6.3) + 1 / (5.2 - 6.3))
    # Eliminating b1 also Lamb-shifts q0/q1 (freq_after = freq + g**2/delta) —
    # documented sequential-composition behavior — so J2 uses those shifted
    # frequencies, not the bare ones.
    q0_after_b1 = 5.0 + 0.08**2 / (5.0 - 6.3)
    q1_after_b1 = 5.2 + 0.08**2 / (5.2 - 6.3)
    j2 = 0.06 * 0.06 / 2 * (1 / (q0_after_b1 - 6.8) + 1 / (q1_after_b1 - 6.8))
    assert np.isclose(float(edge.g), j1 + j2)


def test_bridge_folds_into_user_built_tunable_capacitive_direct_edge():
    """A user-built TunableCapacitive direct edge is folded into (not shadowed by) the bridge exchange."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=6.3, levels=4, label="bus")
    direct = TunableCapacitive(q0, q1, g_0=0.004, label="direct")
    legs = [Capacitive(q0, bus, g=0.08, label="leg0"), Capacitive(q1, bus, g=0.08, label="leg1")]
    chip = Chip([q0, q1, bus], couplings=legs + [direct])

    res = eliminate(chip, "bus")

    (edge,) = res.chip.couplings
    assert edge.label == "direct"
    assert isinstance(edge, TunableCapacitive)
    j = res.effective_params["exchange"]["j_eff"]
    assert np.isclose(float(edge.g_0), 0.004 + float(j))


def _readout_chip():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = Capacitive(q, r, g=0.05, label="cap0")
    probe = ChargeDrive(r, label="probe")
    chip = Chip(
        [q, r],
        couplings=[coupling],
        control_equipment=ControlEquipment([probe]),
    )
    return chip, q, r, coupling


def test_coupling_target_keeps_both_devices_and_emits_crosskerr():
    """Eliminating a coupling (not a device) keeps both endpoints and emits a CrossKerr edge between them."""
    chip, q, r, coupling = _readout_chip()
    res = eliminate(chip, "cap0")
    reduced = res.chip

    assert {d.label for d in reduced.devices} == {"q", "r"}
    edge = reduced.coupling("elim_cap0")
    assert isinstance(edge, CrossKerr)
    assert edge.device_a_label == "q"
    assert edge.device_b_label == "r"


def test_coupling_target_chi_matches_dispersive_shift():
    """The emitted CrossKerr's chi matches the chip's dispersive shift between the two survivors."""
    chip, q, r, coupling = _readout_chip()
    expected_chi = chip.dispersive_shift("q", "r")
    res = eliminate(chip, "cap0")
    edge = res.chip.coupling("elim_cap0")
    assert np.isclose(float(edge.chi), float(expected_chi))


def test_coupling_target_reports_dressed_freq_after():
    """effective_params reports each survivor's dressed post-elimination frequency and Lamb shift."""
    chip, q, r, coupling = _readout_chip()
    expected_q = chip.freq("q", when={"r": 0})
    expected_r = chip.freq("r", when={"q": 0})
    res = eliminate(chip, "cap0")

    assert np.isclose(float(res.effective_params["q"]["freq_after"]), float(expected_q))
    assert np.isclose(float(res.effective_params["r"]["freq_after"]), float(expected_r))
    assert np.isclose(
        float(res.effective_params["q"]["lamb_shift"]), float(expected_q) - q.freq
    )
    assert np.isclose(
        float(res.effective_params["r"]["lamb_shift"]), float(expected_r) - r.freq
    )
    assert np.isclose(float(res.chip["q"].freq), float(expected_q))
    assert np.isclose(float(res.chip["r"].freq), float(expected_r))


def test_coupling_target_validity_reports_g_over_delta():
    """validity reports g/Δ for the eliminated coupling."""
    chip, q, r, coupling = _readout_chip()
    res = eliminate(chip, "cap0")
    validity = res.validity["cap0"]
    assert np.isclose(float(validity["g_over_delta"]), abs(0.05 / (5.0 - 7.0)))


def test_coupling_target_accepts_object_or_label():
    """Coupling-target elimination accepts either the coupling object or its label."""
    chip, q, r, coupling = _readout_chip()
    res = eliminate(chip, coupling)
    assert isinstance(res.chip.coupling("elim_cap0"), CrossKerr)


def test_coupling_target_drive_on_surviving_device_carries_through():
    """A ChargeDrive probing the surviving resonator is the effective-readout-chip flow."""
    chip, q, r, coupling = _readout_chip()
    res = eliminate(chip, "cap0")
    ce = res.chip.control_equipment
    assert ce is not None
    assert [line.label for line in ce.lines] == ["probe"]
    assert ce.lines[0]._target is res.chip["r"]


def test_coupling_target_raises_on_pump_targeting_eliminated_coupling():
    """A ParametricDrive pumping the eliminated coupling with no retarget rule raises ValueError."""
    # ParametricDrive only accepts a modulable coupling (parametric_interaction hook),
    # so the doomed pump here targets a TunableCapacitive edge, not a plain Capacitive.
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    tc = TunableCapacitive(q, r, g_0=0.05, label="tc0")
    pump = ParametricDrive(tc, label="pump")
    wired = Chip(
        [q, r],
        couplings=[tc],
        control_equipment=ControlEquipment([pump]),
    )
    with pytest.raises(ValueError, match="pumps the eliminated coupling 'tc0'"):
        eliminate(wired, "tc0")


def test_pair_keys_survive_device_vs_coupling_scan_order_mismatch():
    """Survivor-pair bookkeeping is keyed in device order, whatever order the legs are scanned in."""
    # Real chips declare a center mode before its outer neighbors, so the
    # coupling-scan order of a mode's survivors need not match the chip's
    # device order — the two sides of every ("J", a, b) lookup must agree
    # anyway (regression: study 05's chained coupler elimination).
    qa = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="qa")
    center = Resonator(freq=6.3, levels=4, label="center")
    qb = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="qb")
    bus = Resonator(freq=6.8, levels=4, label="bus")
    couplings = [
        # Legs scanned qb-first while the device list runs qa < center < qb.
        Capacitive(qb, bus, g=0.06, label="leg_b"),
        Capacitive(qa, bus, g=0.06, label="leg_a"),
        Capacitive(qa, center, g=0.08, label="leg_ca"),
        Capacitive(qb, center, g=0.08, label="leg_cb"),
    ]
    chip = Chip([qa, center, qb, bus], couplings=couplings)

    step1 = eliminate(chip, "center")
    step2 = eliminate(step1.chip, "bus")

    (edge,) = step2.chip.couplings
    assert type(edge) is Capacitive
    assert step2.effective_params["exchange"]["between"] in (("qa", "qb"), ("qb", "qa"))
    # Both folds contribute; the second J uses the first fold's Lamb-shifted
    # frequencies, so assert composition structurally and the magnitude scale.
    assert abs(float(edge.g)) > abs(float(step1.effective_params["exchange"]["j_eff"])) * 0.5


def test_result_mappings_accept_objects_and_either_pair_order():
    """Result mappings accept objects as well as labels as keys; pairs are unordered."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.4, anharmonicity=-0.24, levels=3, label="q2")
    bus = Resonator(freq=6.3, levels=4, label="bus")
    legs = [Capacitive(q, bus, g=0.08, label=f"leg{i}") for i, q in enumerate((q0, q1, q2))]
    chip = Chip([q0, q1, q2, bus], couplings=legs)
    res = eliminate(chip, bus)

    assert res.effective_params[q0] is res.effective_params["q0"]
    assert res.validity[legs[0]] is res.validity["leg0"]
    exchange = res.effective_params["exchange"]
    assert exchange[("q1", "q0")] is exchange[("q0", "q1")]
    assert exchange[(q1, q0)] is exchange[("q0", "q1")]
    assert (q2, q0) in exchange and ("nope", "q0") not in exchange
    assert exchange.get((q0, "q2")) is exchange[("q0", "q2")]

    # The coupling-target path hands back the same widened mappings.
    r = Resonator(freq=7.0, levels=4, label="r")
    qq = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="qq")
    cap = Capacitive(qq, r, g=0.05, label="cap")
    res2 = eliminate(Chip([qq, r], couplings=[cap]), cap)
    assert res2.effective_params[qq] is res2.effective_params["qq"]
    assert res2.validity[cap] is res2.validity["cap"]
