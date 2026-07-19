"""Tests for EliminationResult.describe() — the eliminate() fold report (spec §6.6)."""

from __future__ import annotations

import re

import pytest

from quchip import (
    Capacitive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    FluxDrive,
    FluxTunableTransmon,
    Resonator,
)
from quchip.chip.transformations import eliminate


def _bridge_chip():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=6.3, levels=4, label="bus")
    couplings = [Capacitive(q0, bus, g=0.08, label="leg0"), Capacitive(q1, bus, g=0.08, label="leg1")]
    return Chip([q0, q1, bus], couplings=couplings)


def test_describe_reports_correct_before_after_freq_for_bridge():
    """The bridge report states each survivor's exact before/after freq and a passing validity mark."""
    res = eliminate(_bridge_chip(), "bus")
    report = res.describe()

    q0_after = float(res.effective_params["q0"]["freq_after"])
    q0_before = q0_after - float(res.effective_params["q0"]["lamb_shift"])
    q1_after = float(res.effective_params["q1"]["freq_after"])
    q1_before = q1_after - float(res.effective_params["q1"]["lamb_shift"])

    assert f"{q0_before:.6g} → {q0_after:.6g} GHz" in report
    assert f"{q1_before:.6g} → {q1_after:.6g} GHz" in report
    assert "✓" in report
    assert "dropped:" in report


def test_describe_reports_correct_purcell_folded_t1():
    """A leaf elimination's Purcell fold shows the exact before/after T1 in microseconds."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=30_000.0)
    r = Resonator(freq=7.0, quality_factor=5000.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08, label="cap0")])

    res = eliminate(chip, "r")
    report = res.describe()

    t1_after = float(res.chip["q"].T1)
    purcell_rate = float(res.effective_params["q"]["purcell_rate"])
    t1_before = 1.0 / (1.0 / t1_after - purcell_rate)

    assert f"{t1_before / 1e3:.4g} → {t1_after / 1e3:.4g} µs" in report
    assert "Purcell" in report


def test_describe_zz_is_placeholder_under_sw_and_a_number_under_exact():
    """The ZZ line is the method="exact" placeholder under sw, and the real number under exact."""
    chip = _bridge_chip()

    sw_report = eliminate(chip, "bus", method="sw").describe()
    assert 'ZZ(q0, q1) = —   (available under method="exact")' in sw_report

    exact_report = eliminate(chip, "bus", method="exact").describe()
    match = re.search(r"ZZ\(q0, q1\) = ([-\d.eE+]+) MHz", exact_report)
    assert match is not None
    reported_zz_ghz = float(match.group(1)) / 1e3
    assert reported_zz_ghz == pytest.approx(float(chip.dispersive_shift("q0", "q1")), rel=1e-3)


def test_describe_shows_the_gain_retarget_line_for_a_converted_flux_drive():
    """A FluxDrive converted to a ParametricDrive + Gain pump reports the swap in the fold report."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    fc = FluxTunableTransmon(freq=6.3, anharmonicity=-0.2, levels=3, label="fc")
    couplings = [Capacitive(q0, fc, g=0.08, label="leg0"), Capacitive(q1, fc, g=0.08, label="leg1")]
    flux = FluxDrive(fc, label="flux_fc")
    chip = Chip([q0, q1, fc], couplings=couplings, control_equipment=ControlEquipment([flux]))

    report = eliminate(chip, "fc").describe()

    assert "drive 'flux_fc'" in report
    assert "FluxDrive('fc')" in report
    assert "ParametricDrive on 'elim_fc'" in report
    assert "Gain" in report


def test_describe_exchange_edge_names_the_capacitive_strength_g():
    """The fold report's edge line names a fixed mediated edge's strength 'g', never '<traced>' when concrete."""
    res = eliminate(_bridge_chip(), "bus")
    report = res.describe()

    edge_label = res.effective_params["exchange"]["folded_into"]
    edge = res.chip.coupling_map[edge_label]
    assert type(edge).__name__ == "Capacitive"
    assert f"{edge_label}': Capacitive(g = " in report
    assert "<traced>" not in report


def test_describe_exchange_edge_names_the_tunable_capacitive_strength_g_0():
    """The fold report's edge line names a frequency-controlled mode's mediated edge strength 'g_0'."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    fc = FluxTunableTransmon(freq=6.3, anharmonicity=-0.2, levels=3, label="fc")
    couplings = [Capacitive(q0, fc, g=0.08, label="leg0"), Capacitive(q1, fc, g=0.08, label="leg1")]
    chip = Chip([q0, q1, fc], couplings=couplings)

    res = eliminate(chip, "fc")
    report = res.describe()

    edge_label = res.effective_params["exchange"]["folded_into"]
    edge = res.chip.coupling_map[edge_label]
    assert type(edge).__name__ == "TunableCapacitive"
    assert f"{edge_label}': TunableCapacitive(g_0 = " in report
    assert "<traced>" not in report


@pytest.mark.optional_backend
def test_describe_never_raises_on_a_fully_traced_result():
    """describe() stays exception-free when every effective parameter is a JAX tracer."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    def run(g):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0", T1=30_000.0)
        q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=2, label="q1")
        bus = Resonator(freq=6.3, levels=3, label="bus", quality_factor=5000.0)
        chip = Chip(
            [q0, q1, bus],
            couplings=[Capacitive(q0, bus, g=g, label="leg0"), Capacitive(q1, bus, g=0.08, label="leg1")],
            backend="dynamiqs",
        )
        res = eliminate(chip, "bus")
        report = res.describe()
        assert isinstance(report, str)
        assert "<traced>" in report
        return jnp.real(res.effective_params["exchange"]["j_eff"])

    jax.grad(run)(0.08)
