"""Tests for the eliminate() fold report from EliminationResult.describe()."""

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


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_describe_reports_intrinsic_t1_separately_from_inherited_loss(method):
    """A shared transformed channel is not presented as a change in intrinsic T1."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=30_000.0)
    r = Resonator(freq=7.0, internal_quality_factor=5000.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.08, label="cap0")])
    res = eliminate(chip, "r", method=method)
    report = res.describe()
    assert res.chip["q"].T1 == 30_000.0
    assert "intrinsic T1: 30" in report
    assert "Purcell contribution to 1→0 lowering:" in report
    assert "1/ns (separate channel)" in report


def test_describe_zz_availability_depends_on_method():
    """The report marks ZZ unavailable under SW and gives its exact value."""
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


@pytest.mark.optional_backend
@pytest.mark.validation
def test_describe_never_raises_on_a_fully_traced_result():
    """describe() stays exception-free when every effective parameter is a JAX tracer."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    def run(g):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0", T1=30_000.0)
        q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=2, label="q1")
        bus = Resonator(freq=6.3, levels=3, label="bus", internal_quality_factor=5000.0)
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

    jax.jit(jax.grad(run))(0.08)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_deferred_chi_retains_original_model_after_source_edits(backend):
    """Deferred diagnostics retain their model across edits and traced reads."""
    import jax

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = Capacitive(q, r, g=0.08)
    chip = Chip([q, r], [coupling], backend=backend)
    expected = float(chip.dispersive_shift("q", "r"))
    reduced = eliminate(chip, "r")
    q.freq = 5.5
    r.freq = 6.9
    coupling.g = 0.04
    q.levels = 4
    assert float(jax.jit(lambda: reduced.effective_params["q"]["chi"])()) == pytest.approx(expected, abs=1e-12)
    assert float(reduced.effective_params["q"]["chi"]) == pytest.approx(expected, abs=1e-12)
