"""Ladder rung 3 (spec Sec. 9): method="sw" vs method="exact" bus elimination agree; exact zz is exact.

Reuses the standard bridge fixture (a bus resonator mediating two
capacitively coupled qubits) to compare ``eliminate()``'s two device-target
routes: the perturbative second-order Schrieffer-Wolff route
(``method="sw"``) and the exact-from-dressing route (``method="exact"``, one
full diagonalization). See ``tests/extended/test_sw_vs_exact_grad.py`` for
the companion gradient check.
"""

from __future__ import annotations

import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Resonator
from quchip.chip.transformations import eliminate

_LEG_G = 0.08
_Q0_FREQ = 5.0
_Q1_FREQ = 5.2
_BUS_FREQ = 6.3


def _bridge_chip() -> Chip:
    q0 = DuffingTransmon(freq=_Q0_FREQ, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=_Q1_FREQ, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=_BUS_FREQ, levels=4, label="bus")
    couplings = [
        Capacitive(q0, bus, g=_LEG_G, rwa=True, label="leg0"),
        Capacitive(q1, bus, g=_LEG_G, rwa=True, label="leg1"),
    ]
    return Chip([q0, q1, bus], couplings=couplings, frame="rotating", rwa=True)


def test_sw_and_exact_freq_after_agree_within_dispersive_tolerance():
    """Both eliminate() routes agree on each survivor's post-reduction freq to (g/Delta)^2 relative."""
    full = _bridge_chip()
    sw = eliminate(full, "bus", method="sw")
    exact = eliminate(full, "bus", method="exact")

    # g/Delta = 0.08/1.3 = 0.0615 (q0 leg), 0.08/1.1 = 0.0727 (q1 leg, the larger); tolerance = 0.0727^2 = 5.29e-3.
    tol = (_LEG_G / (_BUS_FREQ - _Q1_FREQ)) ** 2
    # Measured: q0 rel = 2.03e-5, q1 rel = 3.79e-5 — both far under the bound.
    for label in ("q0", "q1"):
        f_sw = float(sw.effective_params[label]["freq_after"])
        f_exact = float(exact.effective_params[label]["freq_after"])
        rel = abs(f_sw - f_exact) / abs(f_exact)
        assert rel < tol, (label, f_sw, f_exact, rel)


def test_exact_zz_matches_chip_dispersive_shift_exactly():
    """method="exact"'s exchange["zz"] equals chip.dispersive_shift, read off the same dressed spectrum."""
    full = _bridge_chip()
    exact = eliminate(full, "bus", method="exact")
    zz = exact.effective_params["exchange"]["zz"]
    assert zz == pytest.approx(full.dispersive_shift("q0", "q1"), rel=1e-12)


def test_min_block_gap_exceeds_coupling_scale():
    """The Sylvester generator's block gap is symmetric across legs and exceeds the leg coupling g (validity margin)."""
    full = _bridge_chip()
    result = eliminate(full, "bus", method="sw")
    gap_leg0 = float(result.validity["leg0"]["min_block_gap"])
    gap_leg1 = float(result.validity["leg1"]["min_block_gap"])
    assert gap_leg0 == pytest.approx(gap_leg1)
    # Measured min_block_gap ~= 1.1 GHz (tighter q1-bus detuning) vs g = 0.08 GHz -- comfortably inside
    # eliminate()'s declared-valid perturbative regime (g_over_delta < 0.1).
    assert gap_leg0 > _LEG_G
