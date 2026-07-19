"""Eliminated single-qubit model reproduces the full qubit+resonator physics."""
import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Resonator, simulate
from quchip.chip.transformations import eliminate


def test_purcell_decay_matches_full_solve():
    """The eliminated single-qubit model's Purcell decay matches the full qubit+lossy-resonator solve."""
    # Full: excited qubit + lossy resonator -> qubit decays at the Purcell rate.
    g = 0.04
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=7.0, quality_factor=2000.0, levels=4, label="r")
    full = Chip([q, r], couplings=[Capacitive(q, r, g=g)])

    reduced = eliminate(full, "r").chip
    tlist = np.linspace(0.0, 300.0, 150)

    full_excited = simulate(
        full, [], tlist,
        initial_state=full.bare_state({q: 1}),
        e_ops={q: q.projector(1, 1)},
        check_truncation=False,
    )
    red_q = reduced["q"]
    red_excited = simulate(
        reduced, [], tlist,
        initial_state=reduced.bare_state({red_q: 1}),
        e_ops={red_q: red_q.projector(1, 1)},
        check_truncation=False,
    )

    full_curve = np.real(full_excited.expect("q"))
    red_curve = np.real(red_excited.expect("q"))
    # g/Delta = 0.04/2.0 = 0.02, (g/Delta)^2 = 4e-4: the leading-order dispersive scale for
    # this config. Measured max curve difference = 1.35e-3, the same order of magnitude
    # (Purcell decay integrates the dispersive residual over the trace, an O(1) factor above
    # the pointwise (g/Delta)^2 scale); abs < 0.05 keeps a ~37x margin over the measured value.
    assert np.max(np.abs(full_curve - red_curve)) < 0.05


def test_lamb_shift_matches_full_dressed_frequency():
    """The eliminated qubit's Lamb-shifted 0->1 frequency matches the full chip's exact dressed frequency."""
    g = 0.04
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    full = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
    reduced = eliminate(full, "r").chip
    # Reduced bare 0->1 vs full dressed 0->1 (resonator in vacuum). eliminate()'s Lamb shift
    # reads the chip's exact dressed pull (one full diagonalization), not a leading-order
    # (g/Delta)^2 = 4e-4 estimate, so agreement is near machine precision (measured rel =
    # 6.4e-8); rel=2e-3 is a loose, defensible ceiling well above that floor.
    full_01 = full.freq("q", when={"r": 0})
    assert reduced["q"].freq == pytest.approx(float(full_01), rel=2e-3)
