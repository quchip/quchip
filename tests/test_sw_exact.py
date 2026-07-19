"""Exact-from-dressing reduction route: labeled energies, ZZ, and the collision guard (spec §6.2)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Resonator
from quchip.chip.sw import (
    bare_hamiltonian,
    exact_reduction,
    extract_pair_parameters,
    h_effective_second_order,
    mode_blocks,
    sylvester_generator,
)


def _bridge_chip(bus_freq: float = 6.3) -> Chip:
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=bus_freq, levels=4, label="bus")
    return Chip(
        [q0, q1, bus],
        couplings=[Capacitive(q0, bus, g=0.08, label="leg0"), Capacitive(q1, bus, g=0.08, label="leg1")],
        rwa=True,
    )


def test_exact_freq_after_matches_conditional_dressed_freq():
    """Exact reduction's freq_after equals each qubit's dressed frequency with the bus and other qubit in |0⟩."""
    chip = _bridge_chip()
    params = exact_reduction(chip, "bus", ["q0", "q1"])
    for surv, other in (("q0", "q1"), ("q1", "q0")):
        expected = chip.freq(surv, when={"bus": 0, other: 0})
        assert abs(float(params[surv]["freq_after"]) - float(expected)) < 1e-10


def test_exact_zz_matches_dispersive_shift():
    """Exact reduction's ZZ parameter for a qubit pair equals the chip's dispersive shift between them."""
    chip = _bridge_chip()
    params = exact_reduction(chip, "bus", ["q0", "q1"])
    assert float(params[("zz", "q0", "q1")]) == pytest.approx(float(chip.dispersive_shift("q0", "q1")), abs=1e-14)


def test_sw_and_exact_exchange_agree_at_second_order():
    """Exact and second-order SW reductions agree on the exchange coupling J within the fourth-order bound."""
    chip = _bridge_chip()
    exact = exact_reduction(chip, "bus", ["q0", "q1"])

    h, labels, dims = bare_hamiltonian(chip, chip.backend)
    p_mask, _ = mode_blocks(dims, labels, "bus")
    s, _ = sylvester_generator(h, p_mask)
    h_eff = h_effective_second_order(h, s, p_mask)
    sw_params = extract_pair_parameters(h_eff, np.flatnonzero(p_mask), labels, dims, "bus")

    j_exact = abs(complex(exact[("J", "q0", "q1")]))
    j_sw = abs(complex(sw_params[("J", "q0", "q1")]))
    # The routes differ at 4th order; the natural bound sums both legs:
    # (g/Δ₁)² + (g/Δ₂)² = 0.0038 + 0.0053 = 0.0091 (measured: 0.0089).
    tolerance = (0.08 / 1.3) ** 2 + (0.08 / 1.1) ** 2
    assert abs(j_exact - j_sw) / j_sw < tolerance


def test_duplicates_guard_raises_on_label_collision():
    """A bus degenerate with a qubit hybridizes 50/50 — bare labels stop meaning anything."""
    chip = _bridge_chip(bus_freq=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="[Nn]ear-degenerate"):
            exact_reduction(chip, "bus", ["q0", "q1"])
