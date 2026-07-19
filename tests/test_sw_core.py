"""Schrieffer-Wolff kernels reproduce the closed-form 2nd-order results (spec §6.2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from quchip import Capacitive, Chip, DuffingTransmon, Resonator
from quchip.chip.sw import (
    bare_hamiltonian,
    h_effective_second_order,
    mode_blocks,
    extract_pair_parameters,
    pathway_attribution,
    sylvester_generator,
)


def _two_qubit_exchange(omega_a: float, omega_b: float, g: float) -> jnp.ndarray:
    """H for two two-level systems with an RWA exchange, C-order product basis (na, nb)."""
    h = jnp.diag(jnp.array([0.0, omega_b, omega_a, omega_a + omega_b], dtype=complex))
    h = h.at[1, 2].set(g).at[2, 1].set(g)  # |01><10| + h.c.
    return h


def test_two_level_exchange_reproduces_level_repulsion():
    """Eliminating either qubit shifts its partner's frequency by g²/Δ, sign flipped by which mode is eliminated."""
    omega_a, omega_b, g = 5.0, 5.3, 0.02
    delta = omega_a - omega_b
    h = _two_qubit_exchange(omega_a, omega_b, g)
    dims, labels = (2, 2), ["a", "b"]

    p_mask, _ = mode_blocks(dims, labels, "b")
    s, _ = sylvester_generator(h, p_mask)
    h_eff = h_effective_second_order(h, s, p_mask)
    params = extract_pair_parameters(h_eff, np.flatnonzero(p_mask), labels, dims, "b")
    assert abs(float(params["a"]["freq_after"]) - (omega_a + g**2 / delta)) < 1e-12

    # Repulsion is symmetric: eliminating the other mode pushes the other way.
    p_mask_a, _ = mode_blocks(dims, labels, "a")
    s_a, _ = sylvester_generator(h, p_mask_a)
    h_eff_a = h_effective_second_order(h, s_a, p_mask_a)
    params_a = extract_pair_parameters(h_eff_a, np.flatnonzero(p_mask_a), labels, dims, "a")
    assert abs(float(params_a["b"]["freq_after"]) - (omega_b - g**2 / delta)) < 1e-12


def test_dispersive_lamb_shift_on_a_real_chip():
    """Eliminating the resonator gives the qubit the dispersive Lamb shift g²/Δ; the P/Q gap equals the detuning."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05, label="c")], rwa=True)

    h, labels, dims = bare_hamiltonian(chip, chip.backend)
    p_mask, _ = mode_blocks(dims, labels, "r")
    s, min_gap = sylvester_generator(h, p_mask)
    h_eff = h_effective_second_order(h, s, p_mask)
    params = extract_pair_parameters(h_eff, np.flatnonzero(p_mask), labels, dims, "r")

    lamb = float(params["q"]["freq_after"]) - 5.0
    expected = 0.05**2 / (5.0 - 7.0)
    assert abs(lamb - expected) / abs(expected) < 1e-9
    # The relevant P<->Q gap is the qubit-resonator detuning itself.
    assert abs(float(min_gap) - 2.0) < 0.3


def _bridge_h():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=6.3, levels=4, label="bus")
    chip = Chip(
        [q0, q1, bus],
        couplings=[Capacitive(q0, bus, g=0.08, label="leg0"), Capacitive(q1, bus, g=0.08, label="leg1")],
        rwa=True,
    )
    return bare_hamiltonian(chip, chip.backend)


def test_bridge_exchange_matches_yan_formula():
    """Eliminating the bus yields an effective qubit-qubit exchange J matching the Yan formula g0*g1/2*(1/Δ0+1/Δ1)."""
    h, labels, dims = _bridge_h()
    p_mask, _ = mode_blocks(dims, labels, "bus")
    s, _ = sylvester_generator(h, p_mask)
    h_eff = h_effective_second_order(h, s, p_mask)
    params = extract_pair_parameters(h_eff, np.flatnonzero(p_mask), labels, dims, "bus")

    j = params[("J", "q0", "q1")]
    expected = 0.08 * 0.08 / 2.0 * (1.0 / (5.0 - 6.3) + 1.0 / (5.2 - 6.3))
    assert abs(float(j.real) - expected) / abs(expected) < 1e-6


def test_degenerate_cross_block_with_zero_coupling_is_guarded():
    """Exactly degenerate P/Q levels with no matrix element between them: no NaN, finite grad."""
    omega = 5.0
    dims, labels = (2, 2), ["a", "b"]
    p_mask, _ = mode_blocks(dims, labels, "b")

    def ground_energy(g):
        h = jnp.diag(jnp.array([0.0, omega, omega, 2.0 * omega], dtype=complex))
        h = h.at[0, 3].set(g).at[3, 0].set(g)  # couples |00> <-> |11| only
        s, _ = sylvester_generator(h, p_mask)
        assert bool(jnp.all(jnp.isfinite(s)))
        h_eff = h_effective_second_order(h, s, p_mask)
        return h_eff[0, 0].real

    grad = jax.grad(ground_energy)(0.05)
    assert np.isfinite(float(grad))
    # d/dg of the 2nd-order shift g^2/(0 - 2*omega) = -g/omega.
    assert abs(float(grad) - (-0.05 / omega)) < 1e-12


def test_pathway_attribution_sums_to_commutator_element():
    """Summed pathway amounts equal the commutator element 0.5*(S@V-V@S); only the bus pathway contributes."""
    h, labels, dims = _bridge_h()
    p_mask, _ = mode_blocks(dims, labels, "bus")
    s, _ = sylvester_generator(h, p_mask)

    p_index = np.flatnonzero(p_mask)
    occ = np.array(np.unravel_index(np.arange(int(np.prod(dims))), dims))
    i_idx = int(np.flatnonzero((occ[0] == 1) & (occ[1] == 0) & (occ[2] == 0))[0])
    j_idx = int(np.flatnonzero((occ[0] == 0) & (occ[1] == 1) & (occ[2] == 0))[0])
    assert i_idx in p_index and j_idx in p_index

    paths = pathway_attribution(h, s, p_mask, i_idx, j_idx)
    total = sum(amount for _, amount in paths)
    e_diag = jnp.diag(jnp.diagonal(h))
    v = h - e_diag
    expected = (0.5 * (s @ v - v @ s))[i_idx, j_idx]
    assert abs(complex(total) - complex(expected)) < 1e-12
    # The bus single-excitation state is the only virtual pathway here.
    bus_idx = int(np.flatnonzero((occ[0] == 0) & (occ[1] == 0) & (occ[2] == 1))[0])
    assert [k for k, _ in paths] == [bus_idx]


def test_collapse_transform_yields_purcell_rate():
    """The mode's transformed jump operator hands the survivor a (g/Δ)²·κ decay."""
    from quchip.chip.sw import purcell_rate_from, transform_collapse

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r", quality_factor=1e4)
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05, label="c")], rwa=True)
    backend = chip.backend

    h, labels, dims = bare_hamiltonian(chip, backend)
    p_mask, _ = mode_blocks(dims, labels, "r")
    s, _ = sylvester_generator(h, p_mask)

    # Transform the UNIT lowering operator of the eliminated mode; the rate
    # multiplies back in at the end (purcell_rate_from's contract).
    b_local = r.annihilation_operator() if hasattr(r, "annihilation_operator") else None
    if b_local is None:
        import qutip

        b_local = qutip.destroy(4)
    b_full = jnp.asarray(backend.to_array(backend.embed(b_local, labels.index("r"), dims)), dtype=complex)
    c_eff = transform_collapse(b_full, s, p_mask)

    p_index = np.flatnonzero(p_mask)
    occ = np.array(np.unravel_index(p_index, dims))
    ground = int(np.flatnonzero((occ[0] == 0) & (occ[1] == 0))[0])
    q_excited = int(np.flatnonzero((occ[0] == 1) & (occ[1] == 0))[0])
    amplitude = c_eff[ground, q_excited]

    kappa = 2.0 * np.pi * 7.0 / 1e4
    rate = purcell_rate_from(amplitude, kappa)
    expected = (0.05 / 2.0) ** 2 * kappa  # (g/Δ)²·κ, |Δ| = 2.0
    assert abs(float(rate) - expected) / expected < 1e-6


def test_no_quality_factor_means_no_channel():
    """Q=None → the mode has no photon-loss channel; the caller's static decision bit."""
    r = Resonator(freq=7.0, levels=4, label="r_bare")
    assert r.collapse_operators() == []
