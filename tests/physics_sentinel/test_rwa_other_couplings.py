"""RWA band policy beyond Capacitive: diagonal, longitudinal, multi-photon, and circuit-level couplings."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Chip, Coupling, CrossKerr, DuffingTransmon, Fluxonium, QuantumSequence, Resonator, simulate
from quchip.declarative import CouplingModel, Scalar, parameter


def _arr(chip, op):
    return np.asarray(chip.backend.to_array(op))


def _qr(tag: str):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label=f"q{tag}")
    r = Resonator(freq=7.0, levels=4, label=f"r{tag}")
    return q, r


def _longitudinal(a, b, bk):
    return bk.tensor(a.number_operator(), b.lowering_operator() + bk.dag(b.lowering_operator()))


def test_cross_kerr_is_rwa_invariant():
    """Diagonal coupling: every band is (0, 0), so RWA changes nothing and drops nothing."""
    q, r = _qr("ck_t")
    chip_rwa = Chip([q, r], [CrossKerr(q, r, chi=-0.002)], rwa=True)
    q2, r2 = _qr("ck_f")
    chip_full = Chip([q2, r2], [CrossKerr(q2, r2, chi=-0.002, rwa=False)], rwa=False)
    np.testing.assert_allclose(
        _arr(chip_rwa, chip_rwa.hamiltonian()), _arr(chip_full, chip_full.hamiltonian()), atol=1e-14
    )

    q3, r3 = _qr("ck_rot")
    chip_rot = Chip([q3, r3], [CrossKerr(q3, r3, chi=-0.002)], frame="rotating", rwa=True)
    problem = QuantumSequence(chip_rot).build_problem(
        tlist=np.linspace(0.0, 10.0, 11), initial_state=chip_rot.bare_state()
    )
    # All bands carry concretely-zero frame carriers: fully folded into H0, nothing dropped.
    assert [t for t in problem.hamiltonian.dynamic_terms if t.origin == "coupling"] == []
    assert problem.hamiltonian.dropped_terms == ()


def test_longitudinal_coupling_masks_to_zero_with_advisories():
    """n̂_a (b + b†) has no excitation-conserving band: RWA removes it entirely and says so."""
    qa, rb = _qr("lg_m")
    chip = Chip(
        [qa, rb], [Coupling(qa, rb, g=0.05, interaction=_longitudinal, label="long")], frame="rotating", rwa=True
    )
    qa2, rb2 = _qr("lg_b")
    bare = Chip([qa2, rb2], [], frame="rotating")
    with pytest.warns(UserWarning, match="vanishes entirely under the resolved RWA"):
        h_masked = _arr(chip, chip.hamiltonian())
    np.testing.assert_allclose(h_masked, _arr(bare, bare.hamiltonian()), atol=1e-14)

    problem = QuantumSequence(chip).build_problem(tlist=np.linspace(0.0, 10.0, 11), initial_state=chip.bare_state())
    records = problem.hamiltonian.dropped_terms
    assert {rec.band_weights for rec in records} == {(0, -1), (0, 1)}
    assert all(float(rec.frequency) == 7.0 for rec in records)

    # Escape hatch: rwa=False keeps both bands as explicit carriers at ±ω_r.
    qa3, rb3 = _qr("lg_f")
    chip_full = Chip(
        [qa3, rb3],
        [Coupling(qa3, rb3, g=0.05, interaction=_longitudinal, rwa=False, label="longf")],
        frame="rotating",
        rwa=True,
    )
    problem_full = QuantumSequence(chip_full).build_problem(
        tlist=np.linspace(0.0, 10.0, 11), initial_state=chip_full.bare_state()
    )
    assert len([t for t in problem_full.hamiltonian.dynamic_terms if t.origin == "coupling"]) == 2
    assert problem_full.hamiltonian.dropped_terms == ()


def test_two_photon_exchange_survives_rwa():
    """g(a²b†² + h.c.) sits in the (±2, ∓2) bands — number conserving, retained, and it drives dynamics."""
    ka = Resonator(freq=5.0, levels=3, label="ka2ph")
    kb = Resonator(freq=5.0, levels=3, label="kb2ph")
    g = 0.01

    def two_photon(a, b, bk):
        low_a, low_b = a.lowering_operator(), b.lowering_operator()
        return bk.tensor(low_a @ low_a, bk.dag(low_b) @ bk.dag(low_b)) + bk.tensor(
            bk.dag(low_a) @ bk.dag(low_a), low_b @ low_b
        )

    chip = Chip([ka, kb], [Coupling(ka, kb, g=g, interaction=two_photon, label="twoph")], frame="rotating", rwa=True)
    tlist = np.linspace(0.0, 30.0, 121)
    # The |2⟩ ↔ |0⟩ exchange deliberately occupies the top Fock level of the
    # 3-level ladders, so the truncation safety net is switched off.
    result = simulate(
        chip,
        [],
        tlist,
        initial_state=chip.bare_state({ka: 2, kb: 0}),
        e_ops={ka: ka.number_operator(), kb: kb.number_operator()},
        check_truncation=False,
    )
    n_a = np.real(np.asarray(result.expect("ka2ph")))
    n_b = np.real(np.asarray(result.expect("kb2ph")))
    assert float(n_a.min()) < 0.01
    assert float(n_b.max()) > 1.99
    # <0,2|a^2 b^{dagger 2}|2,0> = 2, giving full-contrast Rabi with half period 1/(2*2g*2) = 12.5 ns at g = 10 MHz.
    half_period = tlist[int(np.argmax(n_b))]
    assert abs(half_period - 12.5) < 0.5


class _BlueCoupling(CouplingModel):
    """Two-mode-squeezing-retaining coupling: its RWA keeps the |Δa + Δb| = 2 bands instead."""

    _type_prefix = "blue"

    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b):
        return self.g * (a.x * b.x)

    def rwa_keeps_band(self, delta_a: int, delta_b: int) -> bool:
        return abs(delta_a + delta_b) == 2 or (delta_a, delta_b) == (0, 0)


def test_custom_rwa_keeps_band_override():
    """A coupling declaring a non-default retained band set gets it, with no engine involvement."""
    ba = Resonator(freq=5.0, levels=3, label="ba_blue")
    bb = Resonator(freq=5.2, levels=3, label="bb_blue")
    chip = Chip([ba, bb], [_BlueCoupling(ba, bb, g=0.03)], rwa=True)
    h = _arr(chip, chip.hamiltonian())
    idx_01, idx_10, idx_00, idx_11 = 1, 3, 0, 4
    assert abs(h[idx_01, idx_10]) < 1e-14  # exchange band dropped by the override
    np.testing.assert_allclose(abs(h[idx_00, idx_11]), 0.03, atol=1e-12)  # squeezing band kept


def test_fluxonium_dense_charge_coupling_assembles_hermitian():
    """A circuit-level eigenbasis charge operator populates many bands; the mask must stay Hermitian."""
    fl = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=5, label="fl_rwa")
    rr = Resonator(freq=5.0, levels=4, label="rr_rwa")

    def charge_coupling(a, b, bk):
        # charge_coupling_operator() is a dense array-like; bk.tensor coerces it.
        return bk.tensor(a.charge_coupling_operator(), b.lowering_operator() + bk.dag(b.lowering_operator()))

    chip = Chip([fl, rr], [Coupling(fl, rr, g=0.05, interaction=charge_coupling, label="flc")],
                frame="rotating", rwa=True)
    h = _arr(chip, chip.hamiltonian())
    assert float(np.max(np.abs(h - h.conj().T))) < 1e-12

    problem = QuantumSequence(chip).build_problem(tlist=np.linspace(0.0, 5.0, 6), initial_state=chip.bare_state())
    kept = [t for t in problem.hamiltonian.dynamic_terms if t.origin == "coupling"]
    assert len(kept) == 2  # (Δ_fl, Δ_r) = (±1, ∓1): the only populated number-conserving off-diagonal bands
    assert len(problem.hamiltonian.dropped_terms) > 0
