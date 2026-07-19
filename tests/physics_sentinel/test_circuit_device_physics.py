"""Physics sentinels for CircuitDevice subclasses.

Two-tier validation: hardcoded eigenvalues in ``refs/`` (always run) plus a live
scqubits cross-check (runs when scqubits is installed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quchip.devices import ChargeBasisTransmon, DuffingTransmon, Fluxonium


REFS_DIR = Path(__file__).parent / "refs"


def _load_ref(name: str) -> dict:
    return json.loads((REFS_DIR / name).read_text())


def _to_ndarray(op) -> np.ndarray:
    """Convert a QuTiP Qobj or a JAX array to a dense ndarray."""
    if hasattr(op, "full"):
        return op.full()
    return np.asarray(op)


def test_fluxonium_eigvals_match_scqubits_reference():
    """Always-on: hardcoded scqubits reference eigenvalues match quchip diagonalization."""
    ref = _load_ref("fluxonium_scqubits_reference.json")
    q = Fluxonium(
        E_C=ref["params"]["E_C"],
        E_J=ref["params"]["E_J"],
        E_L=ref["params"]["E_L"],
        phi_ext=ref["params"]["phi_ext"],
        levels=ref["params"]["levels"],
        num_basis=400,
        phi_max=5 * np.pi,
    )
    eigvals = np.asarray(q.eigenenergies())
    ref_vals = np.asarray(ref["eigvals_shifted"])
    np.testing.assert_allclose(eigvals, ref_vals, atol=5e-3, rtol=5e-3)


def test_fluxonium_eigvals_match_scqubits_live():
    """Cross-library: quchip agrees with scqubits at runtime (requires scqubits)."""
    scq = pytest.importorskip("scqubits")
    flx = scq.Fluxonium(EJ=4.0, EC=1.0, EL=1.0, flux=0.5, cutoff=110, truncated_dim=10)
    scq_eigvals = np.asarray(flx.eigenvals(evals_count=10))
    scq_shifted = scq_eigvals - scq_eigvals[0]

    q = Fluxonium(
        E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=10,
        num_basis=400, phi_max=5 * np.pi,
    )
    quchip_eigvals = np.asarray(q.eigenenergies())
    np.testing.assert_allclose(quchip_eigvals, scq_shifted, atol=5e-3, rtol=5e-3)


def test_charge_basis_transmon_koch_reference():
    """ChargeBasisTransmon reproduces Koch 2007 transmon-regime 0→1 transition."""
    q = ChargeBasisTransmon(E_C=0.2, E_J=20.0, n_g=0.0, levels=5, num_basis=61)
    freq = float(q.freq)
    # Asymptotic: sqrt(8 E_J E_C) - E_C = sqrt(32) - 0.2 ≈ 5.46 GHz
    assert 5.0 < freq < 6.0


def test_charge_basis_transmon_matches_scqubits_live():
    """ChargeBasisTransmon eigenspectrum matches scqubits.Transmon to 3 digits."""
    scq = pytest.importorskip("scqubits")
    tr_scq = scq.Transmon(EJ=20.0, EC=0.2, ng=0.0, ncut=30, truncated_dim=5)
    scq_eigs = np.asarray(tr_scq.eigenvals(evals_count=5))
    scq_shifted = scq_eigs - scq_eigs[0]

    q = ChargeBasisTransmon(E_C=0.2, E_J=20.0, n_g=0.0, levels=5, num_basis=61)
    quchip_eigs = np.asarray(q.eigenenergies())
    np.testing.assert_allclose(quchip_eigs, scq_shifted, atol=1e-2, rtol=1e-2)


def test_charge_basis_transmon_matches_duffing_in_transmon_regime():
    """ChargeBasisTransmon.from_frequency matches DuffingTransmon's 0→1 transition to <1%."""
    freq_req = 5.0
    anh_req = -0.25
    q_cb = ChargeBasisTransmon.from_frequency(freq=freq_req, anharmonicity=anh_req, levels=3)
    q_duff = DuffingTransmon(freq=freq_req, anharmonicity=anh_req, levels=3)

    gap_cb = float(q_cb.freq)
    gap_duff = float(q_duff.freq)
    assert abs(gap_cb - gap_duff) / gap_duff < 0.01

    eig_cb = np.asarray(q_cb.eigenenergies())
    duff_12 = freq_req + anh_req  # 1→2 for Duffing
    cb_12 = eig_cb[2] - eig_cb[1]
    # Higher-order cosine terms shift the exact 1→2 gap a few percent from the Duffing value.
    assert abs(cb_12 - duff_12) / duff_12 < 0.05


def test_collapse_op_duffing_limit_reproduces_ladder_decay():
    """ChargeBasisTransmon's Fermi-golden-rule |1>→|0> rate matches sqrt(1/T1) to 1% in the transmon regime."""
    T1 = 30_000.0
    q = ChargeBasisTransmon.from_frequency(
        freq=5.0, anharmonicity=-0.25, levels=3, T1=T1, coupling_channel="charge",
    )
    ops = q.collapse_operators()

    # Find a 3×3 jump operator whose (0, 1) entry matches sqrt(1/T1).
    expected = np.sqrt(1.0 / T1)
    match = None
    for op in ops:
        arr = _to_ndarray(op)
        if arr.shape == (3, 3) and np.isclose(abs(arr[0, 1]), expected, rtol=0.01):
            match = arr
            break
    assert match is not None, f"No |0><1| ladder-matching jump op found among {len(ops)} ops"


def test_collapse_op_ladder_mode_matches_duffing_structurally():
    """collapse_model='ladder' returns the inherited BaseDevice channel (destroy(levels))."""
    T1 = 30_000.0
    q_ladder = ChargeBasisTransmon(
        E_C=0.25, E_J=25.0, levels=3, T1=T1, collapse_model="ladder",
    )
    q_duff = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, T1=T1)

    ladder_ops = q_ladder.collapse_operators()
    duff_ops = q_duff.collapse_operators()

    assert len(ladder_ops) == 1
    assert len(duff_ops) == 1

    ladder_arr = _to_ndarray(ladder_ops[0])
    duff_arr = _to_ndarray(duff_ops[0])
    assert np.allclose(ladder_arr, duff_arr, atol=1e-10)


def test_t1_decay_mesolve_matches_between_charge_basis_and_duffing():
    """mesolve T1 decay of |1⟩ agrees to <2% between ChargeBasisTransmon and DuffingTransmon."""
    import qutip

    T1 = 30_000.0
    q_cb = ChargeBasisTransmon.from_frequency(
        freq=5.0, anharmonicity=-0.25, levels=3, T1=T1, num_basis=61, coupling_channel="charge",
    )
    q_duff = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, T1=T1)

    tlist = np.linspace(0.0, T1 / 3, 21)
    psi0 = qutip.basis(3, 1)

    H_cb = qutip.Qobj(_to_ndarray(q_cb.hamiltonian()))
    c_ops_cb = [qutip.Qobj(_to_ndarray(op)) for op in q_cb.collapse_operators()]

    # DuffingTransmon's hamiltonian and collapse ops are already Qobj; feed them straight.
    H_duff = q_duff.hamiltonian()
    c_ops_duff = q_duff.collapse_operators()

    e_op = qutip.basis(3, 1) * qutip.basis(3, 1).dag()
    p1_cb = qutip.mesolve(H_cb, psi0, list(tlist), c_ops_cb, e_ops=[e_op]).expect[0]
    p1_duff = qutip.mesolve(H_duff, psi0, list(tlist), c_ops_duff, e_ops=[e_op]).expect[0]

    np.testing.assert_allclose(p1_cb, p1_duff, atol=0.02, rtol=0.02)
