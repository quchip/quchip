"""Static-ZZ pathway attribution and des-Cloizeaux effective-Hamiltonian extraction."""

from __future__ import annotations

import itertools

import numpy as np

from quchip import Capacitive, Chip, DuffingTransmon
from quchip.analysis import analyze_static_zz, effective_hamiltonian
from quchip.chip.sw import bare_hamiltonian, sylvester_generator


def _two_qubit_chip(g: float = 0.01) -> Chip:
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    return Chip([q0, q1], couplings=[Capacitive(q0, q1, g=g)])


def test_analyze_static_zz_matches_dispersive_shift_exactly():
    chip = _two_qubit_chip()
    result = analyze_static_zz(chip, "q0", "q1")
    assert float(result.zz) == float(chip.dispersive_shift("q0", "q1"))


def test_pathway_sum_matches_half_commutator_element():
    """The pathway amounts sum to the (1_a, 1_b) element of ½[S, V] they attribute."""
    chip = _two_qubit_chip()
    result = analyze_static_zz(chip, "q0", "q1")

    h, labels, dims = bare_hamiltonian(chip)
    idx_a, idx_b = labels.index("q0"), labels.index("q1")
    occupations = np.indices(dims).reshape(len(dims), -1)
    p_mask = (occupations[idx_a] <= 1) & (occupations[idx_b] <= 1)
    s, _ = sylvester_generator(h, p_mask)
    v = h - np.diag(np.diagonal(h))
    occ = [1, 1]
    i_idx = int(np.ravel_multi_index(tuple(occ), dims))

    half_commutator = 0.5 * (s @ v - v @ s)
    expected = complex(half_commutator[i_idx, i_idx])
    pathway_sum = complex(sum(amount for _, amount in result.pathways))
    assert abs(pathway_sum - expected) < 1e-12


def test_effective_hamiltonian_eigenvalues_match_labeled_energies_list_form():
    chip = _two_qubit_chip()
    result = effective_hamiltonian(chip, ["q0", "q1"])
    expected = [float(chip.energy({"q0": a, "q1": b})) for a, b in itertools.product(range(2), range(2))]

    eigenvalues = np.linalg.eigvalsh(np.asarray(result.h_eff))
    assert np.allclose(np.sort(eigenvalues), np.sort(expected), atol=1e-10)


def test_effective_hamiltonian_eigenvalues_match_labeled_energies_dict_form():
    """The ``{device: levels}`` form keeps a single device's leakage ladder."""
    chip = _two_qubit_chip()
    result = effective_hamiltonian(chip, {"q0": 3})
    expected = [float(chip.energy({"q0": n, "q1": 0})) for n in range(3)]

    eigenvalues = np.linalg.eigvalsh(np.asarray(result.h_eff))
    assert np.allclose(np.sort(eigenvalues), np.sort(expected), atol=1e-10)


def test_static_zz_describe_runs():
    chip = _two_qubit_chip()
    text = analyze_static_zz(chip, "q0", "q1").describe()
    assert "Static ZZ(q0, q1)" in text


def test_effective_hamiltonian_describe_runs():
    chip = _two_qubit_chip()
    text = effective_hamiltonian(chip, ["q0", "q1"]).describe()
    assert "Effective Hamiltonian" in text


def test_projected_charge_basis_devices_index_resolved_dimensions():
    """Eigen-projected devices flatten bare labels with retained dims, not the authored charge cutoff."""
    from quchip import ChargeBasisTransmon, Exact

    q = ChargeBasisTransmon(E_C=0.2, E_J=15.0, num_basis=31, levels=3, basis="eigen", label="q")
    r = ChargeBasisTransmon(E_C=0.22, E_J=17.0, num_basis=31, levels=4, basis="eigen", label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.01)], approximation=Exact())
    assert chip.resolve().dims == (3, 4)

    result = effective_hamiltonian(chip, {"q": 2, "r": 2})
    h_eff = np.asarray(result.h_eff)
    assert h_eff.shape == (4, 4)
    assert np.allclose(h_eff, h_eff.conj().T)

    # des Cloizeaux keeps the selected dressed eigenvalues exactly.
    expected = sorted(chip.energy(q=i, r=j) for i in range(2) for j in range(2))
    assert np.allclose(np.sort(np.linalg.eigvalsh(h_eff)), expected)
