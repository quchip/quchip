"""Dressed queries and solves share the explicitly selected static model."""

import numpy as np
import pytest

from quchip import Capacitive, ChargeDrive, Chip, DuffingTransmon, Exact, QuantumSequence, Qubit, RWA, eliminate


@pytest.mark.parametrize("approximation", [RWA(), Exact(), RWA(keep_bands={(1, 1), (-1, -1)})])
def test_dressed_queries_follow_retained_hamiltonian(approximation):
    """Frequencies, Kerr, states and drive elements share an independently built Hamiltonian."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = DuffingTransmon(freq=6.8, anharmonicity=-0.3, levels=3, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.08, label="qr")], approximation=approximation)
    drive = ChargeDrive(q, label="xy")
    chip.wire(drive)
    a = np.diag(np.sqrt([1.0, 2.0]), 1)
    h = np.diag((np.array([0, 5, 9.75])[:, None] + np.array([0, 6.8, 13.3])).ravel())
    exchange = np.kron(a.T, a) + np.kron(a, a.T)
    # The charge product i(a - a†) i(b - b†) carries -g on the counter-rotating terms.
    counter = -(np.kron(a, a) + np.kron(a.T, a.T))
    if isinstance(approximation, Exact):
        h += 0.08 * (exchange + counter)
    else:
        h += 0.08 * (exchange if approximation.keep_bands is None else counter)
    values, vectors = np.linalg.eigh(h)
    indices = np.argmax(abs(vectors), axis=1)  # Dispersive pair: labels are unambiguous.
    energies = values[indices].reshape(3, 3)
    ground, excited = vectors[:, indices[0]], vectors[:, indices[3]]
    frequency = energies[1, 0] - energies[0, 0]
    pull = energies[1, 1] - energies[1, 0] - energies[0, 1] + energies[0, 0]
    assert chip.freq(q) == pytest.approx(frequency, abs=1e-12)
    assert chip.dispersive_shift(q, r) == pytest.approx(pull, abs=1e-12)
    assert chip.kerr_matrix()[q, r] == pytest.approx(pull, abs=1e-12)
    assert abs(np.vdot(chip.backend.to_array(chip.state(q=1)).ravel(), excited)) == pytest.approx(1, abs=1e-12)
    element = excited @ np.kron(1j * (a - a.T), np.eye(3)) @ ground
    assert abs(chip.drive_matrix_elements(q)[drive]) == pytest.approx(abs(element), abs=1e-12)
    np.testing.assert_allclose(chip.dress().eigenvalues, values, atol=1e-12)
    chip.set_frame("rotating")
    assert chip.freq(q) == pytest.approx(frequency, abs=1e-12)
    # Exact reduction diagonalizes the selected Hamiltonian, including custom bands.
    reduced = eliminate(chip, r, method="exact")
    assert reduced.chip.freq(q) == pytest.approx(frequency, abs=1e-12)
    assert float(reduced.effective_params["q"]["chi"]) == pytest.approx(pull, abs=1e-12)
    edge_removed = eliminate(chip, "qr", method="exact").chip
    np.testing.assert_allclose(edge_removed.dress().eigenvalues, values, atol=1e-12)


@pytest.mark.parametrize("approximation, override", [(RWA(), Exact()), (Exact(), RWA())])
def test_solve_override_does_not_reuse_incompatible_dressing(approximation, override):
    """A solve override selects its own ground state without replacing the chip's cached spectrum."""
    q, r = Qubit(freq=5.0, label="q"), Qubit(freq=6.8, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.2)], approximation=approximation, frame="lab")
    frequency = chip.freq(q)
    problem = QuantumSequence(chip).build_problem(tlist=np.array([0.0, 1.0]), approximation=override)
    dressed = problem.engine_result.dress()
    ground = dressed.eigenstates[dressed.state_map[(0, 0)]]
    assert abs(chip.backend.overlap(ground, problem.initial_state)) == pytest.approx(1, abs=1e-12)
    assert chip.freq(q) == frequency


@pytest.mark.parametrize("approximation", [RWA(), Exact()])
def test_dressed_frequency_matches_free_evolution(approximation):
    """A dressed superposition precesses at chip.freq under the selected solve approximation."""
    q, r = Qubit(freq=5.0, label="q"), Qubit(freq=6.8, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.08)], approximation=approximation, frame="lab")
    ground, excited = chip.state(q=0), chip.state(q=1)
    times = np.linspace(0, 4000, 101)
    result = QuantumSequence(chip).simulate(tlist=times, initial_state=(ground + excited) / np.sqrt(2), states="all")
    g, e = (chip.backend.to_array(state).ravel() for state in (ground, excited))
    states = np.stack([chip.backend.to_array(result.state_at(t)).ravel() for t in times])
    coherence = (states @ g.conj()) * (states @ e.conj()).conj()
    np.testing.assert_allclose(coherence, 0.5 * np.exp(2j * np.pi * float(chip.freq(q)) * times), atol=1e-9)


@pytest.mark.optional_backend
@pytest.mark.parametrize("approximation", [RWA(), Exact()])
def test_dressed_frequency_gradient_matches_two_qubit_spectrum(approximation):
    """JIT differentiation retains the selected bands and matches the analytic two-qubit spectrum."""
    pytest.importorskip("dynamiqs")
    import jax

    q, r = Qubit(freq=5.0, label="q"), Qubit(freq=6.8, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.08, label="qr")], approximation=approximation, backend="dynamiqs")
    g, delta, total = 0.08, 1.8, 11.8
    splitting = np.sqrt(delta**2 + 4 * g**2)
    even = np.sqrt(total**2 + 4 * g**2) if isinstance(approximation, Exact) else total
    expected_gradient = -2 * g / splitting + (2 * g / even if isinstance(approximation, Exact) else 0)
    value, gradient = jax.jit(jax.value_and_grad(lambda g: chip.with_params({"qr.g": g}).freq(q)))(g)
    assert value == pytest.approx((even - splitting) / 2, abs=1e-12)
    assert gradient == pytest.approx(expected_gradient, abs=1e-12)
