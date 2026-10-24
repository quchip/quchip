"""Reductions record their approximations on retained terms and can diagonalize effective terms.

The black-box Hamiltonian of a lumped transmon–resonator circuit is its two
linear modes plus the junction cosine as one ``EffectiveTerms``. Exact
diagonalization of that term must keep the chip's spectrum, and its node-basis
circuit model is an independent reference for the dressed values it reports.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.constants import e, h, hbar

from quchip import (
    Capacitive,
    ChargeBasisTransmon,
    Chip,
    DuffingTransmon,
    EffectiveTerms,
    Exact,
    Resonator,
    RWA,
    eliminate,
)

C_Q, C_R, C_G, L_R, L_J = 90e-15, 400e-15, 6e-15, 1.2e-9, 11e-9
CAPACITANCE = np.array([[C_Q + C_G, -C_G], [-C_G, C_R + C_G]])
JOSEPHSON = (hbar / (2 * e)) ** 2 / L_J / h / 1e9  # GHz


def _modes():
    """Return linear mode frequencies in GHz and each mode's reduced junction phase."""
    omega2, vectors = sla.eigh(np.diag([1 / L_J, 1 / L_R]), CAPACITANCE)
    freqs = np.sqrt(omega2) / (2 * np.pi * 1e9)
    flux = vectors[0]
    participations = flux**2 / (L_J * omega2)
    return freqs, np.sign(flux) * np.sqrt(participations * freqs / (2 * JOSEPHSON))


def _quadrature(levels, mode):
    """Return a + a† of one mode on the product space of ``levels``."""
    lower = np.diag(np.sqrt(np.arange(1.0, levels[mode])), 1)
    factors = [np.eye(level) for level in levels]
    factors[mode] = lower + lower.T
    return np.kron(*factors)


def _black_box(levels, *, extra_devices=()):
    """Linear modes q and r plus -E_J [cos(phi) - 1 + phi^2 / 2] on their Fock product space."""
    freqs, phases = _modes()
    phase = phases[0] * _quadrature(levels, 0) + phases[1] * _quadrature(levels, 1)
    junction = -JOSEPHSON * (sla.cosm(phase) - np.eye(phase.shape[0]) + phase @ phase / 2)
    modes = [
        Resonator(freq=freqs[0], levels=levels[0], label="q"),
        Resonator(freq=freqs[1], levels=levels[1], label="r", internal_quality_factor=1.5e4),
    ]
    terms = EffectiveTerms(("q", "r"), tuple(levels), 0.5 * (junction + junction.T), label="junctions",
                           notes=("Exact junction cosine.",))
    return Chip([*modes, *extra_devices], effective_terms=[terms])


def _spectrum(chip, count):
    return np.linalg.eigvalsh(np.asarray(chip.hamiltonian().matrix()))[:count]


def test_device_and_coupling_reductions_record_their_approximations_on_the_retained_terms():
    """physics_notes() of a reduced chip states the reduction that produced its retained terms."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=5, label="r", internal_quality_factor=1e4)
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05, label="qr")])

    device_notes = eliminate(chip, r).chip.physics_notes()["effective:retained_r"]
    edge_notes = eliminate(chip, "qr", method="exact").chip.physics_notes()["effective:retained_qr"]

    assert "Adiabatic elimination (method='sw'): steady-state (vacuum) reduction." in device_notes
    assert any(note.startswith("Dropped: counter-rotating terms") for note in device_notes)
    assert "Retained Lindblad channels: internal_photon_loss." in device_notes
    assert any(note.startswith("Removed 'qr' and retained its complete exact") for note in edge_notes)


@pytest.mark.unit
def test_effective_term_labels_cannot_shadow_devices_or_couplings():
    """Device, coupling and effective-term labels share one namespace, so each target resolves once."""
    q = Resonator(freq=5.0, levels=2, label="q")
    r = Resonator(freq=6.0, levels=2, label="r")

    for label in ("q", "qr"):
        terms = EffectiveTerms(("q",), (2,), np.zeros((2, 2)), label=label)
        with pytest.raises(ValueError, match="collide with device or coupling labels"):
            Chip([q, r], [Capacitive(q, r, g=0.01, label="qr")], effective_terms=[terms])


@pytest.mark.unit
def test_effective_terms_reduction_rejects_unsupported_routes_and_unknown_labels():
    """Only the exact route is implemented, and an unknown label names every target kind."""
    chip = _black_box((4, 3))

    with pytest.raises(NotImplementedError, match="method='exact' only"):
        eliminate(chip, "junctions")
    with pytest.raises(KeyError, match="no device, coupling or effective terms"):
        eliminate(chip, "unknown", method="exact")


def test_exact_reduction_diagonalizes_the_terms_and_keeps_the_spectrum():
    """The retained chip is diagonal in its product basis, keeps its spectrum and reports its dressed values."""
    chip = _black_box((8, 4))
    source_matrix = np.asarray(chip.hamiltonian().matrix())
    kerr = chip.kerr_matrix()

    result = eliminate(chip, "junctions", method="exact")
    reduced = result.chip
    matrix = np.asarray(reduced.hamiltonian().matrix())

    # A unitary change of coordinates: eigvalsh roundoff on GHz-scale entries is below 1e-10.
    np.testing.assert_allclose(_spectrum(reduced, 20), _spectrum(chip, 20), atol=1e-10)
    np.testing.assert_allclose(matrix, np.diag(np.diag(matrix)), atol=1e-10)
    np.testing.assert_array_equal(np.asarray(chip.hamiltonian().matrix()), source_matrix)
    assert [device.label for device in reduced.devices] == ["q", "r"]
    params = result.effective_params
    assert float(params["q"]["freq_after"]) == pytest.approx(float(chip.freq("q")), abs=1e-10)
    assert float(params["q"]["anharmonicity"]) == pytest.approx(float(kerr["q", "q"]), abs=1e-10)
    assert float(params["q"]["cross_kerr"]["r"]) == pytest.approx(float(kerr["q", "r"]), abs=1e-10)
    assert float(params["r"]["cross_kerr"]["q"]) == pytest.approx(float(kerr["q", "r"]), abs=1e-10)
    notes = reduced.physics_notes()["effective:retained_junctions"]
    assert "junctions: Exact junction cosine." in notes
    assert any(note.startswith("Diagonalized effective terms 'junctions' exactly") for note in notes)
    assert "retained_junctions : q (8), r (4)" in reduced.describe()


@pytest.mark.parametrize("approximation", [RWA(), Exact()])
def test_exact_reduction_keeps_other_edges_authored_and_the_spectrum(approximation):
    """A spectator edge keeps its authored strength; the retained correction carries its rotated part."""
    chip = _black_box((6, 3), extra_devices=[Resonator(freq=6.1, levels=3, label="s")])
    edge = Capacitive(chip["r"], chip["s"], g=0.03, label="rs")
    chip = Chip(chip.devices, [edge], effective_terms=chip.effective_terms, approximation=approximation)

    reduced = eliminate(chip, "junctions", method="exact").chip

    assert reduced.approximation == approximation
    assert [coupling.label for coupling in reduced.couplings] == ["rs"]
    assert reduced.coupling_map["rs"].g == pytest.approx(0.03)
    np.testing.assert_allclose(_spectrum(reduced, 30), _spectrum(chip, 30), rtol=0, atol=1e-10)
    np.testing.assert_allclose(reduced.dress().eigenvalues, chip.dress().eigenvalues, rtol=0, atol=1e-10)


@pytest.mark.validation
def test_reported_dressed_values_match_the_node_basis_circuit():
    """Diagonalizing the black-box junction term gives the circuit's qubit frequency, anharmonicity and chi."""
    inverse = np.linalg.inv(CAPACITANCE)
    omega_r = np.sqrt(inverse[1, 1] / L_R)
    charge_zpf = np.sqrt(hbar * omega_r / (2 * inverse[1, 1]))
    transmon = ChargeBasisTransmon(E_C=e**2 * inverse[0, 0] / (2 * h * 1e9), E_J=JOSEPHSON,
                                   n_g=0.25, num_basis=31, levels=8, label="q")
    readout = Resonator(freq=omega_r / (2 * np.pi * 1e9), levels=8, label="r")
    g = inverse[0, 1] * 2 * e * charge_zpf / (h * 1e9)
    circuit = Chip([transmon, readout], [Capacitive(transmon, readout, g=g, label="qr")], approximation=Exact())
    kerr = circuit.kerr_matrix()

    params = eliminate(_black_box((20, 8)), "junctions", method="exact").effective_params["q"]

    # Measured: 3e-10 GHz in f_q, 3e-8 relative in alpha and 4e-8 in chi, the Fock truncation of both models.
    assert float(params["freq_after"]) == pytest.approx(float(circuit.freq("q")), abs=1e-7)
    assert float(params["anharmonicity"]) == pytest.approx(float(kerr["q", "q"]), rel=1e-5)
    assert float(params["cross_kerr"]["r"]) == pytest.approx(float(kerr["q", "r"]), rel=1e-5)
