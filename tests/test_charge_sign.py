"""Every built-in device uses the textbook charge sign [φ, n] = +i, and ChargePhaseDrive follows it."""

import warnings

import numpy as np
import pytest

from quchip import (
    RWA,
    ChargeBasisTransmon,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Exact,
    FluxTunableTransmon,
    Fluxonium,
    Gaussian,
    KerrCavity,
    QuantumSequence,
    Qubit,
    Resonator,
    Square,
)
from quchip.extensions import ChargePhaseDrive

DEVICES = {
    "duffing": lambda: DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q"),
    "resonator": lambda: Resonator(freq=6.0, levels=3, label="q"),
    "kerr_cavity": lambda: KerrCavity(freq=6.0, kerr=0.001, levels=3, label="q"),
    "qubit": lambda: Qubit(freq=5.0, label="q"),
    "flux_tunable": lambda: FluxTunableTransmon(freq=5.0, anharmonicity=-0.25, flux_bias=0.2, asymmetry=0.3,
                                                levels=3, label="q"),
    "charge_basis": lambda: ChargeBasisTransmon(E_C=0.25, E_J=12.5, num_basis=21, levels=3, basis="eigen",
                                                label="q"),
    "charge_basis_offset": lambda: ChargeBasisTransmon(E_C=1.0, E_J=1.0, n_g=0.25, num_basis=21, levels=3,
                                                       basis="eigen", label="q"),
    "fluxonium": lambda: Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=3, num_basis=100,
                                   basis="eigen", label="q"),
}


def _matrix(operator):
    return np.asarray(operator.matrix() if hasattr(operator, "matrix") else operator)


def _upward_elements(device):
    """Return ``(<1|Q|0>, <1|P|0>)`` in the isolated energy basis."""
    vectors = np.asarray(Chip([device]).resolve().bases[device.label].energy_vectors)
    charge, phase = _matrix(device.charge_coupling_operator()), _matrix(device.phase_coupling_operator())
    return vectors[:, 1].conj() @ charge @ vectors[:, 0], vectors[:, 1].conj() @ phase @ vectors[:, 0]


@pytest.mark.unit
@pytest.mark.parametrize("name", [name for name in DEVICES if name != "qubit"])
def test_charge_over_phase_element_is_positive_imaginary(name):
    """<1|Q|0>/<1|P|0> is +i times a positive number, the sign of [φ, n] = +i, whatever the eigenvector phases."""
    charge, phase = _upward_elements(DEVICES[name]())
    ratio = charge / phase
    # Energy eigenvectors of these real Hamiltonians are real up to one phase each, which cancels in the ratio.
    assert abs(ratio.real) < 1e-10 * abs(ratio)
    assert ratio.imag > 0


@pytest.mark.unit
def test_fock_charge_is_i_times_a_dagger_minus_a():
    """On a Fock device Q = i(a† − a), so <1|Q|0> = +i, as in Krantz et al. 2019, Eq. 77."""
    q = DEVICES["qubit"]()
    np.testing.assert_allclose(_matrix(q.charge_coupling_operator()), [[0, -1j], [1j, 0]], atol=1e-15)


def _excited_population(device, freq, amplitude, approximation):
    chip = Chip([device], frame="rotating", approximation=approximation)
    chip.wire(ChargePhaseDrive(device, label="iq"))
    sequence = QuantumSequence(chip)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sequence.schedule("iq", envelope=Gaussian(duration=40.0, amplitude=amplitude), freq=freq)
        result = sequence.simulate(tlist=[0.0, 40.0], states="final", options={"atol": 1e-10, "rtol": 1e-8})
    return float(np.real(result.population("q", 1)[-1]))


@pytest.mark.parametrize("approximation", [RWA(), Exact()], ids=["rwa", "exact"])
@pytest.mark.parametrize("name", ["duffing", "charge_basis", "fluxonium"])
def test_charge_phase_drive_prefers_the_resonant_sideband(name, approximation):
    """A carrier at +f couples with w₊ = |Q + iP|/2 and one at −f with w₋ = |Q − iP|/2, so P1 = sin²(2π w A a)."""
    device = DEVICES[name]()
    charge, phase = _upward_elements(device)
    resonant, counter = abs(charge + 1j * phase) / 2, abs(charge - 1j * phase) / 2
    assert resonant > counter
    freq = float(Chip([device]).freq(device))
    amplitude = 0.01 / resonant
    angle = 2 * np.pi * 0.01 * float(np.real(Gaussian(duration=40.0, amplitude=1.0).area()))
    # Two-level rotation by the sideband weight times the pulse area. The tolerance allows the AC Stark shift and
    # leakage from the third level (about 1e-3 on the Duffing transmon) and the counter-rotating terms of Exact().
    assert _excited_population(device, freq, amplitude, approximation) == pytest.approx(np.sin(angle) ** 2, abs=3e-3)
    assert _excited_population(device, -freq, amplitude, approximation) == pytest.approx(
        np.sin(angle * counter / resonant) ** 2, abs=3e-3)


@pytest.mark.unit
@pytest.mark.parametrize("n_g", [0.0, 0.25])
def test_charge_basis_sine_obeys_the_commutator_of_the_textbook_sign(n_g):
    """With e^{iφ} raising n, [H, n] = i E_J sin φ, so (E_u − E_l) <u|n|l> = i E_J <u|sin φ|l> on every pair."""
    device = ChargeBasisTransmon(E_C=1.0, E_J=1.0, n_g=n_g, num_basis=21, levels=4, label="q")
    space = device.local_space()
    energies, vectors = np.linalg.eigh(np.asarray(device.unresolved_hamiltonian().matrix()))
    charge = vectors.conj().T @ np.asarray(space.matrix("n")) @ vectors
    sine = vectors.conj().T @ np.asarray(space.matrix("sin_phi")) @ vectors
    # The relation is an exact matrix identity in the truncated charge basis, up to diagonalization roundoff.
    np.testing.assert_allclose((energies[:, None] - energies[None, :]) * charge, 1j * 1.0 * sine, atol=1e-10)


def test_virtual_z_between_two_half_pi_pulses_sets_the_handedness():
    """Two real π/2 charge pulses with vz(π/4) between them take |0⟩ to the Bloch vector (1/2, −1/2, −1/√2)."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q")
    chip = Chip([q], frame="rotating", approximation=RWA())
    chip.wire(ChargeDrive(q, label="xy"))
    envelope = Square(duration=20.0)
    amplitude = float(chip.rotation_amplitude(q, "xy", envelope, angle=np.pi / 2))
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=envelope.with_params({"amplitude": amplitude}), freq=chip.freq(q))
    sequence.vz(q, np.pi / 4)
    sequence.schedule("xy", envelope=envelope.with_params({"amplitude": amplitude}), freq=chip.freq(q))
    result = sequence.simulate(tlist=[0.0, 40.0], states="final", options={"atol": 1e-12, "rtol": 1e-10})
    ground, excited = np.asarray(result.final_state.full()).ravel()
    bloch = [2 * np.real(np.conj(ground) * excited), 2 * np.imag(np.conj(ground) * excited),
             abs(ground) ** 2 - abs(excited) ** 2]
    # m = i: a real pulse turns about +y, and vz(θ) turns later pulse axes by +θ about z.
    np.testing.assert_allclose(bloch, [0.5, -0.5, -1 / np.sqrt(2)], atol=1e-8)
