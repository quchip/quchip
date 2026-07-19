"""Tests for KerrCavity Hamiltonian eigenvalues, TwoPhotonDrive channels, and cat-state prep."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.devices.kerr_cavity import KerrCavity
from quchip.control.drives_two_photon import TwoPhotonDrive


# ======================================================================
# Per-module label-counter reset
# ======================================================================

@pytest.fixture(autouse=True)
def _reset_labels():
    from quchip.utils.labeling import reset_label_counters
    reset_label_counters()
    yield
    reset_label_counters()


# ======================================================================
# KerrCavity Hamiltonian
# ======================================================================

class TestKerrCavityHamiltonian:
    """Verify KerrCavity eigenvalues against the analytical formula."""

    def test_eigenvalues_analytical(self):
        """E_n = omega*n - K*n*(n-1)."""
        omega = 5.0
        K = 1.0
        levels = 8
        cav = KerrCavity(freq=omega, kerr=K, levels=levels, label="cav")
        from quchip.backend import get_default_backend
        backend = get_default_backend()
        evals = np.sort(np.real(np.array(backend.eigenenergies(cav.hamiltonian()))))
        expected = np.array([omega * n - K * n * (n - 1) for n in range(levels)])
        expected_sorted = np.sort(expected)
        npt.assert_allclose(evals, expected_sorted, atol=1e-10)

    def test_zero_kerr_is_harmonic(self):
        """K=0 should give equally spaced harmonic oscillator levels."""
        omega = 3.0
        cav = KerrCavity(freq=omega, kerr=0.0, levels=6, label="cav")
        from quchip.backend import get_default_backend
        backend = get_default_backend()
        evals = np.sort(np.real(np.array(backend.eigenenergies(cav.hamiltonian()))))
        expected = np.array([omega * n for n in range(6)])
        npt.assert_allclose(evals, expected, atol=1e-10)

    def test_hamiltonian_is_hermitian(self):
        """H must be Hermitian."""
        cav = KerrCavity(freq=4.0, kerr=0.5, levels=10, label="cav")
        H = cav.hamiltonian()
        H_arr = np.array(H.full()) if hasattr(H, "full") else np.asarray(H)
        npt.assert_allclose(H_arr, H_arr.conj().T, atol=1e-12)

    def test_repr(self):
        """Repr includes label, freq, kerr, levels."""
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=10, label="c")
        r = repr(cav)
        assert "KerrCavity" in r
        assert "c" in r
        assert "5.0" in r
        assert "1.0" in r

    def test_negative_kerr_raises(self):
        """Negative kerr should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            KerrCavity(freq=5.0, kerr=-0.1, levels=5, label="bad")

    def test_zero_freq_raises(self):
        """Zero or negative freq should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            KerrCavity(freq=0.0, kerr=1.0, levels=5, label="bad")

    def test_auto_label(self):
        """Auto-label should use 'kerr_cavity_' prefix."""
        from quchip.utils.labeling import reset_label_counters
        reset_label_counters()
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=5)
        assert cav.label.startswith("kerr_cavity_")

    def test_computational_property(self):
        """KerrCavity.computational is False; the Pauli surface addresses bare Fock, not the cat manifold."""
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=5, label="cav")
        assert cav.computational is False

    def test_state_version_increments(self):
        """Mutation of freq should increment state_version."""
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=5, label="cav")
        v0 = cav.state_version
        cav.freq = 5.1
        assert cav.state_version == v0 + 1


# ======================================================================
# TwoPhotonDrive
# ======================================================================

class TestTwoPhotonDrive:
    """Verify TwoPhotonDrive local_channels returns correct operator and modulation."""

    def test_local_channels_length(self):
        """local_channels should return exactly one channel."""
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=10, label="cav")
        d2 = TwoPhotonDrive(target=cav)
        channels = d2.local_channels(cav)
        assert len(channels) == 1

    def test_coupling_operator_is_hermitian(self):
        """a^2 + a_dag^2 must be Hermitian."""
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=10, label="cav")
        d2 = TwoPhotonDrive(target=cav)
        channels = d2.local_channels(cav)
        op = channels[0].operator
        op_arr = np.array(op.full()) if hasattr(op, "full") else np.asarray(op)
        npt.assert_allclose(op_arr, op_arr.conj().T, atol=1e-12)

    def test_coupling_operator_shape(self):
        """Coupling operator must have shape (levels, levels)."""
        levels = 8
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=levels, label="cav")
        d2 = TwoPhotonDrive(target=cav)
        channels = d2.local_channels(cav)
        op = channels[0].operator
        op_arr = np.array(op.full()) if hasattr(op, "full") else np.asarray(op)
        assert op_arr.shape == (levels, levels)

    def test_modulation_is_single_tone(self):
        """Channel modulation should be SINGLE_TONE."""
        from quchip.control.signal_spec import DriveModulation
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=10, label="cav")
        d2 = TwoPhotonDrive(target=cav)
        channels = d2.local_channels(cav)
        assert channels[0].modulation == DriveModulation.SINGLE_TONE

    def test_type_prefix(self):
        """_type_prefix should be 'two_photon'."""
        assert TwoPhotonDrive._type_prefix == "two_photon"

    def test_auto_label(self):
        """Auto-label should use 'two_photon_' prefix."""
        from quchip.utils.labeling import reset_label_counters
        reset_label_counters()
        d2 = TwoPhotonDrive()
        assert d2.label.startswith("two_photon_")

    def test_operator_offdiagonal_structure(self):
        """a^2 + a_dag^2 must have zeros on diagonal (no weight-0 band)."""
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=8, label="cav")
        d2 = TwoPhotonDrive(target=cav)
        channels = d2.local_channels(cav)
        op = channels[0].operator
        op_arr = np.array(op.full()) if hasattr(op, "full") else np.asarray(op)
        npt.assert_allclose(np.diag(op_arr), 0.0, atol=1e-12)

    def test_schedule_without_freq_rejected_at_sequence_layer(self):
        """``seq.schedule`` on a TwoPhotonDrive requires an explicit freq (no natural default)."""
        import pytest
        from quchip import Chip, QuantumSequence
        from quchip.control.envelopes import Square

        cav = KerrCavity(freq=5.0, kerr=1.0, levels=10, label="cav")
        d2 = TwoPhotonDrive(target=cav)
        chip = Chip([cav])
        chip.wire(d2)
        seq = QuantumSequence(chip)
        with pytest.raises(ValueError, match="explicit freq"):
            seq.schedule(d2, envelope=Square(duration=20.0, amplitude=0.1))


# ======================================================================
# Cat-state preparation via adiabatic ramp (rotating frame)
# ======================================================================

class TestCatStatePreparation:
    """Physics integration test: rotating-frame adiabatic ramp -> cat state."""

    def test_mean_photon_number_at_alpha_squared(self):
        """After adiabatic ramp, <n> should approach eps2/K = alpha^2."""
        from quchip import Chip, QuantumSequence
        from quchip.control.envelopes import LinearRamp

        K = 1.0        # GHz
        omega = 5.0    # GHz
        eps2_max = 2.0 * K    # alpha^2 = 2.0
        N_fock = 20
        t_ramp = 40.0  # ns  (adiabatic: slow compared to 1/(2K) = 0.5 ns)
        T_total = 45.0
        n_steps = 450

        cav = KerrCavity(freq=omega, kerr=K, levels=N_fock, label="cav")
        d2 = TwoPhotonDrive(target=cav)

        chip = Chip([cav], frame="rotating")
        chip.wire(d2)

        n_op = cav.number_operator()
        e_ops = {cav: [n_op]}

        seq = QuantumSequence(chip)
        # In rotating frame, drive at 2*omega; amplitude doubled for RWA factor
        seq.schedule(d2, envelope=LinearRamp(T_total, t_ramp, amplitude=2 * eps2_max),
                     freq=2 * omega)

        tlist = np.linspace(0, T_total, n_steps)
        result = seq.simulate(tlist=tlist, e_ops=e_ops, options={"nsteps": 5000})

        n_final = float(np.real(result.expect_final(cav, index=0)))
        assert abs(n_final - eps2_max / K) < 0.5, (
            f"<n> = {n_final:.3f} but expected approx {eps2_max/K:.1f}"
        )
