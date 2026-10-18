"""Tests for KerrCavity Hamiltonian eigenvalues, TwoPhotonDrive channels, and cat-state prep."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quchip.devices.kerr_cavity import KerrCavity
from quchip.control.drives_two_photon import TwoPhotonDrive


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
        evals = np.sort(np.linalg.eigvalsh(cav.hamiltonian().matrix(backend=backend)).real)
        expected = np.array([omega * n - K * n * (n - 1) for n in range(levels)])
        expected_sorted = np.sort(expected)
        npt.assert_allclose(evals, expected_sorted, atol=1e-10)


    def test_auto_label(self):
        """Auto-label should use 'kerr_cavity_' prefix."""
        from quchip.utils.labeling import reset_label_counters
        reset_label_counters()
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=5)
        assert cav.label.startswith("kerr_cavity_")

    def test_computational_property(self):
        """A Kerr cavity is not automatically treated as a computational qubit."""
        cav = KerrCavity(freq=5.0, kerr=1.0, levels=5, label="cav")
        assert cav.computational is False

class TestTwoPhotonDrive:
    """Verify the two-photon operator and carrier scheduling."""


    def test_auto_label(self):
        """Auto-label should use 'two_photon_' prefix."""
        from quchip.utils.labeling import reset_label_counters
        reset_label_counters()
        d2 = TwoPhotonDrive()
        assert d2.label.startswith("two_photon_")


class TestCatStatePreparation:
    """Physics integration test: rotating-frame adiabatic ramp -> cat state."""

    def test_mean_photon_number_at_alpha_squared(self):
        """After adiabatic ramp, <n> should approach eps2/K = alpha^2."""
        from quchip import Chip, QuantumSequence
        from quchip.control.envelopes import LinearRamp

        K = 0.01       # GHz; the retained Fock ladder stays energy ordered
        omega = 5.0    # GHz
        eps2_max = 2.0 * K    # alpha^2 = 2.0
        N_fock = 20
        t_ramp = 40.0 / K  # ns; fixed adiabaticity relative to Kerr splitting
        T_total = 45.0 / K
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
        # Prepare the Fock vacuum; the truncated negative-Kerr Hamiltonian's
        # lowest energy lies at the cutoff and is not the cat-preparation state.
        result = seq.simulate(tlist=tlist, initial_state=chip.bare_state({cav: cav.basis_state(0)}),
                              e_ops=e_ops, options={"nsteps": 5000})

        n_final = float(np.real(result.expect_final(cav, index=0)))
        assert abs(n_final - eps2_max / K) < 0.5, (
            f"<n> = {n_final:.3f} but expected approx {eps2_max/K:.1f}"
        )
