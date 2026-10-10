"""Tests for FluxTunableTransmon, GaussianEdge, and QuantumSequence.flux_to.

Core tests — no backend simulation required for most; the flux_to test uses
the QuTiP backend for a minimal end-to-end check.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest


# ===========================================================================
# FluxTunableTransmon
# ===========================================================================


class TestFluxTunableTransmon:

    def test_hamiltonian_eigenvalues(self):
        """Duffing H eigenvalues: ground = 0, first ≈ freq, second ≈ 2*freq + alpha."""
        from quchip import FluxTunableTransmon

        freq = 4.47
        alpha = -0.2006
        q = FluxTunableTransmon(freq=freq, anharmonicity=alpha, levels=3)
        H = q.hamiltonian()
        evals = sorted(np.linalg.eigvalsh(H.matrix()).real)
        npt.assert_allclose(evals[0], 0.0, atol=1e-10)
        npt.assert_allclose(evals[1], freq, rtol=1e-8)
        npt.assert_allclose(evals[2], 2 * freq + alpha, rtol=1e-6)


    def test_frequency_at_nonzero_bias(self):
        """frequency_at(flux_bias) matches freq when constructed at that bias."""
        from quchip import FluxTunableTransmon

        freq = 4.0
        q = FluxTunableTransmon(freq=freq, anharmonicity=-0.20, flux_bias=0.1)
        npt.assert_allclose(float(q.frequency_at(0.1)), freq, rtol=1e-6)


    def test_flux_for_frequency_at_bias_matches_construction(self):
        """flux_for_frequency(q.freq) should return the original flux_bias."""
        from quchip import FluxTunableTransmon

        flux_bias = 0.15
        q = FluxTunableTransmon(freq=4.2, anharmonicity=-0.20, flux_bias=flux_bias)
        phi = float(q.flux_for_frequency(q.freq))
        npt.assert_allclose(phi, flux_bias, atol=1e-5)

    def test_asymmetric_squid(self):
        """Non-zero asymmetry should affect frequency at nonzero flux."""
        from quchip import FluxTunableTransmon

        q_sym = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, asymmetry=0.0)
        q_asym = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, asymmetry=0.1)
        npt.assert_allclose(float(q_sym.frequency_at(0.0)), float(q_asym.frequency_at(0.0)), rtol=1e-4)
        f_sym = float(q_sym.frequency_at(0.3))
        f_asym = float(q_asym.frequency_at(0.3))
        assert f_asym > f_sym, "Asymmetric SQUID should have higher freq at non-zero flux"

    def test_jax_traceable_construction(self):
        """jit over FluxTunableTransmon construction and frequency_at must not fail."""
        import jax

        from quchip import FluxTunableTransmon

        @jax.jit
        def get_freq_at(f, alpha, phi):
            q = FluxTunableTransmon(freq=f, anharmonicity=alpha)
            return q.frequency_at(phi)

        result = get_freq_at(4.47, -0.2006, 0.0)
        npt.assert_allclose(float(result), 4.47, rtol=1e-6)

    def test_jax_traceable_freq_sweep(self):
        """Sweeping freq over jnp.linspace — frequency_at stays traceable."""
        import jax
        import jax.numpy as jnp

        from quchip import FluxTunableTransmon

        @jax.jit
        def sweep(freqs):
            return jnp.array([
                FluxTunableTransmon(freq=f, anharmonicity=-0.20).frequency_at(0.1)
                for f in freqs
            ])

        freqs = jnp.linspace(4.0, 5.0, 5)
        results = sweep(freqs)
        assert results.shape == (5,)
        # Flux reduces frequency.
        assert jnp.all(results < freqs)


    def test_flux_bias_mutation_retunes_frequency_and_hamiltonian(self):
        """Changing the operating bias preserves the SQUID calibration and retunes the local model."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, flux_bias=0.0, levels=3)
        expected_frequency = float(q.frequency_at(0.3))
        E_J_max_before = float(q._E_J_max)
        H_before = np.asarray(q.hamiltonian().matrix())

        q.flux_bias = 0.3

        H_after = np.asarray(q.hamiltonian().matrix())
        npt.assert_allclose(q.freq, expected_frequency, rtol=1e-10)
        npt.assert_allclose(float(q._E_J_max), E_J_max_before, rtol=1e-10)
        assert not np.allclose(H_before, H_after)
        npt.assert_allclose(np.linalg.eigvalsh(H_after)[1], expected_frequency, rtol=1e-10)


    def test_flux_bias_and_anharmonicity_rebind_use_the_new_calibration(self):
        """Grouped flux rebinding applies calibration parameters before moving the bias."""
        from quchip import Chip, FluxTunableTransmon

        q = FluxTunableTransmon(
            freq=4.47,
            anharmonicity=-0.2006,
            flux_bias=0.0,
            label="c",
        )
        chip = Chip([q])
        expected = chip.with_params({"c.anharmonicity": -0.24})["c"]
        expected.flux_bias = 0.3

        rebound = chip.with_params(
            {"c.flux_bias": 0.3, "c.anharmonicity": -0.24}
        )

        npt.assert_allclose(rebound.parameters["c.freq"], expected.freq, rtol=1e-10)
        npt.assert_allclose(
            float(rebound["c"]._E_J_max), float(expected._E_J_max), rtol=1e-10
        )

    def test_flux_bias_has_one_period_inverse_design_bounds(self):
        """Explicit flux optimization stays inside one canonical SQUID period."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006)

        assert q.tunable_param_bounds("flux_bias", 0.0) == (-0.5, 0.5)

    def test_flux_bias_rebind_round_trips_and_remains_jax_differentiable(self):
        """A rebound calibration serializes and its Hamiltonian remains differentiable in flux."""
        import jax
        import jax.numpy as jnp

        pytest.importorskip("dynamiqs")

        from quchip import Chip, FluxTunableTransmon
        from quchip.backend.dynamiqs import DynamiqsBackend

        q = FluxTunableTransmon(
            freq=4.47,
            anharmonicity=-0.2006,
            flux_bias=0.0,
            levels=3,
            label="c",
        )
        chip = Chip([q], backend=DynamiqsBackend())
        rebound = chip.with_params({"c.flux_bias": 0.2})
        restored = Chip.from_dict(rebound.to_dict())
        assert restored.parameters["c.flux_bias"] == pytest.approx(0.2)
        assert restored.parameters["c.freq"] == pytest.approx(rebound.parameters["c.freq"])

        def first_transition(flux_bias):
            shifted = chip.with_params({"c.flux_bias": flux_bias})
            matrix = shifted.hamiltonian().matrix(backend=shifted.backend)
            energies = jnp.linalg.eigvalsh(matrix)
            return energies[1] - energies[0]

        gradient = jax.jit(jax.grad(first_transition))(0.2)
        assert jnp.isfinite(gradient)
        assert abs(float(gradient)) > 1.0e-6


# ===========================================================================
# Flux-dependent charge scale
# ===========================================================================


def _charge_scale(flux, asymmetry):
    """Return the closed-form scale ``(cos²(πΦ) + d² sin²(πΦ))^(1/8)``."""
    return (np.cos(np.pi * flux) ** 2 + asymmetry**2 * np.sin(np.pi * flux) ** 2) ** 0.125


def _qubit_resonator_exchange(qubit, g=0.010):
    """Return ``|<1_q 0_r|H|0_q 1_r>|`` for a two-level resonator coupled capacitively to *qubit*."""
    from quchip import Capacitive, Chip, Resonator

    resonator = Resonator(freq=5.0, levels=2, label="r")
    matrix = np.asarray(Chip([qubit, resonator], [Capacitive(qubit, resonator, g=g)]).hamiltonian().matrix())
    return abs(matrix[2, 1])


class TestChargeScale:

    @pytest.mark.parametrize("flux_bias, exchange_mhz", [(0.0, 10.000), (0.25, 9.170), (0.3, 8.756)])
    def test_capacitive_exchange_follows_the_charge_scale(self, flux_bias, exchange_mhz):
        """A flux bias that retunes the qubit scales its 10 MHz sweet-spot exchange by s(Φ)."""
        from quchip import Capacitive, Chip, FluxTunableTransmon, Resonator
        from quchip.analysis import effective_hamiltonian_between_states

        q = FluxTunableTransmon(freq=6.0, anharmonicity=-0.2, levels=3, label="q")
        q.flux_bias = flux_bias
        r = Resonator(freq=5.0, levels=3, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.010)])

        exchange = abs(complex(effective_hamiltonian_between_states(chip, (1, 0), (0, 1))[0, 1]))

        assert 1e3 * exchange == pytest.approx(exchange_mhz, abs=5e-4)
        assert exchange == pytest.approx(0.010 * _charge_scale(flux_bias, 0.0), rel=1e-12)

    def test_charge_scale_follows_flux_bias_rebinding(self):
        """A rebound flux bias scales the exchange and charge drive by s and the phase drive by 1/s."""
        from quchip import Capacitive, ChargeDrive, Chip, FluxTunableTransmon, PhaseDrive, Resonator

        q = FluxTunableTransmon(freq=6.0, anharmonicity=-0.2, asymmetry=0.3, levels=3, label="q")
        r = Resonator(freq=5.0, levels=2, label="r")
        coupled = Chip([q, r], [Capacitive(q, r, g=0.010)])
        scale = _charge_scale(0.3, 0.3)

        exchange = abs(np.asarray(coupled.with_params({"q.flux_bias": 0.3}).hamiltonian().matrix())[2, 1])

        assert exchange == pytest.approx(0.010 * scale, rel=1e-12)
        assert abs(np.asarray(coupled.hamiltonian().matrix())[2, 1]) == pytest.approx(0.010, rel=1e-12)
        for drive_type, factor in ((ChargeDrive, scale), (PhaseDrive, 1.0 / scale)):
            driven = Chip([FluxTunableTransmon(freq=6.0, anharmonicity=-0.2, asymmetry=0.3, levels=3, label="q")])
            driven.wire(drive_type("q", label="line"))
            rebound = driven.with_params({"q.flux_bias": 0.3})

            assert abs(rebound.drive_matrix_elements("q")["line"]) == pytest.approx(factor, rel=1e-12)
            assert abs(driven.drive_matrix_elements("q")["line"]) == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.optional_backend
    def test_charge_scale_gradient_matches_its_analytic_derivative(self):
        """jax.grad of the exchange element in flux_bias equals g ds/dΦ."""
        pytest.importorskip("dynamiqs")
        import jax
        import jax.numpy as jnp

        from quchip import Capacitive, Chip, FluxTunableTransmon, Resonator
        from quchip.backend.dynamiqs import DynamiqsBackend

        asymmetry, g = 0.2, 0.010
        q = FluxTunableTransmon(freq=6.0, anharmonicity=-0.2, asymmetry=asymmetry, levels=3, label="q")
        r = Resonator(freq=5.0, levels=2, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=g)], backend=DynamiqsBackend())

        def exchange(flux_bias):
            matrix = chip.with_params({"q.flux_bias": flux_bias}).hamiltonian().matrix(backend=chip.backend)
            return jnp.real(matrix[2, 1])  # <1_q 0_r|H|0_q 1_r> = g s(Φ)

        def scale_derivative(flux):
            x = np.cos(np.pi * flux) ** 2 + asymmetry**2 * np.sin(np.pi * flux) ** 2
            return 0.125 * x**-0.875 * np.pi * np.sin(2.0 * np.pi * flux) * (asymmetry**2 - 1.0)

        gradient = jax.jit(jax.grad(exchange))
        for flux in (0.0, 0.25, 0.3):
            assert float(gradient(flux)) == pytest.approx(g * scale_derivative(flux), rel=1e-10, abs=1e-15)

    @pytest.mark.validation
    def test_charge_scale_converges_to_the_charge_basis_transmon(self):
        """The exchange ratio J(Φ)/J(0) approaches the charge-basis result as (E_J,max/E_C)^(-1/2)."""
        from quchip import ChargeBasisTransmon, FluxTunableTransmon

        E_C = 0.2

        def largest_deviation(ratio):
            E_J_max = ratio * E_C

            def charge_basis(flux):
                return ChargeBasisTransmon(
                    E_C=E_C, E_J=E_J_max * np.cos(np.pi * flux), levels=2, basis="eigen", num_basis=41, label="q"
                )

            def tunable(flux):
                sweet_spot_freq = np.sqrt(8.0 * E_C * E_J_max) - E_C
                q = FluxTunableTransmon(freq=sweet_spot_freq, anharmonicity=-E_C, levels=2, label="q")
                q.flux_bias = flux
                return q

            reference = _qubit_resonator_exchange(charge_basis(0.0))
            sweet_spot = _qubit_resonator_exchange(tunable(0.0))
            return max(
                abs(
                    (_qubit_resonator_exchange(tunable(flux)) / sweet_spot)
                    / (_qubit_resonator_exchange(charge_basis(flux)) / reference)
                    - 1.0
                )
                for flux in (0.1, 0.2, 0.3)
            )

        deviations = [largest_deviation(ratio) for ratio in (60.0, 120.0, 240.0)]

        # The anharmonic correction to the charge element has relative order sqrt(E_C/E_J) (Koch et al. 2007),
        # so each doubling of E_J,max/E_C divides the remaining deviation by about sqrt(2).
        assert deviations[0] / deviations[1] == pytest.approx(np.sqrt(2.0), rel=0.1)
        assert deviations[1] / deviations[2] == pytest.approx(np.sqrt(2.0), rel=0.1)
        assert deviations[2] < 0.1 / np.sqrt(240.0)


# ===========================================================================
# GaussianEdge envelope
# ===========================================================================


class TestGaussianEdge:


    def test_jax_traceable(self):
        """Envelope values should preserve JAX arrays."""
        import jax.numpy as jnp

        from quchip import GaussianEdge

        env = GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=0.1)
        t = jnp.linspace(0.0, 80.0, 100)
        w = env.value(t)
        assert w.shape == (100,)


# ===========================================================================
# QuantumSequence.flux_to
# ===========================================================================


class TestFluxTo:
    def _make_chip(self):
        from quchip import Chip, FluxDrive, FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, label="QB")
        fdrv = FluxDrive(target=q, label="QB_z")
        chip = Chip([q], frame="rotating", backend="qutip")
        chip.wire(fdrv)
        return chip, q, fdrv


    def test_flux_to_does_not_mutate_envelope_template(self):
        """The original envelope object passed as envelope is not mutated."""
        from quchip import GaussianEdge, QuantumSequence

        chip, q, _ = self._make_chip()
        seq = QuantumSequence(chip)
        template = GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=None)
        seq.flux_to(q, target_freq=4.0, envelope=template)
        assert template.amplitude is None
