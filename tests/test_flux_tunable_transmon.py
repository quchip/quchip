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
    def test_basic_construction_zero_flux(self):
        """freq=4.47, anharmonicity=-0.2006 at flux_bias=0 — round-trip exact."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, flux_bias=0.0, levels=3)
        npt.assert_allclose(q.freq, 4.47, rtol=1e-10)
        npt.assert_allclose(q.anharmonicity, -0.2006, rtol=1e-10)

    def test_hamiltonian_eigenvalues(self):
        """Duffing H eigenvalues: ground = 0, first ≈ freq, second ≈ 2*freq + alpha."""
        from quchip import FluxTunableTransmon

        freq = 4.47
        alpha = -0.2006
        q = FluxTunableTransmon(freq=freq, anharmonicity=alpha, levels=3)
        H = q.hamiltonian()
        H_qt = H
        evals = sorted(H_qt.eigenenergies())
        npt.assert_allclose(evals[0], 0.0, atol=1e-10)
        npt.assert_allclose(evals[1], freq, rtol=1e-8)
        npt.assert_allclose(evals[2], 2 * freq + alpha, rtol=1e-6)

    def test_frequency_at_zero_flux_matches_freq(self):
        """frequency_at(flux_bias) should reproduce the construction freq."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, flux_bias=0.0)
        npt.assert_allclose(float(q.frequency_at(0.0)), q.freq, rtol=1e-6)

    def test_frequency_at_nonzero_bias(self):
        """frequency_at(flux_bias) matches freq when constructed at that bias."""
        from quchip import FluxTunableTransmon

        freq = 4.0
        q = FluxTunableTransmon(freq=freq, anharmonicity=-0.20, flux_bias=0.1)
        npt.assert_allclose(float(q.frequency_at(0.1)), freq, rtol=1e-6)

    def test_frequency_decreases_with_flux(self):
        """SQUID dispersion is monotonically decreasing in Φ/Φ₀ ∈ [0, 0.5)."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, flux_bias=0.0)
        freqs = [float(q.frequency_at(phi)) for phi in [0.0, 0.1, 0.2, 0.3, 0.4]]
        for i in range(len(freqs) - 1):
            assert freqs[i] > freqs[i + 1], f"Expected decreasing, got {freqs}"

    def test_flux_for_frequency_roundtrip(self):
        """flux_for_frequency is the inverse of frequency_at."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, flux_bias=0.0)
        target = 3.8
        phi = float(q.flux_for_frequency(target))
        recovered = float(q.frequency_at(phi))
        npt.assert_allclose(recovered, target, rtol=1e-5)

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

    def test_repr_contains_label(self):
        """repr(device) includes the device label."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, label="QB2")
        assert "QB2" in repr(q)

    def test_serialization_roundtrip(self):
        """to_dict / from_dict roundtrip reproduces all parameters exactly."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(
            freq=4.47, anharmonicity=-0.2006, flux_bias=0.05, asymmetry=0.02,
            levels=3, label="QB_test",
        )
        d = q.to_dict()
        q2 = FluxTunableTransmon.from_dict(d)
        npt.assert_allclose(q2.freq, q.freq, rtol=1e-10)
        npt.assert_allclose(q2.anharmonicity, q.anharmonicity, rtol=1e-10)
        npt.assert_allclose(q2.flux_bias, q.flux_bias, rtol=1e-10)
        npt.assert_allclose(q2.asymmetry, q.asymmetry, rtol=1e-10)
        assert q2.levels == q.levels
        assert q2.label == q.label

    def test_flux_bias_mutation_leaves_hamiltonian_unchanged(self):
        """Mutating flux_bias leaves hamiltonian() unchanged; the local H depends only on freq/anharmonicity."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, flux_bias=0.0, levels=3)
        H_before = np.asarray(q.hamiltonian().full())
        q.flux_bias = 0.3
        H_after = np.asarray(q.hamiltonian().full())
        npt.assert_allclose(H_before, H_after, atol=1e-12)

    def test_anharmonicity_zero_rejected_at_construction(self):
        """anharmonicity=0 raises ValueError; E_C = |anharmonicity| would divide by zero in the SQUID inversion."""
        from quchip import FluxTunableTransmon

        with pytest.raises(ValueError, match="anharmonicity"):
            FluxTunableTransmon(freq=4.47, anharmonicity=0.0)

    def test_flux_for_frequency_raises_for_unattainable_concrete_target(self):
        """flux_for_frequency raises ValueError for a concrete target outside the attainable range."""
        from quchip import FluxTunableTransmon

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, flux_bias=0.0)
        with pytest.raises(ValueError, match="unattainable"):
            q.flux_for_frequency(100.0)


# ===========================================================================
# GaussianEdge envelope
# ===========================================================================


class TestGaussianEdge:
    def test_plateau_at_constant_amplitude(self):
        """Middle of the pulse should be close to amplitude."""
        from quchip import GaussianEdge

        amp = 0.5
        dur = 100.0
        edge = 10.0
        env = GaussianEdge(duration=dur, edge_duration=edge, sigmas=3, amplitude=amp)
        t_mid = np.array([dur / 2.0])
        w = env.waveform(t_mid)
        npt.assert_allclose(np.abs(w), amp, atol=1e-6)

    def test_edges_below_amplitude(self):
        """Edges (t=0 and t=duration) should be well below plateau amplitude."""
        from quchip import GaussianEdge

        amp = 1.0
        env = GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=amp)
        t_edges = np.array([0.0, 80.0])
        w = env.waveform(t_edges)
        assert np.all(np.abs(w) < 0.01 * amp)

    def test_negative_amplitude(self):
        """Negative amplitude (flux down-shift) should work."""
        from quchip import GaussianEdge

        env = GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=-0.3)
        t = np.linspace(0, 80.0, 200)
        w = env.waveform(t)
        npt.assert_allclose(w[100].real, -0.3, atol=1e-4)

    def test_jax_traceable(self):
        """Waveform should work with jax.numpy namespace."""
        import jax.numpy as jnp

        from quchip import GaussianEdge

        env = GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=0.1)
        t = jnp.linspace(0.0, 80.0, 100)
        w = env.waveform(t, xp=jnp)
        assert w.shape == (100,)

    def test_serialization_roundtrip(self):
        """to_dict / from_dict roundtrip preserves all parameters."""
        from quchip.control.envelopes import GaussianEdge

        env = GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=4, amplitude=0.25)
        d = env.to_dict()
        env2 = GaussianEdge.from_dict(d)
        assert env2.duration == env.duration
        assert env2.edge_duration == env.edge_duration
        assert env2.sigmas == env.sigmas
        assert env2.amplitude == env.amplitude

    def test_rejects_edge_too_large(self):
        """Two edge durations exceeding the pulse duration raises ValueError."""
        from quchip import GaussianEdge

        with pytest.raises(ValueError, match="edge_duration"):
            GaussianEdge(duration=10.0, edge_duration=6.0)  # 2*6=12 > 10


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

    def test_schedules_correct_delta_omega(self):
        """flux_to sets amplitude = target_freq - chip.freq(device)."""
        from quchip import GaussianEdge, QuantumSequence

        chip, q, _ = self._make_chip()
        seq = QuantumSequence(chip)
        target_freq = 4.0
        h = seq.flux_to(
            q,
            target_freq=target_freq,
            envelope=GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=None),
        )
        from quchip.control.sequence import _PulseEntry

        entry = seq._entries[h._entry_index]
        assert isinstance(entry, _PulseEntry)
        expected_amp = target_freq - float(chip.freq(q))
        npt.assert_allclose(float(entry.envelope.amplitude), expected_amp, rtol=1e-8)

    def test_flux_to_accepts_target_freq_from_chip(self):
        """target_freq may be chip.freq(other_device) — a runtime float."""
        from quchip import Capacitive, Chip, FluxDrive, FluxTunableTransmon, GaussianEdge, Resonator, QuantumSequence

        q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, label="QB")
        r = Resonator(freq=4.2, levels=3, label="R")
        fdrv = FluxDrive(target=q, label="QB_z")
        coup = Capacitive(q, r, g=0.05)
        chip = Chip([q, r], [coup], frame="rotating", backend="qutip")
        chip.wire(fdrv)

        seq = QuantumSequence(chip)
        target_freq = chip.freq(r)
        h = seq.flux_to(
            q,
            target_freq=target_freq,
            envelope=GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=0.0),
        )

        entry = seq._entries[h._entry_index]
        expected_amp = float(target_freq) - float(chip.freq(q))
        npt.assert_allclose(float(entry.envelope.amplitude), expected_amp, rtol=1e-6)

    def test_flux_to_does_not_mutate_envelope_template(self):
        """The original envelope object passed as envelope is not mutated."""
        from quchip import GaussianEdge, QuantumSequence

        chip, q, _ = self._make_chip()
        seq = QuantumSequence(chip)
        template = GaussianEdge(duration=80.0, edge_duration=20.0, sigmas=3, amplitude=None)
        seq.flux_to(q, target_freq=4.0, envelope=template)
        assert template.amplitude is None


def test_flux_for_frequency_below_minimum_raises():
    """A concrete target below the attainable minimum (-E_C) raises instead of silently inverting."""
    from quchip import FluxTunableTransmon

    q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, flux_bias=0.0)
    with pytest.raises(ValueError):
        q.flux_for_frequency(-0.3)
