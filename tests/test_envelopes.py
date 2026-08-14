"""Unit tests for pulse envelope waveform generation.

Analytical envelope definitions determine the reference waveforms; no backend
or chip is required.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import quchip
from quchip.control.envelopes import Gaussian, GaussianDRAG, Square, SquareWithGaussianEdges


def test_envelope_is_the_only_public_authoring_base():
    """The 0.2 surface exposes one envelope base without compatibility aliases."""
    assert hasattr(quchip, "Envelope")
    assert not hasattr(quchip, "BaseEnvelope")
    assert not hasattr(quchip, "EnvelopeShape")


def test_square_rejects_envelope_owned_global_phase():
    """Global phase belongs to scheduling, not the local square shape."""
    with pytest.raises(TypeError, match="phase"):
        Square(duration=10.0, amplitude=0.5, phase=0.2)


# ======================================================================
# Square envelope
# ======================================================================


class TestSquare:
    """Tests for the constant-amplitude Square envelope."""

    def test_constant_amplitude(self):
        """All samples should equal the given amplitude (magnitude)."""
        amp = 2.5
        sq = Square(duration=50.0, amplitude=amp)
        t = np.linspace(0, 50.0, 100)
        w = sq.value(t)
        npt.assert_allclose(np.abs(w), amp, atol=1e-14)

    def test_defaults_all_real_ones(self):
        """Default amplitude produces an all-real, all-ones local shape."""
        sq = Square(duration=10.0)
        t = np.linspace(0, 10.0, 50)
        w = sq.value(t)
        npt.assert_allclose(w.real, 1.0, atol=1e-14)
        npt.assert_allclose(w.imag, 0.0, atol=1e-14)

    def test_duration_stored(self):
        """The duration attribute should match the constructor argument."""
        sq = Square(duration=42.0, amplitude=0.5)
        assert sq.duration == 42.0


# ======================================================================
# Gaussian envelope
# ======================================================================


class TestGaussian:
    """Tests for the Gaussian window envelope."""

    def test_peak_at_center(self):
        """Max amplitude occurs at or within 1 sample of the center index."""
        g = Gaussian(duration=100.0, amplitude=3.0)
        N = 100
        t = np.linspace(0, 100.0, N)
        w = g.value(t)
        peak_idx = np.argmax(np.abs(w))
        center_idx = N // 2
        assert abs(peak_idx - center_idx) <= 1

    def test_symmetry(self):
        """Waveform is approximately symmetric: |w[k]| ≈ |w[N-1-k]|."""
        # Integer-indexed sampling gives |k| and |N-1-k| centre-distances
        # differing by 1 sample; worst-case asymmetry at 1000 samples is ~1.8%.
        g = Gaussian(duration=100.0, amplitude=1.0)
        N = 1000
        t = np.linspace(0, 100.0, N)
        w = g.value(t)
        mag = np.abs(w)
        npt.assert_allclose(mag, mag[::-1], rtol=0.02)

    def test_edge_value_analytical(self):
        """The edge-to-center ratio should equal exp(-sigmas² / 2)."""
        sigmas = 3.0
        amp = 1.0
        g = Gaussian(duration=50.0, sigmas=sigmas, amplitude=amp)
        edge = np.abs(g.value(np.asarray([0.0]))[0])
        peak = np.abs(g.value(np.asarray([25.0]))[0])
        expected_ratio = np.exp(-(sigmas**2) / 2)  # exp(-4.5)
        npt.assert_allclose(edge / peak, expected_ratio, rtol=1e-6)

    def test_amplitude_scaling(self):
        """The waveform magnitude at the center should equal the amplitude."""
        amp = 5.0
        g = Gaussian(duration=100.0, amplitude=amp)
        center = np.abs(g.value(np.asarray([50.0]))[0])
        npt.assert_allclose(center, amp, rtol=1e-10)

    def test_complex_output(self):
        """Gaussian waveform dtype should be complex."""
        g = Gaussian(duration=10.0)
        t = np.linspace(0, 10.0, 50)
        w = g.value(t)
        assert np.issubdtype(w.dtype, np.complexfloating)

    def test_default_sigmas(self):
        """Default sigmas parameter should be 3."""
        g = Gaussian(duration=20.0)
        assert g.sigmas == 3

    def test_custom_sigmas_edge_ratio(self):
        """Edge ratio scales with sigmas: exp(-sigmas²/2)."""
        sigmas = 2.0
        g = Gaussian(duration=40.0, sigmas=sigmas)
        edge = np.abs(g.value(np.asarray([0.0]))[0])
        peak = np.abs(g.value(np.asarray([20.0]))[0])
        expected_ratio = np.exp(-(sigmas**2) / 2)  # exp(-2)
        npt.assert_allclose(edge / peak, expected_ratio, rtol=1e-6)

    def test_wider_sigma_less_edge_decay(self):
        """Smaller sigmas produce larger edge values (wider Gaussian)."""
        N = 200
        t = np.linspace(0, 50.0, N)
        g_narrow = Gaussian(duration=50.0, sigmas=3.0)
        g_wide = Gaussian(duration=50.0, sigmas=1.0)
        edge_narrow = np.abs(g_narrow.value(t)[0])
        edge_wide = np.abs(g_wide.value(t)[0])
        assert edge_wide > edge_narrow
        peak_wide = np.abs(g_wide.value(np.asarray([25.0]))[0])
        npt.assert_allclose(edge_wide / peak_wide, np.exp(-0.5), rtol=1e-6)


# ======================================================================
# SquareWithGaussianEdges envelope
# ======================================================================


class TestSquareWithGaussianEdges:
    """Tests for the flat-top pulse with Gaussian ramp edges."""

    def test_plateau_value(self):
        """Mid-pulse samples (plateau) should equal the amplitude."""
        env = SquareWithGaussianEdges(duration=40.0, amplitude=0.7, edge_frac=0.25)
        mid = np.abs(env.value(np.asarray([20.0]))[0])
        npt.assert_allclose(mid, 0.7, atol=1e-14)

    def test_ramp_endpoints(self):
        """At t = edge, value is at the peak amplitude; at t = 0, attenuated by exp(-(2*sigmas)^2/2)."""
        # sigma = edge / (2 N_σ), so the boundary (t=0) sits 2 N_σ sigmas from the ramp peak (t=edge).
        sigmas = 3.0
        env = SquareWithGaussianEdges(duration=40.0, amplitude=1.0, edge_frac=0.25, sigmas=sigmas)
        peak = np.abs(env.value(np.asarray([10.0]))[0])
        start = np.abs(env.value(np.asarray([0.0]))[0])
        npt.assert_allclose(peak, 1.0, atol=1e-14)
        npt.assert_allclose(start, np.exp(-((2 * sigmas) ** 2) / 2), rtol=1e-6)

    def test_edge_frac_rejects_out_of_range(self):
        """edge_frac must be in (0, 0.5]."""
        with pytest.raises(ValueError):
            SquareWithGaussianEdges(duration=10.0, amplitude=1.0, edge_frac=0.0)
        with pytest.raises(ValueError):
            SquareWithGaussianEdges(duration=10.0, amplitude=1.0, edge_frac=0.6)

    def test_shape_invariant_under_duration_rescale(self):
        """At matching fractional positions, the waveform is the same."""
        a = SquareWithGaussianEdges(duration=40.0, amplitude=1.0, edge_frac=0.25)
        b = SquareWithGaussianEdges(duration=100.0, amplitude=1.0, edge_frac=0.25)
        fracs = np.linspace(0.0, 1.0, 11)
        wa = a.value(fracs * a.duration)
        wb = b.value(fracs * b.duration)
        npt.assert_allclose(wa, wb, atol=1e-14)

    def test_edge_duration_property(self):
        """edge_duration equals edge_frac * duration."""
        env = SquareWithGaussianEdges(duration=40.0, amplitude=1.0, edge_frac=0.2)
        npt.assert_allclose(env.edge_duration, 8.0)

    def test_roundtrip_serialization(self):
        """to_dict()/from_dict() round-trip preserves the waveform."""
        env = SquareWithGaussianEdges(duration=30.0, amplitude=0.5, edge_frac=0.3, sigmas=2.5)
        d = env.to_dict()
        restored = SquareWithGaussianEdges.from_dict(d)
        t = np.linspace(0, 30.0, 50)
        npt.assert_allclose(env.value(t), restored.value(t), atol=1e-14)


# ======================================================================
# Envelope.sample — vectorized evaluation
# ======================================================================


class TestSample:
    """Verify ``sample(tlist)`` evaluates ``value(tlist)`` elementwise."""

    def test_sample_matches_value_for_each_subclass(self):
        """sample(tlist) matches value(tlist) for every envelope subclass."""
        cases = [
            Square(duration=20.0, amplitude=0.7),
            Gaussian(duration=20.0, amplitude=0.5, sigmas=2.5),
            SquareWithGaussianEdges(duration=20.0, amplitude=0.6, edge_frac=0.2),
        ]
        t = np.linspace(0, 20.0, 51)
        for env in cases:
            npt.assert_allclose(env.sample(t), env.value(t), atol=1e-14)

    def test_sample_real_flag(self):
        """sample(..., real=True) returns the I part of a complex shape."""
        env = GaussianDRAG(duration=10.0, amplitude=1.2, beta=0.4)
        t = np.linspace(0, 10.0, 30)
        w = env.sample(t)
        r = env.sample(t, real=True)
        npt.assert_allclose(r, w.real, atol=1e-14)

    def test_sample_accepts_list(self):
        """sample() accepts a plain Python list of times."""
        env = Gaussian(duration=10.0, amplitude=1.0)
        out = env.sample([0.0, 5.0, 10.0])
        assert out.shape == (3,)

    def test_sample_jax_traced(self):
        """Traced JAX input stays JAX-shaped; no Python-float concretization."""
        import jax
        import jax.numpy as jnp

        env = SquareWithGaussianEdges(duration=20.0, amplitude=0.5, edge_frac=0.25)

        @jax.jit
        def energy(tlist):
            # sum of |E(t)|^2, keeps the pipeline traced end-to-end.
            w = env.sample(tlist)
            return jnp.sum(jnp.abs(w) ** 2)

        t = jnp.linspace(0.0, 20.0, 41)
        val = float(energy(t))
        assert val > 0.0
