"""Unit tests for single-mode band decomposition and demod helpers."""

from __future__ import annotations


import numpy as np

from quchip.engine.bands import decompose_bands
from quchip.engine.observables import BandMeta, recombine_expect


def test_decompose_bands_completeness() -> None:
    """Sum of all bands reconstructs the original matrix."""
    rng = np.random.default_rng(123)
    mat = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))

    bands = decompose_bands(mat, 5)
    reconstructed = sum(bands.values())
    np.testing.assert_allclose(reconstructed, mat, atol=1e-14)


def test_decompose_bands_supports_jitted_jax_inputs() -> None:
    """Single-mode decomposition should remain usable under ``jax.jit``."""
    import jax
    import jax.numpy as jnp

    matrix = jnp.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [2.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j],
            [0.0 + 0.0j, 4.0 + 0.0j, 0.0 + 0.0j],
        ]
    )

    bands = jax.jit(lambda op: decompose_bands(op, 3))(matrix)
    active_weights = {weight for weight, band in bands.items() if np.linalg.norm(np.asarray(band)) > 1e-15}

    assert active_weights == {-1, 1}
    np.testing.assert_allclose(np.asarray(bands[-1]), np.asarray(np.tril(matrix, k=-1)), atol=1e-14)
    np.testing.assert_allclose(np.asarray(bands[1]), np.asarray(np.triu(matrix, k=1)), atol=1e-14)


def test_recombine_remodulate() -> None:
    """Remodulate direction applies exp(-i·ω·w·t) per band (rotating→lab)."""
    tlist = np.array([0.0, 0.125, 0.25])
    freqs = {"q": 5.0, "r": 7.0}

    s_q_p = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
    s_q_m = np.array([0.0 + 2.0j, 0.0 + 2.0j, 0.0 + 2.0j])
    s_qr_1 = np.array([0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j])
    s_qr_2 = np.array([0.0 + 0.25j, 0.0 + 0.25j, 0.0 + 0.25j])

    flat_expect = [s_q_p, s_q_m, s_qr_1, s_qr_2]
    meta = [
        BandMeta(key="q", weight=1, device_labels="q"),
        BandMeta(key="q", weight=-1, device_labels="q"),
        BandMeta(key=("q", "r"), weight=(1, -1), device_labels=("q", "r")),
        BandMeta(key=("q", "r"), weight=(-1, 1), device_labels=("q", "r")),
    ]

    band_sum, phase_corrected = recombine_expect(
        flat_expect,
        meta,
        tlist,
        freqs,
        direction="remodulate",
    )

    # Remodulate: exp(-i·ω·w·t)
    phase_q_p = np.exp(-1j * 2 * np.pi * freqs["q"] * (+1) * tlist)
    phase_q_m = np.exp(-1j * 2 * np.pi * freqs["q"] * (-1) * tlist)
    phase_qr_1 = np.exp(-1j * 2 * np.pi * (freqs["q"] * (+1) + freqs["r"] * (-1)) * tlist)
    phase_qr_2 = np.exp(-1j * 2 * np.pi * (freqs["q"] * (-1) + freqs["r"] * (+1)) * tlist)

    expected_q = phase_q_p * s_q_p + phase_q_m * s_q_m
    expected_qr = phase_qr_1 * s_qr_1 + phase_qr_2 * s_qr_2

    # band_sum = direct accumulation
    np.testing.assert_allclose(band_sum["q"], s_q_p + s_q_m, atol=1e-14)
    np.testing.assert_allclose(band_sum[("q", "r")], s_qr_1 + s_qr_2, atol=1e-14)

    # phase_corrected = remodulated (exp(-i·ω·w·t) applied)
    np.testing.assert_allclose(phase_corrected["q"], expected_q, atol=1e-14)
    np.testing.assert_allclose(phase_corrected[("q", "r")], expected_qr, atol=1e-14)


def test_recombine_demodulate() -> None:
    """Demodulate direction applies exp(+i·ω·w·t) per band (lab→slowly varying)."""
    tlist = np.array([0.0, 0.125, 0.25])
    freqs = {"q": 5.0, "r": 7.0}

    s_q_p = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
    s_q_m = np.array([0.0 + 2.0j, 0.0 + 2.0j, 0.0 + 2.0j])

    flat_expect = [s_q_p, s_q_m]
    meta = [
        BandMeta(key="q", weight=1, device_labels="q"),
        BandMeta(key="q", weight=-1, device_labels="q"),
    ]

    band_sum, phase_corrected = recombine_expect(
        flat_expect,
        meta,
        tlist,
        freqs,
        direction="demodulate",
    )

    # Demodulate: exp(+i·ω·w·t) — conjugate of remodulate
    phase_q_p = np.exp(+1j * 2 * np.pi * freqs["q"] * (+1) * tlist)
    phase_q_m = np.exp(+1j * 2 * np.pi * freqs["q"] * (-1) * tlist)

    expected_q = phase_q_p * s_q_p + phase_q_m * s_q_m

    np.testing.assert_allclose(band_sum["q"], s_q_p + s_q_m, atol=1e-14)
    np.testing.assert_allclose(phase_corrected["q"], expected_q, atol=1e-14)
