"""Stationary spectra and normally ordered output-field correlations."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Chip, Port, Resonator, VNA


def test_coherent_resonator_output_has_unit_g1_and_g2() -> None:
    resonator = Resonator(freq=6.0, levels=8, label="r")
    port = Port(resonator, rate=0.04, label="p")
    vna = VNA(Chip([resonator], ports=[port]), input=port, outputs=[port])
    vna.pump(port, freq=6.0, amplitude=0.02)
    delays = np.array([0.0, 2.0, 7.0])

    g1 = vna.g1(port, delays)
    g2 = vna.g2(port, delays)

    np.testing.assert_allclose(g1.values, np.ones(3), atol=2e-6)
    np.testing.assert_allclose(g2.values, np.ones(3), atol=2e-5)
    assert g1.normalization == "G1(tau) / G1(0)"
    assert g2.normalization == "G2(tau) / G1(0)^2"


def test_cross_port_correlations_retain_both_field_labels() -> None:
    """Cross-port coherence uses the selected initial and delayed output fields."""
    resonator = Resonator(freq=6.0, levels=8, label="r")
    input_port = Port(resonator, rate=0.03, label="in")
    output_port = Port(resonator, rate=0.04, label="out")
    vna = VNA(
        Chip([resonator], ports=[input_port, output_port]),
        input=input_port,
        outputs=[input_port, output_port],
    )
    vna.pump(input_port, freq=6.0, amplitude=0.02)
    delays = np.array([0.0, 2.0, 7.0])

    g1 = vna.g1(output_port, delays, input=input_port)
    g2 = vna.g2(output_port, delays, input=input_port)

    assert g1.input_port == g2.input_port == "in"
    assert g1.output_port == g2.output_port == "out"
    np.testing.assert_allclose(np.abs(g1.values), np.ones(3), atol=2e-6)
    np.testing.assert_allclose(g2.values, np.ones(3), atol=2e-5)


def test_vacuum_output_has_zero_fluctuation_spectrum() -> None:
    resonator = Resonator(freq=6.0, levels=5, label="r")
    port = Port(resonator, rate=0.04, label="p")
    vna = VNA(Chip([resonator], ports=[port]), input=port, outputs=[port])
    frequencies = np.array([-0.1, 0.0, 0.1])

    result = vna.output_spectrum(port, frequencies=frequencies)

    np.testing.assert_allclose(result.fluctuation_spectrum, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.coherent_flux, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.output_photon_flux, 0.0, atol=1e-12)
    assert result.fourier_convention == "2 Re integral_0^inf d tau exp(+i 2 pi f tau) C(tau)"


def test_thermal_output_has_g2_zero_near_two() -> None:
    resonator = Resonator(
        freq=6.0,
        levels=14,
        label="r",
        T1=20.0,
        thermal_population=0.2,
    )
    port = Port(resonator, rate=0.03, label="p")
    vna = VNA(Chip([resonator], ports=[port]), input=port, outputs=[port])

    result = vna.g2(port, [0.0])

    np.testing.assert_allclose(result.values, [2.0], atol=2e-5)


def test_qutip_and_dynamiqs_stationary_output_analysis_agree() -> None:
    """Both backends lower the same spectrum and regression queries independently."""
    pytest.importorskip("dynamiqs")

    def outputs(backend: str):
        resonator = Resonator(
            freq=6.0,
            levels=6,
            label="r",
            T1=20.0,
            thermal_population=0.15,
        )
        port = Port(resonator, rate=0.03, label="p")
        vna = VNA(Chip([resonator], ports=[port], backend=backend), input=port, outputs=[port])
        spectrum = vna.output_spectrum(port, frequencies=[-0.05, 0.0, 0.05])
        return spectrum.fluctuation_spectrum, vna.g1(port, [0.0, 2.0]).values, vna.g2(port, [0.0, 2.0]).values

    qutip_values = outputs("qutip")
    dynamiqs_values = outputs("dynamiqs")

    for qutip_value, dynamiqs_value in zip(qutip_values, dynamiqs_values):
        np.testing.assert_allclose(np.asarray(dynamiqs_value), qutip_value, atol=2e-7)


def test_qutip_output_analysis_is_not_capped_by_engine_dense_dimension() -> None:
    """QuTiP output analysis uses its native stationary lowering beyond the old dense cap."""
    resonator = Resonator(freq=6.0, levels=17, label="r", T1=20.0)
    port = Port(resonator, rate=0.04, label="p")
    vna = VNA(Chip([resonator], ports=[port]), input=port, outputs=[port])

    result = vna.output_spectrum(port, frequencies=[0.0])

    np.testing.assert_allclose(result.fluctuation_spectrum, 0.0, atol=1e-10)
