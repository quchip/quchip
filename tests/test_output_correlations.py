"""Stationary spectra and normally ordered output-field correlations."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Chip, Port, PortNetwork, Resonator, VNA


def test_coherent_resonator_output_has_unit_g1_and_g2() -> None:
    resonator = Resonator(freq=6.0, levels=8, label="r")
    port = Port(resonator, rate=0.04, label="p")
    vna = VNA(
        Chip([resonator], port_network=PortNetwork.from_ports([port])),
        ports=[port],
    )
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
        Chip(
            [resonator],
            port_network=PortNetwork.from_ports([input_port, output_port]),
        ),
        ports=[input_port, output_port],
    )
    frequency, amplitude = 6.003, 0.01 + 0.01j
    vna.pump(input_port, freq=frequency, amplitude=amplitude)
    delays = np.array([0.0, 2.0, 7.0])

    g1 = vna.g1(output_port, delays, input=input_port)
    g2 = vna.g2(output_port, delays, input=input_port)

    assert g1.input_port == g2.input_port == "in"
    assert g1.output_port == g2.output_port == "out"
    means = VNA(vna.chip).finite_power(frequency, amplitude, input=input_port)
    cross = np.conj(means.mean(output_port))*means.mean(input_port)
    np.testing.assert_allclose(g1.unnormalized, cross, atol=1e-10)
    np.testing.assert_allclose(g1.values, cross/abs(cross), atol=2e-6)
    np.testing.assert_allclose(g2.values, np.ones(3), atol=2e-5)


def test_vacuum_output_has_zero_fluctuation_spectrum() -> None:
    resonator = Resonator(freq=6.0, levels=5, label="r")
    port = Port(resonator, rate=0.04, label="p")
    vna = VNA(
        Chip([resonator], port_network=PortNetwork.from_ports([port])),
        ports=[port],
    )
    frequencies = np.array([-0.1, 0.0, 0.1])

    result = vna.output_spectrum(port, frequencies=frequencies)

    np.testing.assert_allclose(result.total_fluctuation_spectrum, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.signal_coherent_flux, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.signal_photon_flux, 0.0, atol=1e-12)
    assert result.fourier_convention == "2 Re integral_0^inf d tau exp(-j 2 pi f tau) C_engineering(tau), j = i"


def test_thermal_output_has_g2_zero_near_two() -> None:
    resonator = Resonator(
        freq=6.0,
        levels=14,
        label="r",
        T1=20.0,
        thermal_occupation=0.2,
    )
    port = Port(resonator, rate=0.03, label="p")
    vna = VNA(
        Chip([resonator], port_network=PortNetwork.from_ports([port])),
        ports=[port],
    )

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
            thermal_occupation=0.15,
        )
        port = Port(resonator, rate=0.03, label="p")
        vna = VNA(
            Chip(
                [resonator],
                port_network=PortNetwork.from_ports([port]),
                backend=backend,
            ),
            ports=[port],
        )
        spectrum = vna.output_spectrum(port, frequencies=[-0.05, 0.0, 0.05])
        return spectrum.total_fluctuation_spectrum, vna.g1(port, [0.0, 2.0]).values, vna.g2(port, [0.0, 2.0]).values

    qutip_values = outputs("qutip")
    dynamiqs_values = outputs("dynamiqs")

    for qutip_value, dynamiqs_value in zip(qutip_values, dynamiqs_values):
        np.testing.assert_allclose(np.asarray(dynamiqs_value), qutip_value, atol=2e-7)


@pytest.mark.validation
def test_qutip_output_analysis_is_not_capped_by_engine_dense_dimension() -> None:
    """QuTiP output analysis uses its native stationary lowering beyond the old dense cap."""
    resonator = Resonator(freq=6.0, levels=17, label="r", T1=20.0)
    port = Port(resonator, rate=0.04, label="p")
    vna = VNA(
        Chip([resonator], port_network=PortNetwork.from_ports([port])),
        ports=[port],
    )

    result = vna.output_spectrum(port, frequencies=[0.0])

    np.testing.assert_allclose(result.total_fluctuation_spectrum, 0.0, atol=1e-10)


def test_detuned_thermal_spectrum_uses_the_physical_upper_sideband() -> None:
    """Thermal fluorescence peaks at the mode frequency relative to the chosen carrier."""
    resonator = Resonator(freq=6.01, levels=12, label="r", T1=20.0, thermal_occupation=0.2)
    port = Port(resonator, rate=0.03, label="p")
    vna = VNA(Chip([resonator], port_network=PortNetwork.from_ports([port])))
    vna.pump(port, freq=6.0, amplitude=0.0)
    offsets = np.array([-0.01, 0.0, 0.01])
    result = vna.output_spectrum(port, frequencies=offsets)
    expected = 0.03*0.05*0.2 / ((0.08/2)**2 + (2*np.pi*(offsets-0.01))**2)
    np.testing.assert_allclose(result.total_fluctuation_spectrum, expected, atol=1e-9)
    measured = VNA(vna.chip).measure(6., 0., input=port, noise_frequencies=[-.02, -.01, 0., .01, .02])
    np.testing.assert_allclose(measured.noise_spectrum(port)[1:-1], expected, atol=1e-9)
