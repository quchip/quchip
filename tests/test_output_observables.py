"""Transient output fields derived from the solve-bound SLH model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import quchip

from quchip import Chip, PortNetwork, QuantumSequence, Resonator, Square


def test_package_root_keeps_one_input_output_concept() -> None:
    """Implementation handles and per-moment request types stay off the package root."""
    assert "PortNetwork" in quchip.__all__
    assert {
        "CoherentInput",
        "ControlEndpoint",
        "FieldTerminal",
        "OutputAmplitude",
        "OutputPhotonFlux",
        "OutputQuadrature",
        "SLHComponent",
    }.isdisjoint(quchip.__all__)


def _one_port_chip(
    *,
    rate: float = 0.04,
    delay: float = 0.0,
    backend: str = "qutip",
) -> tuple[Chip, Resonator, object]:
    resonator = Resonator(freq=0.2, levels=4, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=rate)
    cable = network.delay("cable", duration=delay)
    network.link(port, cable)
    plane = network.expose("readout", at=cable.port(2))
    chip = Chip([resonator], port_network=network, frame="lab", backend=backend)
    return chip, resonator, plane


def test_exposure_is_the_complete_transient_input_output_handle() -> None:
    """One external-plane object schedules input and returns all output moments."""
    beta = 0.12 + 0.05j
    chip, resonator, plane = _one_port_chip()
    sequence = QuantumSequence(chip)
    sequence.schedule(
        plane.input,
        envelope=Square(duration=2.0, amplitude=abs(beta)),
        phase=np.angle(beta),
    )
    times = np.linspace(0.0, 2.0, 41)

    result = sequence.simulate(
        tlist=times,
        initial_state={resonator: 1},
        e_ops={plane: plane.output},
        partition=False,
    )
    field = result.output(plane)

    assert plane.label == "readout"
    assert chip.port_network is not None
    assert chip.port_network.external_port(plane) == plane
    assert chip.port_network.external_ports == (plane,)
    assert "_input_key" not in repr(plane)
    assert "_output_key" not in repr(plane)
    assert field.exposure == "readout"
    assert set(result.outputs) == {"readout"}
    np.testing.assert_allclose(field.times, times)
    np.testing.assert_allclose(field.quadrature(), np.real(field.amplitude))
    np.testing.assert_allclose(
        field.quadrature(phase=np.pi / 2),
        np.imag(field.amplitude),
        atol=2e-8,
    )
    assert field.final_amplitude == field.amplitude[-1]
    assert field.final_photon_flux == field.photon_flux[-1]
    with pytest.raises(FrozenInstanceError):
        plane.label = "other"  # type: ignore[misc]


def test_output_amplitude_is_s_beta_plus_l_expectation() -> None:
    """Output amplitude evaluates the resolved S beta plus L relation."""
    rate = 0.04
    beta = 0.12 + 0.05j
    chip, resonator, plane = _one_port_chip(rate=rate)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        plane.input,
        envelope=Square(duration=2.0, amplitude=abs(beta)),
        phase=np.angle(beta),
    )

    result = sequence.simulate(
        tlist=np.linspace(0.0, 2.0, 41),
        e_ops={plane: plane.output, resonator: resonator.lowering_operator()},
        partition=False,
    )

    lowering = result.observable_traces["r"]
    assert not isinstance(lowering, list)
    expected = beta + np.sqrt(rate) * lowering.raw
    np.testing.assert_allclose(result.output(plane).amplitude, expected, atol=2e-8)


def test_output_quadrature_is_an_arbitrary_post_solve_projection() -> None:
    """One complex field trace supports any quadrature phase after the solve."""
    chip, _, plane = _one_port_chip()
    sequence = QuantumSequence(chip)
    sequence.schedule(
        plane.input,
        envelope=Square(duration=1.0, amplitude=0.15),
        phase=0.21,
    )
    field = sequence.simulate(
        tlist=np.linspace(0.0, 1.0, 21),
        e_ops={plane: plane.output},
        partition=False,
    ).output(plane)

    phase = 0.37
    np.testing.assert_allclose(
        field.quadrature(phase),
        np.real(np.exp(-1j * phase) * field.amplitude),
        atol=2e-8,
    )


def test_output_photon_flux_includes_spontaneous_emission() -> None:
    """Normally ordered output flux includes spontaneous emission from L."""
    rate = 0.04
    chip, resonator, plane = _one_port_chip(rate=rate)
    times = np.linspace(0.0, 4.0, 41)

    field = QuantumSequence(chip).simulate(
        tlist=times,
        initial_state={resonator: 1},
        e_ops={plane: plane.output},
        partition=False,
    ).output(plane)

    np.testing.assert_allclose(field.photon_flux, rate * np.exp(-rate * times), atol=2e-7)


def test_scattering_routes_the_direct_coherent_background() -> None:
    """Complete output traces include coherent background routed by scattering."""
    left = Resonator(freq=0.2, levels=2, label="left")
    right = Resonator(freq=0.3, levels=2, label="right")
    network = PortNetwork(scattering=[[0.0, 1.0], [1.0, 0.0]], label="swap")
    network.port("left", target=left, rate=0.04)
    network.port("right", target=right, rate=0.09)
    left_plane = network.external_port("left")
    right_plane = network.external_port("right")
    assert network.external_port("left") == left_plane
    assert hash(network.external_port("left")) == hash(left_plane)
    chip = Chip([left, right], port_network=network, frame="lab")
    sequence = QuantumSequence(chip)
    sequence.schedule(left_plane.input, envelope=Square(duration=0.2, amplitude=0.2))

    result = sequence.simulate(
        tlist=np.linspace(0.0, 0.2, 5),
        e_ops={"right": left_plane.output, "other": right_plane.output},
        partition=False,
    )

    assert result.output(left_plane).amplitude[0] == pytest.approx(0.0)
    assert result.output(right_plane).amplitude[0] == pytest.approx(0.2)


def test_one_plane_cannot_be_requested_twice() -> None:
    """A solve has one canonical trace for each external plane."""
    chip, _, plane = _one_port_chip()

    with pytest.raises(ValueError, match="readout.*only once"):
        QuantumSequence(chip).simulate(
            tlist=np.linspace(0.0, 0.2, 3),
            e_ops={"first": plane.output, "second": plane.output},
            partition=False,
        )


def test_output_field_retains_the_markov_boundary_before_reference_delay() -> None:
    """Field traces expose both reported-plane and pre-delay boundary values."""
    delay = 0.2
    chip, resonator, plane = _one_port_chip(delay=delay)
    times = np.linspace(0.0, 1.0, 11)

    field = QuantumSequence(chip).simulate(
        tlist=times,
        initial_state={resonator: 1},
        e_ops={plane: plane.output},
        partition=False,
    ).output("readout")

    assert field.photon_flux[0] == pytest.approx(0.0)
    assert field.raw_photon_flux[0] == pytest.approx(0.04)
    np.testing.assert_allclose(
        field.photon_flux[times >= delay],
        field.raw_photon_flux[:-2],
        atol=2e-7,
    )


def test_reference_delay_applies_on_both_sides_of_direct_scattering() -> None:
    """Reference delay shifts incident and reported coherent fields reciprocally."""
    delay = 0.2
    chip, _, plane = _one_port_chip(delay=delay)
    sequence = QuantumSequence(chip)
    sequence.schedule(plane.input, envelope=Square(duration=0.5, amplitude=0.2))
    times = np.linspace(0.0, 0.8, 9)

    amplitude = sequence.simulate(
        tlist=times,
        e_ops={plane: plane.output},
        partition=False,
    ).output(plane).amplitude

    np.testing.assert_allclose(amplitude[times < 2 * delay], 0.0, atol=2e-8)
    assert amplitude[4] == pytest.approx(0.2, abs=2e-4)


def test_unknown_exposure_reports_the_available_boundary() -> None:
    """Exposure lookup rejects unknown labels and reports valid choices."""
    chip, _, _ = _one_port_chip()
    assert chip.port_network is not None

    with pytest.raises(KeyError, match="missing.*readout"):
        chip.port_network.external_port("missing")


def test_output_field_batch_preserves_grid_and_post_solve_analysis() -> None:
    """Batched field extraction keeps sweep axes and arbitrary quadrature phase."""
    chip, _, plane = _one_port_chip()
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(plane.input, envelope=Square(duration=0.5, amplitude=0.1))

    result = sequence.simulate_batch(
        pulse.vary("amplitude", [0.1, 0.2]),
        tlist=np.linspace(0.0, 0.5, 6),
        e_ops={plane: plane.output},
        progress=False,
    )
    field = result.output(plane)

    assert field.amplitude.shape == (2, 6)
    assert field.photon_flux.shape == (2, 6)
    np.testing.assert_allclose(field.amplitude[:, 0], [0.1, 0.2], atol=2e-8)
    np.testing.assert_allclose(field.quadrature(), np.real(field.amplitude))


def _tilt(frequency, *, slope):
    return 0.8 * np.exp(1j * slope * frequency)


def test_transient_filters_apply_at_the_carrier() -> None:
    """Inbound filters scale beta at the pulse carrier; outbound filters scale the reported field."""

    def build(with_filter: bool):
        resonator = Resonator(freq=0.2, levels=12, label="r")
        network = PortNetwork(label="line")
        port = network.port("coupler", target=resonator, rate=0.04)
        if with_filter:
            section = network.filter("tilt", transfer=_tilt, slope=1.3)
            network.link(port, section)
            plane = network.expose("readout", at=section.port(2))
        else:
            plane = network.expose("readout", at=port)
        return Chip([resonator], port_network=network, frame="rotating"), plane

    times = np.linspace(0.0, 30.0, 61)
    traces = []
    for with_filter in (False, True):
        chip, plane = build(with_filter)
        sequence = QuantumSequence(chip)
        sequence.schedule(plane.input, envelope=Square(duration=20.0, amplitude=0.02), freq=0.2)
        traces.append(sequence.simulate(tlist=times, e_ops={plane: plane.output}).output(plane))
    plain, filtered = traces
    factor = _tilt(0.2, slope=1.3)

    np.testing.assert_allclose(filtered.raw_amplitude, factor * plain.raw_amplitude, atol=1e-5)
    np.testing.assert_allclose(filtered.amplitude, factor**2 * plain.amplitude, atol=1e-5)
    np.testing.assert_allclose(filtered.photon_flux, abs(factor) ** 4 * plain.photon_flux, atol=1e-5)


def test_outbound_filter_needs_a_known_output_carrier() -> None:
    """A lab-frame output has no carrier to evaluate an outbound filter at."""
    resonator = Resonator(freq=0.2, levels=4, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=0.04)
    section = network.filter("tilt", transfer=_tilt, slope=1.3)
    network.link(port, section)
    plane = network.expose("readout", at=section.port(2))
    chip = Chip([resonator], port_network=network, frame="lab")

    with pytest.raises(ValueError, match="carrier"):
        QuantumSequence(chip).simulate(
            tlist=np.linspace(0.0, 1.0, 5),
            initial_state={resonator: 1},
            e_ops={plane: plane.output},
        )


def test_transient_output_through_an_amplifier_scales_by_root_gain() -> None:
    """Reported transient amplitude scales by sqrt(G) and flux by G; added noise is spectral only."""

    def build(gain: float | None):
        resonator = Resonator(freq=0.2, levels=8, label="r")
        network = PortNetwork(label="fridge")
        port = network.port("coupler", target=resonator, rate=0.04)
        circulator = network.circulator("circ")
        network.link(port, circulator.port(2))
        drive = network.expose("drive", at=circulator.port(1))
        if gain is None:
            readout = network.expose("readout", at=circulator.port(3))
        else:
            amplifier = network.amplifier("hemt", gain=gain, added_noise=1.0)
            network.link(circulator.port(3), amplifier)
            readout = network.expose("readout", at=amplifier.port(2))
        return Chip([resonator], port_network=network, frame="rotating"), drive, readout

    times = np.linspace(0.0, 20.0, 41)
    fields = []
    for gain in (None, 25.0):
        chip, drive, readout = build(gain)
        sequence = QuantumSequence(chip)
        sequence.schedule(drive.input, envelope=Square(duration=10.0, amplitude=0.02), freq=0.2)
        fields.append(sequence.simulate(tlist=times, e_ops={readout: readout.output}).output(readout))
    plain, amplified = fields
    np.testing.assert_allclose(amplified.amplitude, 5.0 * plain.amplitude, atol=1e-6)
    np.testing.assert_allclose(amplified.photon_flux, 25.0 * plain.photon_flux, atol=1e-6)
