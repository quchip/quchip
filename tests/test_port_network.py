"""Composable SLH field boundaries and their resolved physics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from quchip import Chip, Port, PortNetwork, QuantumSequence, Resonator


def _lowering(levels: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, levels)), 1).astype(complex)


def test_network_owns_ports_and_unconnected_ports_are_identity_exposures() -> None:
    """Unconnected quantum ports resolve as identity external exposures."""
    resonator = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(label="feedline")
    left = network.port("left", target=resonator, rate=0.01)
    right = network.port("right", target=resonator, rate=0.04)

    chip = Chip([resonator], port_network=network)
    resolved = chip.resolve().slh

    assert chip.port_network is network
    assert chip.ports == (left, right)
    assert [channel.key for channel in resolved.external_channels] == ["left", "right"]
    np.testing.assert_allclose(resolved.S, np.eye(2))
    np.testing.assert_allclose(resolved.L[0].to_dense(), np.sqrt(0.01) * _lowering(3))
    np.testing.assert_allclose(resolved.L[1].to_dense(), np.sqrt(0.04) * _lowering(3))


def test_direct_scattering_mapping_uses_output_input_order() -> None:
    """Scattering mappings use the documented output-input key order."""
    resonator = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(
        label="feedline",
        scattering={("right", "left"): 1.0, ("left", "right"): 1.0},
    )
    network.port("left", target=resonator, rate=0.01)
    network.port("right", target=resonator, rate=0.04)

    resolved = Chip([resonator], port_network=network).resolve().slh

    np.testing.assert_allclose(network.S, [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(resolved.S, [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), np.sqrt(0.04) * _lowering(3))
    np.testing.assert_allclose(resolved.L[1].to_dense(), np.sqrt(0.01) * _lowering(3))


def test_concrete_nonunitary_scattering_is_rejected() -> None:
    """A concrete scalar scattering boundary must be unitary."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(scattering=[[0.9]], label="lossy")
    network.port("readout", target=resonator, rate=0.01)

    with pytest.raises(ValueError, match="unitary"):
        Chip([resonator], port_network=network).resolve()


def test_cascade_requires_an_explicit_remaining_boundary() -> None:
    """A cascade hides connected terminals until the remaining boundary is exposed."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)

    with pytest.raises(ValueError, match="free terminals"):
        Chip([first, second], port_network=network).resolve()


def test_instantaneous_feedback_cycle_is_rejected_explicitly() -> None:
    """Instantaneous network feedback cycles fail before model resolution."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="loop")
    port = network.port("chip_port", target=resonator, rate=0.04)
    line = network.through("line")
    network.connect(port.output, line.input)
    network.connect(line.output, port.input)

    with pytest.raises(ValueError, match="feedback|cycle"):
        Chip([resonator], port_network=network).resolve()


def test_cascade_generates_series_coupling_and_hamiltonian() -> None:
    """Series composition produces both combined coupling and an SLH Hamiltonian."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)
    network.expose("feedline", input=a.input, output=b.output)

    resolved = Chip([first, second], port_network=network).resolve().slh
    identity = np.eye(2)
    l_a = np.sqrt(0.04) * np.kron(_lowering(2), identity)
    l_b = np.sqrt(0.09) * np.kron(identity, _lowering(2))
    product = l_b.conj().T @ l_a
    expected_h = (product - product.conj().T) / (2j)

    assert [channel.key for channel in resolved.external_channels] == ["feedline"]
    np.testing.assert_allclose(resolved.S, [[1.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), l_a + l_b)
    generated = [
        term for term in resolved.H.static_terms if term.origin == "network"
    ]
    assert len(generated) == 1
    np.testing.assert_allclose(generated[0].operator.to_dense(), expected_h)


def test_cascade_rejects_mixed_rotating_frame_frequencies() -> None:
    """Static SLH composition refuses channels with a missing relative carrier."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)
    network.expose("feedline", input=a.input, output=b.output)
    chip = Chip(
        [first, second],
        port_network=network,
        frame={"a": 5.0, "b": 6.0},
    )

    with pytest.raises(ValueError, match="different rotating-frame frequencies"):
        chip.resolve()


def test_phase_component_enters_series_coupling_and_generated_hamiltonian() -> None:
    """A phase shifter rotates both series coupling and its generated Hamiltonian."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    phase = network.phase_shift("phase", phase=np.pi / 2)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, phase)
    network.cascade(phase, b)
    network.expose("feedline", input=a.input, output=b.output)

    resolved = Chip([first, second], port_network=network).resolve().slh
    identity = np.eye(2)
    l_a = np.sqrt(0.04) * np.kron(_lowering(2), identity)
    l_b = np.sqrt(0.09) * np.kron(identity, _lowering(2))
    propagated = 1j * l_a
    product = l_b.conj().T @ propagated

    np.testing.assert_allclose(resolved.S, [[1j]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), propagated + l_b)
    generated = [term for term in resolved.H.static_terms if term.origin == "network"]
    np.testing.assert_allclose(
        generated[0].operator.to_dense(),
        (product - product.conj().T) / (2j),
    )


def test_sequence_template_retains_composed_input_free_slh() -> None:
    """Sequence assembly preserves the resolved input-free network model."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)
    network.expose("feedline", input=a.input, output=b.output)
    chip = Chip([first, second], port_network=network)

    resolved = QuantumSequence(chip).resolve().slh

    assert [channel.key for channel in resolved.external_channels] == ["feedline"]
    np.testing.assert_allclose(resolved.S, [[1.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), chip.resolve().slh.L[0].to_dense())


def test_exposure_delay_is_reference_plane_metadata_only() -> None:
    """Exposure delay moves the reference plane without entering instantaneous SLH."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.01)
    network.expose("readout", input=port.input, output=port.output, delay=0.25)

    resolved = Chip([resonator], port_network=network).resolve().slh

    assert resolved.external_channels[0].reference_delay == 0.25
    np.testing.assert_allclose(resolved.S, [[1.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), np.sqrt(0.01) * _lowering(2))


def test_attenuator_uses_power_transmission_and_hidden_vacuum_dilation() -> None:
    """Power attenuation resolves through a unitary hidden-vacuum dilation."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    loss = network.attenuator("cold_loss", eta=0.64)
    network.connect(port.output, loss.input)
    network.expose("readout", input=port.input, output=loss.output)

    resolved = Chip([resonator], port_network=network).resolve().slh
    coupling = np.sqrt(0.04) * _lowering(2)

    assert [channel.key for channel in resolved.channels] == [
        "readout",
        "hidden.cold_loss.vacuum",
    ]
    assert [channel.accessibility for channel in resolved.channels] == ["exposed", "hidden"]
    np.testing.assert_allclose(resolved.S, [[0.8, 0.6], [-0.6, 0.8]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), 0.8 * coupling)
    np.testing.assert_allclose(resolved.L[1].to_dense(), -0.6 * coupling)
    dissipative_strength = sum(
        operator.to_dense().conj().T @ operator.to_dense() for operator in resolved.L
    )
    np.testing.assert_allclose(dissipative_strength, coupling.conj().T @ coupling)


def test_network_dilation_precedes_stable_identity_hidden_baths() -> None:
    """Network vacuum channels precede identity scattering for device baths."""
    resonator = Resonator(
        freq=6.0,
        levels=2,
        internal_quality_factor=100_000,
        label="r",
    )
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    loss = network.attenuator("cold_loss", eta=0.64)
    network.connect(port.output, loss.input)
    network.expose("readout", input=port.input, output=loss.output)

    resolved = Chip([resonator], port_network=network).resolve().slh

    assert [channel.key for channel in resolved.channels[:2]] == [
        "readout",
        "hidden.cold_loss.vacuum",
    ]
    assert resolved.channels[2].collapse.source == "r"
    np.testing.assert_allclose(
        resolved.S,
        [[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
    )


def test_from_ports_builds_an_explicit_identity_network() -> None:
    """Existing port objects can seed the sole explicit network boundary."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    port = Port(resonator, rate=0.01, label="readout")

    chip = Chip([resonator], port_network=PortNetwork.from_ports([port]))

    assert chip.port_network is not None
    assert chip.port_network.ports == (port,)
    np.testing.assert_allclose(chip.resolve().slh.S, [[1.0]])


def test_chip_has_one_network_construction_path() -> None:
    """Ports enter a chip through its sole network-owned boundary."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    port = Port(resonator, rate=0.01, label="readout")

    with pytest.raises(TypeError, match="ports"):
        Chip([resonator], ports=[port])  # type: ignore[call-arg]


def test_second_network_cannot_silently_replace_the_first() -> None:
    """A chip rejects replacing an already attached field network."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    first = PortNetwork(label="first")
    first.port("readout", target=resonator, rate=0.01)
    second = PortNetwork(label="second")
    second.port("other", target=resonator, rate=0.01)
    chip = Chip([resonator], port_network=first)

    with pytest.raises(ValueError, match="disconnect_network"):
        chip.connect_network(second)

    assert chip.disconnect_network() is first
    assert chip.port_network is None
    assert chip.ports == ()


def test_network_graph_round_trips_and_clone_remains_independent() -> None:
    """Serialization preserves the graph while cloning isolates mutable structure."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    loss = network.attenuator("cold_loss", eta=0.64)
    network.connect(port.output, loss.input)
    network.expose("readout", input=port.input, output=loss.output, delay=0.1)
    chip = Chip([resonator], port_network=network)

    restored = Chip.from_dict(json.loads(json.dumps(chip.to_dict())))
    cloned = chip.clone()

    np.testing.assert_allclose(restored.resolve().slh.S, chip.resolve().slh.S)
    np.testing.assert_allclose(restored.resolve().slh.L[0].to_dense(), chip.resolve().slh.L[0].to_dense())
    assert restored.resolve().slh.external_channels[0].reference_delay == 0.1
    assert cloned.port_network is not chip.port_network
    assert cloned.ports[0] is not chip.ports[0]
    cloned.ports[0].rate = 0.09
    assert chip.ports[0].rate == 0.04
    rebound = chip.with_params({"network.component.cold_loss.eta": 0.25})
    np.testing.assert_allclose(rebound.resolve().slh.S[:2, :2], [[0.5, np.sqrt(0.75)], [-np.sqrt(0.75), 0.5]])
    np.testing.assert_allclose(chip.resolve().slh.S[:2, :2], [[0.8, 0.6], [-0.6, 0.8]])


def test_scattering_and_exposure_delay_are_bindable_network_parameters() -> None:
    """Scattering entries and exposure delays rebind through stable parameter paths."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(
        label="line",
        scattering={("readout", "readout"): 1.0},
    )
    port = network.port("chip_port", target=resonator, rate=0.04)
    network.expose("readout", input=port.input, output=port.output, delay=0.1)
    chip = Chip([resonator], port_network=network)

    rebound = chip.with_params(
        {
            "network.scattering.readout.readout": -1.0,
            "network.exposure.readout.delay": 0.2,
        }
    )

    np.testing.assert_allclose(rebound.resolve().slh.S, [[-1.0]])
    assert rebound.resolve().slh.external_channels[0].reference_delay == 0.2
    np.testing.assert_allclose(chip.resolve().slh.S, [[1.0]])
    assert chip.resolve().slh.external_channels[0].reference_delay == 0.1


def test_attenuator_power_transmission_is_jax_differentiable() -> None:
    """Attenuator transmission remains differentiable through network resolution."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    resonator = Resonator(freq=6.0, levels=2, label="r")

    def transmitted_coupling(eta):
        network = PortNetwork(label="line")
        port = network.port("chip_port", target=resonator, rate=0.04)
        loss = network.attenuator("cold_loss", eta=eta)
        network.connect(port.output, loss.input)
        network.expose("readout", input=port.input, output=loss.output)
        value = Chip([resonator], port_network=network).resolve().slh.L[0].to_dense()[0, 1]
        return jnp.real(value)

    np.testing.assert_allclose(transmitted_coupling(0.64), 0.16)
    np.testing.assert_allclose(jax.grad(transmitted_coupling)(0.64), 0.125)
