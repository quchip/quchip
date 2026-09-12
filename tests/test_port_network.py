"""Composable SLH field boundaries and their resolved physics."""

from __future__ import annotations

from typing import Any

import json
import warnings

import numpy as np
import pytest

from quchip import Chip, PortNetwork, QuantumSequence, Resonator
from quchip.chip.couplings import Capacitive


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


def test_attached_network_edits_update_future_calculations_and_parameter_paths() -> None:
    """Network authoring stays live while earlier resolved channels stay captured."""
    resonator = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(label="feedline")
    left = network.port("left", target=resonator, rate=0.01)
    chip = Chip([resonator], port_network=network)
    before = chip.resolve().slh
    clone = chip.clone()

    right = network.port("right", target=resonator, rate=0.04)

    assert chip.ports == (left, right)
    assert chip.port("right") is right
    assert chip.parameters["port.right.rate"] == 0.04
    assert chip.settings["ports"] == (("left", ("r",)), ("right", ("r",)))
    after = chip.resolve().slh
    assert [channel.key for channel in before.external_channels] == ["left"]
    assert [channel.key for channel in after.external_channels] == ["left", "right"]
    np.testing.assert_allclose(after.L[1].to_dense(), np.sqrt(0.04) * _lowering(3))
    assert [port.label for port in clone.ports] == ["left"]

    rebound = chip.with_params({"port.right.rate": 0.09})
    assert rebound.port("right").rate == 0.09
    assert right.rate == 0.04
    assert Chip.from_dict(chip.to_dict()).parameters["port.right.rate"] == 0.04
    assert chip.disconnect_network() is network
    assert chip.ports == ()


def test_new_network_port_targets_are_validated_on_the_next_calculation() -> None:
    """An attached graph cannot hide a new port whose target is absent."""
    resonator = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork()
    network.port("left", target=resonator, rate=0.01)
    chip = Chip([resonator], port_network=network)
    chip.resolve()
    network.port("bad", target="missing", rate=0.02)

    with pytest.raises((ValueError, KeyError), match="missing"):
        chip.resolve()


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

    with pytest.raises(ValueError, match="not statically known to be equal"):
        chip.resolve()


def test_phase_component_enters_series_coupling_and_generated_hamiltonian() -> None:
    """A phase shifter rotates both series coupling and its generated Hamiltonian."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    phase = network.phase_shift("phase", phase=np.pi / 2)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, phase, b)
    network.expose("feedline", input=a, output=b)

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


def test_cascade_and_expose_accept_terminals_ports_and_components() -> None:
    """One cascade call chains mixed endpoints; ambiguous shorthand fails loudly."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    line = network.phase_shift("line", phase=0.0)
    splitter = network.beam_splitter("splitter")
    network.cascade(port, line, splitter.input_terminal("left"))
    network.expose("readout", input=port, output=splitter.output_terminal("left"))

    with pytest.raises(AttributeError, match="multiple inputs"):
        network.expose("other", input=splitter, output=splitter.output_terminal("right"))
    with pytest.raises(ValueError, match="output terminal"):
        network.cascade(splitter.input_terminal("right"), port)

    network.expose("spare", input=splitter.input_terminal("right"), output=splitter.output_terminal("right"))

    resolved = Chip([resonator], port_network=network).resolve().slh
    transmitted = np.sqrt(0.5) * np.sqrt(0.04) * _lowering(2)
    np.testing.assert_allclose(resolved.L[0].to_dense(), transmitted)
    np.testing.assert_allclose(resolved.L[1].to_dense(), -transmitted)


def test_delay_section_is_reference_plane_metadata_only() -> None:
    """A linked delay decorates both legs of the plane without entering instantaneous SLH."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.01)
    cable = network.delay("cable", duration=0.25)
    network.link(port, cable)
    network.expose("readout", at=cable.port(2))

    resolved = Chip([resonator], port_network=network).resolve().slh
    plane = resolved.external_channels[0].reference

    assert [element.duration for element in plane.inbound] == [0.25]
    assert [element.duration for element in plane.outbound] == [0.25]
    np.testing.assert_allclose(resolved.S, [[1.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), np.sqrt(0.01) * _lowering(2))


def test_delay_section_after_a_circulator_decorates_only_the_reached_legs() -> None:
    """Reference sections apply per propagation path; interior sections are rejected."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="fridge")
    port = network.port("chip_port", target=resonator, rate=0.04)
    circulator = network.circulator("circ")
    line = network.delay("line", duration=1.5)
    network.link(port, circulator.port(2))
    network.link(circulator.port(3), line)
    network.expose("drive", at=circulator.port(1))
    network.expose("readout", at=line.port(2))
    chip = Chip([resonator], port_network=network)

    resolved = chip.resolve().slh
    drive_plane, readout_plane = (channel.reference for channel in resolved.external_channels)

    assert drive_plane.inbound == () and drive_plane.outbound == ()
    assert [element.label for element in readout_plane.inbound] == ["line"]
    assert [element.label for element in readout_plane.outbound] == ["line"]
    rebound = chip.with_params({"network.component.line.duration": 2.5}).resolve().slh
    assert rebound.external_channels[1].reference.outbound[0].duration == 2.5


def test_reference_section_and_downstream_loss_share_an_external_run() -> None:
    """A delay followed by matched loss retains the exact external CW transmission."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    interior = PortNetwork(label="interior")
    inner_port = interior.port("chip_port", target=resonator, rate=0.04)
    inner_loss = interior.attenuator("loss", eta=0.5)
    inner_cable = interior.delay("cable", duration=0.1)
    interior.link(inner_port, inner_cable, inner_loss)
    interior.expose("readout", at=inner_loss.port(2))
    chip = Chip([resonator], port_network=interior)
    from quchip import VNA
    actual = VNA(chip).sweep([6.0]).s("readout", "readout")[0]
    np.testing.assert_allclose(actual, -0.5*np.exp(4j*np.pi*6.0*0.1), atol=1e-10)


def test_attenuator_is_a_reciprocal_two_sided_vacuum_dilation() -> None:
    """A linked attenuator attenuates both directions through hidden vacuum channels."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    loss = network.attenuator("cold_loss", eta=0.64)
    network.link(port, loss)
    network.expose("readout", at=loss.port(2))

    resolved = Chip([resonator], port_network=network).resolve().slh
    coupling = np.sqrt(0.04) * _lowering(2)

    assert [channel.key for channel in resolved.channels] == [
        "readout",
        "hidden.cold_loss.vacuum_1",
        "hidden.cold_loss.vacuum_2",
    ]
    assert [channel.accessibility for channel in resolved.channels] == ["exposed", "hidden", "hidden"]
    np.testing.assert_allclose(
        resolved.S,
        [[0.64, 0.6, 0.48], [-0.48, 0.8, -0.36], [-0.6, 0.0, 0.8]],
    )
    np.testing.assert_allclose(resolved.L[0].to_dense(), 0.8 * coupling)
    np.testing.assert_allclose(resolved.L[1].to_dense(), -0.6 * coupling)
    np.testing.assert_allclose(resolved.L[2].to_dense(), 0.0 * coupling)
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
    network.link(port, loss)
    network.expose("readout", at=loss.port(2))

    resolved = Chip([resonator], port_network=network).resolve().slh

    assert [channel.key for channel in resolved.channels[:3]] == [
        "readout",
        "hidden.cold_loss.vacuum_1",
        "hidden.cold_loss.vacuum_2",
    ]
    assert resolved.channels[3].collapse.source == "r"
    np.testing.assert_allclose(resolved.S[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(resolved.S[:3, 3], 0.0)


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
    network.link(port, loss)
    cable = network.delay("cable", duration=0.1)
    network.link(loss, cable)
    network.expose("readout", at=cable.port(2))
    chip = Chip([resonator], port_network=network)

    restored = Chip.from_dict(json.loads(json.dumps(chip.to_dict())))
    cloned = chip.clone()

    np.testing.assert_allclose(restored.resolve().slh.S, chip.resolve().slh.S)
    np.testing.assert_allclose(restored.resolve().slh.L[0].to_dense(), chip.resolve().slh.L[0].to_dense())
    assert restored.resolve().slh.external_channels[0].reference == chip.resolve().slh.external_channels[0].reference
    assert cloned.port_network is not chip.port_network
    assert cloned.ports[0] is not chip.ports[0]
    cloned.ports[0].rate = 0.09
    assert chip.ports[0].rate == 0.04
    rebound = chip.with_params({"network.component.cold_loss.eta": 0.25})
    np.testing.assert_allclose(rebound.resolve().slh.S[0, :3], [0.25, np.sqrt(0.75), 0.5 * np.sqrt(0.75)])
    np.testing.assert_allclose(chip.resolve().slh.S[0, :3], [0.64, 0.6, 0.48])


def test_scattering_entries_are_bindable_network_parameters() -> None:
    """Scattering entries rebind through stable parameter paths."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(
        label="line",
        scattering={("readout", "readout"): 1.0},
    )
    network.port("chip_port", target=resonator, rate=0.04)
    network.expose("readout", at=network.ports[0])
    chip = Chip([resonator], port_network=network)

    rebound = chip.with_params({"network.scattering.readout.readout": -1.0})

    np.testing.assert_allclose(rebound.resolve().slh.S, [[-1.0]])
    np.testing.assert_allclose(chip.resolve().slh.S, [[1.0]])


def test_attenuator_power_transmission_is_jax_differentiable() -> None:
    """Attenuator transmission remains differentiable through network resolution."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    resonator = Resonator(freq=6.0, levels=2, label="r")

    def transmitted_coupling(eta):
        network = PortNetwork(label="line")
        port = network.port("chip_port", target=resonator, rate=0.04)
        loss = network.attenuator("cold_loss", eta=eta)
        network.link(port, loss)
        network.expose("readout", at=loss.port(2))
        value = Chip([resonator], port_network=network).resolve().slh.L[0].to_dense()[0, 1]
        return jnp.real(value)

    np.testing.assert_allclose(transmitted_coupling(0.64), 0.16)
    np.testing.assert_allclose(jax.grad(transmitted_coupling)(0.64), 0.125)


def test_circulator_and_isolator_route_a_reflection_readout() -> None:
    """Linked sides wire both directions; permutation rows compile terminal by terminal."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="fridge")
    port = network.port("chip_port", target=resonator, rate=0.04)
    circulator = network.circulator("circ")
    isolator = network.isolator("iso")
    network.link(port, circulator.port(2))
    network.link(circulator.port(3), isolator)
    network.expose("drive", at=circulator.port(1))
    network.expose("readout", at=isolator.port(2))
    chip = Chip([resonator], port_network=network)

    with pytest.raises(ValueError, match="side"):
        network.expose("ambiguous", at=isolator)

    resolved = chip.resolve().slh
    coupling = np.sqrt(0.04) * _lowering(2)

    assert [channel.key for channel in resolved.channels] == ["drive", "readout", "hidden.iso.load"]
    np.testing.assert_allclose(resolved.S, [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(resolved.L[0].to_dense(), 0.0 * coupling)
    np.testing.assert_allclose(resolved.L[1].to_dense(), coupling)
    assert not [term for term in resolved.H.static_terms if term.origin == "network"]

    restored = Chip.from_dict(json.loads(json.dumps(chip.to_dict())))
    np.testing.assert_allclose(restored.resolve().slh.S, resolved.S)
    np.testing.assert_allclose(chip.clone().resolve().slh.L[1].to_dense(), coupling)


def test_link_rejects_directional_components_and_keeps_true_feedback_explicit() -> None:
    """Only sided components link; a bidirectional line between two ports is feedback."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    directional = PortNetwork(label="directional")
    with pytest.raises(ValueError, match="side"):
        directional.link(
            directional.port("a_port", target=first, rate=0.04),
            directional.phase_shift("line", phase=0.1),
        )

    network = PortNetwork(label="loop")
    network.link(
        network.port("a_port", target=first, rate=0.04),
        network.port("b_port", target=second, rate=0.09),
    )
    with pytest.raises(ValueError, match="feedback|cycle"):
        Chip([first, second], port_network=network).resolve()


def test_every_builtin_component_round_trips_as_kind_and_parameters() -> None:
    """Serialized networks rebuild built-in components from their factory kind and parameters."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    network = PortNetwork(label="fridge")
    port = network.port("chip_port", target=resonator, rate=0.04)
    circ = network.circulator("circ", ports=4)
    iso = network.isolator("iso")
    loss = network.attenuator("loss", eta=0.25)
    splitter = network.beam_splitter("split", eta=0.3)
    hybrid = network.hybrid90("hyb")
    swap = network.permutation("swap", order=[1, 0])
    phase = network.phase_shift("phase", phase=0.4)
    through = network.through("thru")
    custom = network.component("custom", scattering=[[0.0, 1j], [1j, 0.0]], terminals=("a", "b"))
    cable = network.delay("cable", duration=1.5)
    hemt = network.amplifier("hemt", gain=50.0, added_noise=1.0)
    network.link(port, circ.port(2))
    network.link(circ.port(3), iso, loss, cable, hemt)
    network.expose("readout", at=hemt.port(2))
    network.expose("drive", at=circ.port(1))
    network.expose("spare", at=circ.port(4))
    network.cascade(splitter.output_terminal("left"), phase, through, hybrid.input_terminal("left"))
    network.cascade(splitter.output_terminal("right"), swap.input_terminal("0"))
    network.cascade(hybrid.output_terminal("left"), custom.input_terminal("a"))
    network.cascade(swap.output_terminal("0"), custom.input_terminal("b"))
    network.expose("aux_a", input=splitter.input_terminal("left"), output=custom.output_terminal("a"))
    network.expose("aux_b", input=splitter.input_terminal("right"), output=custom.output_terminal("b"))
    network.expose("hyb_side", input=hybrid.input_terminal("right"), output=hybrid.output_terminal("right"))
    network.expose("swap_side", input=swap.input_terminal("1"), output=swap.output_terminal("1"))
    chip = Chip([resonator], port_network=network)

    payload = json.loads(json.dumps(chip.to_dict()))
    restored = Chip.from_dict(payload)

    kinds = {item["label"]: item for item in payload["port_network"]["components"]}
    assert kinds["circ"] == {"label": "circ", "kind": "circulator", "parameters": {"ports": 4}}
    assert kinds["swap"]["parameters"] == {"order": [1, 0]}
    assert kinds["custom"]["kind"] == "scattering" and kinds["custom"]["terminals"] == ["a", "b"]
    assert restored.port_network is not None
    assert restored.port_network.to_dict() == chip.port_network.to_dict()
    original, rebuilt = chip.resolve().slh, restored.resolve().slh
    np.testing.assert_allclose(rebuilt.S, original.S)
    assert [c.key for c in rebuilt.channels] == [c.key for c in original.channels]
    assert restored.parameters == chip.parameters
    assert "component.circ.ports" not in network.parameters
    assert "component.swap.order" not in network.parameters
    with pytest.raises(TypeError, match="kind"):
        PortNetwork.from_dict({"components": [{"label": "x", "kind": "warp_drive", "parameters": {}}]})
    with pytest.raises(TypeError, match="no kind"):
        PortNetwork.from_dict({"components": [{"label": "x", "parameters": {}}]})


def _open_system(network: PortNetwork, chip: Chip) -> tuple[Any, dict[str, str]]:
    """Every core terminal as one channel of an unconnected SLH triple (ports carry L, components none).

    ``network`` must have no connections so every port resolves to its own exposed channel.
    """
    from quchip.engine import concatenate
    from quchip.engine.ir import CollapseTerm, HamiltonianProgram, ResolvedSLH, SLHChannel

    resolved_ports = {channel.key: channel for channel in chip.resolve().slh.external_channels}
    zero = next(iter(resolved_ports.values())).coupling.scaled(0.0)
    systems = []
    for component in network.components:
        matrix = network._component_matrix(component)
        size = len(component.output_names)
        channels = []
        for index, name in enumerate(component.output_names):
            key = f"{component.label}.{name}"
            local = component._local_ports[index]
            coupling = zero if local is None else resolved_ports[local.label].coupling
            collapse = CollapseTerm(operator=coupling, rate=1.0, source=key, channel="port", frame_frequency=None)
            channels.append(SLHChannel(key=key, accessibility="exposed", collapse=collapse, coupling_operator=coupling))
        systems.append(
            ResolvedSLH(
                scattering=np.asarray(matrix, dtype=complex),
                hamiltonian=HamiltonianProgram(),
                channels=tuple(channels),
                support=np.ones((size, size), dtype=bool),
            )
        )
    return concatenate(*systems), {}


def _close_connections(network: PortNetwork, system: Any) -> Any:
    """Close every connection with feedback_reduce, following the merged channel keys it creates."""
    from quchip.engine import feedback_reduce

    current_input = {channel.key: channel.key for channel in system.channels}
    current_output = dict(current_input)
    for input_key, output_key in network._connections.items():
        source = current_output[".".join(output_key)]
        sink = current_input[".".join(input_key)]
        system = feedback_reduce(system, output=source, input=sink)
        merged = f"{source}->{sink}"
        for terminal, key in current_input.items():
            if key == source:
                current_input[terminal] = merged
        for terminal, key in current_output.items():
            if key == sink:
                current_output[terminal] = merged
    return system


def test_feedback_loop_matches_gough_james_reduction() -> None:
    """A lossy ring through a beam splitter compiles to the algebraic feedback reduction."""

    first = Resonator(freq=5.0, levels=3, label="a")
    second = Resonator(freq=5.0, levels=3, label="b")
    network = PortNetwork(label="ring")
    port_a = network.port("pa", target=first, rate=0.04)
    port_b = network.port("pb", target=second, rate=0.09)
    splitter = network.beam_splitter("bs", eta=0.36)
    phase = network.phase_shift("phi", phase=0.7)
    network.cascade(port_a, splitter.input_terminal("left"))
    network.cascade(splitter.output_terminal("left"), port_b, phase, splitter.input_terminal("right"))
    network.expose("ring", input=port_a.input, output=splitter.output_terminal("right"))
    compiled = Chip([first, second], port_network=network).resolve().slh

    bare = PortNetwork(label="bare")
    bare.port("pa", target=Resonator(freq=5.0, levels=3, label="a"), rate=0.04)
    bare.port("pb", target=Resonator(freq=5.0, levels=3, label="b"), rate=0.09)
    open_chip = Chip(list(device for port in bare.ports for device in port._targets), port_network=bare)
    oracle = _close_connections(network, _open_system(network, open_chip)[0])

    assert [channel.key for channel in compiled.channels] == ["ring"]
    np.testing.assert_allclose(compiled.S, oracle.S, atol=1e-12)
    np.testing.assert_allclose(compiled.L[0].to_dense(), oracle.L[0].to_dense(), atol=1e-12)
    compiled_h = sum(term.operator.to_dense() for term in compiled.H.static_terms if term.origin == "network")
    oracle_h = sum(term.operator.to_dense() for term in oracle.H.static_terms if term.origin == "network")
    np.testing.assert_allclose(compiled_h, oracle_h, atol=1e-12)
    assert compiled.feeds(0, 0)


def test_feedback_rejects_reference_cycle() -> None:
    """A delay inside an instantaneous loop is not Markovian and is refused by name."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="loop")
    port = network.port("p", target=resonator, rate=0.04)
    splitter = network.beam_splitter("bs", eta=0.5)
    cable = network.delay("cable", duration=1.0)
    network.cascade(port, splitter.input_terminal("left"))
    network.cascade(splitter.output_terminal("left"), cable.input_terminal("1"))
    network.cascade(cable.output_terminal("2"), splitter.input_terminal("right"))
    network.expose("out", input=port.input, output=splitter.output_terminal("right"))

    with pytest.raises(ValueError, match="cable.*feedback loop"):
        Chip([resonator], port_network=network).resolve()


def test_feedback_rejects_singular_concrete_loop() -> None:
    """A loop whose round trip is exactly unity has no instantaneous solution."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="loop")
    port = network.port("p", target=resonator, rate=0.04)
    line = network.through("line")
    network.connect(port.output, line.input)
    network.connect(line.output, port.input)

    with pytest.raises(ValueError, match="singular"):
        Chip([resonator], port_network=network).resolve()


def test_feedback_gain_is_jittable_and_differentiable() -> None:
    """A traced loop phase flows through the compiled ring gain."""
    jax = pytest.importorskip("jax")
    resonator = Resonator(freq=5.0, levels=2, label="r")

    def reflection(value: Any) -> Any:
        network = PortNetwork(label="ring")
        port = network.port("p", target=resonator, rate=0.04)
        splitter = network.beam_splitter("bs", eta=0.5)
        phase = network.phase_shift("phi", phase=value)
        network.cascade(port, splitter.input_terminal("left"))
        network.cascade(splitter.output_terminal("left"), phase, splitter.input_terminal("right"))
        network.expose("out", input=port.input, output=splitter.output_terminal("right"))
        slh = Chip([resonator], port_network=network).resolve().slh
        return jax.numpy.abs(slh.S[0, 0]) ** 2

    value, gradient = jax.jit(jax.value_and_grad(reflection))(jax.numpy.asarray(0.3))
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))
    np.testing.assert_allclose(float(value), float(reflection(0.3)), rtol=1e-6)


def _ring(eta: Any) -> Chip:
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="ring")
    port = network.port("p", target=resonator, rate=0.04)
    splitter = network.beam_splitter("bs", eta=eta)
    phase = network.phase_shift("phi", phase=0.3)
    network.cascade(port, splitter.input_terminal("left"))
    network.cascade(splitter.output_terminal("left"), phase, splitter.input_terminal("right"))
    network.expose("out", input=port.input, output=splitter.output_terminal("right"))
    return Chip([resonator], port_network=network)


def test_loop_support_is_structural_not_numerical() -> None:
    """Reachability through a loop follows the wiring, not the current coefficient values."""
    transparent = _ring(1.0).resolve().slh
    mixing = _ring(0.36).resolve().slh
    np.testing.assert_array_equal(transparent.support, mixing.support)
    assert transparent.feeds(0, 0)


def test_many_channel_loop_with_small_determinant_is_not_singular() -> None:
    """Five near-resonant loops in one component have a tiny determinant but a benign condition number."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="mixer")
    network.port("p", target=resonator, rate=0.04)
    detuning = 0.002
    names = tuple(str(index) for index in range(6))
    mixer = network.component("mix", scattering=np.diag(np.full(6, np.exp(1j * detuning))), terminals=names)
    for name in names[1:]:
        network.connect(mixer.output_terminal(name), mixer.input_terminal(name))
    network.expose("probe", input=mixer.input_terminal("0"), output=mixer.output_terminal("0"))

    resolved = Chip([resonator], port_network=network).resolve().slh

    assert abs(np.linalg.det(np.eye(5) - np.exp(1j * detuning) * np.eye(5))) < 1e-12
    probe = [channel.key for channel in resolved.channels].index("probe")
    np.testing.assert_allclose(resolved.S[probe, probe], np.exp(1j * detuning))


def _two_line_chip() -> Chip:
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=5.4, levels=2, label="b")
    network = PortNetwork(label="lines")
    port_a = network.port("pa", target=first, rate=0.02)
    port_b = network.port("pb", target=second, rate=0.03)
    loss = network.attenuator("loss", eta=0.5)
    cable = network.delay("cable", duration=1.0)
    network.link(port_a, loss)
    network.expose("line_a", at=loss.port(2))
    network.link(port_b, cable)
    network.expose("line_b", at=cable.port(2))
    return Chip([first, second], port_network=network)


def test_restrict_keeps_the_subgraphs_touching_selected_ports() -> None:
    """restrict() copies every component, connection, and plane reachable from the chosen ports."""
    chip = _two_line_chip()
    network = chip.port_network
    assert network is not None
    full = chip.resolve().slh

    line_a = network.restrict(["pa"])

    assert [port.label for port in line_a.ports] == ["pa"]
    assert {component.label for component in line_a.components} == {"pa", "loss"}
    assert [exposure.label for exposure in line_a.external_ports] == ["line_a"]
    sub = Chip([Resonator(freq=5.0, levels=2, label="a")], port_network=line_a).resolve().slh
    keys = [channel.key for channel in full.channels]
    rows = [keys.index(channel.key) for channel in sub.channels]
    np.testing.assert_allclose(sub.S, np.asarray(full.S)[np.ix_(rows, rows)])
    assert "component.loss.eta" in line_a.parameters and "component.cable.duration" not in line_a.parameters

    line_b = network.restrict(["pb"])
    assert [exposure.label for exposure in line_b.external_ports] == ["line_b"]
    resolved_b = Chip([Resonator(freq=5.4, levels=2, label="b")], port_network=line_b).resolve().slh
    assert [element.label for element in resolved_b.external_channels[0].reference.outbound] == ["cable"]


def test_restrict_rejects_subgraphs_spanning_other_ports() -> None:
    """A subgraph that also touches an unselected port cannot be cut without changing the dynamics."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=5.4, levels=2, label="b")
    network = PortNetwork(label="cascade")
    port_a = network.port("pa", target=first, rate=0.02)
    port_b = network.port("pb", target=second, rate=0.03)
    network.cascade(port_a, port_b)
    network.expose("feedline", input=port_a.input, output=port_b.output)

    with pytest.raises(ValueError, match="pb"):
        network.restrict(["pa"])
    with pytest.raises(KeyError):
        network.restrict(["missing"])


def _readout_line() -> PortNetwork:
    """A reusable fridge output line: cold attenuator, isolator, cable, amplifier."""
    line = PortNetwork(label="readout_line")
    loss = line.attenuator("att", eta=0.5)
    isolator = line.isolator("iso")
    cable = line.delay("cable", duration=2.0)
    hemt = line.amplifier("hemt", gain=100.0, added_noise=2.0)
    line.link(loss, isolator, cable, hemt)
    line.expose("chip", at=loss.port(1))
    line.expose("room", at=hemt.port(2))
    return line


def test_include_instantiates_a_block_per_line_with_prefixed_labels() -> None:
    """One block definition wires two resonators through identical prefixed copies."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=5.4, levels=2, label="b")
    template = _readout_line()
    network = PortNetwork(label="fridge")
    for resonator, prefix in ((first, "r1"), (second, "r2")):
        port = network.port(f"{prefix}_port", target=resonator, rate=0.02)
        line = network.include(template, prefix=prefix)
        network.link(port, line.port("chip"))
        network.expose(f"{prefix}_out", at=line.port("room"))
    chip = Chip([first, second], port_network=network)

    labels = {component.label for component in network.components}
    assert {"r1/att", "r1/iso", "r1/cable", "r1/hemt", "r2/att", "r2/hemt"} <= labels
    assert {"component.r1/att.eta", "component.r2/cable.duration"} <= set(network.parameters)
    assert len(template.components) == 4
    assert [exposure.label for exposure in template.external_ports] == ["chip", "room"]

    resolved = chip.resolve().slh
    keys = [channel.key for channel in resolved.channels]
    assert keys[:2] == ["r1_out", "r2_out"]
    assert [element.label for element in resolved.external_channels[0].reference.outbound] == ["r1/cable", "r1/hemt"]

    by_hand = PortNetwork(label="one")
    port = by_hand.port("r1_port", target=Resonator(freq=5.0, levels=2, label="a"), rate=0.02)
    loss = by_hand.attenuator("att", eta=0.5)
    hemt = by_hand.amplifier("hemt", gain=100.0, added_noise=2.0)
    by_hand.link(port, loss, by_hand.isolator("iso"), by_hand.delay("cable", duration=2.0), hemt)
    by_hand.expose("r1_out", at=hemt.port(2))
    lone = Chip([Resonator(freq=5.0, levels=2, label="a")], port_network=by_hand).resolve().slh
    np.testing.assert_allclose(resolved.S[0, 0], lone.S[0, 0])
    np.testing.assert_allclose(resolved.L[0].to_dense(), np.kron(lone.L[0].to_dense(), np.eye(2)))
    rebound = chip.with_params({"network.component.r1/att.eta": 0.25})
    assert rebound.port_network is not None and rebound.port_network.parameters["component.r1/att.eta"] == 0.25


def test_include_interfaces_and_rejections() -> None:
    """Asymmetric interfaces use input()/output(); templates with ports or boundary scattering are refused."""
    template = PortNetwork(label="splitter_block")
    splitter = template.beam_splitter("bs", eta=0.5)
    phase = template.phase_shift("phi", phase=0.2)
    template.cascade(splitter.output_terminal("left"), phase)
    template.expose("in", input=splitter.input_terminal("left"), output=phase.output)
    template.expose("aux", input=splitter.input_terminal("right"), output=splitter.output_terminal("right"))

    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="host")
    port = network.port("p", target=resonator, rate=0.02)
    block = network.include(template, prefix="b")
    network.cascade(port, block.input("in"))
    network.expose("through", input=block.input("aux"), output=block.output("in"))
    network.expose("side", input=port.input, output=block.output("aux"))
    resolved = Chip([resonator], port_network=network).resolve().slh
    assert [channel.key for channel in resolved.channels] == ["through", "side"]
    assert block.component("phi").label == "b/phi"
    with pytest.raises(ValueError, match="input\\(\\)|output\\(\\)"):
        block.port("in")
    with pytest.raises(KeyError):
        block.port("missing")

    with_ports = PortNetwork(label="with_ports")
    with_ports.port("q", target=resonator, rate=0.01)
    with pytest.raises(ValueError, match="ports"):
        network.include(with_ports, prefix="x")
    with_boundary = PortNetwork(label="boundary", scattering={("a", "a"): -1.0})
    passthrough = with_boundary.through("t")
    with_boundary.expose("a", input=passthrough, output=passthrough)
    with pytest.raises(ValueError, match="scattering"):
        network.include(with_boundary, prefix="y")
    with pytest.raises(ValueError, match="[Pp]refix"):
        network.include(template, prefix="b")
    before = template.to_dict()
    with pytest.raises(ValueError, match="include itself"):
        template.include(template, prefix="again")
    assert template.to_dict() == before


def test_copied_component_parameters_do_not_alias_authored_arrays() -> None:
    """Graph copying owns numerical parameters while retaining callable identity."""
    eta = np.asarray(0.64)
    template = PortNetwork(label="template")
    loss = template.attenuator("loss", eta=eta)
    template.expose("chip", at=loss.port(1))
    template.expose("room", at=loss.port(2))
    copied = template.copy()
    host = PortNetwork(label="host")
    host.include(template, prefix="a")
    eta[...] = 0.25
    assert template.parameters["component.loss.eta"] == pytest.approx(0.25)
    assert copied.parameters["component.loss.eta"] == pytest.approx(0.64)
    assert host.parameters["component.a/loss.eta"] == pytest.approx(0.64)


def test_include_keeps_filter_callables_by_reference() -> None:
    """Filters inside a block stay callable and sweepable under their prefixed parameter path."""

    def lowpass(frequency, cutoff):
        return 1.0 / (1.0 + 1j * frequency / cutoff)

    template = PortNetwork(label="filtered")
    stage = template.filter("lp", transfer=lowpass, cutoff=8.0)
    template.expose("chip", at=stage.port(1))
    template.expose("room", at=stage.port(2))
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="host")
    port = network.port("p", target=resonator, rate=0.02)
    block = network.include(template, prefix="f")
    network.link(port, block.port("chip"))
    network.expose("out", at=block.port("room"))
    chip = Chip([resonator], port_network=network)

    resolved = chip.resolve().slh
    element = resolved.external_channels[0].reference.outbound[0]
    assert element.label == "f/lp"
    np.testing.assert_allclose(element.transfer(8.0, **dict(element.parameters)), lowpass(8.0, 8.0))
    rebound = chip.with_params({"network.component.f/lp.cutoff": 4.0})
    assert rebound.port_network is not None
    assert rebound.port_network.parameters["component.f/lp.cutoff"] == 4.0


def test_directional_component_side_error_is_actionable() -> None:
    """A directional splitter has no physical sides and says which accessors to use instead."""
    network = PortNetwork(label="line")
    splitter = network.beam_splitter("bs", eta=0.5)
    with pytest.raises(ValueError, match="directional.*input_terminal"):
        splitter.port("left")
    with pytest.raises(ValueError, match="directional"):
        network.link(network.port("p", target=Resonator(freq=5.0, levels=2, label="r"), rate=0.02), splitter)


def _cascaded_pair() -> Chip:
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=5.0, levels=2, label="b")
    network = PortNetwork(label="line")
    port_a = network.port("a_port", target=first, rate=0.05)
    port_b = network.port("b_port", target=second, rate=0.05)
    network.cascade(port_a, port_b)
    network.expose("feedline", input=port_a.input, output=port_b.output)
    return Chip([first, second], port_network=network, frame=5.0)


def test_cascaded_degenerate_modes_solve_and_transfer_the_excitation() -> None:
    """Pitch-and-catch between two identical modes: n_b peaks at 4/e^2 at t = 2/kappa."""
    chip = _cascaded_pair()
    times = np.linspace(0.0, 120.0, 241)
    result = QuantumSequence(chip).simulate(
        times,
        e_ops={"b": chip["b"].number_operator()},
        initial_state=chip.bare_state({"a": 1, "b": 0}),
        partition=False,
        )
    occupation = np.asarray(result.expect("b")).real
    np.testing.assert_allclose(occupation[np.argmin(np.abs(times - 40.0))], 4.0 / np.e**2, atol=2e-3)
    assert times[np.argmax(occupation)] == pytest.approx(40.0, abs=1.0)


def test_state_warns_when_the_cascade_hybridizes_the_requested_label() -> None:
    """Degenerate cascaded modes dress into a 50/50 pair; state() says so and points to bare_state."""
    chip = _cascaded_pair()
    with pytest.warns(UserWarning, match=r"assignment overlap 0\.500.*bare_state"):
        dressed = chip.state({"a": 1, "b": 0})
    index = np.ravel_multi_index((1, 0), (2, 2))
    populations = np.abs(np.asarray(chip.backend.to_array(dressed)).ravel()) ** 2
    assert populations[index] == pytest.approx(0.5, abs=1e-6)
    bare = np.abs(np.asarray(chip.backend.to_array(chip.bare_state({"a": 1, "b": 0}))).ravel()) ** 2
    assert bare[index] == pytest.approx(1.0)


def test_state_stays_quiet_for_well_separated_labels() -> None:
    """A weakly coupled, detuned pair keeps every label above the warning overlap."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    chip = Chip([first, second], couplings=[Capacitive(first, second, g=0.01)])
    assert chip.dress().assignment_overlaps[(1, 0)] > 0.9
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        chip.state({"a": 1, "b": 0})


def test_two_port_giant_atom_reproduces_kockum_rates_and_lamb_shift() -> None:
    """Two coupling points a phase phi apart: Gamma = 2 gamma (1 + cos phi) and Delta = +gamma sin phi."""
    gamma = 0.02
    for phi in (np.pi / 2, -np.pi / 2, np.pi):
        atom = Resonator(freq=5.0, levels=2, label="q")
        network = PortNetwork(label="wg")
        near = network.port("near", target=atom, rate=gamma)
        far = network.port("far", target=atom, rate=gamma)
        network.cascade(near, network.phase_shift("phi", phase=phi), far)
        network.expose("feedline", input=near.input, output=far.output)
        slh = Chip([atom], port_network=network, frame=5.0).resolve().slh
        coupling = slh.L[0].to_dense()
        np.testing.assert_allclose(abs(coupling[0, 1]) ** 2, 2 * gamma * (1 + np.cos(phi)), atol=1e-12)
        generated = [term for term in slh.H.static_terms if term.origin == "network"]
        assert bool(generated) is not bool(np.isclose(np.sin(phi), 0.0))  # a vanishing cross term is dropped
        shift = sum(term.operator.to_dense()[1, 1].real for term in generated)
        np.testing.assert_allclose(shift, gamma * np.sin(phi), atol=1e-12)


def test_connecting_a_network_invalidates_the_dressed_cache() -> None:
    """Warm dressing before a cascade is attached, then the assigned states must follow the new terms."""
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=5.0, levels=2, label="b")
    chip = Chip([first, second], frame=5.0)
    assert chip.dress().assignment_overlaps[(1, 0)] == pytest.approx(1.0)

    network = PortNetwork(label="line")
    port_a = network.port("a_port", target=first, rate=0.05)
    port_b = network.port("b_port", target=second, rate=0.05)
    network.cascade(port_a, port_b)
    network.expose("feedline", input=port_a.input, output=port_b.output)
    chip.connect_network(network)

    assert chip.dress().assignment_overlaps[(1, 0)] == pytest.approx(0.5, abs=1e-6)
    with pytest.warns(UserWarning, match="assignment overlap 0.500"):
        chip.state({"a": 1, "b": 0})
