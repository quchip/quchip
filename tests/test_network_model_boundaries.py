"""Derived and exported models cannot silently omit network interactions."""

import numpy as np
import pytest

from quchip import Chip, Exact, PortNetwork, Resonator
from quchip.inverse_design.subsystems import build_local_subsystem


pytestmark = pytest.mark.unit


def _cascade_chip():
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)
    network.expose("feedline", input=a.input, output=b.output)
    return Chip([first, second], port_network=network, frame="lab", approximation=Exact())


def test_full_fit_subsystem_preserves_network_hamiltonian_and_channels():
    source = _cascade_chip()
    source.set_state_order("b", "a")
    local = build_local_subsystem(source, ("a", "b"))
    expected, actual = source.resolve(), local.resolve()
    np.testing.assert_allclose(actual.hamiltonian().matrix(), expected.hamiltonian().matrix())
    assert len(actual.slh.L) == len(expected.slh.L)
    np.testing.assert_allclose(actual.slh.L[0].to_dense(), expected.slh.L[0].to_dense())
    np.testing.assert_allclose(
        local.backend.to_array(local.bare_state("10")), source.backend.to_array(source.bare_state("10")),
    )
    assert local.port_network is not source.port_network


def test_partial_fit_subsystem_refuses_to_discard_network():
    with pytest.raises(NotImplementedError, match="PortNetwork"):
        build_local_subsystem(_cascade_chip(), ("a",))
    first = Resonator(freq=5.0, levels=2, label="a")
    second = Resonator(freq=6.0, levels=2, label="b")
    lowering = np.diag([1.0], 1)
    network = PortNetwork(label="line")
    network.port(
        "joint", target=(first, second), rate=0.04,
        operator=np.kron(lowering, np.eye(2)) + np.kron(np.eye(2), lowering),
    )
    with pytest.raises(NotImplementedError, match="port 'joint'"):
        build_local_subsystem(Chip([first, second], port_network=network), ("a",))


def test_scqubits_export_refuses_to_discard_network():
    pytest.importorskip("scqubits")
    from quchip.interop.scqubits import to_scqubits

    with pytest.raises(NotImplementedError, match="PortNetwork"):
        to_scqubits(_cascade_chip())
