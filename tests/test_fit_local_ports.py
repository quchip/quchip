"""Local fit evaluation keeps independent ports and refuses to cut a shared field network."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, PortNetwork, Resonator, fit_a_dress
from quchip.inverse_design.subsystems import build_local_subsystem


def _readout_chip(line: str | None = None) -> Chip:
    """Two transmons with one readout each, optionally connected to a field network."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    r0 = Resonator(freq=7.0, levels=3, label="r0")
    r1 = Resonator(freq=7.3, levels=3, label="r1")
    network = None
    if line == "independent":
        network = PortNetwork(label="readout")
        for readout in (r0, r1):
            network.expose(readout.label, at=network.port(readout.label, target=readout, rate=0.006))
    elif line == "feedline":
        network = PortNetwork(label="readout")
        first = network.port("r0", target=r0, rate=0.006)
        second = network.port("r1", target=r1, rate=0.006)
        network.cascade(first, second)
        network.expose("feedline", input=first.input, output=second.output)
    elif line == "joint":
        lowering = np.diag(np.sqrt([1.0, 2.0]), 1)
        operator = np.kron(lowering, np.eye(3)) + np.kron(np.eye(3), lowering)
        network = PortNetwork(label="readout")
        network.port("joint", target=(r0, r1), rate=0.006, operator=operator)
    return Chip(
        [q0, q1, r0, r1],
        [Capacitive(q0, r0, g=-1.0e-4, label="q0_r0"), Capacitive(q1, r1, g=-1.2e-4, label="q1_r1")],
        frame=5.1,
        port_network=network,
    )


@pytest.mark.unit
def test_local_subsystem_keeps_only_the_ports_of_kept_devices() -> None:
    """A local subsystem keeps each port whose targets it holds, with its rate and external plane."""
    chip = _readout_chip("independent")

    local = build_local_subsystem(chip, ("q0", "r0"))

    assert local.port_network is not None
    assert [port.label for port in local.ports] == ["r0"]
    assert local.port("r0").resolve_targets(local) == ("r0",)
    assert local.port("r0").rate == chip.port("r0").rate
    assert [plane.label for plane in local.port_network.external_ports] == ["r0"]
    assert build_local_subsystem(chip, ("q0", "q1")).port_network is None
    assert [port.label for port in chip.ports] == ["r0", "r1"]


def test_local_fit_with_independent_ports_matches_the_portless_fit() -> None:
    """Independent ports add no Hamiltonian term, so the local fit is the same with and without them."""
    expected = fit_a_dress(_readout_chip(), evaluator="local")

    result = fit_a_dress(_readout_chip("independent"), evaluator="local")

    assert result.converged
    # Both fits evaluate identical Hamiltonians, so only round-off can separate them.
    assert result.final_params == pytest.approx(expected.final_params, rel=1e-12)
    assert [report.final for report in result.final_targets] == pytest.approx(
        [report.final for report in expected.final_targets], rel=1e-12
    )


@pytest.mark.unit
def test_local_fit_refuses_a_feedline_that_leaves_the_subsystem() -> None:
    """A kept readout cascaded with a discarded readout cannot be cut from the shared feedline."""
    with pytest.raises(NotImplementedError, match=r"PortNetwork.*also touches ports \['r1'\]"):
        fit_a_dress(_readout_chip("feedline"), evaluator="local")


@pytest.mark.unit
def test_local_subsystem_refuses_a_port_on_kept_and_discarded_devices() -> None:
    """A port that also acts on a discarded device cannot be kept or dropped."""
    with pytest.raises(NotImplementedError, match="port 'joint'"):
        build_local_subsystem(_readout_chip("joint"), ("q0", "r0"))
