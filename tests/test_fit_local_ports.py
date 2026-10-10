"""Local fit evaluation keeps the ports of each neighborhood."""

from __future__ import annotations

import pytest

from quchip import Capacitive, Chip, DuffingTransmon, PortNetwork, Resonator, fit_a_dress
from quchip.inverse_design.subsystems import build_local_subsystem


def _readout_chip(ports: bool) -> Chip:
    """A transmon with a readout and a second readout, optionally with one port per readout."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    r0 = Resonator(freq=7.0, levels=3, label="r0")
    r1 = Resonator(freq=7.3, levels=3, label="r1")
    network = None
    if ports:
        network = PortNetwork(label="readout")
        for readout in (r0, r1):
            network.expose(readout.label, at=network.port(readout.label, target=readout, rate=0.006))
    return Chip([q0, r0, r1], [Capacitive(q0, r0, g=-1.0e-4, label="q0_r0")], frame=5.1, port_network=network)


@pytest.mark.unit
def test_local_subsystem_keeps_only_the_ports_of_kept_devices() -> None:
    """A local subsystem keeps each port whose targets it holds, and no network without one."""
    chip = _readout_chip(ports=True)

    assert [port.label for port in build_local_subsystem(chip, ("q0", "r0")).ports] == ["r0"]
    assert build_local_subsystem(chip, ("q0",)).port_network is None


def test_local_fit_with_independent_ports_matches_the_portless_fit() -> None:
    """Independent ports add no Hamiltonian term, so the local fit is the same with and without them."""
    expected = fit_a_dress(_readout_chip(ports=False), evaluator="local")

    result = fit_a_dress(_readout_chip(ports=True), evaluator="local")

    assert result.converged
    assert result.final_params == pytest.approx(expected.final_params, rel=1e-12)
