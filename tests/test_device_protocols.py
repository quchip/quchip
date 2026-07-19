"""Runtime-checkable Protocols for physical-operator drive dispatch."""

from __future__ import annotations


from quchip.devices import DuffingTransmon, Resonator
from quchip.devices.protocols import ChargeCoupled, FluxCoupled, PhaseCoupled


def test_duffing_transmon_is_not_charge_coupled():
    """DuffingTransmon is Fock-eigenbasis — no physical charge operator."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25)
    assert not isinstance(q, ChargeCoupled)
    assert not isinstance(q, PhaseCoupled)
    assert not isinstance(q, FluxCoupled)


def test_resonator_is_not_protocol_conformant():
    """Resonator exposes none of the physical coupling-operator protocols."""
    r = Resonator(freq=7.0, levels=6)
    assert not isinstance(r, ChargeCoupled)
    assert not isinstance(r, PhaseCoupled)
    assert not isinstance(r, FluxCoupled)


def test_mock_conformant_class_is_recognized():
    """A class that happens to expose the method is recognized."""

    class FakeFluxonium:
        def charge_coupling_operator(self):
            return None

        def phase_coupling_operator(self):
            return None

        def flux_coupling_operator(self):
            return None

    obj = FakeFluxonium()
    assert isinstance(obj, ChargeCoupled)
    assert isinstance(obj, PhaseCoupled)
    assert isinstance(obj, FluxCoupled)
