"""Runtime-checkable Protocols for physical-operator drive dispatch."""

from __future__ import annotations


from quchip.devices import DuffingTransmon, Resonator
from quchip.devices.protocols import ChargeCoupled, FluxCoupled, PhaseCoupled


def test_duffing_transmon_exposes_standard_fock_channels():
    """DuffingTransmon exposes conventional oscillator control quadratures."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25)
    assert isinstance(q, ChargeCoupled)
    assert isinstance(q, PhaseCoupled)
    assert isinstance(q, FluxCoupled)


def test_resonator_exposes_standard_fock_channels():
    """Resonator exposes conventional oscillator control quadratures."""
    r = Resonator(freq=7.0, levels=6)
    assert isinstance(r, ChargeCoupled)
    assert isinstance(r, PhaseCoupled)
    assert isinstance(r, FluxCoupled)


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
