"""Drive classes dispatch to physical operators via Protocols when available."""

from __future__ import annotations

import numpy as np

from quchip.backend.qutip import QuTiPBackend
from quchip.control.drive import ChargeDrive, FluxDrive, PhaseDrive
from quchip.devices import DuffingTransmon


class _FakePhysicalDevice:
    """Exposes all three coupling operators — tests the Protocol-conformant path."""

    levels = 3
    label = "fake"

    def __init__(self):
        self._charge = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.3], [0.0, 0.3, 0.0]], dtype=complex
        )
        self._phase = np.array(
            [[0.0, 0.5, 0.0], [0.5, 0.0, 0.2], [0.0, 0.2, 0.0]], dtype=complex
        )
        self._flux = np.array(
            [[0.0, 0.7, 0.0], [0.7, 0.0, 0.4], [0.0, 0.4, 0.0]], dtype=complex
        )
        self._connected_drives = []

    def charge_coupling_operator(self):
        return self._charge

    def phase_coupling_operator(self):
        return self._phase

    def flux_coupling_operator(self):
        return self._flux

    def connect(self, drive):
        self._connected_drives.append(drive)


def test_charge_drive_falls_through_for_duffing():
    """DuffingTransmon does not implement the Protocol — structural path wins."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25)
    drive = ChargeDrive(target=q)
    (channel,) = drive.local_channels(q)

    backend = QuTiPBackend()
    a = backend.destroy(q.levels)
    expected = 1j * (a - backend.dag(a))
    assert np.allclose(np.array(channel.operator.full()), np.array(expected.full()))


def test_charge_drive_prefers_physical_operator_for_protocol_device():
    """A ChargeCoupled-conformant device's physical operator is used over the structural fallback."""
    fake = _FakePhysicalDevice()
    drive = ChargeDrive()
    (channel,) = drive.local_channels(fake)
    assert np.allclose(channel.operator, fake._charge)


def test_phase_drive_prefers_physical_operator():
    """A PhaseCoupled-conformant device's physical operator is used over the structural fallback."""
    fake = _FakePhysicalDevice()
    drive = PhaseDrive()
    (channel,) = drive.local_channels(fake)
    assert np.allclose(channel.operator, fake._phase)


def test_flux_drive_prefers_physical_operator():
    """A FluxCoupled-conformant device's physical operator is used over the structural fallback."""
    fake = _FakePhysicalDevice()
    drive = FluxDrive()
    (channel,) = drive.local_channels(fake)
    assert np.allclose(channel.operator, fake._flux)


def test_flux_drive_falls_back_to_number_operator_for_duffing():
    """DuffingTransmon does not conform to FluxCoupled — structural path."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25)
    drive = FluxDrive(target=q)
    (channel,) = drive.local_channels(q)

    backend = QuTiPBackend()
    expected = backend.number(q.levels)
    assert np.allclose(np.array(channel.operator.full()), np.array(expected.full()))
