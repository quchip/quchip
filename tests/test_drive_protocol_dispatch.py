"""Drive classes dispatch to physical operators via Protocols when available."""

from __future__ import annotations

import numpy as np
import quchip

from quchip import Chip, Gaussian, QuantumSequence
from quchip.control.drive import ChargeDrive, FluxDrive, PhaseDrive
from quchip.declarative import DeviceModel, LocalOps, Scalar, parameter
from quchip.devices import ChargeCoupled, DuffingTransmon, FluxCoupled, PhaseCoupled


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


def test_duffing_declares_fock_drive_capabilities():
    """A Fock device owns the conventional operators its drives consume."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25)
    assert isinstance(q, quchip.FockDevice)
    assert isinstance(q, ChargeCoupled)
    assert isinstance(q, PhaseCoupled)
    assert isinstance(q, FluxCoupled)
    assert ChargeDrive(target=q).local_channels(q)[0].operator == q.charge_coupling_operator()
    assert PhaseDrive(target=q).local_channels(q)[0].operator == q.phase_coupling_operator()
    assert FluxDrive(target=q).local_channels(q)[0].operator == q.flux_coupling_operator()


class _CapabilityFreeDevice(DeviceModel):
    freq: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op: LocalOps, p: object):
        return self.freq * op.n


def test_drive_rejects_device_without_requested_capability():
    """A drive never invents a physical operator absent from its target device."""
    device = _CapabilityFreeDevice(freq=5.0, levels=3, label="plain")

    with np.testing.assert_raises_regex(TypeError, "charge_coupling_operator"):
        ChargeDrive(target=device).local_channels(device)


def test_charge_drive_prefers_physical_operator_for_protocol_device():
    """A ChargeCoupled device supplies the charge-drive operator."""
    fake = _FakePhysicalDevice()
    drive = ChargeDrive()
    (channel,) = drive.local_channels(fake)
    assert np.allclose(channel.operator, fake._charge)


def test_phase_drive_prefers_physical_operator():
    """A PhaseCoupled device supplies the phase-drive operator."""
    fake = _FakePhysicalDevice()
    drive = PhaseDrive()
    (channel,) = drive.local_channels(fake)
    assert np.allclose(channel.operator, fake._phase)


def test_flux_drive_prefers_physical_operator():
    """A FluxCoupled device supplies the flux-drive operator."""
    fake = _FakePhysicalDevice()
    drive = FluxDrive()
    (channel,) = drive.local_channels(fake)
    assert np.allclose(channel.operator, fake._flux)


def test_dynamiqs_keeps_symbolic_fock_drive_bands_sparse():
    """Device-authored Fock operators lower to sparse solver bands."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    q.reference_freq = q.freq
    chip = Chip([q], frame={q: q.freq}, rwa=True, backend="dynamiqs")
    drive = ChargeDrive(q, label="xy")
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Gaussian(duration=10.0, amplitude=0.02),
        freq=q.freq,
    )

    result = sequence.resolve()
    drive_terms = [term for term in result.dynamic_terms if term.origin == "drive"]
    assert drive_terms
    assert {term.operator.layout for term in drive_terms} == {"dia"}
