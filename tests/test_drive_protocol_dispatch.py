"""Drive classes dispatch to physical operators via Protocols when available."""

from __future__ import annotations

from quchip.approximations import RWA

import numpy as np
import pytest
import quchip

from quchip import Chip, Gaussian, QuantumSequence
from quchip.control.drive import ChargeDrive, FluxDrive, PhaseDrive
from quchip.control.signal import AnalyticSignal
from quchip.declarative import DeviceModel, LocalOps, Scalar, parameter
from quchip.devices import ChargeCoupled, DuffingTransmon, FluxCoupled, PhaseCoupled
from quchip.engine.ir import Constant


_UNIT_SIGNAL = AnalyticSignal(Constant(1.0 + 0.0j))


class _FakePhysicalDevice:
    """Exposes all three coupling operators — tests the Protocol-conformant path."""

    levels = 3
    label = "fake"

    def __init__(self):
        self._charge = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.3], [0.0, 0.3, 0.0]], dtype=complex)
        self._phase = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.2], [0.0, 0.2, 0.0]], dtype=complex)
        self._flux = np.array([[0.0, 0.7, 0.0], [0.7, 0.0, 0.4], [0.0, 0.4, 0.0]], dtype=complex)
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
    np.testing.assert_allclose(
        ChargeDrive(target=q).hamiltonian(q, _UNIT_SIGNAL).matrix(t=0.0),
        q.charge_coupling_operator().matrix(),
    )
    np.testing.assert_allclose(
        PhaseDrive(target=q).hamiltonian(q, _UNIT_SIGNAL).matrix(t=0.0),
        q.phase_coupling_operator().matrix(),
    )
    np.testing.assert_allclose(
        FluxDrive(target=q).hamiltonian(q, _UNIT_SIGNAL).matrix(t=0.0),
        q.flux_coupling_operator().matrix(),
    )


class _CapabilityFreeDevice(DeviceModel):
    freq: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op: LocalOps, p: object):
        return self.freq * op.n


def test_drive_rejects_device_without_requested_capability():
    """A drive never invents a physical operator absent from its target device."""
    device = _CapabilityFreeDevice(freq=5.0, levels=3, label="plain")

    with np.testing.assert_raises_regex(TypeError, "charge_coupling_operator"):
        ChargeDrive(target=device).hamiltonian(device, _UNIT_SIGNAL)


def test_charge_drive_prefers_physical_operator_for_protocol_device():
    """A ChargeCoupled device supplies the charge-drive operator."""
    fake = _FakePhysicalDevice()
    drive = ChargeDrive()
    assert np.allclose(drive.hamiltonian(fake, _UNIT_SIGNAL).matrix(t=0.0), fake._charge)


def test_phase_drive_prefers_physical_operator():
    """A PhaseCoupled device supplies the phase-drive operator."""
    fake = _FakePhysicalDevice()
    drive = PhaseDrive()
    assert np.allclose(drive.hamiltonian(fake, _UNIT_SIGNAL).matrix(t=0.0), fake._phase)


def test_flux_drive_prefers_physical_operator():
    """A FluxCoupled device supplies the flux-drive operator."""
    fake = _FakePhysicalDevice()
    drive = FluxDrive()
    assert np.allclose(drive.hamiltonian(fake, _UNIT_SIGNAL).matrix(t=0.0), fake._flux)


@pytest.mark.optional_backend
def test_dynamiqs_keeps_symbolic_fock_drive_bands_sparse():
    """Device-authored Fock operators lower to sparse solver bands."""
    pytest.importorskip("dynamiqs")
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    q.reference_freq = q.freq
    chip = Chip([q], frame={q: q.freq}, approximation=RWA(), backend="dynamiqs")
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
