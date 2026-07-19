"""Dynamiqs smoke coverage for Kerr-cavity operators."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.extended, pytest.mark.optional_backend]

pytest.importorskip("dynamiqs")
pytest.importorskip("jax")

from quchip.backend import set_default_backend  # noqa: E402
from quchip.control.drives_two_photon import TwoPhotonDrive  # noqa: E402
from quchip.devices.kerr_cavity import KerrCavity  # noqa: E402


def test_kerr_cavity_and_two_photon_drive_build_with_dynamiqs_backend() -> None:
    """Operator products must use backend matrix multiplication, not elementwise multiplication."""
    set_default_backend("dynamiqs")

    cav = KerrCavity(freq=5.0, kerr=0.25, levels=5, label="cav")
    drive = TwoPhotonDrive(target=cav)

    hamiltonian = cav.hamiltonian()
    channel = drive.local_channels(cav)[0]

    assert hamiltonian.to_jax().shape == (5, 5)
    assert channel.operator.to_jax().shape == (5, 5)
