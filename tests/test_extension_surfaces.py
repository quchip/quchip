"""Tests for the declarative device, noise-channel, and batch-handle extension surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Chip, ChargeDrive, DuffingTransmon, Gaussian, QuantumSequence
from quchip.declarative.models import DeviceModel
from quchip.declarative.parameters import Scalar, parameter
from quchip.devices.base import BaseDevice, NoiseChannel


class SpinHalf(DeviceModel):
    """Minimal spin-like device authored purely on the sigma surface."""

    _type_prefix = "spin"
    _default_levels = 2

    freq: Scalar = parameter(positive=True)

    def local_hamiltonian(self, op):
        # H = -(freq/2)·sigma_z, so |1> sits `freq` above |0>.
        return (-0.5 * self.freq) * op.sigma_z


def test_sigma_ops_author_a_spin_device():
    """A sigma_z-authored device gives a diagonal Hamiltonian split by ±freq/2."""
    spin = SpinHalf(freq=4.0, levels=2)
    chip = Chip([spin])
    h = np.asarray(chip.backend.to_array(chip.hamiltonian()))
    np.testing.assert_allclose(h, np.diag([-2.0, 2.0]), atol=1e-12)


class DampedTransmon(DuffingTransmon):
    """Transmon with an extra device-declared two-photon-loss channel."""

    _type_prefix = "damped"

    two_photon_rate: Scalar = parameter(default=None, positive=True)

    _noise_channels = BaseDevice._noise_channels + (
        NoiseChannel("two_photon_loss", ("two_photon_rate",), lambda dev: (
            [np.sqrt(dev.two_photon_rate) * (dev.lowering_operator() @ dev.lowering_operator())]
            if dev.two_photon_rate is not None
            else []
        )),
    )


def test_noise_channel_declaration_composes_without_override():
    """A device-declared noise channel composes with the built-in T1 channel unmodified."""
    quiet = DampedTransmon(freq=5.0, anharmonicity=-0.3, levels=3)
    assert quiet.collapse_operators() == []

    noisy = DampedTransmon(freq=5.0, anharmonicity=-0.3, levels=3, T1=10_000.0, two_photon_rate=1e-4)
    ops = noisy.collapse_operators()
    # Built-in T1 relaxation channel plus the declared two-photon channel.
    assert len(ops) == 2
    from quchip.backend import get_default_backend

    a2 = np.asarray(get_default_backend().to_array(ops[-1]))
    lowering = np.diag(np.sqrt(np.arange(1, 3)).astype(complex), k=1)
    expected = np.sqrt(1e-4) * (lowering @ lowering)
    np.testing.assert_allclose(a2, expected, atol=1e-12)


def _sequence_with_pulse():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3)
    chip = Chip([q], frame="rotating", rwa=True)
    drive = ChargeDrive(target=q)
    chip.wire(drive)
    seq = QuantumSequence(chip)
    handle = seq.schedule(drive, envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3.0), freq=5.0)
    return q, drive, seq, handle


def test_stale_handle_is_rejected_not_silently_misapplied():
    """A batch handle raises once its owning sequence has been mutated out from under it."""
    q, drive, seq, handle = _sequence_with_pulse()
    seq.delay(q, 5.0)
    seq._entries.pop(0)  # shifts entry indices out from under the handle
    with pytest.raises(RuntimeError, match="modified after the handle"):
        handle.vary("amplitude", [0.01, 0.02])


def test_duplicate_axis_names_rejected_upfront():
    """Building a batch with duplicate axis names across handles raises before solving."""
    q, drive, seq, handle = _sequence_with_pulse()
    ax1 = handle.vary("amplitude", [0.01, 0.02], name="amps")
    ax2 = handle.vary("freq", [4.9, 5.1], name="amps")
    with pytest.raises(ValueError, match="Duplicate batch axis name"):
        seq.build_batch(ax1, ax2, tlist=np.linspace(0.0, 20.0, 11))
