from __future__ import annotations

import jax
import jax.tree_util as jtu

from quchip.declarative import DeviceModel, Scalar, parameter


class _Oscillator(DeviceModel):
    freq: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op):
        return self.freq * op.n


def test_device_model_pytree_round_trip_preserves_declared_noise_and_reference_freq():
    """DeviceModel pytree round-trip preserves declared params, noise params, and a reference-frequency override."""
    device = _Oscillator(freq=5.0, levels=3, T1=30_000.0)
    device.reference_freq = 4.9
    leaves, treedef = jtu.tree_flatten(device)
    restored = jtu.tree_unflatten(treedef, leaves)
    assert restored.freq == device.freq
    assert restored.T1 == device.T1
    assert restored.reference_freq == device.reference_freq


def test_gradient_flows_through_declared_parameter_of_device_built_inside_grad():
    """A gradient flows through a declared parameter of a device constructed inside a jax.grad-transformed function."""

    def energy(freq):
        device = _Oscillator(freq=freq, levels=3)
        return device.freq**2

    grad = jax.grad(energy)(5.0)
    assert grad == 10.0
