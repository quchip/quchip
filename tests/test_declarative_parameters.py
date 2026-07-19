from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from quchip import DeviceModel, Scalar, parameter


class ToyDevice(DeviceModel):
    freq: Scalar = parameter(positive=True)
    detuning: Scalar = parameter(default=0.0)
    approximation = None

    def local_hamiltonian(self, op):
        return (self.freq + self.detuning) * op.n


def test_parameter_fields_generate_constructor_and_attributes():
    """Declared parameter fields become constructor kwargs and attributes; unset fields take their declared defaults."""
    dev = ToyDevice(freq=5.0, levels=3, label="q")
    assert dev.freq == 5.0
    assert dev.detuning == 0.0
    assert dev.levels == 3
    assert dev.label == "q"


def test_positive_parameter_validation_uses_concrete_only_path():
    """A parameter declared positive=True rejects a concrete negative value at construction, raising ValueError."""
    with pytest.raises(ValueError, match="freq must be positive"):
        ToyDevice(freq=-1.0)


def test_traced_positive_parameter_does_not_concretize():
    """Positive-parameter validation skips concretization for traced values, staying differentiable via jax.grad."""
    def build(x):
        dev = ToyDevice(freq=x)
        return dev.freq * 2.0

    assert jax.grad(build)(jnp.asarray(5.0)) == 2.0


def test_parameter_pytree_leaves_include_declared_fields():
    """Flattening a DeviceModel pytree yields each declared parameter as its own leaf."""
    dev = ToyDevice(freq=jnp.asarray(5.0), detuning=jnp.asarray(0.1))
    leaves, _ = jtu.tree_flatten(dev)
    assert any(leaf is dev.freq for leaf in leaves)
    assert any(leaf is dev.detuning for leaf in leaves)


def test_parameter_pytree_roundtrip_preserves_base_state():
    """DeviceModel pytree round-trip preserves declared parameter values and static base state (label, levels)."""
    dev = ToyDevice(freq=jnp.asarray(5.0), detuning=jnp.asarray(0.1), levels=4, label="q")
    leaves, treedef = jtu.tree_flatten(dev)
    restored = jtu.tree_unflatten(treedef, leaves)
    assert restored.label == "q"
    assert restored.levels == 4
    assert float(restored.freq) == 5.0
    assert float(restored.detuning) == pytest.approx(0.1)


def test_parameter_pytree_roundtrip_preserves_noise_params():
    """DeviceModel pytree round-trip preserves T1 and T2 noise parameters."""
    dev = ToyDevice(freq=5.0, levels=3, T1=100.0, T2=80.0)
    leaves, treedef = jtu.tree_flatten(dev)
    restored = jtu.tree_unflatten(treedef, leaves)
    assert restored.T1 == 100.0
    assert restored.T2 == 80.0


def test_parameter_pytree_traceable_through_jit():
    """A DeviceModel instance passes through jax.jit as an argument, with its declared parameters traced correctly."""
    @jax.jit
    def freq_doubled(dev):
        return dev.freq * 2.0

    dev = ToyDevice(freq=jnp.asarray(5.0), levels=3, label="q")
    assert float(freq_doubled(dev)) == 10.0
