from __future__ import annotations

import jax
import pytest

from quchip.declarative import CouplingModel, DeviceModel, EnvelopeShape, Scalar, parameter


class _Oscillator(DeviceModel):
    freq: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op):
        return self.freq * op.n


class _ExchangeCoupling(CouplingModel):
    g: Scalar = parameter(positive=True, unit="GHz")

    def interaction(self, a, b):
        return self.g * (a.a * b.adag + a.adag * b.a)


class _LinearEnvelope(EnvelopeShape):
    duration: Scalar = parameter(positive=True, unit="ns")

    def value(self, t):
        return t / self.duration


def test_coupling_model_rejects_negative_write_of_positive_field():
    """A CouplingModel subclass rejects a negative post-construction write to a positive=True field."""
    q0 = _Oscillator(freq=5.0, levels=3)
    q1 = _Oscillator(freq=5.2, levels=3)
    coupling = _ExchangeCoupling(q0, q1, g=0.01)
    with pytest.raises(ValueError):
        coupling.g = -0.01


def test_coupling_model_accepts_traced_write_of_positive_field():
    """A traced value written to a positive=True CouplingModel field passes unchecked."""
    q0 = _Oscillator(freq=5.0, levels=3)
    q1 = _Oscillator(freq=5.2, levels=3)
    coupling = _ExchangeCoupling(q0, q1, g=0.01)

    def write_and_read(value):
        coupling.g = value
        return coupling.g

    jax.grad(write_and_read)(-0.01)


def test_envelope_shape_rejects_negative_write_of_positive_field():
    """An EnvelopeShape subclass rejects a negative post-construction write to a positive=True field."""
    env = _LinearEnvelope(duration=20.0)
    with pytest.raises(ValueError):
        env.duration = -5.0


def test_envelope_shape_accepts_traced_write_of_positive_field():
    """A traced value written to a positive=True EnvelopeShape field passes unchecked."""
    env = _LinearEnvelope(duration=20.0)

    def write_and_read(value):
        env.duration = value
        return env.duration

    jax.grad(write_and_read)(-5.0)


def test_envelope_shape_validates_after_pytree_round_trip():
    """An envelope rebuilt via tree_unflatten still rejects a negative write to a positive=True field."""
    env = _LinearEnvelope(duration=20.0)
    leaves, treedef = jax.tree_util.tree_flatten(env)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    with pytest.raises(ValueError):
        restored.duration = -1.0
