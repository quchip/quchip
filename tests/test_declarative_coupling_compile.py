"""Tests for CouplingModel compile-path guards: endpoint order and dynamic-source validation."""

from __future__ import annotations

import pytest

from quchip.declarative import CouplingModel, DeviceModel, DynamicScalar, Scalar, parameter


class _Oscillator(DeviceModel):
    freq: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op):
        return self.freq * op.n


class _ForwardCoupling(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b):
        return self.g * (a.a * b.adag + a.adag * b.a)


class _ReversedCoupling(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b):
        return self.g * (b.a * a.adag + b.adag * a.a)


class _DynamicCouplingRwaNone(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b):
        return self.g * (a.a * b.adag + a.adag * b.a)

    def time_dependent(self, a, b):
        return DynamicScalar(object()) * (a.a * b.adag + a.adag * b.a)

    def rwa_time_dependent(self, a, b):
        return None


class _RwaOnPolicy:
    """Minimal stand-in for the chip's RWA-resolution surface."""

    def resolve_rwa(self, coupling):
        return True


def test_forward_endpoint_order_compiles():
    """An interaction authored in (a, b) endpoint order compiles to a backend operator."""
    q0 = _Oscillator(freq=5.0, levels=3)
    q1 = _Oscillator(freq=5.2, levels=3)
    coupling = _ForwardCoupling(q0, q1, g=0.01)
    assert coupling.interaction_hamiltonian() is not None


def test_reversed_endpoint_order_raises():
    """An interaction authored in reversed (b, a) order raises instead of silently mis-embedding."""
    q0 = _Oscillator(freq=5.0, levels=3)
    q1 = _Oscillator(freq=5.2, levels=3)
    coupling = _ReversedCoupling(q0, q1, g=0.01)
    with pytest.raises(TypeError):
        coupling.interaction_hamiltonian()


def test_rwa_override_returning_none_raises_valueerror():
    """A valid full dynamic form with an RWA override returning None raises the method-named ValueError."""
    q0 = _Oscillator(freq=5.0, levels=3)
    q1 = _Oscillator(freq=5.2, levels=3)
    coupling = _DynamicCouplingRwaNone(q0, q1, g=0.01)
    with pytest.raises(ValueError, match="rwa_time_dependent"):
        coupling.dynamic_interaction_terms(_RwaOnPolicy())
