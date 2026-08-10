"""Tests for CouplingModel compile-path guards: endpoint order and dynamic-source validation."""

from __future__ import annotations

import pytest

from quchip import Chip, Square
from quchip.declarative import CouplingModel, DeviceModel, DynamicScalar, Scalar, parameter


class _Oscillator(DeviceModel):
    freq: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op, p):
        return p.freq * op.n


class _ForwardCoupling(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        return p.g * (a.a * b.adag + a.adag * b.a)


class _ReversedCoupling(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        return p.g * (b.a * a.adag + b.adag * a.a)


class _DynamicCoupling(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        return p.g * (a.a * b.adag + a.adag * b.a)

    def time_dependent(self, a, b, p):
        _ = p
        return DynamicScalar(Square(duration=10.0, amplitude=0.02)) * (a.x * b.x)


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


def test_intrinsic_coupling_is_projected_then_rwa_resolved_by_engine():
    """An intrinsic coupling authors one form and the engine removes counter-rotating bands."""
    q0 = _Oscillator(freq=5.0, levels=3, label="q0")
    q1 = _Oscillator(freq=5.2, levels=3, label="q1")
    coupling = _DynamicCoupling(q0, q1, g=0.01, label="dynamic")
    result = Chip([q0, q1], [coupling], frame="lab", rwa=True).engine_result()

    assert len([term for term in result.dynamic_terms if term.operator.tag == "coupling_dynamic"]) == 2
    intrinsic_drops = [term for term in result.dropped_terms if term.operator.startswith("intrinsic")]
    assert {term.band_weights for term in intrinsic_drops} == {(-1, -1), (1, 1)}
