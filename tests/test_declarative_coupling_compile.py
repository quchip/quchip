"""Tests for CouplingModel compile-path guards and intrinsic terms."""

from __future__ import annotations

from quchip.approximations import RWA


from quchip import Chip
from quchip.declarative import (
    CosineCoefficient,
    CouplingModel,
    DeviceModel,
    TimeDependentTerm,
    Scalar,
    parameter,
)

import pytest


pytestmark = pytest.mark.unit


class _Oscillator(DeviceModel):
    freq: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op, p):
        return p.freq * op.n


class _ForwardCoupling(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        return p.g * (a.a * b.adag + a.adag * b.a)


class _DynamicCoupling(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        return p.g * (a.a * b.adag + a.adag * b.a)

    def time_terms(self, a, b, p):
        _ = p
        return (
            TimeDependentTerm(
                operator=a.x * b.x,
                coefficient=CosineCoefficient(0.02, 0.1),
            ),
        )


def test_forward_endpoint_order_compiles():
    """An interaction authored in (a, b) endpoint order compiles to a backend operator."""
    q0 = _Oscillator(freq=5.0, levels=3)
    q1 = _Oscillator(freq=5.2, levels=3)
    coupling = _ForwardCoupling(q0, q1, g=0.01)
    assert coupling.interaction_hamiltonian() is not None


@pytest.mark.parametrize("frame", ["lab", "rotating"])
def test_time_dependent_coupling_is_projected_then_rwa_resolved_by_engine(frame):
    """A time-dependent coupling authors one form and the engine removes counter-rotating bands."""
    q0 = _Oscillator(freq=5.0, levels=3, label="q0")
    q1 = _Oscillator(freq=5.2, levels=3, label="q1")
    coupling = _DynamicCoupling(q0, q1, g=0.01, label="dynamic")
    result = Chip([q0, q1], [coupling], frame=frame, approximation=RWA()).resolve()

    assert len([term for term in result.dynamic_terms if term.operator.tag == "coupling_dynamic"]) == 2
    dynamic_drops = [
        term for term in result.dropped_terms
        if term.operator.startswith("time-dependent coupling")
    ]
    assert {term.band_weights for term in dynamic_drops} == {(-1, -1), (1, 1)}
    for term in dynamic_drops:
        assert term.amplitude is None
        assert term.frequency is None
        assert term.frame_frequency == pytest.approx(0 if frame == "lab" else 10.2)
