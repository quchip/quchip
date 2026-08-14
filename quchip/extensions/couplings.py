"""Reference coupling models with time dependence and coupling-owned loss."""

from __future__ import annotations

from typing import Any

from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.dynamics import CosineCoefficient, TimeDependentTerm
from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import CouplingModel
from quchip.declarative.parameters import UNBOUND, Scalar, parameter


class ModulatedCapacitive(CouplingModel):
    """Capacitive interaction with a prescribed sinusoidal strength variation."""

    _type_prefix = "modulated_capacitive"

    static_strength: Scalar = parameter(default=UNBOUND, unit="GHz", symbol=r"g_0")
    modulation_amplitude: Scalar = parameter(default=UNBOUND, unit="GHz", symbol=r"\delta g")
    modulation_frequency: Scalar = parameter(
        default=UNBOUND,
        positive=True,
        unit="GHz",
        symbol=r"\nu_m",
    )
    modulation_phase: Scalar = parameter(default=0.0, unit="rad", symbol=r"\phi_m")

    def interaction(self, a: Any, b: Any, p: Any) -> PhysicsExpr:
        return p.static_strength * a.charge * b.charge

    def time_terms(self, a: Any, b: Any, p: Any) -> tuple[TimeDependentTerm, ...]:
        return (
            TimeDependentTerm(
                operator=a.charge * b.charge,
                coefficient=CosineCoefficient(
                    amplitude=p.modulation_amplitude,
                    frequency=p.modulation_frequency,
                    phase=p.modulation_phase,
                ),
            ),
        )


class CollectiveDecayCoupling(CouplingModel):
    """Exchange coupling with an equal-phase collective decay channel."""

    _type_prefix = "collective_decay"

    exchange_strength: Scalar = parameter(default=UNBOUND, unit="GHz", symbol="g")
    decay_rate: Scalar = parameter(
        default=UNBOUND,
        nonnegative=True,
        unit="1/ns",
        symbol=r"\gamma_c",
        noise=True,
        kw_only=True,
    )

    def interaction(self, a: Any, b: Any, p: Any) -> Any:
        return p.exchange_strength * (a.a * b.adag + a.adag * b.a)

    def dissipation(self, a: Any, b: Any, p: Any) -> tuple[CollapseChannel, ...]:
        return (
            CollapseChannel(
                a.a * b.I + a.I * b.a,
                p.decay_rate,
                "collective_decay",
            ),
        )
