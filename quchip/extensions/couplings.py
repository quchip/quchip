"""Reference coupling models with time dependence and coupling-owned loss."""

from __future__ import annotations

from typing import Any

from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.dynamics import CosineCoefficient, TimeDependentTerm
from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import CouplingModel
from quchip.declarative.parameters import UNBOUND, Scalar, parameter


class ModulatedCapacitive(CouplingModel):
    r"""Capacitive interaction with a prescribed sinusoidal strength variation.

    The static term is :math:`g_0 n_a^{(q)}n_b^{(q)}` and the modulation is
    :math:`\delta g\cos(2\pi\nu_m t+\phi_m)n_a^{(q)}n_b^{(q)}`.

    Parameters
    ----------
    device_a, device_b : device or str
        Coupled endpoints or their chip labels.
    static_strength : float
        Static capacitive strength :math:`g_0` in GHz. Can be signed.
    modulation_amplitude : float
        Modulation amplitude :math:`\delta g` in GHz. Can be signed.
    modulation_frequency : float
        Modulation frequency :math:`\nu_m` in GHz. Must be positive.
    modulation_phase : float, default 0.0
        Modulation phase in radians.
    label : str or None, default None
        Coupling label.
    """

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
        """Return the static capacitive interaction.

        Parameters
        ----------
        a, b : EndpointOps
            Ordered endpoint operator namespaces.
        p : ParameterNamespace
            Bound coupling parameters.
        """
        return p.static_strength * a.charge * b.charge

    def time_terms(self, a: Any, b: Any, p: Any) -> tuple[TimeDependentTerm, ...]:
        """Return the sinusoidally modulated capacitive interaction term.

        Parameters
        ----------
        a, b : EndpointOps
            Ordered endpoint operator namespaces.
        p : ParameterNamespace
            Bound coupling parameters.
        """
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
    r"""Exchange coupling with an equal-phase collective decay channel.

    Parameters
    ----------
    device_a, device_b : device or str
        Coupled endpoints or their chip labels.
    exchange_strength : float
        Exchange rate :math:`g` in GHz.
    decay_rate : float
        Non-negative collective Lindblad rate :math:`\gamma_c` in 1/ns.
    label : str or None, default None
        Coupling label.

    Notes
    -----
    The collapse operator is proportional to ``a + b``, a shared bath channel,
    not two independent relaxation channels.
    """

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
        """Return the exchange interaction ``g(a b† + a† b)``.

        Parameters
        ----------
        a, b : EndpointOps
            Ordered endpoint operator namespaces.
        p : ParameterNamespace
            Bound coupling parameters.
        """
        return p.exchange_strength * (a.a * b.adag + a.adag * b.a)

    def dissipation(self, a: Any, b: Any, p: Any) -> tuple[CollapseChannel, ...]:
        """Return the shared-bath collapse channel proportional to ``a + b``.

        Parameters
        ----------
        a, b : EndpointOps
            Ordered endpoint operator namespaces.
        p : ParameterNamespace
            Bound coupling parameters.
        """
        return (
            CollapseChannel(
                a.a * b.I + a.I * b.a,
                p.decay_rate,
                "collective_decay",
            ),
        )
