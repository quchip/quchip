"""Reference device models, including device-owned time dependence and loss."""

from __future__ import annotations

from typing import Any

from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.dynamics import CosineCoefficient, TimeDependentTerm
from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.fock import FockDevice
from quchip.devices.kerr_cavity import KerrCavity


class FrequencyModulatedMode(FockDevice):
    """Harmonic mode with a prescribed sinusoidal frequency variation."""

    _type_prefix = "frequency_modulated_mode"
    _default_levels = 10
    approximation = (
        "Single harmonic mode in a fixed Fock basis with an externally prescribed "
        "sinusoidal frequency coefficient."
    )

    frequency: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz", symbol=r"\omega_0")
    modulation_amplitude: Scalar = parameter(default=UNBOUND, unit="GHz", symbol=r"\delta\omega")
    modulation_frequency: Scalar = parameter(
        default=UNBOUND,
        positive=True,
        unit="GHz",
        symbol=r"\nu_m",
    )
    modulation_phase: Scalar = parameter(default=0.0, unit="rad", symbol=r"\phi_m")

    @property
    def freq(self) -> Any:
        """Bare reference frequency in GHz."""
        return self.frequency

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        return p.frequency * op.n

    def time_terms(self, op: LocalOps, p: Any) -> tuple[TimeDependentTerm, ...]:
        return (
            TimeDependentTerm(
                operator=op.n,
                coefficient=CosineCoefficient(
                    amplitude=p.modulation_amplitude,
                    frequency=p.modulation_frequency,
                    phase=p.modulation_phase,
                ),
            ),
        )


class LossyKerrCavity(KerrCavity):
    """Kerr cavity with an intrinsic two-photon-loss channel."""

    _type_prefix = "lossy_kerr_cavity"

    two_photon_loss_rate: Scalar = parameter(
        default=UNBOUND,
        nonnegative=True,
        unit="1/ns",
        symbol=r"\kappa_2",
        noise=True,
        kw_only=True,
    )

    def dissipation(self, op: Any, p: Any) -> tuple[CollapseChannel, ...]:
        return super().dissipation(op, p) + (
            CollapseChannel(op.a @ op.a, p.two_photon_loss_rate, "two_photon_loss"),
        )
