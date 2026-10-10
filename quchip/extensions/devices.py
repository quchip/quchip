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
    r"""Harmonic mode with a prescribed sinusoidal frequency variation.

    The authored Hamiltonian is :math:`H_0=\omega_0 n` and the time term is
    :math:`\delta\omega\cos(2\pi\nu_m t+\phi_m)n`. Frequencies are ordinary GHz
    and ``t`` is in ns.

    Parameters
    ----------
    frequency : float
        Bare mode frequency :math:`\omega_0` in GHz. Must be positive.
    modulation_amplitude : float
        Frequency excursion :math:`\delta\omega` in GHz. Can be signed.
    modulation_frequency : float
        Modulation frequency :math:`\nu_m` in GHz. Must be positive.
    modulation_phase : float, default 0.0
        Phase :math:`\phi_m` in radians.
    levels : int, default 10
        Fock truncation dimension.
    label : str or None, default None
        Device label. ``None`` selects an automatic label.
    T1 : float or None, default None
        Energy-relaxation time in ns. ``None`` disables T1 relaxation.
    T2 : float or None, default None
        Total 0-1 coherence time in ns. If both are set, ``T2 <= 2*T1``.
    thermal_occupation : float or None, default None
        Dimensionless mean bath occupation. ``None`` disables absorption.

    References
    ----------
    See Didier et al., *Phys. Rev. A* 97, 022330 (2018),
    https://doi.org/10.1103/PhysRevA.97.022330, for parametrically modulated
    transmon models.
    """

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
        """Return the static harmonic Hamiltonian.

        Parameters
        ----------
        op : LocalOps
            Local operator namespace.
        p : ParameterNamespace
            Bound symbolic parameters.
        """
        return p.frequency * op.n

    def time_terms(self, op: LocalOps, p: Any) -> tuple[TimeDependentTerm, ...]:
        """Return the sinusoidal frequency-modulation term.

        Parameters
        ----------
        op : LocalOps
            Local operator namespace.
        p : ParameterNamespace
            Bound symbolic parameters.
        """
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
    r"""Kerr cavity with an intrinsic two-photon-loss channel.

    Parameters
    ----------
    two_photon_loss_rate : float
        Non-negative two-photon Lindblad rate :math:`\kappa_2` in 1/ns.
    freq, kerr, levels, label, T1, T2, thermal_occupation
        Inherited :class:`~quchip.devices.kerr_cavity.KerrCavity` parameters.
    """

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
        """Return inherited channels plus two-photon loss at ``kappa_2``.

        Parameters
        ----------
        op : LocalOps
            Local operator namespace.
        p : ParameterNamespace
            Bound symbolic parameters.
        """
        return super().dissipation(op, p) + (
            CollapseChannel(op.a @ op.a, p.two_photon_loss_rate, "two_photon_loss"),
        )
