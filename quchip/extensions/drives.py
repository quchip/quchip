"""Reference drives with multi-observable coupling and drive-owned loss."""

from __future__ import annotations

from typing import Any, ClassVar

from quchip.control.drive import ChargeDrive, DeviceDrive
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.base import BaseDevice
from quchip.devices.protocols import ChargeCoupled, PhaseCoupled


class ChargePhaseDrive(DeviceDrive):
    r"""Map delivered I and Q quadratures to charge and phase observables.

    The drive contributes :math:`H_d(t) = I(t)\, \hat Q + Q(t)\, \hat P`, with
    :math:`\hat Q` the device's ``charge_coupling_operator()`` and
    :math:`\hat P` its ``phase_coupling_operator()``. With the analytic signal
    :math:`s = I + iQ` and the Fock operators :math:`\hat Q = i(\hat a -
    \hat a^\dagger)` and :math:`\hat P = \hat a + \hat a^\dagger`, this is
    :math:`H_d = i(s^* \hat a - s\, \hat a^\dagger)`. A positive carrier then
    drives the resonant sideband with the phase convention of
    :class:`~quchip.control.drive.ChargeDrive` and twice its rate. The drive
    has no counter-rotating term there, so ``RWA()`` drops none of it. The
    other sideband cancels only on a device whose upward matrix elements obey
    :math:`\langle u|\hat Q|l\rangle = -i\langle u|\hat P|l\rangle`, such as
    an unscaled Fock device. On other devices, for example a charge-basis
    transmon or a fluxonium, both sidebands couple with the device's own
    weights. A pulse scheduled without ``freq`` takes the carrier
    ``chip.freq(target)``.

    Parameters
    ----------
    target : device or None, default None
        Optional target device. ``None`` allows attachment through the normal
        drive connection API.
    label : str or None, default None
        Drive label.
    """

    _type_prefix = "charge_phase"
    carrier_harmonic: ClassVar[int | None] = 1

    def hamiltonian(self, target: Any, signal: Any) -> Any:
        """Return the I/Q Hamiltonian for a charge-and-phase coupled target.

        Parameters
        ----------
        target : device
            Device implementing charge and phase coupling protocols.
        signal : signal
            Delivered I/Q signal.
        """
        if not isinstance(target, ChargeCoupled) or not isinstance(target, PhaseCoupled):
            raise TypeError(
                f"ChargePhaseDrive requires {type(target).__name__} to define "
                "charge_coupling_operator() and phase_coupling_operator()."
            )
        return (
            signal.i * target.charge_coupling_operator()
            + signal.q * target.phase_coupling_operator()
        )


class LossyChargeDrive(ChargeDrive):
    """Charge-control line with an effective target-relaxation rate.

    Parameters
    ----------
    line_loss_rate : float
        Non-negative effective relaxation rate in 1/ns.
    target : device or None, default None
        Optional target passed to :class:`~quchip.control.drive.ChargeDrive`.
    label : str or None, default None
        Drive label.
    """

    _type_prefix = "lossy_charge"
    line_loss_rate: Scalar = parameter(
        default=UNBOUND,
        nonnegative=True,
        unit="1/ns",
        noise=True,
        kw_only=True,
    )

    def dissipation(self, device: BaseDevice, op: Any, p: Any) -> tuple[CollapseChannel, ...]:
        """Return the effective line-loss collapse channel.

        Parameters
        ----------
        device : BaseDevice
            Target device.
        op : operator namespace
            Target lowering operator namespace.
        p : ParameterNamespace
            Bound drive parameters.
        """
        return (
            CollapseChannel(
                op.a,
                p.line_loss_rate,
                "line_relaxation",
            ),
        )
