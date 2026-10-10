"""Reference drives with multi-observable coupling and drive-owned loss."""

from __future__ import annotations

from typing import Any

from quchip.control.drive import ChargeDrive, DeviceDrive
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.base import BaseDevice
from quchip.devices.protocols import ChargeCoupled, PhaseCoupled


class ChargePhaseDrive(DeviceDrive):
    """Map delivered I and Q quadratures to charge and phase observables.

    Parameters
    ----------
    target : device or None, default None
        Optional target device. ``None`` allows attachment through the normal
        drive connection API.
    label : str or None, default None
        Drive label.
    """

    _type_prefix = "charge_phase"

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
            - signal.q * target.phase_coupling_operator()
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
