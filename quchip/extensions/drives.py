"""Reference drives with multi-observable coupling and drive-owned loss."""

from __future__ import annotations

from typing import Any

from quchip.control.drive import ChargeDrive, DeviceDrive
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.base import BaseDevice
from quchip.devices.protocols import ChargeCoupled, PhaseCoupled


class ChargePhaseDrive(DeviceDrive):
    """Map delivered I and Q to charge and phase observables."""

    _type_prefix = "charge_phase"

    def hamiltonian(self, target: Any, signal: Any) -> Any:
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
    """Charge-control line with an effective target-relaxation rate."""

    _type_prefix = "lossy_charge"
    line_loss_rate: Scalar = parameter(
        default=UNBOUND,
        nonnegative=True,
        unit="1/ns",
        noise=True,
        kw_only=True,
    )

    def dissipation(self, device: BaseDevice, op: Any, p: Any) -> tuple[CollapseChannel, ...]:
        return (
            CollapseChannel(
                op.a,
                p.line_loss_rate,
                "line_relaxation",
            ),
        )
