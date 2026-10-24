"""Reference drives with multi-observable coupling and drive-owned loss."""

from __future__ import annotations

from typing import Any, ClassVar

from quchip.control.drive import ChargeDrive, DeviceDrive
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.expr import as_operator_expr
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.base import BaseDevice
from quchip.devices.protocols import ChargeCoupled, PhaseCoupled


class ChargePhaseDrive(DeviceDrive):
    r"""Map delivered I and Q quadratures to charge and phase observables.

    The drive contributes :math:`H_d(t) = I(t)\, \hat Q - Q(t)\, \hat P`, with
    :math:`\hat Q` the device's ``charge_coupling_operator()`` and
    :math:`\hat P` its ``phase_coupling_operator()``. :math:`I` and :math:`Q`
    are the quadratures of the delivered analytic signal :math:`s = I + iQ`,
    carrier included. With the Fock operators
    :math:`\hat Q = i(\hat a^\dagger - \hat a)` and
    :math:`\hat P = \hat a + \hat a^\dagger`, this is
    :math:`H_d = i(s\, \hat a^\dagger - s^* \hat a)`. A positive carrier then
    drives the resonant sideband with the phase convention of
    :class:`~quchip.control.drive.ChargeDrive` and twice its rate. The drive
    has no counter-rotating term there, so ``RWA()`` drops none of it.

    An upward transition :math:`l \to u` couples to :math:`s` with the weight
    :math:`(\langle u|\hat Q|l\rangle + i\langle u|\hat P|l\rangle)/2` and to
    :math:`s^*` with :math:`(\langle u|\hat Q|l\rangle - i\langle u|\hat
    P|l\rangle)/2`. With the charge sign :math:`[\varphi, n] = +i` of every
    built-in device, :math:`\langle u|\hat Q|l\rangle / \langle u|\hat
    P|l\rangle` is :math:`+i` times a positive number on the low transitions,
    so the resonant sideband is the stronger one. The other sideband cancels
    only where the ratio is exactly :math:`+i`, as on an unscaled Fock device.
    A charge-basis transmon or a fluxonium couples both sidebands with its
    own weights. A pulse scheduled without ``freq`` takes the carrier
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
        label: str = getattr(target, "label")
        dims = (getattr(target, "local_space")().dimension,)
        charge = as_operator_expr(target.charge_coupling_operator(), labels=(label,), dims=dims,
                                  name=rf"\hat Q_{{{label}}}")
        phase = as_operator_expr(target.phase_coupling_operator(), labels=(label,), dims=dims,
                                 name=rf"\hat P_{{{label}}}")
        return signal.i * charge - signal.q * phase


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
