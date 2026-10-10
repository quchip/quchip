"""Runtime-checkable Protocols for physical-operator drive dispatch.

Devices expose their *physical* charge / phase / flux operators in their
authored local basis. Drives require these declarations so that their matrix
elements stay physically explicit. The engine then applies the device's
resolved local-basis transformation together with every other attached
operator.

These Protocols are :func:`typing.runtime_checkable`, so
``isinstance(device, ChargeCoupled)`` works at runtime. A device conforms when
it defines the named method, without explicit subclassing. This includes
:class:`~quchip.interop.eigenbasis.EigenbasisDevice` and external devices.

The accessors follow the common operator extension contract. Symbolic
expressions are preferred, but matrices and pure JAX callables stay valid. The
engine resolves every form through the same local-basis boundary.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quchip.backend.protocol import Operator


@runtime_checkable
class ChargeCoupled(Protocol):
    """Device exposes the physical charge operator in its authored local basis.

    :class:`~quchip.control.drive.ChargeDrive` dispatches against this
    Protocol and emits drives using :meth:`charge_coupling_operator`.
    """

    def charge_coupling_operator(self) -> Operator:
        """Return the physical charge operator in the authored local basis."""
        ...


@runtime_checkable
class PhaseCoupled(Protocol):
    """Device exposes the physical phase-space coupling operator.

    Returns :math:`\\sin\\hat\\varphi` on a charge-basis transmon (where
    :math:`\\hat\\varphi` is not single-valued in the integer charge basis) or
    :math:`\\hat\\varphi` on a fluxonium (where it is well-defined). Used by
    :class:`~quchip.control.drive.PhaseDrive`.
    """

    def phase_coupling_operator(self) -> Operator:
        """Return the physical phase-space coupling operator in the authored basis."""
        ...


@runtime_checkable
class FluxCoupled(Protocol):
    """Device exposes the physical flux-line coupling operator.

    For a fluxonium this is :math:`\\hat\\varphi`. Used by
    :class:`~quchip.control.drive.FluxDrive`.
    """

    def flux_coupling_operator(self) -> Operator:
        """Return the physical flux-line coupling operator in the authored basis."""
        ...


@runtime_checkable
class FrequencyControlled(Protocol):
    """Device exposes a frequency-vs-flux relation, i.e. it is frequency-tunable.

    :func:`~quchip.chip.transformations.eliminate_device.reduce_device` uses
    ``isinstance(mode, FrequencyControlled)`` to decide whether an eliminated
    mode's mediated-exchange fold should stay tunable. If so, it emits a
    :class:`~quchip.chip.couplings.TunableCapacitive`, not a fixed
    :class:`~quchip.chip.couplings.Capacitive`.
    :class:`~quchip.devices.transmon.flux_tunable.FluxTunableTransmon`
    satisfies this Protocol structurally, with no explicit subclassing.
    """

    def frequency_at(self, flux: Any) -> Any:
        """Return the device's transition frequency at a flux bias.

        Parameters
        ----------
        flux : float or array-like
            Reduced flux bias in units of ``Phi_0``.
        """
        ...
