"""Runtime-checkable Protocols for physical-operator drive dispatch.

Circuit-level devices (fluxonium, charge-basis transmon) expose their
*physical* charge / phase / flux operators in the truncated eigenbasis.
Drives prefer these over the structural Fock-ladder fallback so that
drive matrix elements are physically correct.

These Protocols are :func:`typing.runtime_checkable` so that
``isinstance(device, ChargeCoupled)`` works at runtime. A device
conforms by defining the named method — no explicit subclassing
required (matches :class:`~quchip.interop.eigenbasis.EigenbasisDevice`
and any future third-party device).

The accessors return the operator as a dense, trace-safe array-like —
not a backend-native operator — so a traced device parameter flows
through the eigenbasis projection without concretization. Backend
composition entry points (``Backend.tensor``, ``Backend.dag``) coerce
array-likes via :meth:`~quchip.backend.protocol.Backend.coerce_operator`,
so the accessors' output composes directly with native operators.

See ``docs/superpowers/specs/2026-04-21-circuit-level-devices-design.md``
for the full rationale.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quchip.backend.protocol import Operator


@runtime_checkable
class ChargeCoupled(Protocol):
    """Device exposes the physical charge operator in its truncated eigenbasis.

    :class:`~quchip.control.drive.ChargeDrive` dispatches against this
    Protocol and emits drives using :meth:`charge_coupling_operator`
    rather than the structural ``1j*(a - a†)`` from
    :meth:`BaseDevice.lowering_operator`.
    """

    def charge_coupling_operator(self) -> Operator:
        """Return the physical charge operator in the truncated eigenbasis."""
        ...


@runtime_checkable
class PhaseCoupled(Protocol):
    """Device exposes the physical phase-space coupling operator.

    Returns :math:`V^\\dagger \\sin\\hat\\varphi V` on a charge-basis transmon
    (where :math:`\\hat\\varphi` is not single-valued in the integer charge
    basis) or :math:`V^\\dagger \\hat\\varphi V` on a fluxonium (where
    :math:`\\hat\\varphi` is well-defined). Used by
    :class:`~quchip.control.drive.PhaseDrive`.
    """

    def phase_coupling_operator(self) -> Operator:
        """Return the physical phase-space coupling operator in the eigenbasis."""
        ...


@runtime_checkable
class FluxCoupled(Protocol):
    """Device exposes the physical flux-line coupling operator.

    For a fluxonium this is :math:`V^\\dagger \\hat\\varphi V`; for a
    flux-tunable transmon (future follow-up) it will be a flux-modulated
    term. Used by :class:`~quchip.control.drive.FluxDrive`.
    """

    def flux_coupling_operator(self) -> Operator:
        """Return the physical flux-line coupling operator in the eigenbasis."""
        ...


@runtime_checkable
class FrequencyControlled(Protocol):
    """Device exposes a frequency-vs-flux relation, i.e. it is frequency-tunable.

    :func:`~quchip.chip.transformations.eliminate_device.reduce_device` uses
    ``isinstance(mode, FrequencyControlled)`` to decide whether an eliminated
    mode's mediated-exchange fold should stay tunable — emitting a
    :class:`~quchip.chip.couplings.TunableCapacitive` — rather than a fixed
    :class:`~quchip.chip.couplings.Capacitive`.
    :class:`~quchip.devices.transmon.flux_tunable.FluxTunableTransmon`
    satisfies this Protocol structurally, with no explicit subclassing.
    """

    def frequency_at(self, flux: Any) -> Any:
        """Return the device's transition frequency at the given flux bias."""
        ...
