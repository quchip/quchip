"""Solve-time coherent field controls.

Field inputs share the sequence scheduling grammar with classical drive lines,
but remain independent of :class:`~quchip.control.drive.BaseDrive` and
:class:`~quchip.control.equipment.ControlEquipment`. Their target is an
external SLH exposure rather than a device or coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from quchip.control.signal import AnalyticSignal
from quchip.utils.labeling import auto_label, resolve_label


@runtime_checkable
class ControlEndpoint(Protocol):
    """Structural endpoint accepted by the sequence scheduling grammar.

    Existing :class:`~quchip.control.drive.BaseDrive` implementations satisfy
    this protocol without inheritance. A field endpoint supplies the same
    label, target-label, and signal-building boundary while owning different
    downstream physics.
    """

    label: str

    @property
    def target_label(self) -> str | None:
        """Return the endpoint's device, coupling, or field-exposure label."""
        ...

    def signal(self, pulse: Any, target: Any) -> AnalyticSignal:
        """Build the complete analytic signal for one scheduled pulse."""
        ...


@dataclass(frozen=True, init=False)
class CoherentInput:
    r"""Coherent incident field bound to one external network exposure.

    The scheduled analytic signal is interpreted directly as
    ``beta(t) = A(t) exp(i theta) exp(-i 2*pi*f*t)`` in ``1/sqrt(ns)``.
    Consequently ``abs(beta)**2`` is photon flux in photons/ns.
    """

    exposure: str
    label: str

    def __init__(self, exposure: str | Any, *, label: str | None = None) -> None:
        exposure_label = resolve_label(exposure)
        object.__setattr__(self, "exposure", exposure_label)
        object.__setattr__(
            self,
            "label",
            label if label is not None else auto_label("coherent_input"),
        )

    @property
    def target_label(self) -> str:
        """Return the external exposure receiving the incident field."""
        return self.exposure

    def signal(self, pulse: Any, target: Any = None) -> AnalyticSignal:
        """Build beta from the existing envelope, phase, carrier, and timing grammar."""
        _ = target
        return AnalyticSignal.from_pulse(pulse)
