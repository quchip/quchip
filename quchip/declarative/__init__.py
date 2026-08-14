"""Declarative physics model API."""

from __future__ import annotations

from typing import Any

from quchip.declarative import qnp
from quchip.declarative.dynamics import CosineCoefficient, TimeCoefficient, TimeDependentTerm
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.expr import (
    PhysicsExpr,
    as_operator_expr,
    as_scalar_expr,
    as_state_expr,
)
from quchip.declarative.models import CouplingModel, DeviceModel
from quchip.declarative.ops import EndpointOps, LocalOps
from quchip.declarative.parameters import (
    Parameter,
    Scalar,
    Setting,
    parameter,
    setting,
)

# ``Envelope`` is loaded lazily because its control module imports the
# declarative parameter primitives defined by this package.

__all__ = [
    "CouplingModel",
    "CollapseChannel",
    "CosineCoefficient",
    "DeviceModel",
    "EndpointOps",
    "Envelope",
    "LocalOps",
    "Parameter",
    "PhysicsExpr",
    "TimeDependentTerm",
    "Scalar",
    "Setting",
    "TimeCoefficient",
    "as_operator_expr",
    "as_scalar_expr",
    "as_state_expr",
    "parameter",
    "setting",
    "qnp",
]


def __getattr__(name: str) -> Any:
    """Lazily expose Envelope without triggering the control import cycle."""
    if name == "Envelope":
        from quchip.control.envelopes import Envelope

        return Envelope
    raise AttributeError(f"module 'quchip.declarative' has no attribute {name!r}")
