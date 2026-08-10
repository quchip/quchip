"""Declarative physics model API."""

from __future__ import annotations

from typing import Any

from quchip.declarative import qnp
from quchip.declarative.expr import (
    DynamicScalar,
    PhysicsExpr,
    as_operator_expr,
    as_scalar_expr,
    as_state_expr,
)
from quchip.declarative.models import CouplingModel, DeviceModel
from quchip.declarative.ops import EndpointOps, LocalOps
from quchip.declarative.parameters import (
    Modulation,
    Parameter,
    Scalar,
    parameter,
)

# ``EnvelopeShape`` lives in :mod:`quchip.declarative.envelope_shape`, which
# imports :class:`quchip.control.envelopes.BaseEnvelope`, which imports
# ``quchip.control``, which imports ``quchip.engine``, which imports back
# into this package — a cycle. The lazy import here lets ``DeviceModel``
# (e.g. via ``Resonator(DeviceModel)`` during ``quchip.devices``
# initialization) load first without tripping it.

__all__ = [
    "CouplingModel",
    "DeviceModel",
    "DynamicScalar",
    "EndpointOps",
    "EnvelopeShape",
    "LocalOps",
    "Modulation",
    "Parameter",
    "PhysicsExpr",
    "Scalar",
    "as_operator_expr",
    "as_scalar_expr",
    "as_state_expr",
    "parameter",
    "qnp",
]


def __getattr__(name: str) -> Any:
    """Lazily expose EnvelopeShape without triggering the control import cycle."""
    if name == "EnvelopeShape":
        from quchip.declarative.envelope_shape import EnvelopeShape

        return EnvelopeShape
    raise AttributeError(f"module 'quchip.declarative' has no attribute {name!r}")
