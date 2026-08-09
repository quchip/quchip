from __future__ import annotations

from quchip.declarative import DeviceModel, EnvelopeShape, Scalar, parameter
from quchip.declarative.parameters import UNBOUND


def test_device_model_parameter_without_default_can_remain_unbound_after_defaulted_field():
    """Declaration order does not prevent a later parameter from remaining symbolic."""
    class SymbolicDevice(DeviceModel):
        a: Scalar = parameter(default=1.0)
        b: Scalar = parameter()

    device = SymbolicDevice()
    assert device.a == 1.0
    assert device.b is UNBOUND

def test_inherited_envelope_parameter_without_default_can_remain_unbound():
    """Inherited defaults and symbolic child parameters compose without constructor ordering rules."""

    class BaseEnv(EnvelopeShape):
        duration: Scalar = parameter(default=10.0)

    class SymbolicEnvelope(BaseEnv):
        edge: Scalar = parameter()

    envelope = SymbolicEnvelope()
    assert envelope.duration == 10.0
    assert envelope.edge is UNBOUND
