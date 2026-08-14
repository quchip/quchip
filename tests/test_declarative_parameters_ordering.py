from __future__ import annotations

import inspect

import pytest

from quchip.declarative import DeviceModel, Envelope, Scalar, TimeCoefficient, parameter
from quchip.declarative.parameters import UNBOUND


def test_device_model_parameter_can_explicitly_remain_unbound_after_defaulted_field():
    """Declaration order does not prevent a later parameter from remaining symbolic."""
    class SymbolicDevice(DeviceModel):
        a: Scalar = parameter(default=1.0)
        b: Scalar = parameter(default=UNBOUND)

    device = SymbolicDevice()
    assert device.a == 1.0
    assert device.b is UNBOUND

def test_inherited_envelope_parameter_can_explicitly_remain_unbound():
    """Inherited defaults and symbolic child parameters compose without constructor ordering rules."""

    class BaseEnv(Envelope):
        duration: Scalar = parameter(default=10.0)

        def value(self, t):
            return t

    class SymbolicEnvelope(BaseEnv):
        edge: Scalar = parameter(default=UNBOUND)

    envelope = SymbolicEnvelope()
    assert envelope.duration == 10.0
    assert envelope.edge is UNBOUND


@pytest.mark.parametrize("base", [DeviceModel, Envelope, TimeCoefficient])
def test_keyword_only_fields_do_not_constrain_positional_declaration_order(base):
    """Constructor synthesis partitions keyword-only fields after positional fields."""

    class MixedOrder(base):
        option: Scalar = parameter(default=1.0, kw_only=True)
        amplitude: Scalar = parameter()

        def local_hamiltonian(self, op, p):
            return p.amplitude * op.n

        def value(self, time):
            return self.amplitude * time

    signature = inspect.signature(MixedOrder)
    assert tuple(signature.parameters)[:2] == ("amplitude", "option")
    assert signature.parameters["amplitude"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["option"].kind is inspect.Parameter.KEYWORD_ONLY
