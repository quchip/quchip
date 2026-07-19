from __future__ import annotations

import pytest

from quchip.declarative import DeviceModel, EnvelopeShape, Scalar, parameter


def test_device_model_required_after_optional_raises_naming_class_and_fields():
    """A DeviceModel subclass with a required field after an optional one raises, naming the class and both fields."""
    with pytest.raises(TypeError, match=r"Bad.*'b'.*'a'"):

        class Bad(DeviceModel):
            a: Scalar = parameter(default=1.0)
            b: Scalar = parameter()


def test_device_model_inherited_required_after_optional_raises():
    """An optional field inherited from a DeviceModel base, followed by a required field in the subclass, raises."""

    class Base(DeviceModel):
        a: Scalar = parameter(default=1.0)

    with pytest.raises(TypeError, match=r"Child.*'b'.*'a'"):

        class Child(Base):
            b: Scalar = parameter()


def test_envelope_shape_required_after_optional_raises_naming_class_and_fields():
    """An EnvelopeShape subclass with a required field after an optional one raises, naming the class and fields."""
    with pytest.raises(TypeError, match=r"BadEnvelope.*'edge'.*'duration'"):

        class BadEnvelope(EnvelopeShape):
            duration: Scalar = parameter(default=10.0)
            edge: Scalar = parameter()


def test_envelope_shape_inherited_required_after_optional_raises():
    """An optional field inherited from an EnvelopeShape base, followed by a required field in the subclass, raises."""

    class BaseEnv(EnvelopeShape):
        duration: Scalar = parameter(default=10.0)

    with pytest.raises(TypeError, match=r"ChildEnv.*'edge'.*'duration'"):

        class ChildEnv(BaseEnv):
            edge: Scalar = parameter()
