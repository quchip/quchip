from __future__ import annotations

import pytest

from quchip.declarative import Envelope, Scalar, parameter


class _NoDuration(Envelope):
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t):
        return self.amplitude


def test_envelope_shape_without_declared_duration_raises_at_construction():
    """An Envelope subclass that omits ``duration`` fails at construction."""
    with pytest.raises(TypeError, match=r"_NoDuration.*duration"):
        _NoDuration()
