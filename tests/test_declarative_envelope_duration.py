from __future__ import annotations

import pytest

from quchip.declarative import EnvelopeShape, Scalar, parameter


class _NoDuration(EnvelopeShape):
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t):
        return self.amplitude


def test_envelope_shape_without_declared_duration_raises_at_construction():
    """An EnvelopeShape subclass that never declares ``duration`` raises TypeError at construction."""
    with pytest.raises(TypeError, match=r"_NoDuration.*duration"):
        _NoDuration()
