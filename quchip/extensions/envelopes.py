"""Reference scheduled envelope authored through the declarative surface."""

from __future__ import annotations

from typing import Any

from quchip.declarative import Envelope, Scalar, parameter, qnp
from quchip.declarative.parameters import UNBOUND


class CosineEnvelope(Envelope):
    r"""Raised-cosine pulse with zero endpoints and peak amplitude at mid-pulse.

    .. math:: E(t) = \frac{A}{2}\left[1-\cos(2\pi t/\tau)\right]
    """

    duration: Scalar = parameter(default=UNBOUND, positive=True, unit="ns")
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t: Any) -> Any:
        return qnp.asarray(
            0.5 * self.amplitude * (1.0 - qnp.cos(2.0 * qnp.pi * t / self.duration)),
            dtype=complex,
        )
