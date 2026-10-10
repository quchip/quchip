"""Reference scheduled envelope authored through the declarative surface."""

from __future__ import annotations

from typing import Any

from quchip.declarative import Envelope, Scalar, parameter, qnp
from quchip.declarative.parameters import UNBOUND


class CosineEnvelope(Envelope):
    r"""Raised-cosine pulse with zero endpoints and peak amplitude at mid-pulse.

    .. math:: E(t) = \frac{A}{2}\left[1-\cos(2\pi t/\tau)\right]
    Parameters
    ----------
    duration : float
        Pulse duration in ns. Must be positive.
    amplitude : float, default 1.0
        Peak real amplitude. The returned envelope is complex with zero
        quadrature.

    References
    ----------
    See Motzoi et al., *Phys. Rev. Lett.* 103, 110501 (2009),
    https://doi.org/10.1103/PhysRevLett.103.110501, for smooth pulse shaping.
    """

    duration: Scalar = parameter(default=UNBOUND, positive=True, unit="ns")
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t: Any) -> Any:
        """Evaluate the envelope at local times ``t`` in ns.

        Parameters
        ----------
        t : scalar or array-like
            Time relative to pulse start.
        """
        return qnp.asarray(
            0.5 * self.amplitude * (1.0 - qnp.cos(2.0 * qnp.pi * t / self.duration)),
            dtype=complex,
        )
