"""Transmon device models.

* :class:`DuffingTransmon` — weakly anharmonic Duffing approximation
  (valid in the transmon regime :math:`E_J \\gg E_C`; Koch et al.
  PRA **76**, 042319 (2007)).
* :class:`FluxTunableTransmon` — SQUID-dispersion flux-tunable transmon
  (symmetric or asymmetric), suitable for parametric/flux-driven operations
  and tunable couplers.  Inherits from :class:`~quchip.devices.base.BaseDevice`
  directly; constructor takes physical dressed parameters.
* :class:`ChargeBasisTransmon` — exact charge-basis diagonalization;
  captures charge-dispersion with :math:`n_g` outside the deep transmon
  regime.
"""

from quchip.devices.transmon.charge_basis import ChargeBasisTransmon
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.devices.transmon.flux_tunable import FluxTunableTransmon

__all__ = ["ChargeBasisTransmon", "DuffingTransmon", "FluxTunableTransmon"]
