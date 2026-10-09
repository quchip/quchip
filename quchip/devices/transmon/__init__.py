"""Transmon device models.

* :class:`DuffingTransmon`: weakly anharmonic Duffing approximation. Valid in
  the transmon regime :math:`E_J \\gg E_C` (Koch et al. PRA **76**, 042319
  (2007)).
* :class:`FluxTunableTransmon`: SQUID-dispersion flux-tunable transmon
  (symmetric or asymmetric). It applies to parametric/flux-driven operations
  and tunable couplers, inherits directly from
  :class:`~quchip.devices.base.BaseDevice`, and its constructor takes physical
  dressed parameters.
* :class:`ChargeBasisTransmon`: exact charge-basis diagonalization that
  captures charge-dispersion with :math:`n_g` outside the deep transmon regime.
"""

from quchip.devices.transmon.charge_basis import ChargeBasisTransmon
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.devices.transmon.flux_tunable import FluxTunableTransmon

__all__ = ["ChargeBasisTransmon", "DuffingTransmon", "FluxTunableTransmon"]
