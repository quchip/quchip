"""Device models — truncated-Hilbert-space quantum systems owned by a chip.

A device declares its **local** Hamiltonian; couplings and drives
contribute their own local Hamiltonians. See :mod:`quchip.devices.base` for
the full device protocol.

Public models
-------------
* :class:`Resonator` — linear harmonic mode, ``H = omega * n_hat``.
* :class:`DuffingTransmon` — Duffing-anharmonic transmon qubit,
  ``H = omega * n + (alpha/2) * n * (n - I)``.
* :class:`FluxTunableTransmon` — SQUID-dispersion flux-tunable transmon;
  ``freq``/``anharmonicity`` are the calibrated local transition
  parameters at the stored ``flux_bias``.
* :class:`KerrCavity` — Kerr-nonlinear resonator,
  ``H = omega * n_hat - K * n_hat * (n_hat - I)``.
* :class:`CircuitDevice` — abstract base for circuit-level devices
  built by diagonalizing a native-basis Hamiltonian.
* :class:`Fluxonium` — circuit-level fluxonium in the phase basis.
* :class:`ChargeBasisTransmon` — circuit-level transmon in the
  integer charge basis.

Coupling Protocols (for drive dispatch)
---------------------------------------
* :class:`ChargeCoupled`, :class:`PhaseCoupled`, :class:`FluxCoupled`,
  :class:`FrequencyControlled`
"""

from quchip.devices.circuit import CircuitDevice
from quchip.devices.fluxonium import Fluxonium
from quchip.devices.kerr_cavity import KerrCavity
from quchip.devices.protocols import ChargeCoupled, FluxCoupled, FrequencyControlled, PhaseCoupled
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.charge_basis import ChargeBasisTransmon
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.devices.transmon.flux_tunable import FluxTunableTransmon

__all__ = [
    "ChargeBasisTransmon",
    "ChargeCoupled",
    "CircuitDevice",
    "DuffingTransmon",
    "FluxCoupled",
    "FluxTunableTransmon",
    "Fluxonium",
    "FrequencyControlled",
    "KerrCavity",
    "PhaseCoupled",
    "Resonator",
]
