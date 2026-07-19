"""Linear-resonator device model.

Hamiltonian:

.. math:: H = \\omega \\, \\hat{n}

where :math:`\\hat{n} = a^\\dagger a` is the Fock-basis number operator
and :math:`\\omega` is the bare cavity frequency.

Approximation
-------------
Strictly harmonic / non-interacting: no Kerr, no cross-Kerr, no drive
backaction beyond what couplings/drives themselves introduce. This is
the ideal cavity / transmission-line-resonator mode — suitable for
readout cavities, filter modes, photonic oscillators, and cavity-QED
benchmarks where anharmonicity is either absent or modelled
separately. For Kerr / anharmonic cavities, use a device that owns an
explicit ``(K/2) n(n-1)`` term (see ``examples/kerr_cat_qubit.py``).

Optional dissipation
--------------------
Passing ``quality_factor = Q`` adds a single photon-loss collapse
operator ``sqrt(kappa) a`` with :math:`\\kappa = 2\\pi\\,f/Q`
(angular decay rate, rad/ns).

**Quality-factor convention (physics, not a unit conversion).**
``quality_factor`` is defined against the *ordinary* frequency
``freq`` (GHz) carried by this class. The resulting decay rate is
:math:`\\kappa = 2\\pi\\,f/Q` (angular, rad/ns). The :math:`2\\pi`
here is intrinsic to the physical definition of Q — not an
ordinary→angular units conversion bolted on at the engine boundary.
Concretely, Q counts cycles of the *ordinary* oscillation per
e-folding of energy, so energy decays as
:math:`e^{-t/\\tau} = e^{-\\kappa t}` with
:math:`\\kappa = \\omega/Q = 2\\pi f/Q`. For this reason the
:math:`2\\pi` lives in the resonator's photon-loss noise channel and
must not be moved to the units boundary in ``stage2_assembly.py``.

Noise hooks inherited from :class:`~quchip.devices.base.BaseDevice`
(``T1``, ``T2``, ``thermal_population``) produce the Lindblad
channels described in that base class. For circuit-QED conventions
see Krantz et al., *Applied Physics Reviews* **6**, 021318 (2019), §V.

References
----------
* Walls & Milburn, *Quantum Optics*, 2nd ed. (Springer, 2008), Ch. 7.
* Blais, Grimsmo, Girvin & Wallraff, *Circuit quantum electrodynamics*,
  *Reviews of Modern Physics* **93**, 025005 (2021).

Example
-------
>>> from quchip.chip import Chip
>>> from quchip.devices import Resonator
>>> r = Resonator(freq=7.2, levels=6, label="readout")
>>> chip = Chip(devices=[r])
>>> r.freq, r.levels
(7.2, 6)
"""

from __future__ import annotations


from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp
import numpy as np

from quchip.backend.protocol import Operator
from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import DeviceModel
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import Scalar, parameter
from quchip.devices.base import BaseDevice, NoiseChannel


def _photon_loss_channel(device: Any) -> list[Operator]:
    """Cavity photon-loss channel ``sqrt(κ)·a`` with ``κ = 2π·freq/Q``.

    The ``2π`` is intrinsic to the physical definition of Q (cycles of the
    ordinary oscillation per e-folding of energy), not a units conversion —
    see the module docstring. Empty when ``quality_factor`` is unset.
    """
    if device.quality_factor is None:
        return []
    sqrt_kappa = jnp.sqrt(2 * np.pi * device.freq / device.quality_factor)
    return [sqrt_kappa * device.lowering_operator()]


class Resonator(DeviceModel):
    """Linear microwave / photonic resonator — pure harmonic oscillator.

    Parameters
    ----------
    freq : float
        Bare cavity frequency ω in GHz. Must be positive. May be a JAX
        tracer for sweeps / gradients.
    quality_factor : float | None, optional
        Loaded Q, defined against the *ordinary* frequency ``freq``
        (GHz). When set, adds a photon-loss Lindblad channel
        ``sqrt(2*pi*freq/Q) a`` — i.e. decay rate
        ``kappa = 2*pi*freq/Q`` (angular, rad/ns). The ``2*pi`` here is
        part of the physical definition of Q (energy e-folds per
        ordinary cycle divided by Q), not a units-boundary conversion.
        Must be positive. Like every noise parameter it may be set — or
        cleared with ``None`` — after construction; the next simulate
        reflects the current value.
    levels : int, default 10
        Fock-space truncation. Choose comfortably above the maximum
        expected photon occupation.
    label : str | None, default None
        If omitted, auto-generated as ``resonator_{idx}`` via the shared
        labeling counter.
    **noise_kwargs
        Forwarded verbatim to :class:`BaseDevice` — ``T1``, ``T2``,
        ``thermal_population``.

    Example
    -------
    >>> from quchip.devices import Resonator
    >>> r = Resonator(freq=7.2, quality_factor=10_000, levels=8)
    >>> len(r.collapse_operators()) >= 1
    True
    """

    _type_prefix: str = "resonator"
    _default_levels: int = 10
    tunable_param_names = ("freq",)

    freq: Scalar = parameter(positive=True, unit="GHz")
    quality_factor: Scalar = parameter(default=None, positive=True)

    # --- generated __init__ stub (tools/gen_device_stubs.py); do not edit ---
    if TYPE_CHECKING:
        def __init__(
            self,
            freq: Scalar,
            quality_factor: Scalar = None,
            *,
            levels: int = 10,
            label: str | None = None,
            T1: float | None = None,
            T2: float | None = None,
            thermal_population: float | None = None,
        ) -> None: ...
    # --- end generated stub ---

    approximation = "Linear harmonic oscillator with no Kerr or cross-Kerr self-interaction."

    def local_hamiltonian(self, op: LocalOps) -> PhysicsExpr:
        """Return the harmonic oscillator Hamiltonian ``H = freq * n``."""
        return self.freq * op.n

    def physics_notes(self) -> list[str]:
        """Return declared harmonic-oscillator and dissipation assumptions."""
        notes = super().physics_notes()
        notes.append("Linear harmonic oscillator (no Kerr, no cross-Kerr self-interaction)")
        if self.quality_factor is not None:
            notes.append("Dissipation: photon loss at rate κ = 2π·ω/Q")
        return notes

    # Base T1/T2/thermal channels plus cavity photon loss when ``Q`` is set —
    # one declaration, no collapse_operators override.
    _noise_channels: ClassVar[tuple[NoiseChannel, ...]] = BaseDevice._noise_channels + (
        NoiseChannel("photon_loss", ("quality_factor",), _photon_loss_channel),
    )

    def intrinsic_decay_rate(self) -> Any | None:
        """Combined lowering-channel rate: ``κ = 2π·freq/Q`` photon loss plus the thermal-emission rate.

        Both :attr:`quality_factor` and ``T1``/``thermal_population`` build
        independent lowering-operator collapse channels on this device (the
        ``photon_loss`` :class:`~quchip.devices.base.NoiseChannel`, a pure
        loss channel unaffected by ``thermal_population``, and the inherited
        thermal-emission channel — see
        :meth:`~quchip.devices.base.BaseDevice.intrinsic_decay_rate` for its
        ``(n̄+1)/T1`` / ``n̄+1`` formulas); this hook reports their summed
        rate rather than either alone, so a caller reading one scalar decay
        rate (e.g. an adiabatic-elimination Purcell fold) does not
        under-count decay when both are set. ``None`` only when neither is
        set.
        """
        kappa = None if self.quality_factor is None else 2 * np.pi * self.freq / self.quality_factor
        thermal_rate = super().intrinsic_decay_rate()
        if kappa is None and thermal_rate is None:
            return None
        if kappa is None:
            return thermal_rate
        if thermal_rate is None:
            return kappa
        return kappa + thermal_rate
