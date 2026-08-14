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
must not be moved to the units boundary in ``assembly.py``.

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


from typing import Any, ClassVar

import numpy as np

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.fock import FockDevice


class Resonator(FockDevice):
    """Linear microwave / photonic resonator — pure harmonic oscillator.

    Parameters
    ----------
    freq : float
        Bare cavity frequency ω in GHz. Must be positive. May be a JAX
        tracer for sweeps / gradients.
    quality_factor : float | None, optional
        Loaded Q referenced to the ordinary frequency ``freq`` in GHz.
        When set, adds a photon-loss Lindblad channel
        ``sqrt(2*pi*freq/Q) a`` with angular decay rate
        ``kappa = 2*pi*freq/Q`` in rad/ns. Must be positive. Like every
        noise parameter, it may be set after construction or cleared with
        ``None``; the next simulation reflects the current value.
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

    _type_prefix: ClassVar[str] = "resonator"
    _default_levels: ClassVar[int] = 10
    tunable_param_names = ("freq",)

    freq: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz", symbol=r"\omega")
    quality_factor: Scalar = parameter(default=None, positive=True, noise=True, kw_only=True)

    approximation = "Linear harmonic oscillator with no Kerr or cross-Kerr self-interaction."

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        """Return the harmonic oscillator Hamiltonian ``H = freq * n``."""
        return p.freq * op.n

    def dissipation(self, op: LocalOps, p: Any) -> tuple[CollapseChannel, ...]:
        channels = super().dissipation(op, p)
        if self.quality_factor is None:
            return channels
        return channels + (
            CollapseChannel(op.a, 2 * np.pi * p.freq / p.quality_factor, "photon_loss"),
        )

    def physics_notes(self) -> list[str]:
        """Return declared harmonic-oscillator and dissipation assumptions."""
        notes = super().physics_notes()
        notes.append("Linear harmonic oscillator (no Kerr, no cross-Kerr self-interaction)")
        if self.quality_factor is not None:
            notes.append("Dissipation: photon loss at rate κ = 2π·ω/Q")
        return notes

    def intrinsic_decay_rate(self) -> Any | None:
        """Combined lowering-channel rate: ``κ = 2π·freq/Q`` photon loss plus the thermal-emission rate.

        Both :attr:`quality_factor` and ``T1``/``thermal_population`` build
        independent lowering-operator collapse channels on this device (the
        ``photon_loss`` channel, a pure loss channel unaffected by
        ``thermal_population``, and the inherited
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
