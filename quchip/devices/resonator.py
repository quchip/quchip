"""Linear-resonator device model.

Hamiltonian:

.. math:: H = \\omega \\, \\hat{n}

where :math:`\\hat{n} = a^\\dagger a` is the Fock-basis number operator
and :math:`\\omega` is the bare cavity frequency.

Approximation
-------------
Strictly harmonic / non-interacting: no Kerr, no cross-Kerr, no drive
backaction other than what couplings/drives themselves introduce. It models the
ideal cavity / transmission-line-resonator mode and applies to readout
cavities, filter modes, photonic oscillators, and cavity-QED benchmarks, where
anharmonicity is absent or modelled separately. For Kerr / anharmonic cavities,
use a device that owns an explicit ``(K/2) n(n-1)`` term (see
``examples/kerr_cat_qubit.py``).

Optional dissipation
--------------------
If you pass ``internal_quality_factor = Q``, the device adds a single unobserved
photon-loss collapse operator ``sqrt(kappa) a`` with :math:`\\kappa = 2\\pi\\,f/Q`
(energy-decay rate, 1/ns).

**Quality-factor convention (physics, not a unit conversion).**
``internal_quality_factor`` is defined against this class's *ordinary*
frequency ``freq`` (GHz), which gives the decay rate
:math:`\\kappa = 2\\pi\\,f/Q` (energy decay, 1/ns). The :math:`2\\pi` here is
intrinsic to the physical definition of Q, not an ordinary→angular units
conversion at the engine boundary. ``Q/(2*pi)`` is the number of
ordinary-frequency cycles per e-folding of energy, so energy decays as
:math:`e^{-t/\\tau} = e^{-\\kappa t}` with
:math:`\\kappa = \\omega/Q = 2\\pi f/Q`.

Noise hooks inherited from :class:`~quchip.devices.base.BaseDevice`
(``T1``, ``T2``, ``thermal_occupation``) produce the Lindblad
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
# Because the 2π is intrinsic to the definition of Q, the `2\\pi` lives in the
# resonator's photon-loss noise channel and must not be moved to the units
# boundary in `assembly.py`.

from __future__ import annotations


from typing import Any, ClassVar

import numpy as np

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.fock import FockDevice


class Resonator(FockDevice):
    """Linear microwave / photonic resonator (pure harmonic oscillator).

    Parameters
    ----------
    freq : float
        Bare cavity frequency ω in GHz. Must be positive. Can be a JAX tracer
        for sweeps / gradients.
    internal_quality_factor : float | None, optional
        Internal Q referenced to the ordinary frequency ``freq`` in GHz. When
        set, it adds a photon-loss Lindblad channel ``sqrt(2*pi*freq/Q) a``
        with energy-decay rate ``kappa = 2*pi*freq/Q`` in 1/ns. Must be
        positive. Like every noise parameter, it can be set after construction
        or cleared with ``None``, and the next simulation uses the current
        value.
    levels : int, default 10
        Fock-space truncation. Choose it well above the maximum expected photon
        occupation.
    label : str | None, default None
        If omitted, the label is ``resonator_{idx}`` with the shared labeling
        counter.
    T1 : float or None, default None
        Energy-relaxation time in ns. ``None`` disables T1 relaxation.
    T2 : float or None, default None
        Total 0-1 coherence time in ns. If both are set, ``T2 <= 2*T1``.
    thermal_occupation : float or None, default None
        Dimensionless mean bath occupation. ``None`` disables absorption.

    References
    ----------
    Blais et al., *Rev. Mod. Phys.* 93, 025005 (2021),
    https://doi.org/10.1103/RevModPhys.93.025005.

    Example
    -------
    >>> from quchip.devices import Resonator
    >>> r = Resonator(freq=7.2, internal_quality_factor=10_000, levels=8)
    >>> len(r.collapse_operators()) >= 1
    True
    """

    _type_prefix: ClassVar[str] = "resonator"
    _default_levels: ClassVar[int] = 10
    tunable_param_names = ("freq",)
    dressed_fit_target_fields = (("freq", "freq"),)
    dressed_fit_param_names = ("freq",)

    freq: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz", symbol=r"\omega")
    internal_quality_factor: Scalar = parameter(default=None, positive=True, noise=True, kw_only=True)

    approximation = "Linear harmonic oscillator with no Kerr or cross-Kerr self-interaction."

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        """Return the harmonic oscillator Hamiltonian ``H = freq * n``.

        Parameters
        ----------
        op : LocalOps
            Fock operator namespace.
        p : ParameterNamespace
            Symbolic ``freq`` value.
        """
        return p.freq * op.n

    def dissipation(self, op: LocalOps, p: Any) -> tuple[CollapseChannel, ...]:
        channels = super().dissipation(op, p)
        if self.internal_quality_factor is None:
            return channels
        return channels + (
            CollapseChannel(op.a, 2 * np.pi * p.freq / p.internal_quality_factor, "internal_photon_loss"),
        )

    def physics_notes(self) -> list[str]:
        """Return declared harmonic-oscillator and dissipation assumptions."""
        notes = super().physics_notes()
        if self.internal_quality_factor is not None:
            notes.append("Internal dissipation: photon loss at rate κ_internal = 2π·ω/Q_internal")
        return notes

    def intrinsic_decay_rate(self) -> Any | None:
        """Return the combined lowering-channel rate: ``κ = 2π·freq/Q`` photon loss plus the thermal-emission rate.

        Both :attr:`internal_quality_factor` and ``T1``/``thermal_occupation`` build
        independent lowering-operator collapse channels on this device. The first is
        the ``internal_photon_loss`` channel, a pure loss channel that
        ``thermal_occupation`` does not affect. The second is the inherited
        thermal-emission channel. See
        :meth:`~quchip.devices.base.BaseDevice.intrinsic_decay_rate` for its
        ``(n̄+1)/T1`` / ``n̄+1`` formulas. This hook reports their summed rate, not
        either rate alone. A caller that reads one scalar decay rate, e.g. an
        adiabatic-elimination Purcell fold, then does not under-count decay when
        both are set. ``None`` only when neither is set.
        """
        kappa = (
            None
            if self.internal_quality_factor is None
            else 2 * np.pi * self.freq / self.internal_quality_factor
        )
        thermal_rate = super().intrinsic_decay_rate()
        if kappa is None and thermal_rate is None:
            return None
        if kappa is None:
            return thermal_rate
        if thermal_rate is None:
            return kappa
        return kappa + thermal_rate
