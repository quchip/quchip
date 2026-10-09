"""Duffing-approximation transmon qubit.

Hamiltonian:

.. math::

    H = \\omega \\, \\hat{n} + \\tfrac{\\alpha}{2} \\, \\hat{n}\\,(\\hat{n} - \\mathbb{1})

where :math:`\\hat{n} = a^\\dagger a`, :math:`\\omega` is the bare
``0 -> 1`` transition frequency and :math:`\\alpha` is the
anharmonicity (conventionally negative for a transmon,
:math:`\\alpha \\sim -200\\ \\text{MHz}`).

Approximation & regime of validity
----------------------------------
The Kerr / Duffing expansion of the transmon's cosine Josephson potential
truncated at quartic order:

.. math:: H_{\\text{full}} = 4 E_C (\\hat{n} - n_g)^2 - E_J \\cos \\hat{\\phi}

expanded to :math:`\\hat{\\phi}^4` after rotating-frame normal
ordering. Validity conditions (Koch et al. 2007):

* Transmon regime, :math:`E_J / E_C \\gtrsim 50`. The charge-dispersion of the
  lowest levels becomes exponentially small in :math:`\\sqrt{8 E_J / E_C}`, so
  offset-charge noise is suppressed and a weakly anharmonic oscillator is a
  good approximation of the qubit.
* Low-lying levels only. Higher levels probe more of the cosine nonlinearity
  and deviate from the quartic truncation.
* Anharmonicity :math:`\\alpha \\approx -E_C`, with
  :math:`\\omega_{01} \\approx \\sqrt{8 E_J E_C} - E_C`.

Not captured: full charge-basis spectrum, higher-order nonlinearities
(:math:`\\hat{\\phi}^6` and beyond), flux-tunability, two-qubit dispersive
shifts beyond what couplings/drives supply.

References
----------
* Koch, Yu, Gambetta, Houck, Schuster, Majer, Blais, Devoret, Girvin
  & Schoelkopf, *Charge-insensitive qubit design derived from the
  Cooper pair box*, *Physical Review A* **76**, 042319 (2007), Eq.
  2.6 (Duffing form); Eqs. 2.11–2.12 (regime of validity).
* Didier, Sete, da Silva & Rigetti, *Analytical modeling of
  parametrically modulated transmon qubits*, *Physical Review A*
  **97**, 022330 (2018) — anharmonic-oscillator sector used in
  pulse-level modelling.
* Krantz, Kjaergaard, Yan, Orlando, Gustavsson & Oliver, *A quantum
  engineer's guide to superconducting qubits*, *Applied Physics
  Reviews* **6**, 021318 (2019) — §III.B for the Duffing form, §V for
  the ``T1`` / ``T2`` / thermal channels that the base class attaches.

Noise hooks inherited from :class:`~quchip.devices.base.BaseDevice`
(``T1``, ``T2``, ``thermal_occupation``) produce the standard
Lindblad channels described in that base class.

Example
-------
>>> from quchip.chip import Chip
>>> from quchip.devices import DuffingTransmon, Resonator
>>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
>>> r = Resonator(freq=7.0, levels=6, label="r")
>>> chip = Chip(devices=[q, r])
>>> q.computational, q.freq, q.anharmonicity
(True, 5.0, -0.25)
"""

from __future__ import annotations


from typing import Any, ClassVar

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.fock import FockDevice


def duffing_expr(op: LocalOps, freq: Scalar, anharmonicity: Scalar) -> PhysicsExpr:
    """Shared Duffing local Hamiltonian ``H = omega n + (alpha/2) n (n - I)``.
    """
    # Both `DuffingTransmon` and `FluxTunableTransmon` build their static local
    # Hamiltonian from this single expression, so both produce the identical
    # declarative term.
    n = op.n
    return freq * n + (0.5 * anharmonicity) * (n @ (n - op.I))


class DuffingTransmon(FockDevice):
    """Transmon modelled as a weakly anharmonic Duffing oscillator.

    Parameters
    ----------
    freq : float
        Bare ``0 -> 1`` transition frequency ω in GHz. Must be positive. Can be
        a JAX tracer for sweeps / gradients.
    anharmonicity : float
        Anharmonicity α in GHz. Usually negative for superconducting transmons
        (for example ``-0.25`` GHz). Can be a JAX tracer.
    levels : int, default 3
        Fock-space truncation. Three levels suffice for leakage-aware
        single-qubit modelling. Increase it for higher-level physics (for
        example iSWAP-family gates through the ``|02>-|11>`` crossing).
    label : str | None, default None
        If omitted, auto-generated as ``duffing_{idx}`` with the shared
        labeling counter.
    T1 : float or None, default None
        Energy-relaxation time in ns. ``None`` disables T1 relaxation.
    T2 : float or None, default None
        Total 0-1 coherence time in ns, not a pure-dephasing time. If both are
        set, it must satisfy ``T2 <= 2*T1``. ``None`` disables T2 noise.
    thermal_occupation : float or None, default None
        Dimensionless mean bath occupation. ``None`` disables thermal
        absorption.

    References
    ----------
    Koch et al., *Phys. Rev. A* 76, 042319 (2007),
    https://doi.org/10.1103/PhysRevA.76.042319.

    Example
    -------
    >>> from quchip.devices import DuffingTransmon
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, T1=30_000.0, T2=20_000.0)
    >>> len(q.collapse_operators()) >= 1
    True
    """

    _type_prefix: ClassVar[str] = "duffing"
    _default_levels: ClassVar[int] = 3
    tunable_param_names = ("freq", "anharmonicity")
    dressed_fit_target_fields = (
        ("freq", "freq"),
        ("anharmonicity", "anharmonicity"),
    )
    dressed_fit_param_names = ("freq", "anharmonicity")
    approximation = "Duffing expansion: cosine Josephson potential truncated at 4th order."
    computational = True

    freq: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz", symbol=r"\omega")
    anharmonicity: Scalar = parameter(default=UNBOUND, unit="GHz", symbol=r"\alpha")

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        """Return the local Duffing Hamiltonian.

        Parameters
        ----------
        op : LocalOps
            Fock operator namespace.
        p : ParameterNamespace
            Symbolic ``freq`` and ``anharmonicity`` values.
        """
        return duffing_expr(op, p.freq, p.anharmonicity)

    def physics_notes(self) -> list[str]:
        """Return declared Duffing-approximation validity notes."""
        notes = super().physics_notes()
        notes.append("Validity: transmon regime E_J/E_C ≳ 50; higher-order cosine terms dropped")
        return notes
