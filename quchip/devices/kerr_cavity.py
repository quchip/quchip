"""KerrCavity: Kerr-nonlinear resonator model.

Hamiltonian:

.. math::

   H = \\omega \\, \\hat{n} - K \\, \\hat{n}(\\hat{n} - I)

where :math:`\\omega` is the cavity frequency (GHz, ordinary) and :math:`K`
is the Kerr nonlinearity (GHz, positive).  Eigenvalues are:

.. math::

   E_n = \\omega n - K n(n-1)

The Kerr term shifts higher Fock levels down by :math:`K` per photon pair,
which gives the anharmonic energy ladder. Together with a two-photon parametric
drive, this ladder stabilises cat states.

Approximation
-------------
This is an effective single-mode model after adiabatic elimination of the SNAIL
or STS-SQUID that supplies the nonlinearity. The Kerr coefficient :math:`K`
captures the leading-order nonlinearity, and higher-order corrections are
ignored. The Hilbert space is truncated at ``levels`` Fock states. To prevent
truncation artefacts, choose ``levels >= 4 * (eps2 / K) + 10``.

References
----------
.. [1] Grimm et al., *Stabilization and operation of a Kerr-cat qubit*,
       Nature 584, 205 (2020). arXiv:1907.12131.
.. [2] Hajr et al., *High-Coherence Kerr-Cat Qubit in 2D Architecture*,
       PRX Quantum 5, 020347 (2024). arXiv:2404.16697.
"""

from __future__ import annotations


from typing import Any, ClassVar

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.fock import FockDevice


class KerrCavity(FockDevice):
    """Kerr-nonlinear resonator supporting cat-qubit stabilisation.

    Hamiltonian:

    .. math::

       H = \\omega \\, \\hat{n} - K \\, \\hat{n}(\\hat{n} - I)

    The nonlinearity :math:`K` shifts the photon-number eigenenergies and makes
    the cavity anharmonic. With a two-photon parametric drive at
    :math:`2\\omega`, the steady state becomes a cat state with amplitude
    :math:`\\alpha = \\sqrt{\\varepsilon_2 / K}`.

    Parameters
    ----------
    freq : float
        Positive cavity frequency :math:`\\omega` in GHz. Can be a JAX tracer
        for sweeps / gradients.
    kerr : float
        Non-negative Kerr nonlinearity :math:`K` in GHz. A positive value
        shifts even-photon levels downward. Usually 1–100 MHz in
        superconducting circuits.
    levels : int
        Fock-space truncation dimension. To prevent truncation artefacts,
        choose at least ``4 * (eps2 / K) + 10``. Default 30.
    label : str | None
        Human-readable label. ``None`` gives an automatic label
        ``kerr_cavity_0``, ``kerr_cavity_1``, …
    T1 : float or None, default None
        Energy-relaxation time in ns. ``None`` disables T1 relaxation.
    T2 : float or None, default None
        Total 0-1 coherence time in ns. If both are set, ``T2 <= 2*T1``.
    thermal_occupation : float or None, default None
        Dimensionless mean bath occupation. ``None`` disables absorption.

    Notes
    -----
    This Hamiltonian is diagonal in the Fock basis and does not itself define a
    computational subspace. With a two-photon parametric drive, you can
    engineer the steady state into a cat-code manifold spanned by the even and
    odd cat states :math:`|C^+_\\alpha\\rangle` and
    :math:`|C^-_\\alpha\\rangle`. Bit-flip errors in that manifold are
    exponentially suppressed, :math:`\\sim e^{-2|\\alpha|^2}`, in the
    stabilized regime. This class's inherited Pauli surface
    (:attr:`computational` is ``False``) addresses the bare Fock ``|0>``,
    ``|1>`` subspace. See :meth:`physics_notes` for the caveat.

    References
    ----------
    .. [1] Grimm et al., Nature 584, 205 (2020). arXiv:1907.12131.
    .. [2] Hajr et al., PRX Quantum 5, 020347 (2024). arXiv:2404.16697.

    Examples
    --------
    >>> from quchip.devices.kerr_cavity import KerrCavity
    >>> cav = KerrCavity(freq=5.0, kerr=1.0, levels=10, label="cav")
    >>> cav.freq, cav.kerr, cav.levels
    (5.0, 1.0, 10)
    """

    _type_prefix: ClassVar[str] = "kerr_cavity"
    _default_levels: ClassVar[int] = 30
    tunable_param_names = ("freq", "kerr")
    approximation = (
        "Kerr-nonlinear cavity effective single-mode model; "
        "SNAIL/STS-SQUID adiabatically eliminated."
    )
    # The inherited Pauli surface (sigma_x/y/z) addresses the bare Fock
    # |0>, |1> subspace, not the cat-code manifold |C+_alpha>, |C-_alpha>.
    # This class does not implement cat-basis Paulis.
    computational = False

    freq: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz", symbol=r"\omega")
    # Non-negative: a positive Kerr shifts even-photon levels downward.
    kerr: Scalar = parameter(default=UNBOUND, nonnegative=True, unit="GHz", symbol="K")

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        """Return :math:`H = \\omega \\hat{n} - K \\hat{n}(\\hat{n} - I)`.

        The Kerr term :math:`\\hat{n}(\\hat{n}-I) = \\hat{n}^2 - \\hat{n}`
        gives eigenvalue contributions :math:`-K n(n-1)` for the
        :math:`n`-photon Fock state.

        Returns
        -------
        PhysicsExpr
            Declarative expression for the Hermitian operator
            ``H = omega*n - K*n*(n-1)`` (GHz), diagonal in the Fock basis.
        Parameters
        ----------
        op : LocalOps
            Fock operator namespace.
        p : ParameterNamespace
            Symbolic ``freq`` and ``kerr`` values.
        """

        n = op.n
        return p.freq * n - p.kerr * (n @ (n - op.I))

    def physics_notes(self) -> list[str]:
        """Return declared Kerr-cavity approximation notes."""
        notes = super().physics_notes()
        notes.append("Kerr Hamiltonian: H = ω·n̂ − K·n̂(n̂−I)")
        notes.append(
            "computational=False: the inherited Pauli surface (sigma_x/y/z) addresses the "
            "bare Fock |0>, |1> subspace, not the cat-code manifold |C+_alpha>, |C-_alpha>; "
            "this class does not implement cat-basis Paulis."
        )
        return notes
