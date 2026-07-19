"""KerrCavity — Kerr-nonlinear resonator model.

Hamiltonian:

.. math::

   H = \\omega \\, \\hat{n} - K \\, \\hat{n}(\\hat{n} - I)

where :math:`\\omega` is the cavity frequency (GHz, ordinary) and :math:`K`
is the Kerr nonlinearity (GHz, positive).  Eigenvalues are:

.. math::

   E_n = \\omega n - K n(n-1)

The Kerr term shifts higher Fock levels down by :math:`K` per pair of
photons, creating the anharmonic energy ladder that stabilises cat states
when combined with a two-photon parametric drive.

Approximation
-------------
This is an effective single-mode model after adiabatic elimination of the
SNAIL or STS-SQUID that provides the nonlinearity.  The Kerr coefficient
:math:`K` captures the leading-order nonlinearity; higher-order corrections
are neglected.  The Hilbert space is truncated at ``levels`` Fock states —
choose ``levels >= 4 * (eps2 / K) + 10`` to avoid truncation artefacts.

References
----------
.. [1] Grimm et al., *Stabilization and operation of a Kerr-cat qubit*,
       Nature 584, 205 (2020). arXiv:1907.12131.
.. [2] Hajr et al., *High-Coherence Kerr-Cat Qubit in 2D Architecture*,
       PRX Quantum 5, 020347 (2024). arXiv:2404.16697.
"""

from __future__ import annotations


from typing import TYPE_CHECKING

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import DeviceModel
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import Scalar, parameter


class KerrCavity(DeviceModel):
    """Kerr-nonlinear resonator supporting cat-qubit stabilisation.

    Hamiltonian:

    .. math::

       H = \\omega \\, \\hat{n} - K \\, \\hat{n}(\\hat{n} - I)

    The nonlinearity :math:`K` shifts the photon-number eigenenergies,
    making the cavity anharmonic.  Combined with a two-photon parametric
    drive at :math:`2\\omega`, the steady state becomes a cat state with
    amplitude :math:`\\alpha = \\sqrt{\\varepsilon_2 / K}`.

    Parameters
    ----------
    freq : float
        Cavity frequency :math:`\\omega` in GHz.  Must be positive.
        May be a JAX tracer for sweeps / gradients.
    kerr : float
        Kerr nonlinearity :math:`K` in GHz.  Non-negative; positive
        value shifts even-photon levels downward.  Typically 1–100 MHz
        in superconducting circuits.
    levels : int
        Fock-space truncation dimension.  Choose at least
        ``4 * (eps2 / K) + 10`` to avoid truncation artefacts.
        Default 30.
    label : str | None
        Human-readable label.  ``None`` → auto-generated
        ``kerr_cavity_0``, ``kerr_cavity_1``, …
    **noise_kwargs
        Forwarded to :class:`~quchip.devices.base.BaseDevice`:
        ``T1``, ``T2``, ``thermal_population``, etc.

    Notes
    -----
    This Hamiltonian is diagonal in the Fock basis and does not itself
    define a computational subspace. Combined with a two-photon parametric
    drive, the steady state can be engineered into a cat-code manifold
    spanned by the even cat state :math:`|C^+_\\alpha\\rangle` and the odd
    cat state :math:`|C^-_\\alpha\\rangle`. Bit-flip errors within that
    manifold are exponentially suppressed, :math:`\\sim e^{-2|\\alpha|^2}`,
    in the stabilized regime. This class's inherited Pauli surface
    (:attr:`computational` is ``False``) addresses the bare Fock ``|0>``,
    ``|1>`` subspace; see :meth:`physics_notes` for the caveat.

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

    _type_prefix: str = "kerr_cavity"
    _default_levels: int = 30
    tunable_param_names = ("freq", "kerr")
    approximation = (
        "Kerr-nonlinear cavity effective single-mode model; "
        "SNAIL/STS-SQUID adiabatically eliminated."
    )
    # The inherited Pauli surface (sigma_x/y/z) addresses the bare Fock
    # |0>, |1> subspace, not the cat-code manifold |C+_alpha>, |C-_alpha>.
    # This class does not implement cat-basis Paulis.
    computational = False

    freq: Scalar = parameter(positive=True, unit="GHz")
    # Non-negative: a positive Kerr shifts even-photon levels downward.
    kerr: Scalar = parameter(nonnegative=True, unit="GHz")

    # --- generated __init__ stub (tools/gen_device_stubs.py); do not edit ---
    if TYPE_CHECKING:
        def __init__(
            self,
            freq: Scalar,
            kerr: Scalar,
            *,
            levels: int = 30,
            label: str | None = None,
            T1: float | None = None,
            T2: float | None = None,
            thermal_population: float | None = None,
        ) -> None: ...
    # --- end generated stub ---

    def local_hamiltonian(self, op: LocalOps) -> PhysicsExpr:
        """Return :math:`H = \\omega \\hat{n} - K \\hat{n}(\\hat{n} - I)`.

        The Kerr term :math:`\\hat{n}(\\hat{n}-I) = \\hat{n}^2 - \\hat{n}`
        gives eigenvalue contributions :math:`-K n(n-1)` for the
        :math:`n`-photon Fock state.

        Returns
        -------
        PhysicsExpr
            Declarative expression for the Hermitian operator
            ``H = omega*n - K*n*(n-1)`` (GHz), diagonal in the Fock basis.
        """
        n = op.n
        return self.freq * n - self.kerr * (n @ (n - op.I))

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
