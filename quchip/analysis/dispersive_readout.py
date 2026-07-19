"""Closed-form dispersive-readout analysis — pointer states, SNR, assignment error.

Maps the dispersive parameters of a qubit–resonator pair — the dispersive pull
``chi`` and resonator linewidth ``kappa`` — plus a readout drive and an
integration time to steady-state figures of merit, with no solver run and no
resonator in the Hilbert space. Composes with
:func:`quchip.chip.transformations.eliminate`, which reports ``chi``/``kappa``
per surviving qubit::

    res = eliminate(chip, "readout_res")
    ro = analyze_dispersive_readout(
        chi=res.effective_params["q0"]["chi"],
        kappa=res.effective_params["q0"]["kappa"],
        tau=500.0, n_photons=2.0,
    )
    ro.snr, ro.assignment_error

Physics (driven, damped linear resonator, steady state of
``d⟨a⟩/dt = −(iδ + κ/2)⟨a⟩ − iε``):

    delta_r ≡ f_r|0 − f_drive            drive placement  [GHz]  (Δ_r = ω_r − ω_d)
    δ_j   = 2π·(delta_r + chi_eff·j)     resonator−drive detuning, qubit in |j⟩  [rad/ns]
    α_j   = −i·ε / (κ/2 + i·δ_j)         coherent pointer state
    n̄_j  = |α_j|²                        steady-state photons (emergent)
    σ     = 1/√(2κτ)                     integrated vacuum-noise blob width
    SNR   = |α₁ − α₀|·√(2κτ)
    p_err = ½·erfc(SNR/(2√2))            two equal Gaussians, optimal discriminant
    Γ_m   = κ·|α₁ − α₀|²/2               measurement-induced dephasing  [1/ns]

χ convention: ``chi`` is the *full* pull ``χ_pull ≡ f_r(qubit |1⟩) − f_r(qubit
|0⟩)`` in GHz — **2×** the σ_z-convention χ of ``H_disp = (ω_r + χσ_z)a†a``.
In the small-χ limit ``Γ_m → 8·χ_σz²·n̄/κ`` with ``χ_σz = π·chi`` in rad/ns.

Unit convention: public inputs ``chi`` and ``delta_r`` are GHz (ordinary
frequency), ``kappa`` is a rate in 1/ns, ``tau`` is in ns, ``eps`` is in
rad/ns. The GHz→rad/ns conversions (2π) happen exactly once, at the public
boundary of :func:`analyze_dispersive_readout` — local physics conversions of
an analysis module, distinct from the engine's own Hamiltonian-assembly 2π
boundary in stage 2.

Everything is closed-form algebra in the array namespace of its inputs, so the
result is JAX-traceable and differentiable end-to-end:
``jax.grad`` of ``result.snr`` with respect to any chip parameter works when
``chi``/``kappa`` come from a traced :func:`eliminate`.

Approximations (declared explicitly): steady state only (no ring-up
transient), linear resonator, 2nd-order dispersive coupling, no
measurement-induced qubit T1. The optional strong-drive correction
``chi_eff = chi/(1 + n̄₀/n_crit)`` is applied only when ``n_crit`` is given.

References
----------
Koch et al., *Charge-insensitive qubit design derived from the Cooper pair
box*, PRA 76, 042319 (2007), §IV — dispersive χ for the transmon.
Gambetta et al., *Qubit-photon interactions in a cavity: Measurement-induced
dephasing and number splitting*, PRA 74, 042318 (2006) — Γ_m.
Krantz et al., *A quantum engineer's guide to superconducting qubits*,
Appl. Phys. Rev. 6, 021318 (2019), §V — dispersive readout, SNR, p_err.
Blais et al., *Circuit quantum electrodynamics*, RMP 93, 025005 (2021) —
general cQED readout theory, n_crit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import erfc as _scipy_erfc

from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer, is_jax_array, is_jax_namespace


def _erfc(x: Any, xp: Any) -> Any:
    """Complementary error function in the given array namespace."""
    if is_jax_namespace(xp):
        from jax.scipy.special import erfc

        return erfc(x)
    return _scipy_erfc(x)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispersiveReadoutResult:
    """Store steady-state dispersive-readout figures of merit.

    All numeric fields stay in the array namespace of the inputs (JAX in, JAX
    out), so the whole result traces under ``jax.jit``/``grad``. ``validity``
    holds ``{"n_over_ncrit", "below_ncrit"}`` when ``n_crit`` was given —
    ``below_ncrit`` is then a *traced* boolean under jit/grad; read it outside
    the traced region or branch with ``jnp.where`` — and is empty otherwise.
    ``chi`` here is χ_pull (``f_r|1 − f_r|0``, 2× the σ_z-convention χ).

    Attributes
    ----------
    pointer_states
        Complex ``α_j`` in the IQ plane, shape ``(levels,)``.
    photon_numbers
        Steady-state photons ``|α_j|²``, shape ``(levels,)``.
    sigma
        Integrated vacuum-noise blob width ``1/√(2κτ)`` (dimensionless,
        α-plane units).
    snr
        ``|α₁ − α₀|·√(2κτ)``.
    assignment_error
        ``½·erfc(SNR/(2√2))`` — optimal linear discriminant between two equal
        Gaussians.
    dephasing_rate
        Measurement-induced dephasing ``Γ_m = κ·|α₁ − α₀|²/2`` in 1/ns.
    chi_eff
        ``chi``, or ``chi/(1 + n̄₀/n_crit)`` when ``n_crit`` was given (GHz).
    validity
        See above.
    notes
        Explicitly dropped physics.
    """

    pointer_states: Any
    photon_numbers: Any
    sigma: Any
    snr: Any
    assignment_error: Any
    dephasing_rate: Any
    chi_eff: Any
    validity: dict[str, Any]
    notes: list[str]

    def summary(self) -> str:
        """Print and return a formatted summary.

        Concrete values only — call it outside ``jax.jit``/``grad`` regions.
        """
        alphas = np.asarray(self.pointer_states)
        photons = np.asarray(self.photon_numbers)
        lines = ["Dispersive readout (steady state):"]
        lines.append("  pointer states α_j = " + ", ".join(f"{a.real:+.4f}{a.imag:+.4f}j" for a in alphas))
        lines.append("  photons n̄_j = " + ", ".join(f"{n:.4f}" for n in photons))
        lines.append(f"  SNR = {float(self.snr):.4f}  (blob width σ = {float(self.sigma):.4f})")
        lines.append(f"  assignment error = {float(self.assignment_error):.3e}")
        lines.append(f"  dephasing Γ_m = {float(self.dephasing_rate):.3e} 1/ns")
        text = "\n".join(lines)
        print(text)
        return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_dispersive_readout(
    chi: Any,
    kappa: Any,
    tau: Any,
    *,
    n_photons: Any = None,
    eps: Any = None,
    delta_r: Any = 0.0,
    n_crit: Any = None,
    levels: int = 2,
) -> DispersiveReadoutResult:
    """Compute closed-form steady-state readout figures of merit from ``(chi, kappa)``.

    Parameters
    ----------
    chi
        Dispersive pull χ_pull ``= f_r(qubit |1⟩) − f_r(qubit |0⟩)`` in GHz —
        2× the σ_z-convention χ. Take it from
        ``eliminate(...).effective_params[qubit]["chi"]``.
    kappa
        Resonator linewidth (rate) in 1/ns, e.g. ``effective_params[...]["kappa"]``.
    tau
        Integration time in ns.
    n_photons
        Target steady-state photon number with the qubit in |0⟩. Exactly one
        of ``n_photons`` and ``eps`` must be given; the drive rate is then
        ``ε = √(n̄₀·((κ/2)² + δ₀²))``.
    eps
        Readout drive rate in rad/ns (power-user path; exactly one of
        ``n_photons`` and ``eps``).
    delta_r
        Detuning of the qubit-in-ground resonator from the drive,
        ``delta_r = f_r|0 − f_drive`` in GHz — the literature's
        ``Δ_r = ω_r − ω_d``. ``0.0`` drives on the qubit-in-ground resonance;
        positive values place the drive *below* ``f_r|0``.
    n_crit
        Critical photon number ``Δ²/(4g²)``. When given, the strong-drive
        collapse ``chi_eff = chi/(1 + n̄₀/n_crit)`` is applied and
        ``validity`` reports ``n_over_ncrit``/``below_ncrit``.
    levels
        Number of qubit levels to compute pointer states for (static Python
        int — it fixes array shapes; 3 includes ``|f⟩``). The pull of level
        ``j`` is the linear-dispersive ``chi·j``.

    Returns
    -------
    DispersiveReadoutResult

    Raises
    ------
    ValueError
        If both or neither of ``n_photons`` and ``eps`` are given (a *static*
        argument-presence check — never a traced-value comparison).

    Examples
    --------
    >>> from quchip import analyze_dispersive_readout
    >>> ro = analyze_dispersive_readout(chi=0.002, kappa=0.005, tau=500.0, n_photons=2.0)
    >>> snr, p_err = ro.snr, ro.assignment_error

    ``chi`` and ``kappa`` are typically taken from
    ``eliminate(chip, "readout_res").effective_params[qubit]``.
    """
    if (n_photons is None) == (eps is None):
        raise ValueError("Provide exactly one of n_photons (target photons) or eps (drive rate, rad/ns).")

    given = tuple(v for v in (chi, kappa, tau, n_photons, eps, delta_r, n_crit) if v is not None)
    xp: Any = np
    if any(is_jax_array(v) for v in given) or contains_tracer(given):
        import jax.numpy as jnp

        xp = jnp

    # The single GHz -> rad/ns boundary of this module (ordinary -> angular).
    delta_0 = TWO_PI * delta_r
    half_kappa = kappa / 2.0
    if eps is None:
        eps = xp.sqrt(n_photons * (half_kappa**2 + delta_0**2))

    alpha_0 = -1j * eps / (half_kappa + 1j * delta_0)
    n_0 = xp.abs(alpha_0) ** 2

    notes = [
        "Steady-state pointer states only: ring-up transients not modeled.",
        "Linear resonator, 2nd-order dispersive approximation.",
        "Measurement-induced qubit T1 not modeled.",
    ]
    # n_crit presence is a static shape/structure decision; the correction
    # itself stays traced.
    if n_crit is None:
        chi_eff = chi
        validity: dict[str, Any] = {}
    else:
        n_over_ncrit = n_0 / n_crit
        chi_eff = chi / (1.0 + n_over_ncrit)
        validity = {"n_over_ncrit": n_over_ncrit, "below_ncrit": n_over_ncrit < 1.0}
        notes.append("Strong-drive correction applied: chi_eff = chi/(1 + n̄₀/n_crit).")

    j = xp.arange(levels)
    delta_j = TWO_PI * (delta_r + chi_eff * j)
    alpha = -1j * eps / (half_kappa + 1j * delta_j)

    separation = xp.abs(alpha[1] - alpha[0])
    snr = separation * xp.sqrt(2.0 * kappa * tau)
    return DispersiveReadoutResult(
        pointer_states=alpha,
        photon_numbers=xp.abs(alpha) ** 2,
        sigma=1.0 / xp.sqrt(2.0 * kappa * tau),
        snr=snr,
        assignment_error=0.5 * _erfc(snr / (2.0 * np.sqrt(2.0)), xp),
        dephasing_rate=kappa * separation**2 / 2.0,
        chi_eff=chi_eff,
        validity=validity,
        notes=notes,
    )
