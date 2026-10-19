"""Calculate steady-state dispersive pointer states and detected information.

The model is a linear resonator with vacuum energy loss and fixed qubit state.
It excludes ring-up transients, thermal noise, and measurement-induced qubit T1.
``kappa`` is the total energy linewidth. ``kappa_detected`` is the observed
part, before detector efficiency. Elimination's intrinsic downward-decay
summary is not a general linewidth or detected rate.

For drive detuning ``delta_j = 2pi*(delta_r + j*chi_eff)``,
``alpha_j = -i*eps/(kappa/2 + i*delta_j)``. The ground-state photon number is
``nbar_0 = |alpha_0|²``. The dephasing rate is
``Gamma_m = kappa*|alpha_1-alpha_0|²/2``. Heterodyne detection gives
``SNR = |alpha_1-alpha_0|*sqrt(2*efficiency*kappa_detected*tau)``.

``chi`` is the full pull ``f_r|1-f_r|0`` in GHz. With
``Z=|0><0|-|1><1|``, the angular coefficient of ``Z*n_r`` is ``-pi*chi``.
The small-pull limit is ``8*(pi*chi)²*nbar_0/kappa`` divided by
``1+(4pi*delta_r/kappa)²``. It reduces to the usual expression on resonance.

The optional ``chi/(1+nbar_0/n_crit)`` interpolation is phenomenological.
It does not extend dispersive theory into a controlled strong-drive regime.
For a two-level Jaynes-Cummings model, ``n_crit=Delta²/(4g²)`` uses the resolved
exchange matrix element. Multilevel dispersive limits depend on both qubit
transition detunings and the occupied levels.

References
----------
Koch et al., PRA 76, 042319 (2007), section III B:
https://arxiv.org/abs/cond-mat/0703002.
Gambetta et al., PRA 74, 042318 (2006):
https://arxiv.org/abs/cond-mat/0602322.
Krantz et al., Applied Physics Reviews 6, 021318 (2019), section V:
https://arxiv.org/abs/1904.06560.
Blais et al., RMP 93, 025005 (2021), section III.3:
https://arxiv.org/abs/2005.12667.
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

    All numeric fields stay in the inputs' array namespace (JAX in, JAX out),
    so the full result traces under ``jax.jit``/``grad``. If ``n_crit`` was
    given, ``validity`` holds ``{"n_over_ncrit", "below_ncrit"}``, and
    otherwise is empty. Under jit/grad, ``below_ncrit`` is a *traced* boolean,
    so read it outside the traced region or branch with ``jnp.where``. Here,
    ``chi`` is the full pull ``f_r|1 − f_r|0`` in GHz.

    Attributes
    ----------
    pointer_states
        Complex ``α_j`` in the IQ plane, shape ``(levels,)``.
    photon_numbers
        Steady-state photons ``|α_j|²``, shape ``(levels,)``.
    sigma
        Integrated vacuum-noise width in the cavity amplitude plane. It is
        ``1/sqrt(2*efficiency*kappa_detected*tau)``, or infinity without detection.
    snr
        Pointer separation divided by ``sigma``.
    assignment_error
        ``½·erfc(SNR/(2√2))``, the optimal linear discriminant between two
        equal Gaussians.
    dephasing_rate
        Measurement-induced dephasing ``Γ_m = κ·|α₁ − α₀|²/2`` in 1/ns.
    chi_eff
        ``chi``, or ``chi/(1 + n̄₀/n_crit)`` when ``n_crit`` was given (GHz).
    validity
        See above.
    notes
        Physics that the analysis explicitly drops.
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

        Concrete values only. Call it outside ``jax.jit``/``grad`` regions.
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
    kappa_detected: Any = None,
    efficiency: Any = 1.0,
) -> DispersiveReadoutResult:
    """Calculate closed-form steady-state readout figures of merit from ``(chi, kappa)``.

    Parameters
    ----------
    chi
        Full pull ``f_r|1-f_r|0`` in ordinary GHz. Its signed Pauli coefficient
        is ``-chi/2`` for ``Z=|0><0|-|1><1|``. Elimination reports this full pull.
    kappa
        Positive total resonator energy linewidth in 1/ns, including observed
        and unobserved vacuum loss. Exclude pure dephasing. Elimination's
        intrinsic downward-decay diagnostic is not a general replacement.
    tau
        Positive integration time in ns.
    n_photons
        Nonnegative ground-state steady photon number, or None. Give exactly
        one of ``n_photons`` and ``eps``. This sets
        ``eps=sqrt(n_photons*((kappa/2)**2+(2*pi*delta_r)**2))``.
    eps
        Complex drive rate in rad/ns, or None. Give exactly one of ``eps`` and
        ``n_photons``. It is the coefficient driving the cavity amplitude.
    delta_r
        Real ``f_r|0-f_drive`` in GHz. The default zero drives the ground-state
        cavity resonance. Positive values place the drive below resonance.
    n_crit
        Positive critical-occupation estimate, or None by default. When given,
        apply the phenomenological interpolation ``chi/(1+nbar_0/n_crit)``.
        Report ``n_over_ncrit`` and ``below_ncrit`` without certifying validity.
    levels
        Static Python integer of at least two, default two. Level ``j`` has
        the assumed pull ``j*chi``. Higher conditional frequencies are not
        derived from a multilevel Hamiltonian.
    kappa_detected
        Observed energy-loss rate in 1/ns, between zero and ``kappa``.
        None defaults to ``kappa``, representing complete collection.
    efficiency
        Detection efficiency between zero and one, default one. Zero gives
        zero SNR, infinite ``sigma``, and assignment error one half.

    Returns
    -------
    DispersiveReadoutResult

    Raises
    ------
    ValueError
        If drive arguments are ambiguous, ``levels`` is invalid, or concrete
        numerical values are nonfinite or outside their stated domains.
        Numerical domain checks are skipped for tracers. Callers must keep
        traced inputs within the same physical domains.

    Examples
    --------
    >>> from quchip import analyze_dispersive_readout
    >>> ro = analyze_dispersive_readout(chi=0.002, kappa=0.005, tau=500.0, n_photons=2.0)
    >>> snr, p_err = ro.snr, ro.assignment_error

    Specify total and detected loss from the physical cavity and receiver.
    For example, ``kappa_detected=0.004`` with ``kappa=0.005`` leaves
    ``0.001/ns`` unobserved.
    """
    if (n_photons is None) == (eps is None):
        raise ValueError("Provide exactly one of n_photons (target photons) or eps (drive rate, rad/ns).")

    if not isinstance(levels, int) or isinstance(levels, bool) or levels < 2:
        raise ValueError("levels must be a Python integer of at least two.")
    if kappa_detected is None:
        kappa_detected = kappa
    domains = (
        ("kappa", kappa, True), ("tau", tau, True), ("n_crit", n_crit, True),
        ("n_photons", n_photons, False), ("kappa_detected", kappa_detected, False),
        ("efficiency", efficiency, False),
    )
    for name, value, positive in domains:
        if value is None or contains_tracer(value):
            continue
        array = np.asarray(value)
        if not np.isrealobj(array) or np.any(~np.isfinite(array)) or np.any(array <= 0 if positive else array < 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    if not contains_tracer(efficiency) and np.any(np.asarray(efficiency) > 1):
        raise ValueError("efficiency must not exceed one.")
    if not contains_tracer((kappa_detected, kappa)) and np.any(np.asarray(kappa_detected) > np.asarray(kappa)):
        raise ValueError("kappa_detected must not exceed kappa.")
    for name, value in (("chi", chi), ("delta_r", delta_r), ("eps", eps)):
        if value is not None and not contains_tracer(value) and np.any(~np.isfinite(np.asarray(value))):
            raise ValueError(f"{name} must be finite.")

    given = tuple(v for v in (chi, kappa, tau, n_photons, eps, delta_r, n_crit, kappa_detected, efficiency)
                  if v is not None)
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
        "Vacuum energy-loss channels and coherent pointer states assumed.",
    ]
    # Argument presence is static. The interpolation remains traced.
    if n_crit is None:
        chi_eff = chi
        validity: dict[str, Any] = {}
    else:
        n_over_ncrit = n_0 / n_crit
        chi_eff = chi / (1.0 + n_over_ncrit)
        validity = {"n_over_ncrit": n_over_ncrit, "below_ncrit": n_over_ncrit < 1.0}
        notes.append("Phenomenological interpolation applied: chi_eff = chi/(1 + n̄₀/n_crit).")

    j = xp.arange(levels)
    delta_j = TWO_PI * (delta_r + chi_eff * j)
    alpha = -1j * eps / (half_kappa + 1j * delta_j)

    separation = xp.abs(alpha[1] - alpha[0])
    collected = efficiency * kappa_detected
    snr = separation * xp.sqrt(2.0 * collected * tau)
    variance_rate = xp.where(collected > 0, 2.0 * collected * tau, 1.0)
    sigma = xp.where(collected > 0, 1.0 / xp.sqrt(variance_rate), xp.inf)
    return DispersiveReadoutResult(
        pointer_states=alpha,
        photon_numbers=xp.abs(alpha) ** 2,
        sigma=sigma,
        snr=snr,
        assignment_error=0.5 * _erfc(snr / (2.0 * np.sqrt(2.0)), xp),
        dephasing_rate=kappa * separation**2 / 2.0,
        chi_eff=chi_eff,
        validity=validity,
        notes=notes,
    )
