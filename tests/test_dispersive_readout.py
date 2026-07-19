"""Dispersive readout — χ/κ reporting in ``eliminate()`` and the closed-form module.

χ convention (χ_pull) used throughout: ``chi = f_r(qubit in |1⟩) − f_r(qubit in |0⟩)``
in GHz — the *full* resonator pull per qubit excitation, which is **2×** the σ_z
convention ``H_disp = (ω_r + χ σ_z) a†a`` of most textbooks.
"""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Resonator
from quchip.chip.transformations import eliminate


# ---------------------------------------------------------------------------
# Part A — χ/κ reported by eliminate()
# ---------------------------------------------------------------------------


def _readout_chip(g=0.05):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=6.5, quality_factor=5000.0, levels=6, label="r")
    return Chip([q, r], couplings=[Capacitive(q, r, g=g)]), q, r


def test_eliminate_reports_chi_matching_dressed_pull_and_duffing_formula():
    """eliminate() reports chi exactly matching the dressed pull and the Duffing formula to ~5%."""
    g = 0.05
    chip, q, r = _readout_chip(g)
    # The exact reference: dressed pull computed directly on the full chip.
    pull = chip.freq(r, when={q: 1}) - chip.freq(r, when={q: 0})

    res = eliminate(chip, r)
    chi = res.effective_params["q"]["chi"]

    # Exact: the rule uses the same dressed-spectrum kernel.
    assert float(chi) == pytest.approx(float(pull), rel=1e-9)

    # Loose: the Duffing analytic χ = 2g²α/(Δ(Δ+α)) (Koch et al., PRA 76,
    # 042319, §IV) is 2nd-order dispersive; the numeric pull carries
    # higher-order corrections, so agreement is only to ~few %.
    delta = 5.0 - 6.5
    alpha = -0.25
    chi_analytic = 2 * g**2 * alpha / (delta * (delta + alpha))
    assert float(chi) == pytest.approx(chi_analytic, rel=0.05)


def test_eliminate_reports_kappa_matching_purcell_kappa():
    """eliminate() reports kappa = 2*pi*f_r/Q (1/ns), the same rate the Purcell fold uses."""
    chip, q, r = _readout_chip()
    res = eliminate(chip, r)
    assert float(res.effective_params["q"]["kappa"]) == pytest.approx(2 * np.pi * 6.5 / 5000.0, rel=1e-9)


def test_eliminate_without_quality_factor_reports_kappa_zero():
    """A resonator with no quality_factor set reports kappa = 0."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=6.5, levels=4, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05)])
    res = eliminate(chip, "r")
    assert res.effective_params["q"]["kappa"] == 0.0


# ---------------------------------------------------------------------------
# Part B — closed-form readout analysis (steady-state pointer states)
# ---------------------------------------------------------------------------


def test_pointer_states_collapse_and_snr_vanishes_as_chi_to_zero():
    """At chi=0 the pointer states coincide, SNR vanishes, and assignment error saturates at 0.5."""
    from quchip.analysis import analyze_dispersive_readout

    ro = analyze_dispersive_readout(chi=0.0, kappa=0.02, tau=500.0, n_photons=2.0)
    assert np.allclose(ro.pointer_states[0], ro.pointer_states[1])
    assert float(ro.snr) == pytest.approx(0.0, abs=1e-12)
    assert float(ro.assignment_error) == pytest.approx(0.5, rel=1e-9)
    assert any("steady-state" in note.lower() for note in ro.notes)


def test_photon_number_reproduces_requested_n_photons():
    """The requested n_photons is reproduced whether driven by n_photons or the equivalent eps."""
    from quchip.analysis import analyze_dispersive_readout

    for delta_r in (0.0, 0.003):
        ro = analyze_dispersive_readout(chi=-5e-4, kappa=0.02, tau=500.0, n_photons=2.0, delta_r=delta_r)
        assert float(ro.photon_numbers[0]) == pytest.approx(2.0, rel=1e-9)

    eps = np.sqrt(2.0 * ((0.02 / 2) ** 2 + (2 * np.pi * 0.003) ** 2))
    ro2 = analyze_dispersive_readout(chi=-5e-4, kappa=0.02, tau=500.0, eps=eps, delta_r=0.003)
    assert float(ro2.photon_numbers[0]) == pytest.approx(2.0, rel=1e-9)


def test_snr_grows_as_sqrt_tau():
    """SNR scales with sqrt(tau): a 4x integration time gives a 2x SNR."""
    from quchip.analysis import analyze_dispersive_readout

    kw = dict(chi=-5e-4, kappa=0.02, n_photons=2.0)
    snr_1 = float(analyze_dispersive_readout(tau=400.0, **kw).snr)
    snr_4 = float(analyze_dispersive_readout(tau=1600.0, **kw).snr)
    assert snr_4 / snr_1 == pytest.approx(2.0, rel=1e-9)


def test_dephasing_rate_matches_small_chi_limit():
    """Small-chi dephasing rate matches Gamma_m = 8*(chi_sigma_z,ang)^2*nbar/kappa (Gambetta PRA 74, 042318)."""
    from quchip.analysis import analyze_dispersive_readout

    # χ_pull = 2χ_σz, so the angular σ_z-convention χ is 2π·(chi/2) = π·chi.
    chi, kappa, nbar = 1e-5, 0.02, 5.0
    ro = analyze_dispersive_readout(chi=chi, kappa=kappa, tau=500.0, n_photons=nbar)
    expected = 8 * (np.pi * chi) ** 2 * nbar / kappa
    assert float(ro.dephasing_rate) == pytest.approx(expected, rel=1e-3)


def test_strong_drive_chi_collapse_with_n_crit():
    """n_crit collapses chi_eff by 1/(1+n/n_crit) and reduces SNR relative to the uncollapsed case."""
    from quchip.analysis import analyze_dispersive_readout

    ro = analyze_dispersive_readout(chi=-5e-4, kappa=0.02, tau=500.0, n_photons=2.0, n_crit=8.0)
    assert float(ro.chi_eff) == pytest.approx(-5e-4 / (1 + 2.0 / 8.0), rel=1e-9)
    assert float(ro.validity["n_over_ncrit"]) == pytest.approx(0.25, rel=1e-9)
    assert bool(ro.validity["below_ncrit"]) is True
    ro0 = analyze_dispersive_readout(chi=-5e-4, kappa=0.02, tau=500.0, n_photons=2.0)
    assert float(ro.snr) < float(ro0.snr)


def test_levels_controls_pointer_state_count():
    """levels sets the number of reported pointer states and photon numbers."""
    from quchip.analysis import analyze_dispersive_readout

    ro = analyze_dispersive_readout(chi=-5e-4, kappa=0.02, tau=500.0, n_photons=2.0, levels=3)
    assert np.shape(ro.pointer_states) == (3,)
    assert np.shape(ro.photon_numbers) == (3,)


def test_exactly_one_of_n_photons_and_eps_is_required():
    """Passing neither or both of n_photons/eps raises ValueError."""
    from quchip.analysis import analyze_dispersive_readout

    with pytest.raises(ValueError, match="xactly one"):
        analyze_dispersive_readout(chi=-5e-4, kappa=0.02, tau=500.0)
    with pytest.raises(ValueError, match="xactly one"):
        analyze_dispersive_readout(chi=-5e-4, kappa=0.02, tau=500.0, n_photons=2.0, eps=0.01)


def test_module_is_jax_traceable_and_differentiable():
    """snr is differentiable and jit-compilable with respect to chi."""
    import jax
    import jax.numpy as jnp

    from quchip.analysis import analyze_dispersive_readout

    def snr(chi):
        return analyze_dispersive_readout(chi=chi, kappa=0.02, tau=500.0, n_photons=2.0).snr

    grad = jax.grad(snr)(jnp.float64(-5e-4))
    assert jnp.isfinite(grad)
    assert float(grad) != 0.0
    jitted = float(jax.jit(snr)(jnp.float64(-5e-4)))
    assert jitted == pytest.approx(float(snr(jnp.float64(-5e-4))), rel=1e-12)
