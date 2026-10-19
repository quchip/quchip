"""Dispersive pull, intrinsic decay diagnostics, and steady-state detected readout."""

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
    r = Resonator(freq=6.5, internal_quality_factor=5000.0, levels=6, label="r")
    return Chip([q, r], couplings=[Capacitive(q, r, g=g)]), q, r


@pytest.mark.validation
def test_eliminate_reports_kappa_matching_purcell_kappa():
    """eliminate() reports kappa = 2*pi*f_r/Q (1/ns), the same rate the Purcell fold uses."""
    chip, q, r = _readout_chip()
    res = eliminate(chip, r)
    assert float(res.effective_params["q"]["kappa"]) == pytest.approx(2 * np.pi * 6.5 / 5000.0, rel=1e-9)


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


@pytest.mark.parametrize("delta_r", [0.0, 0.005])
def test_dephasing_rate_matches_small_chi_limit(delta_r):
    """Small-pull dephasing includes the analytic detuning suppression."""
    from quchip.analysis import analyze_dispersive_readout

    # Z has positive ground eigenvalue, so chi_Z,angular = -pi*chi_pull.
    chi, kappa, nbar = 1e-7, 0.02, 5.0
    ro = analyze_dispersive_readout(chi=chi, kappa=kappa, tau=500.0, n_photons=nbar, delta_r=delta_r)
    expected = 8 * (np.pi * chi) ** 2 * nbar / kappa / (1 + (4 * np.pi * delta_r / kappa) ** 2)
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

    def collected_snr(efficiency):
        return analyze_dispersive_readout(chi=-5e-4, kappa=0.02, tau=500.0,
                                          n_photons=2.0, kappa_detected=0.01, efficiency=efficiency).snr

    value, slope = jax.jit(jax.value_and_grad(collected_snr))(0.5)
    # SNR scales as sqrt(efficiency), with all cavity dynamics held fixed.
    assert float(slope) == pytest.approx(float(value) / (2 * 0.5), rel=1e-12)


@pytest.mark.unit
def test_detected_loss_changes_information_but_not_pointer_states_or_dephasing():
    """Unobserved vacuum loss preserves pointer dynamics while reducing the detected SNR."""
    from quchip import analyze_dispersive_readout

    arguments = dict(chi=-5e-4, kappa=0.02, tau=500.0, n_photons=2.0)
    ideal = analyze_dispersive_readout(**arguments)
    lossy = analyze_dispersive_readout(**arguments, kappa_detected=0.01, efficiency=0.5)
    np.testing.assert_allclose(lossy.pointer_states, ideal.pointer_states, atol=1e-14)
    assert lossy.dephasing_rate == pytest.approx(ideal.dephasing_rate)
    assert lossy.snr == pytest.approx(ideal.snr / 2)
    assert lossy.sigma == pytest.approx(ideal.sigma * 2)
    dark = analyze_dispersive_readout(**arguments, kappa_detected=0.0)
    assert dark.snr == 0.0
    assert dark.assignment_error == 0.5
    assert np.isinf(dark.sigma)


@pytest.mark.unit
@pytest.mark.parametrize("name,value", [
    ("kappa", 0.0), ("tau", -1.0), ("n_photons", -1.0), ("n_crit", 0.0),
    ("levels", 1), ("levels", 2.5), ("kappa_detected", -0.01),
    ("kappa_detected", 0.03), ("efficiency", 1.1), ("kappa", np.nan),
])
def test_readout_rejects_invalid_concrete_domains(name, value):
    """Concrete parameters outside the model's physical domains raise before calculation."""
    from quchip import analyze_dispersive_readout

    arguments = dict(chi=-5e-4, kappa=0.02, tau=500.0, n_photons=2.0)
    arguments[name] = value
    with pytest.raises(ValueError, match=name):
        analyze_dispersive_readout(**arguments)
