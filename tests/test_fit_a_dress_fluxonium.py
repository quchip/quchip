"""Tests that ``fit_a_dress`` fits a Fluxonium chip via ``tunable_params``/``tunable_param_bounds``."""

from __future__ import annotations

import pytest

from quchip import Capacitive, Chip, Resonator, fit_a_dress, set_default_backend
from quchip.devices.fluxonium import Fluxonium


@pytest.fixture
def _dynamiqs_backend():
    """Fluxonium's hamiltonian is JAX-native; fit dressing happens through dynamiqs."""
    pytest.importorskip("dynamiqs")
    set_default_backend("dynamiqs")
    yield
    set_default_backend("qutip")


def test_fluxonium_tunable_params_round_trip() -> None:
    """tunable_params/set_tunable_param/tunable_param_bounds round-trip E_C, E_J, E_L, phi_ext."""
    flux = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=4, num_basis=120)

    params = flux.tunable_params()
    assert set(params) == {"E_C", "E_J", "E_L", "phi_ext"}
    assert pytest.approx(float(params["E_J"])) == 4.0

    flux.set_tunable_param("E_J", 3.5)
    assert pytest.approx(float(flux.E_J)) == 3.5

    bounds = flux.tunable_param_bounds("E_J", 4.0)
    assert bounds[0] > 0 and bounds[1] > bounds[0]
    bounds = flux.tunable_param_bounds("phi_ext", 0.5)
    assert bounds == (-0.5, 0.5)


@pytest.mark.optional_backend
@pytest.mark.usefixtures("_dynamiqs_backend")
def test_fit_a_dress_recovers_fluxonium_dressed_freq() -> None:
    """fit_a_dress recovers a target fluxonium dressed frequency while holding the resonator anchor fixed."""
    flux = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=4, num_basis=120)
    res = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([flux, res], [Capacitive(flux, res, g=0.05)])
    seed_dressed_flux = float(chip.freq(flux))
    target_freq = seed_dressed_flux * 1.10

    # 6 free bare parameters (E_C, E_J, E_L, phi_ext, res.freq, coupling.g) against 3 target
    # residuals: underdetermined by count, and intentionally so — the point of this test is that
    # fit_a_dress moves *some* combination of the 4 fluxonium bare parameters to hit the target
    # without the caller having to pick which one via fit_parameters.
    with pytest.warns(UserWarning, match="underdetermined by count"):
        result = fit_a_dress(
            chip,
            observable_targets={flux: {"freq": target_freq}, res: {"freq": 7.0}},
            max_hilbert_dim=10_000,
        )

    fitted_chip = result.chip
    fitted_flux = result.rebind(flux)
    fitted_res = result.rebind(res)
    achieved = float(fitted_chip.freq(fitted_flux))
    assert achieved == pytest.approx(target_freq, rel=5e-3, abs=5e-3)
    # The resonator anchor holds; coupling does not drift it.
    assert float(fitted_chip.freq(fitted_res)) == pytest.approx(7.0, abs=5e-3)
    bare_keys = (
        f"{flux.label}.E_C",
        f"{flux.label}.E_J",
        f"{flux.label}.E_L",
        f"{flux.label}.phi_ext",
    )
    moved = any(
        abs(result.final_params[k] - result.initial_params[k]) > 1e-6 for k in bare_keys
    )
    assert moved, "optimizer never moved any fluxonium bare parameter"


@pytest.mark.optional_backend
@pytest.mark.usefixtures("_dynamiqs_backend")
def test_fit_a_dress_works_for_duffing_and_fluxonium_with_same_call() -> None:
    """fit_a_dress accepts DuffingTransmon and Fluxonium devices through the same call signature."""
    from quchip import DuffingTransmon

    duff = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="duff")
    flux = Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.5, levels=4, num_basis=120, label="flux")
    chip = Chip([duff, flux], [Capacitive(duff, flux, g=0.005)])

    duff_seed = float(chip.freq(duff))
    flux_seed = float(chip.freq(flux))

    # 7 free bare parameters (duff.freq, duff.anharmonicity, flux's E_C/E_J/E_L/phi_ext,
    # coupling.g) against 4 target residuals: underdetermined by count, expected for the same
    # reason as test_fit_a_dress_recovers_fluxonium_dressed_freq above.
    with pytest.warns(UserWarning, match="underdetermined by count"):
        result = fit_a_dress(
            chip,
            observable_targets={duff: {"freq": duff_seed}, flux: {"freq": flux_seed}},
            max_hilbert_dim=10_000,
        )

    fitted_chip = result.chip
    assert float(fitted_chip.freq(result.rebind(duff))) == pytest.approx(duff_seed, abs=5e-3)
    assert float(fitted_chip.freq(result.rebind(flux))) == pytest.approx(flux_seed, abs=5e-3)
