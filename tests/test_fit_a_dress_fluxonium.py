"""Tests that ``fit_a_dress`` fits a Fluxonium chip via ``tunable_params``/``tunable_param_bounds``."""

from __future__ import annotations

import pytest

from quchip import Capacitive, Chip, fit_a_dress, set_default_backend
from quchip.devices.fluxonium import Fluxonium


@pytest.fixture
def _dynamiqs_backend():
    """Fluxonium's hamiltonian is JAX-native; fit dressing happens through dynamiqs."""
    pytest.importorskip("dynamiqs")
    set_default_backend("dynamiqs")
    yield
    set_default_backend("qutip")


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
            constraints={
                duff: {"freq": duff_seed, "anharmonicity": float(chip.dressed_anharmonicity(duff))},
                flux: {"freq": flux_seed, "anharmonicity": float(chip.dressed_anharmonicity(flux))},
                chip.couplings[0]: {"cross_kerr": None},
            },
            vary={duff: ("freq", "anharmonicity"), flux: tuple(flux.tunable_params()), chip.couplings[0]: ("g",)},
            max_hilbert_dim=10_000,
        )

    fitted_chip = result.chip
    assert float(fitted_chip.freq(result.rebind(duff))) == pytest.approx(duff_seed, abs=5e-3)
    assert float(fitted_chip.freq(result.rebind(flux))) == pytest.approx(flux_seed, abs=5e-3)
