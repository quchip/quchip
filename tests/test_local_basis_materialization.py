from __future__ import annotations

import numpy as np

from quchip import ChargeBasisTransmon, Chip, Fluxonium
from quchip.utils.constants import TWO_PI


def test_charge_basis_device_stays_authored_while_engine_projects_locally() -> None:
    device = ChargeBasisTransmon(
        E_C=0.25,
        E_J=12.0,
        n_g=0.1,
        num_basis=9,
        basis="eigen",
        levels=3,
        label="q",
    )

    authored = device.hamiltonian().matrix()
    result = device.engine_result()
    basis = result.bases["q"]

    assert authored.shape == (9, 9)
    assert result.dims == (3,)
    assert basis.native_dim == 9
    assert basis.resolved_dim == 3
    np.testing.assert_allclose(
        result.matrix() / TWO_PI,
        basis.vectors.conj().T @ authored @ basis.vectors,
        atol=1e-10,
    )


def test_native_is_default_and_device_policy_overrides_the_chip() -> None:
    inherited = ChargeBasisTransmon(
        E_C=0.25,
        E_J=12.0,
        num_basis=9,
        levels=3,
        label="inherited",
    )
    native = ChargeBasisTransmon(
        E_C=0.25,
        E_J=12.0,
        num_basis=9,
        basis="native",
        label="native",
    )

    inherited_result = Chip([inherited], basis="eigen").engine_result()
    native_result = Chip([native], basis="eigen").engine_result()

    assert inherited_result.dims == (3,)
    assert inherited_result.bases["inherited"].kind == "eigen"
    assert native_result.dims == (9,)
    assert native_result.bases["native"].kind == "native"
    np.testing.assert_allclose(
        native_result.matrix() / TWO_PI,
        native.hamiltonian().matrix(),
        atol=1e-10,
    )


def test_fluxonium_uses_the_same_phase_grid_to_local_eigen_path() -> None:
    device = Fluxonium(
        E_C=1.0,
        E_J=4.0,
        E_L=0.9,
        phi_ext=0.5,
        num_basis=31,
        phi_max=3.0 * np.pi,
        basis="eigen",
        levels=4,
        label="flux",
    )

    result = Chip([device]).engine_result()

    assert device.hamiltonian().shape == (31, 31)
    assert result.dims == (4,)
    assert result.bases["flux"].vectors.shape == (31, 4)
