from __future__ import annotations

import numpy as np
import pytest

from quchip import (
    Bath,
    ChargeBasisTransmon,
    ChargeDrive,
    Chip,
    Capacitive,
    ControlEquipment,
    Fluxonium,
    Resonator,
    Square,
)
from quchip.engine import build_problem
from quchip.engine.ir import DriveOp


def test_charge_basis_device_exposes_unresolved_and_projected_hamiltonians() -> None:
    device = ChargeBasisTransmon(
        E_C=0.25,
        E_J=12.0,
        n_g=0.1,
        num_basis=9,
        basis="eigen",
        levels=3,
        label="q",
    )

    authored = device.unresolved_hamiltonian().matrix()
    resolved = device.hamiltonian().matrix()
    result = device.resolve()
    basis = result.bases["q"]

    assert authored.shape == (9, 9)
    assert resolved.shape == (3, 3)
    assert result.dims == (3,)
    assert basis.native_dim == 9
    assert basis.resolved_dim == 3
    np.testing.assert_allclose(
        result.hamiltonian().matrix(),
        basis.vectors.conj().T @ authored @ basis.vectors,
        atol=1e-10,
    )
    np.testing.assert_allclose(resolved, result.hamiltonian().matrix(), atol=1e-10)

    full_basis = ChargeBasisTransmon(
        E_C=0.25,
        E_J=12.0,
        num_basis=3,
        basis="eigen",
        levels=3,
        label="full",
    )
    full_chip = Chip([full_basis])
    authored_state = np.asarray([1.0, 0.0, 0.0])
    problem = build_problem(
        full_chip,
        [],
        np.asarray([0.0, 1.0]),
        initial_state=authored_state,
    )
    full_record = problem.engine_result.bases["full"]
    np.testing.assert_allclose(
        full_chip.backend.to_array(problem.initial_state).reshape(-1),
        full_record.vectors.conj().T @ authored_state,
        atol=1e-10,
    )
    dressed_ground = full_chip.backend.to_array(full_chip.state({full_basis: 0})).reshape(-1)
    assert abs(dressed_ground[0]) == pytest.approx(1.0, abs=1e-10)

    solver_state = full_chip.bare_state({full_basis: 0})
    solver_problem = build_problem(
        full_chip,
        [],
        np.asarray([0.0, 1.0]),
        initial_state=solver_state,
    )
    np.testing.assert_allclose(
        full_chip.backend.to_array(solver_problem.initial_state),
        full_chip.backend.to_array(solver_state),
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

    inherited_result = Chip([inherited], basis="eigen").resolve()
    native_result = Chip([native], basis="eigen").resolve()

    assert inherited_result.dims == (3,)
    assert inherited_result.bases["inherited"].kind == "eigen"
    assert native_result.dims == (9,)
    assert native_result.bases["native"].kind == "native"
    np.testing.assert_allclose(
        native_result.hamiltonian().matrix(),
        native.unresolved_hamiltonian().matrix(),
        atol=1e-10,
    )
    native_chip = Chip([native])
    native_ground = native_chip.backend.to_array(native_chip.state({native: 0})).reshape(-1)
    np.testing.assert_allclose(
        abs(np.vdot(native_result.bases["native"].energy_vectors[:, 0], native_ground)),
        1.0,
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

    result = Chip([device]).resolve()

    assert device.unresolved_hamiltonian().shape == (31, 31)
    assert device.hamiltonian().shape == (4, 4)
    assert result.dims == (4,)
    assert result.bases["flux"].vectors.shape == (31, 4)


def test_attached_physics_uses_one_resolved_local_basis() -> None:
    device = ChargeBasisTransmon(
        E_C=0.25,
        E_J=12.0,
        num_basis=9,
        basis="eigen",
        levels=3,
        label="q",
        T1=30_000.0,
        coupling_channel="charge",
    )
    drive = ChargeDrive(device, label="xy")
    resonator = Resonator(freq=7.0, levels=2, label="r")
    coupling = Capacitive(device, resonator, g=0.02, rwa=False)
    chip = Chip(
        [device, resonator],
        couplings=[coupling],
        control_equipment=ControlEquipment([drive]),
        baths=[Bath("collective_decay", targets=[device, resonator], rate=1e-4)],
    )
    pulse = DriveOp(
        target_label="q",
        drive_label="xy",
        envelope=Square(duration=10.0, amplitude=0.01),
        freq=5.0,
        start_time=0.0,
    )

    problem = build_problem(
        chip,
        [pulse],
        np.linspace(0.0, 10.0, 11),
        initial_state={device: 1, resonator: 0},
        e_ops=chip.e_ops(q=["n", "charge"]),
    )

    assert problem.engine_result.dims == (3, 2)
    assert all(term.operator.shape == (6, 6) for term in problem.engine_result.dynamic_terms)
    assert all(term.operator.shape == (6, 6) for term in problem.engine_result.collapse_terms)
    assert any(term.channel == "collective_decay" for term in problem.engine_result.collapse_terms)
    assert all(np.asarray(chip.backend.to_array(op)).shape == (6, 6) for op in problem.e_ops)
    assert np.asarray(chip.backend.to_array(problem.initial_state)).shape == (6, 1)
    matrix = np.asarray(problem.engine_result.hamiltonian().matrix(t=0.0))
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-10)
