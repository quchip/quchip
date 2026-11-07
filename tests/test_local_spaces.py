from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from quchip import ChargeSpace, CustomSpace, FockSpace, LocalOps, PhaseGridSpace
from quchip.devices.fluxonium import Fluxonium
from quchip.devices.transmon.charge_basis import ChargeBasisTransmon
from quchip.engine.basis import resolve_local_basis


@pytest.mark.unit
def test_charge_space_materializes_charge_and_josephson_operators() -> None:
    space = ChargeSpace(num_basis=5)
    op = LocalOps(label="q", space=space)

    shifted_charge = op.n - 0.2 * op.I
    matrix = (shifted_charge @ shifted_charge - 2.0 * op.cos_phi).matrix()

    charges = np.arange(-2, 3, dtype=float)
    expected = np.diag((charges - 0.2) ** 2)
    expected -= np.eye(5, k=1) + np.eye(5, k=-1)
    np.testing.assert_allclose(matrix, expected)


@pytest.mark.unit
def test_phase_grid_space_materializes_phase_and_charge_operators() -> None:
    space = PhaseGridSpace(points=5, extent=2.0)
    op = LocalOps(label="q", space=space)

    phase = op.phi.matrix()
    charge = op.n.matrix()

    grid = np.linspace(-2.0, 2.0, 5, endpoint=False)
    np.testing.assert_allclose(np.diag(phase), grid)
    np.testing.assert_allclose(charge, charge.conj().T)
    np.testing.assert_allclose(space.matrix("sin_phi"), np.diag(np.sin(grid)), atol=1e-15)
    np.testing.assert_allclose(space.matrix("n2"), -(np.eye(5, k=1) - 2 * np.eye(5) + np.eye(5, k=-1)) / 0.8**2)


@pytest.mark.unit
@pytest.mark.parametrize("space, name", [(PhaseGridSpace(5, 2.0), "cos_phi"), (PhaseGridSpace(5, 2.0), "n"),
                                         (ChargeSpace(5), "sin_phi"), (FockSpace(3), "sigma_y")])
def test_concrete_space_matrices_are_read_only(space, name) -> None:
    with pytest.raises(ValueError):
        space.matrix(name)[0, 0] = 1.0


@pytest.mark.validation
@pytest.mark.parametrize("device, value", [
    (lambda x: Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=x, levels=3, num_basis=60), 0.3),
    (lambda x: Fluxonium(E_C=1.0, E_J=4.0, E_L=1.0, phi_ext=0.3, levels=3, num_basis=60, phi_max=x), 15.0),
    (lambda x: ChargeBasisTransmon(E_C=0.3, E_J=6.0, n_g=x, levels=3, num_basis=21), 0.2),
])
def test_space_device_frequency_gradients_match_central_differences(device, value) -> None:
    gradient = jax.grad(lambda x: device(x).freq)(value)
    step = 1e-5
    reference = (device(value + step).freq - device(value - step).freq) / (2 * step)
    assert gradient == pytest.approx(reference, rel=1e-5, abs=1e-9)


@pytest.mark.unit
def test_custom_space_accepts_named_matrices_and_pure_jax_callables() -> None:
    class ArrayBackend:
        @staticmethod
        def from_array(value, *, dims):
            _ = dims
            return value

        @staticmethod
        def to_array(value):
            return value

    def loss(scale):
        space = CustomSpace(
            dimension=2,
            operators={
                "x": jnp.asarray([[0.0, 1.0], [1.0, 0.0]]),
                "energy": lambda: jnp.diag(jnp.asarray([0.0, 1.0])) * scale,
            },
        )
        op = LocalOps(label="spin", space=space)
        return op["energy"].matrix(backend=ArrayBackend())[1, 1] + 0.0 * op["x"].matrix(
            backend=ArrayBackend()
        )[0, 1]

    assert jax.grad(loss)(2.0) == 1.0


@pytest.mark.unit
def test_eigen_basis_record_exposes_and_applies_the_local_transform() -> None:
    hamiltonian = jnp.asarray([[1.0, 0.2, 0.0], [0.2, 2.0, 0.0], [0.0, 0.0, 5.0]])

    basis = resolve_local_basis(hamiltonian, basis="eigen", levels=2)

    assert basis.native_dim == 3
    assert basis.resolved_dim == 2
    np.testing.assert_allclose(basis.projector, basis.projector.conj().T)
    np.testing.assert_allclose(basis.projector @ basis.projector, basis.projector, atol=1e-12)
    np.testing.assert_allclose(
        basis.transform_operator(hamiltonian),
        np.diag(basis.energies),
        atol=1e-12,
    )


@pytest.mark.unit
def test_eigen_basis_requires_an_explicit_retained_dimension() -> None:
    with pytest.raises(ValueError, match="levels"):
        resolve_local_basis(jnp.eye(3), basis="eigen")


@pytest.mark.validation
def test_local_basis_gradient_ignores_degeneracy_confined_to_discarded_levels() -> None:
    operator = jnp.diag(jnp.asarray([1.0, -1.0, 0.0, 0.0]))

    def observable(coupling):
        hamiltonian = jnp.asarray(
            [
                [0.0, coupling, 0.0, 0.0],
                [coupling, 2.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0, 10.0],
            ]
        )
        basis = resolve_local_basis(hamiltonian, basis="eigen", levels=2)
        projected = basis.transform_operator(operator)
        return basis.energies[1] + jnp.real(projected[0, 0])

    gradient = jax.jit(jax.grad(observable))(0.3)
    finite_difference = (observable(0.300001) - observable(0.299999)) / 0.000002

    assert jnp.isfinite(gradient)
    assert gradient == pytest.approx(finite_difference, rel=1e-5)
