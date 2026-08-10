from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from quchip import ChargeSpace, CustomSpace, FockSpace, LocalOps, PhaseGridSpace
from quchip.engine.basis import resolve_local_basis


def test_fock_space_materializes_authored_operators_without_a_live_device() -> None:
    space = FockSpace(4)
    op = LocalOps(label="mode", space=space)

    matrix = (5.0 * op.n + 0.25 * (op.adag @ op.a)).matrix()

    np.testing.assert_allclose(matrix, np.diag([0.0, 5.25, 10.5, 15.75]))


def test_charge_space_materializes_charge_and_josephson_operators() -> None:
    space = ChargeSpace(num_basis=5)
    op = LocalOps(label="q", space=space)

    shifted_charge = op.n - 0.2 * op.I
    matrix = (shifted_charge @ shifted_charge - 2.0 * op.cos_phi).matrix()

    charges = np.arange(-2, 3, dtype=float)
    expected = np.diag((charges - 0.2) ** 2)
    expected -= np.eye(5, k=1) + np.eye(5, k=-1)
    np.testing.assert_allclose(matrix, expected)


def test_phase_grid_space_materializes_phase_and_charge_operators() -> None:
    space = PhaseGridSpace(points=5, extent=2.0)
    op = LocalOps(label="q", space=space)

    phase = op.phi.matrix()
    charge = op.n.matrix()

    np.testing.assert_allclose(np.diag(phase), np.linspace(-2.0, 2.0, 5, endpoint=False))
    np.testing.assert_allclose(charge, charge.conj().T)


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


def test_eigen_basis_record_exposes_and_applies_the_local_transform() -> None:
    hamiltonian = jnp.asarray([[1.0, 0.2, 0.0], [0.2, 2.0, 0.0], [0.0, 0.0, 5.0]])
    operator = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    basis = resolve_local_basis(hamiltonian, basis="eigen", levels=2)

    assert basis.native_dim == 3
    assert basis.resolved_dim == 2
    np.testing.assert_allclose(basis.projector, basis.vectors @ basis.vectors.conj().T)
    np.testing.assert_allclose(
        basis.transform_operator(operator),
        basis.vectors.conj().T @ operator @ basis.vectors,
    )


def test_eigen_basis_requires_an_explicit_retained_dimension() -> None:
    with pytest.raises(ValueError, match="levels"):
        resolve_local_basis(jnp.eye(3), basis="eigen")


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

    gradient = jax.grad(observable)(0.3)
    finite_difference = (observable(0.300001) - observable(0.299999)) / 0.000002

    assert jnp.isfinite(gradient)
    assert gradient == pytest.approx(finite_difference, rel=1e-5)
