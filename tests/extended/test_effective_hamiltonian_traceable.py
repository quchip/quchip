"""JAX traceability of :func:`quchip.analysis.analyze_static_zz` (needs the dynamiqs backend)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.optional_backend

pytest.importorskip("dynamiqs")

from quchip.analysis import analyze_static_zz, effective_hamiltonian  # noqa: E402
from quchip.analysis.effective_hamiltonian import _inverse_sqrt_hermitian  # noqa: E402
from quchip.backend.dynamiqs import DynamiqsBackend  # noqa: E402
from quchip.chip.chip import Chip  # noqa: E402
from quchip.chip.couplings import Capacitive  # noqa: E402
from quchip.devices.transmon.duffing import DuffingTransmon  # noqa: E402


def _two_qubit_chip(g: jnp.ndarray | float) -> Chip:
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    return Chip([q0, q1], couplings=[Capacitive(q0, q1, g=g)], backend=DynamiqsBackend())


def test_grad_through_static_zz_matches_finite_difference():
    """jax.grad of analyze_static_zz's zz shift w.r.t. g matches a central finite-difference estimate."""
    def loss(g):
        chip = _two_qubit_chip(g)
        return analyze_static_zz(chip, "q0", "q1").zz

    g0 = 0.01
    grad = jax.grad(loss)(jnp.float64(g0))
    assert jnp.isfinite(grad)

    step = 1e-6
    finite_diff = (loss(jnp.float64(g0 + step)) - loss(jnp.float64(g0 - step))) / (2 * step)
    assert abs(float(grad) - float(finite_diff)) / abs(float(finite_diff)) < 1e-4


def test_grad_through_effective_hamiltonian_is_finite():
    """Löwdin projection remains differentiable when the Gram spectrum is degenerate."""
    def exchange(g):
        chip = _two_qubit_chip(g)
        return jnp.real(effective_hamiltonian(chip, ["q0", "q1"]).h_eff[1, 2])

    derivative = jax.grad(exchange)(jnp.float64(0.01))

    assert jnp.isfinite(derivative)


def test_inverse_sqrt_handles_an_ill_conditioned_repeated_gram_spectrum():
    """The traceable inverse square root meets its stated residual tolerance."""
    eigenvalues = jnp.asarray([1.0, 1.0, 1e-2, 1e-2, 1e-4, 1e-4, 1e-8, 1e-8])
    gram = jnp.diag(eigenvalues)

    inverse_sqrt = _inverse_sqrt_hermitian(gram)
    residual = inverse_sqrt @ gram @ inverse_sqrt - jnp.eye(gram.shape[0])

    assert float(jnp.linalg.norm(residual, ord=2)) < 1e-9
