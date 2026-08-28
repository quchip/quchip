"""JAX traceability of the labeled dressed Kerr matrix."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.optional_backend

pytest.importorskip("dynamiqs")

from quchip import Capacitive, Chip, DuffingTransmon, KerrMatrix, Resonator  # noqa: E402
from quchip.backend.dynamiqs import DynamiqsBackend  # noqa: E402


def _matrix(coupling: jnp.ndarray) -> KerrMatrix:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    chip = Chip(
        [q, r],
        [Capacitive(q, r, g=coupling)],
        backend=DynamiqsBackend(),
    )
    return chip.kerr_matrix()


def test_kerr_matrix_jits_and_gradients_flow_through_values_and_lookup() -> None:
    coupling = jnp.asarray(0.05)

    matrix = jax.jit(_matrix)(coupling)
    values_gradient = jax.grad(lambda g: _matrix(g).values[0, 1])(coupling)
    lookup_gradient = jax.grad(lambda g: _matrix(g)["q", "r"])(coupling)

    assert matrix.labels == ("q", "r")
    assert matrix.values.shape == (2, 2)
    assert jnp.isfinite(matrix.values).all()
    assert jnp.isfinite(values_gradient)
    assert jnp.isfinite(lookup_gradient)
    assert values_gradient == pytest.approx(lookup_gradient)
    assert abs(float(lookup_gradient)) > 1e-4
