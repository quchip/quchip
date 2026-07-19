"""Gradients flow through the structural RWA mask inside chip.hamiltonian()."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.optional_backend

pytest.importorskip("dynamiqs")

from quchip import Capacitive, Chip, DuffingTransmon  # noqa: E402
from quchip.backend.dynamiqs import DynamiqsBackend  # noqa: E402


def test_grad_of_dressed_gap_through_rwa_mask():
    """The gradient of the RWA-masked dressed-state gap w.r.t. coupling g is finite and nonzero."""
    def gap(g):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.3, levels=3, label="q1")
        cap = Capacitive(q0, q1, g=g)
        chip = Chip([q0, q1], [cap], rwa=True, backend=DynamiqsBackend())
        h = jnp.asarray(chip.backend.to_array(chip.hamiltonian()))
        evals = jnp.linalg.eigvalsh(h)
        return evals[2] - evals[1]

    value = gap(0.05)
    grad = jax.grad(gap)(0.05)
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))
    assert abs(float(grad)) > 1e-6  # the masked exchange band still carries g-dependence
