"""Ladder rung 3 grad check (spec Sec. 9): jax.grad flows through eliminate(method="exact")'s zz.

Mirrors ``tests/extended/test_effective_hamiltonian_traceable.py``'s pattern:
the dressed-spectrum eigendecomposition inside ``exact_reduction``
(``quchip/chip/sw.py``) needs the JAX-native dynamiqs backend to stay traced
end-to-end. Companion to ``tests/physics_sentinel/test_sw_vs_exact.py``,
which checks the same bridge fixture's sw-vs-exact agreement without a
traced coupling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.optional_backend

pytest.importorskip("dynamiqs")

from quchip import Capacitive, Chip, DuffingTransmon, Resonator  # noqa: E402
from quchip.backend.dynamiqs import DynamiqsBackend  # noqa: E402
from quchip.chip.transformations import eliminate  # noqa: E402

_Q0_FREQ = 5.0
_Q1_FREQ = 5.2
_BUS_FREQ = 6.3
_LEG1_G = 0.08


def _bridge_chip_dynamiqs(leg0_g: jnp.ndarray | float) -> Chip:
    """Standard bridge fixture (leg0's g traced, leg1 fixed) on the dynamiqs backend."""
    q0 = DuffingTransmon(freq=_Q0_FREQ, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=_Q1_FREQ, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=_BUS_FREQ, levels=4, label="bus")
    couplings = [
        Capacitive(q0, bus, g=leg0_g, rwa=True, label="leg0"),
        Capacitive(q1, bus, g=_LEG1_G, rwa=True, label="leg1"),
    ]
    return Chip([q0, q1, bus], couplings=couplings, frame="rotating", rwa=True, backend=DynamiqsBackend())


def test_grad_through_exact_zz_matches_finite_difference():
    """jax.grad of the exact-route residual ZZ w.r.t. one leg's g matches a central finite difference."""

    def loss(g: jnp.ndarray) -> jnp.ndarray:
        chip = _bridge_chip_dynamiqs(g)
        return eliminate(chip, "bus", method="exact").effective_params["exchange"]["zz"]

    g0 = 0.08
    grad = jax.grad(loss)(jnp.float64(g0))
    assert jnp.isfinite(grad)

    step = 1e-5
    finite_diff = (loss(g0 + step) - loss(g0 - step)) / (2 * step)
    # Measured: grad = 0.0334707525, finite_diff = 0.0334707531, relative error = 1.8e-7.
    assert abs(float(grad) - float(finite_diff)) / abs(float(finite_diff)) < 1e-3
