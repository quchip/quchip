"""Gradients flow through scheduled pump parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.optional_backend

pytest.importorskip("dynamiqs")

from quchip import (  # noqa: E402
    Chip,
    ControlEquipment,
    DuffingTransmon,
    ParametricDrive,
    QuantumSequence,
    Square,
    TunableCapacitive,
)


def test_grad_through_pump_amplitude():
    """The gradient of the final q1 population w.r.t. pump amplitude is finite and non-negligible."""
    def loss(amp):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.24, levels=3, label="q1")
        tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
        pump = ParametricDrive(tc, label="pump")
        chip = Chip([q0, q1], couplings=[tc], frame="rotating", rwa=True, backend="dynamiqs")
        chip.connect(ControlEquipment([pump]))
        seq = QuantumSequence(chip)
        seq.pump(tc, envelope=Square(duration=50.0, amplitude=amp))
        result = seq.simulate(
            tlist=jnp.linspace(0.0, 50.0, 51),
            initial_state={"q0": 1, "q1": 0},
            options={"store_states": True, "store_final_state": True},
        )
        return result.population_array(q1, 1)[-1]

    g = jax.grad(loss)(0.004)
    assert np.isfinite(float(g))
    assert abs(float(g)) > 1e-3  # transfer is amplitude-sensitive below the swap point
