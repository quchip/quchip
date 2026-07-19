"""Gradients flow through the reduced chip's scheduled pump amplitude.

The chip under test is not hand-built: it is the output of ``eliminate()`` on
the rung-1 bridge fixture (a bridge coupler folded away, a
``ParametricDrive``-pumped ``TunableCapacitive`` edge left behind, wired
through the retarget rule's ``Gain``). A single loss function must span the
*reduced* model exactly as it would the original — this rung checks that
``jax.grad`` sees through the whole chain: retarget ``Gain``, edge pump,
dynamiqs solve.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.optional_backend

pytest.importorskip("dynamiqs")

from quchip import (  # noqa: E402
    Capacitive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    FluxDrive,
    FluxTunableTransmon,
    QuantumSequence,
    Square,
)
from quchip.chip.transformations import eliminate  # noqa: E402

_LEG_G = 0.08
_BRIDGE_FREQ = 6.3


def _bridge_chip_dynamiqs() -> Chip:
    """Degenerate two-leg bridge (rung-1 fixture) on the dynamiqs backend."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.24, levels=3, label="q1")
    fc = FluxTunableTransmon(freq=_BRIDGE_FREQ, anharmonicity=-0.2, levels=3, label="fc")
    couplings = [
        Capacitive(q0, fc, g=_LEG_G, label="leg0"),
        Capacitive(q1, fc, g=_LEG_G, label="leg1"),
    ]
    flux = FluxDrive(fc, label="cflux")
    return Chip(
        [q0, q1, fc],
        couplings=couplings,
        control_equipment=ControlEquipment([flux]),
        frame="rotating",
        rwa=True,
        backend="dynamiqs",
    )


def test_grad_through_reduced_chip_pump_amplitude():
    """jax.grad of the reduced-chip transfer w.r.t. pump amplitude matches a central finite difference."""
    full = _bridge_chip_dynamiqs()
    res = eliminate(full, "fc")
    reduced = res.chip
    exchange = res.effective_params["exchange"]
    j_eff = float(exchange["j_eff"])
    d_j_domega_c = float(exchange["dJ_domega_c"])

    amp0 = 0.05
    # Duration for a half-swap at amp0 (max d(population)/d(J_total), so the gradient signal
    # is not swamped by the always-on static exchange j_eff nor flattened at a swap extremum).
    j_total0 = j_eff + d_j_domega_c * amp0
    duration = 1.0 / (8.0 * abs(j_total0))

    with warnings.catch_warnings():
        # Degenerate q0/q1 (both 5.0 GHz idle) trip the near-degenerate dressed-state labeling
        # warning; bare_state still returns the exact bare product state (see
        # tests/physics_sentinel/test_eliminate_portability.py).
        warnings.simplefilter("ignore", UserWarning)
        init_state = reduced.bare_state({"q0": 1, "q1": 0})

    def loss(amp: float) -> float:
        seq = QuantumSequence(reduced)
        seq.schedule("cflux", envelope=Square(duration=duration, amplitude=amp))
        result = seq.simulate(
            tlist=jnp.linspace(0.0, duration, 11),
            initial_state=init_state,
            options={"store_states": True, "store_final_state": True},
        )
        return result.population_array(reduced["q1"], 1)[-1]

    grad = jax.grad(loss)(amp0)
    eps = 1e-4
    finite_diff = (loss(amp0 + eps) - loss(amp0 - eps)) / (2 * eps)

    assert np.isfinite(float(grad))
    # Measured: grad = -0.6283194, finite_diff = -0.6283190, relative error = 7.7e-7.
    assert abs(float(grad) - float(finite_diff)) < 0.2 * abs(float(finite_diff))
