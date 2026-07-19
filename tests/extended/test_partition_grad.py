"""Gradient flow through a partitioned simulate (dynamiqs backend)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.optional_backend

jax = pytest.importorskip("jax")
pytest.importorskip("dynamiqs")

from quchip import Capacitive, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence  # noqa: E402
from quchip.backend.dynamiqs import DynamiqsBackend  # noqa: E402
from quchip.results.partitioned import PartitionedSimulationResult  # noqa: E402


def test_grad_flows_through_partitioned_solve():
    """The gradient of a partitioned-solve expectation value w.r.t. drive amplitude is finite and nonzero."""
    def loss(amp):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.24, levels=3, label="q1")
        q2 = DuffingTransmon(freq=5.2, anharmonicity=-0.23, levels=3, label="q2")
        chip = Chip([q0, q1, q2], couplings=[Capacitive(q0, q1, g=0.005)],
                    frame="rotating", rwa=True, backend=DynamiqsBackend())
        d0 = ChargeDrive(target=q0, label="d0")
        d2 = ChargeDrive(target=q2, label="d2")
        chip.wire(d0, d2)
        seq = QuantumSequence(chip)
        seq.schedule(d0, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=amp), freq=chip.freq(q0))
        seq.schedule(d2, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q2))
        result = seq.simulate(tlist=np.linspace(0.0, 20.0, 21), e_ops=chip.e_ops(q0="Z"))
        assert isinstance(result, PartitionedSimulationResult)
        return jax.numpy.real(result.expect("q0")[-1])

    grad = jax.grad(loss)(0.02)
    assert np.isfinite(float(grad)) and abs(float(grad)) > 0.0
