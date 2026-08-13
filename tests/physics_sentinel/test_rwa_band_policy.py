"""RWA band policy sentinels: structural filter ≡ authored JC; shared-frame fold fix."""

from __future__ import annotations

from quchip.approximations import RWA, Exact

import numpy as np
import pytest

from quchip import Capacitive, Chip, Coupling, DuffingTransmon, simulate


def _pair():
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.3, levels=3, label="q1")
    return q0, q1, Capacitive(q0, q1, g=0.05)


def test_rwa_chip_hamiltonian_matches_authored_jc():
    """Structural band filter reproduces the hand-authored beam-splitter form exactly."""
    q0, q1, cap = _pair()
    chip = Chip([q0, q1], [cap], frame={q0: 5.0, q1: 5.3}, approximation=RWA())

    j0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="j0")
    j1 = DuffingTransmon(freq=5.3, anharmonicity=-0.3, levels=3, label="j1")
    jc = Coupling(
        j0,
        j1,
        g=0.05,
        interaction=lambda a, b, bk: (
            bk.tensor(bk.dag(a.lowering_operator()), b.lowering_operator())
            + bk.tensor(a.lowering_operator(), bk.dag(b.lowering_operator()))
        ),
    )
    chip_jc = Chip([j0, j1], [jc], frame={j0: 5.0, j1: 5.3}, approximation=Exact())

    h = np.asarray(chip.resolve().hamiltonian().matrix(t=0.0))
    h_jc = np.asarray(chip_jc.resolve().hamiltonian().matrix(t=0.0))
    np.testing.assert_allclose(h, h_jc, atol=1e-12)


def test_coupling_cannot_override_chip_approximation():
    """Approximation belongs to the engine, not to one coupling."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.3, levels=2, label="q1")
    with pytest.raises(TypeError, match="approximation"):
        Capacitive(q0, q1, g=0.05, approximation=Exact())


def test_shared_scalar_frame_matches_lab_without_rwa():
    """Exact keeps counter-rotating bands consistent across lab and shared frames."""

    def run(frame):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q0")
        q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.3, levels=2, label="q1")
        cap = Capacitive(q0, q1, g=0.05)
        chip = Chip([q0, q1], [cap], frame=frame, approximation=Exact())
        tlist = np.linspace(0.0, 25.0, 51)
        result = simulate(
            chip,
            [],
            tlist,
            initial_state=chip.bare_state(),
            e_ops={q0: q0.number_operator(), q1: q1.number_operator()},
        )
        return np.real(np.asarray(result.expect("q0"))), np.real(np.asarray(result.expect("q1")))

    n0_lab, n1_lab = run("lab")
    n0_shared, n1_shared = run(5.15)

    # A frozen a†b† + ab band would Rabi |00> -> |11> at 2g (period 10 ns);
    # the true counter-rotating response is a ~1e-4 wiggle. The lab-frame
    # reference guards its own sensitivity: populations must stay near zero.
    assert float(np.max(n0_lab)) < 0.01
    np.testing.assert_allclose(n0_shared, n0_lab, atol=1e-3)
    np.testing.assert_allclose(n1_shared, n1_lab, atol=1e-3)
