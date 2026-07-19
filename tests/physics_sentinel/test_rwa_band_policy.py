"""RWA band policy sentinels: structural filter ≡ authored JC; shared-frame fold fix."""

from __future__ import annotations

import numpy as np

from quchip import Capacitive, Chip, Coupling, DuffingTransmon, simulate


def _pair(rwa_flag):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.3, levels=3, label="q1")
    return q0, q1, Capacitive(q0, q1, g=0.05, rwa=rwa_flag)


def test_rwa_chip_hamiltonian_matches_authored_jc():
    """Structural band filter reproduces the hand-authored beam-splitter form exactly."""
    q0, q1, cap = _pair(rwa_flag=True)
    chip = Chip([q0, q1], [cap], frame="rotating", rwa=True)

    j0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="j0")
    j1 = DuffingTransmon(freq=5.3, anharmonicity=-0.3, levels=3, label="j1")
    jc = Coupling(
        j0, j1, g=0.05, rwa=False,
        interaction=lambda a, b, bk: (
            bk.tensor(bk.dag(a.lowering_operator()), b.lowering_operator())
            + bk.tensor(a.lowering_operator(), bk.dag(b.lowering_operator()))
        ),
    )
    chip_jc = Chip([j0, j1], [jc], frame="rotating", rwa=False)

    h = np.asarray(chip.backend.to_array(chip.hamiltonian()))
    h_jc = np.asarray(chip_jc.backend.to_array(chip_jc.hamiltonian()))
    np.testing.assert_allclose(h, h_jc, atol=1e-12)


def test_mixed_policy_respects_per_coupling_override():
    """chip rwa=True + one rwa=False coupling: only the override keeps counter-rotating elements."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.3, levels=2, label="q1")
    q2 = DuffingTransmon(freq=5.6, anharmonicity=-0.3, levels=2, label="q2")
    cap_rwa = Capacitive(q0, q1, g=0.05)              # inherits chip rwa=True
    cap_full = Capacitive(q1, q2, g=0.05, rwa=False)  # per-coupling override
    chip = Chip([q0, q1, q2], [cap_rwa, cap_full], rwa=True)
    h = np.asarray(chip.backend.to_array(chip.hamiltonian()))
    # Basis |q0 q1 q2> with q2 fastest: |110> = index 6, |000> = 0, |011> = 3.
    assert abs(h[0, 6]) < 1e-12   # q0-q1 counter-rotating masked
    assert abs(h[0, 3]) > 1e-3    # q1-q2 counter-rotating survives the override


def test_shared_scalar_frame_matches_lab_without_rwa():
    """rwa=False in a shared nonzero frame keeps counter-rotating bands oscillating, matching the lab frame."""
    def run(frame):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q0")
        q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.3, levels=2, label="q1")
        cap = Capacitive(q0, q1, g=0.05, rwa=False)
        chip = Chip([q0, q1], [cap], frame=frame, rwa=False)
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
