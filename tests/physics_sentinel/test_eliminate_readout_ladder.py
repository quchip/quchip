"""Ladder rung 2 (spec Sec. 9): CrossKerr readout reduction reproduces the full dispersive-readout probe.

``eliminate(chip, coupling)`` on a qubit-resonator exchange edge folds the
edge into a :class:`~quchip.chip.couplings.CrossKerr` while both endpoint
devices survive (the effective-readout-chip flow). A ``ChargeDrive`` probe
wired to the resonator survives the reduction untouched, so the identical
schedule call replays on both chips — this rung checks that the reduced
pointer-separation trajectory agrees with the full qubit+resonator dynamics,
and that the reduction never forces a density-matrix solve the full chip
didn't already need.
"""

from __future__ import annotations

import numpy as np

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    QuantumSequence,
    Resonator,
    Square,
    build_problem,
)
from quchip.chip.transformations import eliminate

_G = 0.05  # qubit-resonator coupling, GHz
_Q_FREQ = 5.0
_R_FREQ = 7.0  # Delta = 2.0 GHz -> g/Delta = 0.025, (g/Delta)^2 = 6.25e-4
_R_LEVELS = 8
_QUALITY_FACTOR = 300.0  # bad-cavity: kappa_ordinary = 7.0/300 = 0.0233 GHz >> |chi|
_AMPLITUDE = 0.02
_DURATION = 80.0  # ~1.9 ring-up times (1/kappa_ordinary ~= 43 ns)


def _readout_chip(*, quality_factor: float | None) -> Chip:
    """Fresh qubit + probed readout resonator, wired with a ``ChargeDrive`` probe on the resonator."""
    q = DuffingTransmon(freq=_Q_FREQ, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=_R_FREQ, levels=_R_LEVELS, quality_factor=quality_factor, label="r")
    readout = ChargeDrive(target=r, label="readout")
    return Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=_G, label="qr")],
        control_equipment=ControlEquipment([readout]),
        frame="rotating",
        rwa=True,
    )


def _probe_pointer(chip: Chip, readout_freq: float, qubit_level: int) -> complex:
    """Replay the readout probe on *chip* by its surviving drive label; return the final resonator <a>."""
    tlist = np.linspace(0.0, _DURATION, 81)
    seq = QuantumSequence(chip)
    seq.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    result = seq.simulate(
        tlist=tlist,
        initial_state=chip.state(q=qubit_level, r=0),
        e_ops=chip.e_ops(r="a"),
    )
    return complex(np.asarray(result.expect_values("r"), dtype=complex)[-1])


def test_readout_pointer_separation_agrees_full_vs_reduced():
    """CrossKerr-reduced pointer separation (phase and magnitude) matches the full qubit+resonator probe."""
    full = _readout_chip(quality_factor=_QUALITY_FACTOR)
    q, r = full["q"], full["r"]
    readout_freq = 0.5 * (full.freq(r, {q: 0}) + full.freq(r, {q: 1}))
    reduced = eliminate(full, "qr").chip

    a_full_0 = _probe_pointer(full, readout_freq, 0)
    a_full_1 = _probe_pointer(full, readout_freq, 1)
    a_red_0 = _probe_pointer(reduced, readout_freq, 0)
    a_red_1 = _probe_pointer(reduced, readout_freq, 1)

    sep_full = a_full_1 - a_full_0
    sep_red = a_red_1 - a_red_0

    tol = (_G / (_R_FREQ - _Q_FREQ)) ** 2  # 6.25e-4
    # Measured: angle diff = 5.06e-4 rad, magnitude relative diff = 0.031% — both
    # under the (g/Delta)^2 = 6.25e-4 bound.
    angle_diff = abs(np.angle(sep_full) - np.angle(sep_red))
    assert angle_diff < tol, (angle_diff, tol)
    mag_rel_diff = abs(abs(sep_full) - abs(sep_red)) / abs(sep_full)
    assert mag_rel_diff < tol, (mag_rel_diff, tol)


def test_reduced_readout_does_not_force_density_matrix_solve():
    """The CrossKerr reduction never forces mesolve when the full chip wouldn't need it either."""
    full = _readout_chip(quality_factor=None)
    q, r = full["q"], full["r"]
    readout_freq = 0.5 * (full.freq(r, {q: 0}) + full.freq(r, {q: 1}))
    reduced = eliminate(full, "qr").chip
    tlist = np.linspace(0.0, _DURATION, 81)

    seq_full = QuantumSequence(full)
    seq_full.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    problem_full = build_problem(full, list(seq_full.scheduled_ops), tlist, initial_state=full.state(q=0, r=0))

    seq_reduced = QuantumSequence(reduced)
    seq_reduced.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    problem_reduced = build_problem(
        reduced, list(seq_reduced.scheduled_ops), tlist, initial_state=reduced.state(q=0, r=0)
    )

    assert problem_full.c_ops == ()
    assert problem_reduced.c_ops == ()
    chosen_full = problem_full.solver or ("mesolve" if problem_full.c_ops else "sesolve")
    chosen_reduced = problem_reduced.solver or ("mesolve" if problem_reduced.c_ops else "sesolve")
    assert chosen_full == chosen_reduced == "sesolve"


def test_reduced_readout_collapse_profile_matches_full_chip():
    """With a lossy resonator, the reduction leaves the collapse-operator profile untouched."""
    full = _readout_chip(quality_factor=_QUALITY_FACTOR)
    q, r = full["q"], full["r"]
    readout_freq = 0.5 * (full.freq(r, {q: 0}) + full.freq(r, {q: 1}))
    reduced = eliminate(full, "qr").chip
    tlist = np.linspace(0.0, _DURATION, 81)

    seq_full = QuantumSequence(full)
    seq_full.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    problem_full = build_problem(full, list(seq_full.scheduled_ops), tlist, initial_state=full.state(q=0, r=0))

    seq_reduced = QuantumSequence(reduced)
    seq_reduced.schedule("readout", envelope=Square(duration=_DURATION, amplitude=_AMPLITUDE), freq=readout_freq)
    problem_reduced = build_problem(
        reduced, list(seq_reduced.scheduled_ops), tlist, initial_state=reduced.state(q=0, r=0)
    )

    assert len(problem_full.c_ops) == len(problem_reduced.c_ops) == 1
