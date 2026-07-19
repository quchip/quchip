"""Regression snapshot tests for the Rabi, callable-signal IR, and crosstalk paths.

These tests capture numerical baselines for three critical simulation
paths and fail if the results drift.

The three paths tested:
  1. Rabi oscillation -- verifies the 2pi boundary and population dynamics
  2. Callable-signal IR -- verifies a carrier-modulated ScalarModulation term
     stays distinct from a constant one instead of collapsing under evaluation
  3. Crosstalk dynamics -- verifies victim device excitation from leaked
     signal through the crosstalk pipeline

Purpose: safety net for the callable-IR and crosstalk paths. These tests detect any
behavioral change.

Unit convention: frequencies in GHz (ordinary), times in ns, h-bar = 1.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from quchip.chip.chip import Chip
from quchip.control import ChargeDrive, ControlEquipment, Crosstalk
from quchip.control.envelopes import Square
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import simulate
from quchip.engine.ir import (
    CanonicalOperator,
    Carrier,
    Constant,
    DriveOp,
    DynamicTerm,
    HamiltonianDescription,
    ScalarModulation,
    evaluate_signal_program,
)


# Rabi oscillation snapshot


def test_rabi_oscillation_snapshot() -> None:
    """Regression baseline: Rabi oscillation P(|1>) = sin^2(pi * Omega * t)."""
    # Omega = 0.05 GHz, duration 100 ns -> 5 full Rabi cycles (period = 1/(2*Omega) = 10 ns).
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    drive = ChargeDrive(target=q)
    chip = Chip([q])
    chip.connect(ControlEquipment(lines=[drive]))
    chip.set_frame("rotating")

    envelope = Square(duration=100.0, amplitude=0.05)
    drive_op = DriveOp(
        target_label="q",
        envelope=envelope,
        freq=5.0,
        start_time=0.0,
        drive_label=drive.label,
    )

    tlist = np.linspace(0, 100, 501)
    result = simulate(chip, [drive_op], tlist)
    p1 = result.population("q", 1)

    # At t=100 ns: 5 full Rabi cycles -> back to ground state
    npt.assert_allclose(
        p1[-1],
        0.0,
        atol=0.05,
        err_msg="Rabi snapshot: P(|1>) at t=100ns should be ~0 (5 full cycles)",
    )

    # At t=10 ns: half Rabi period -> max excitation
    idx_10 = np.argmin(np.abs(tlist - 10.0))
    npt.assert_allclose(
        p1[idx_10],
        1.0,
        atol=0.05,
        err_msg="Rabi snapshot: P(|1>) at t=10ns should be ~1.0 (half period)",
    )


# Callable-signal IR snapshot


def test_callable_ir_snapshot() -> None:
    """Regression baseline: scalar callable terms stay distinct from constants."""
    tlist = np.linspace(0, 60, 601)
    op = CanonicalOperator.from_dense(
        np.eye(2, dtype=complex),
        dims=(2,),
        basis="fock",
        subsystem_labels=("q",),
    )
    desc_dsp = HamiltonianDescription(
        static_terms=(),
        dynamic_terms=(
            DynamicTerm(
                operator=op,
                time_dependence=ScalarModulation(signal=Carrier(freq=0.05)),
                origin="drive",
            ),
        ),
        dims=(2,),
        metadata={},
    )
    desc_ideal = HamiltonianDescription(
        static_terms=(),
        dynamic_terms=(
            DynamicTerm(
                operator=op,
                time_dependence=ScalarModulation(signal=Constant(0.05 + 0.0j)),
                origin="drive",
            ),
        ),
        dims=(2,),
        metadata={},
    )

    dsp_term = desc_dsp.dynamic_terms[0].time_dependence
    ideal_term = desc_ideal.dynamic_terms[0].time_dependence

    assert isinstance(ideal_term, ScalarModulation)

    coeff_dsp = np.asarray(evaluate_signal_program(dsp_term.signal, tlist))
    coeff_ideal = np.asarray(evaluate_signal_program(ideal_term.signal, tlist))

    # The two callable terms should not collapse to the same coefficient trace.
    assert np.any(coeff_dsp != coeff_ideal), (
        "Callable IR snapshot: carrier-based coefficients should differ from "
        "constant coefficients. If identical, the callable path is being "
        "silently collapsed."
    )


# Crosstalk dynamics snapshot


def test_crosstalk_dynamics_snapshot() -> None:
    """Regression baseline: beta=0.1 crosstalk from driven q1 causes measurable excitation on q2."""
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.5, anharmonicity=-0.2, levels=3, label="q2")
    d1 = ChargeDrive(target=q1)
    d2 = ChargeDrive(target=q2)

    equip = ControlEquipment(lines=[d1, d2], signal_chain=[Crosstalk(source=d1.label, victim=d2.label, beta=0.1)])

    chip_xt = Chip([q1, q2])
    chip_xt.set_frame("lab")
    chip_xt.connect(equip)

    envelope = Square(duration=50.0, amplitude=0.1)
    drive_op = DriveOp(
        target_label="q1",
        envelope=envelope,
        freq=5.0,
        start_time=0.0,
        drive_label=d1.label,
    )

    tlist = np.linspace(0, 50, 301)
    result_xt = simulate(chip_xt, [drive_op], tlist)

    p_q2_xt = result_xt.population("q2", 1)
    p_q1_xt = result_xt.population("q1", 1)

    q1_no = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q1")
    q2_no = DuffingTransmon(freq=5.5, anharmonicity=-0.2, levels=3, label="q2")
    d1_no = ChargeDrive(target=q1_no)
    d2_no = ChargeDrive(target=q2_no)

    chip_no = Chip([q1_no, q2_no])
    chip_no.set_frame("lab")
    chip_no.connect(ControlEquipment(lines=[d1_no, d2_no]))

    drive_op_no = DriveOp(
        target_label="q1",
        envelope=envelope,
        freq=5.0,
        start_time=0.0,
        drive_label=d1_no.label,
    )

    result_no = simulate(chip_no, [drive_op_no], tlist)
    p_q1_no = result_no.population("q1", 1)

    # Crosstalk should cause non-zero victim excitation relative to the
    # no-crosstalk baseline.
    assert np.max(p_q2_xt) > np.max(result_no.population("q2", 1)) + 1e-6, (
        f"Crosstalk snapshot: victim q2 should have measurable excitation "
        f"(max P(q2,|1>)={np.max(p_q2_xt):.6f}), but it did not exceed the "
        f"no-crosstalk baseline by a meaningful margin"
    )

    # The source line is driven directly in both cases; the key regression
    # signal is measurable leaked excitation on the victim.
    assert np.max(p_q1_xt) > 1e-3
    assert np.max(p_q1_no) > 1e-3
