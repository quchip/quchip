"""Declarative drive extension contract."""

from __future__ import annotations

from quchip.approximations import Exact

import quchip
import numpy as np
import pytest
from quchip import (
    ChargeDrive,
    CouplingDrive,
    DeviceDrive,
    DuffingTransmon,
    FluxDrive,
    ParametricDrive,
    PhaseDrive,
    TunableCapacitive,
    TwoPhotonDrive,
)
from quchip.engine.ir import DriveOp


def test_drive_authoring_bases_are_public() -> None:
    assert quchip.DeviceDrive is DeviceDrive
    assert quchip.CouplingDrive is CouplingDrive


def test_builtin_drives_expose_authored_hamiltonians() -> None:
    first = DuffingTransmon(5.0, -0.25, levels=3, label="a")
    second = DuffingTransmon(5.2, -0.25, levels=3, label="b")
    coupling = TunableCapacitive(first, second, g_0=0.0, label="tc")
    pulse = DriveOp(
        target_label="a",
        drive_label="charge",
        envelope=quchip.Square(duration=2.0, amplitude=0.01),
        freq=5.0,
    )
    signal = ChargeDrive(first).signal(pulse, first)

    for drive, target in (
        (ChargeDrive(first), first),
        (PhaseDrive(first), first),
        (FluxDrive(first), first),
        (TwoPhotonDrive(first), first),
        (ParametricDrive(coupling), coupling),
    ):
        authored = drive.hamiltonian(target, signal)
        assert authored.labels


def test_custom_drive_compiles_from_physical_iq_quadratures() -> None:
    class QuadratureDrive(DeviceDrive):
        def hamiltonian(self, target, signal):
            return signal.i * target.charge_coupling_operator() - signal.q * target.phase_coupling_operator()

    mode = DuffingTransmon(5.0, -0.25, levels=3, label="q")
    drive = QuadratureDrive(mode, label="quadrature")
    chip = quchip.Chip([mode], control_equipment=quchip.ControlEquipment([drive]))
    sequence = quchip.QuantumSequence(chip)
    sequence.schedule(drive, envelope=quchip.Square(duration=2.0, amplitude=0.01))

    result = sequence.resolve()

    assert any(term.origin == "drive" for term in result.dynamic_terms)
    values = [
        term.time_dependence.signal.evaluate(0.25, xp=np) for term in result.dynamic_terms if term.origin == "drive"
    ]
    assert any(abs(value) > 0.0 for value in values)


def test_drive_hamiltonian_accepts_nonlinear_signal_algebra() -> None:
    class QuadraticFlux(DeviceDrive):
        def hamiltonian(self, target, signal):
            return (signal.i * signal.i) * target.flux_coupling_operator()

    mode = DuffingTransmon(5.0, -0.25, levels=3, label="q")
    drive = QuadraticFlux(mode, label="quadratic")
    chip = quchip.Chip(
        [mode],
        approximation=Exact(),
        control_equipment=quchip.ControlEquipment([drive]),
    )
    sequence = quchip.QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=quchip.Square(duration=2.0, amplitude=0.2),
    )

    terms = [term for term in sequence.resolve().dynamic_terms if term.origin == "drive"]

    assert len(terms) == 1
    assert terms[0].time_dependence.signal.evaluate(1.0, xp=np) == pytest.approx(0.04)


def test_custom_coupling_drive_does_not_require_parametric_interaction() -> None:
    class InteractionDrive(CouplingDrive):
        def hamiltonian(self, target, signal):
            return signal.i * target.interaction_hamiltonian()

    first = DuffingTransmon(5.0, -0.25, levels=2, label="a")
    second = DuffingTransmon(5.2, -0.25, levels=2, label="b")
    coupling = quchip.Capacitive(first, second, g=0.01, label="ab")
    drive = InteractionDrive(coupling, label="edge")
    chip = quchip.Chip(
        [first, second],
        couplings=[coupling],
        approximation=Exact(),
        control_equipment=quchip.ControlEquipment([drive]),
    )
    sequence = quchip.QuantumSequence(chip)
    sequence.schedule(drive, envelope=quchip.Square(duration=2.0, amplitude=0.1))

    terms = sequence.resolve().dynamic_terms
    assert terms
    assert all(term.operator.subsystem_labels == ("a", "b") for term in terms)
    assert any(
        abs(term.time_dependence.signal.evaluate(1.0, xp=np)) > 0.0
        for term in terms
    )
