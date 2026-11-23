"""Dressed weak-drive cross-resonance susceptibility."""

from __future__ import annotations

from quchip.approximations import RWA

import numpy as np
import pytest

from quchip import analyze_cr_susceptibility
from quchip.chip import Capacitive, Chip
from quchip.control import ChargeDrive, ControlEquipment
from quchip.devices.transmon.duffing import DuffingTransmon


def _driven_pair(*, two_control_lines: bool = False, backend: str | None = None):
    control = DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=3, label="control")
    target = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="target")
    drive = ChargeDrive(control, label="control_xy")
    lines = [drive]
    if two_control_lines:
        lines.append(ChargeDrive(control, label="control_xy_2"))
    chip = Chip(
        [control, target],
        [Capacitive(control, target, g=0.004)],
        control_equipment=ControlEquipment(lines),
        approximation=RWA(),
        backend=backend,
    )
    return chip, control, target, drive


@pytest.mark.optional_backend
def test_cr_susceptibility_has_backend_independent_dressed_state_phases() -> None:
    """QuTiP and dynamiqs use the same bare-overlap phase convention."""
    pytest.importorskip("dynamiqs")
    qutip_chip, qutip_control, qutip_target, qutip_drive = _driven_pair(backend="qutip")
    dynamiqs_chip, dynamiqs_control, dynamiqs_target, dynamiqs_drive = _driven_pair(backend="dynamiqs")

    qutip_result = analyze_cr_susceptibility(qutip_chip, qutip_control, qutip_target, drive=qutip_drive)
    dynamiqs_result = analyze_cr_susceptibility(
        dynamiqs_chip,
        dynamiqs_control,
        dynamiqs_target,
        drive=dynamiqs_drive,
    )

    for field in ("m_control_0", "m_control_1", "IX_per_amplitude", "ZX_per_amplitude"):
        np.testing.assert_allclose(
            np.asarray(getattr(qutip_result, field)),
            np.asarray(getattr(dynamiqs_result, field)),
            rtol=1e-8,
            atol=1e-10,
        )


def test_cr_susceptibility_ignores_eigensolver_column_signs(monkeypatch) -> None:
    """Flipping arbitrary dressed eigenvector signs leaves IX and ZX unchanged."""
    chip, control, target, drive = _driven_pair()
    reference = analyze_cr_susceptibility(chip, control, target, drive=drive)
    compute = chip._analysis._compute_array_labeled

    def sign_flipped_eigensystem(engine_result=None):
        eigenvalues, eigenvectors, overlaps, labeling = compute(engine_result)
        signs = np.where(np.arange(eigenvectors.shape[1]) % 2, -1.0, 1.0)
        return eigenvalues, eigenvectors * signs[None, :], overlaps, labeling

    monkeypatch.setattr(chip._analysis, "_compute_array_labeled", sign_flipped_eigensystem)
    flipped = analyze_cr_susceptibility(chip, control, target, drive=drive)

    for field in ("m_control_0", "m_control_1", "IX_per_amplitude", "ZX_per_amplitude"):
        np.testing.assert_allclose(getattr(flipped, field), getattr(reference, field), rtol=1e-12, atol=1e-12)


def test_cr_susceptibility_rejects_missing_or_ambiguous_control_line() -> None:
    """Implicit drive resolution requires exactly one line targeting the control."""
    control = DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=3, label="control")
    target = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="target")
    chip = Chip([control, target], [Capacitive(control, target, g=0.004)])
    with pytest.raises(ValueError, match="control equipment"):
        analyze_cr_susceptibility(chip, control, target)

    ambiguous, control, target, drive = _driven_pair(two_control_lines=True)
    with pytest.raises(ValueError, match=r"Expected one drive.*found 2"):
        analyze_cr_susceptibility(ambiguous, control, target)

    explicit = analyze_cr_susceptibility(ambiguous, control, target, drive=drive)
    assert explicit.drive == drive.label


def test_cr_susceptibility_rejects_same_device_and_wrong_target_drive() -> None:
    """The analysis refuses a self-edge or a line wired to a different device."""
    chip, control, target, drive = _driven_pair()
    with pytest.raises(ValueError, match="different devices"):
        analyze_cr_susceptibility(chip, control, control, drive=drive)

    target_drive = ChargeDrive(target, label="target_xy")
    chip.wire(target_drive)
    with pytest.raises(ValueError, match="does not target control"):
        analyze_cr_susceptibility(chip, control, target, drive=target_drive)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_cr_susceptibility_matches_the_scheduled_carrier(backend):
    """Reported CR coefficients reproduce the dressed scheduled-drive matrix elements."""
    from quchip import QuantumSequence, Square

    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    chip, control, target, drive = _driven_pair(backend=backend)
    response = analyze_cr_susceptibility(chip, control, target, drive=drive)
    sequence = QuantumSequence(chip)
    amplitude = 0.001
    sequence.schedule(drive, envelope=Square(duration=10.0, amplitude=amplitude), freq=chip.freq(target))
    driven = np.asarray(sequence.hamiltonian().matrix(t=0.0)) - np.asarray(chip.hamiltonian().matrix(t=0.0))
    transitions = []
    for level in (0, 1):
        states = []
        for excitation in (0, 1):
            state = np.asarray(chip.backend.to_array(chip.state(control=level, target=excitation))).ravel()
            pivot = state[3 * level + excitation]
            states.append(state * np.conj(pivot) / abs(pivot))
        transitions.append(np.vdot(states[1], driven @ states[0]) / amplitude)
    np.testing.assert_allclose(response.IX_per_amplitude, transitions[0] + transitions[1], atol=1e-12)
    np.testing.assert_allclose(response.ZX_per_amplitude, transitions[0] - transitions[1], atol=1e-12)
