"""Tests for quchip.analysis.cross_resonance: CR coefficient recovery, bloch_model behavior,
and CRHamiltonianResult structure and re-exports."""

from __future__ import annotations

import numpy as np
import pytest

from quchip.analysis.cross_resonance import (
    CRHamiltonianResult,
    CRSusceptibilityResult,
    analyze_cross_resonance,
    analyze_cr_susceptibility,
    bloch_model,
)
from quchip.chip import Capacitive, Chip
from quchip.control import ChargeDrive, ControlEquipment
from quchip.devices.transmon.duffing import DuffingTransmon


def _make_synthetic_data(
    t: np.ndarray,
    px0: float, py0: float, pz0: float,
    px1: float, py1: float, pz1: float,
    td: float = 1e6,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    """Generate synthetic Bloch traces from known parameters."""
    p0 = bloch_model(t, px0, py0, pz0, td, 0.0, 0.0, 0.0)
    p1 = bloch_model(t, px1, py1, pz1, td, 0.0, 0.0, 0.0)
    if noise_level > 0 and rng is not None:
        p0 = p0 + noise_level * rng.standard_normal(p0.shape)
        p1 = p1 + noise_level * rng.standard_normal(p1.shape)
    return p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]


def _closed_form_x_axis_trace(t: np.ndarray, px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Bloch trace for H = pi * px * X from the +Z initial state."""
    phase = 2 * np.pi * abs(px) * t
    sign = np.sign(px)
    return (
        np.zeros_like(t),
        -sign * np.sin(phase),
        np.cos(phase),
    )


# ---------------------------------------------------------------------------
# bloch_model unit tests
# ---------------------------------------------------------------------------


def test_bloch_model_zero_drive():
    """Zero drive → initial state [0, 0, 1] decays to zero (no precession)."""
    t = np.linspace(0, 1e-6, 50)
    td = 5e-7  # short decay
    m = bloch_model(t, 0.0, 0.0, 0.0, td, 0.0, 0.0, 0.0)
    assert m.shape == (3, len(t))
    assert abs(float(m[2, 0]) - 1.0) < 0.01, "Initial Z should be near 1"
    assert float(m[2, -1]) < 0.5, "Z should decay significantly by end"


def test_bloch_model_pure_z_precession():
    """Pure Z drive → no precession (only Z component)."""
    t = np.linspace(0, 1e-6, 100)
    px, py, pz = 0.0, 0.0, 2e6  # 2 MHz Z drive
    m = bloch_model(t, px, py, pz, 1e8, 0.0, 0.0, 0.0)
    assert m.shape == (3, len(t))
    # Initial state |0> is an eigenstate of sigma_z, so a co-axial drive only phases it.
    assert np.all(np.abs(m[2] - 1.0) < 0.01), "Z eigenstate shouldn't precess under Z drive"


def test_bloch_model_pure_x_precession():
    """Pure X drive → Y/Z oscillate at the drive frequency, X stays 0."""
    t = np.linspace(0, 1e-6, 200)
    f = 3e6  # 3 MHz
    m = bloch_model(t, f, 0.0, 0.0, 1e9, 0.0, 0.0, 0.0)
    assert np.max(np.abs(m[0])) < 0.05, "X should stay near 0 for pure X drive"
    assert np.std(m[1]) > 0.1, "Y should oscillate for pure X drive"


# ---------------------------------------------------------------------------
# analyze_cross_resonance unit tests
# ---------------------------------------------------------------------------


def test_recovery_known_coefficients_noise_free():
    """Recover known CR coefficients from exact (noise-free) Bloch trajectories."""
    # Known CR coefficients in Hz
    IX_true, IY_true, IZ_true = 2.3e6, 2.6e6, 0.1e6
    ZX_true, ZY_true, ZZ_true = 2.6e6, -0.2e6, 0.15e6

    # From I and Z components:
    # ctrl0 parameters: IX + ZX, IY + ZY, IZ + ZZ
    # ctrl1 parameters: IX - ZX, IY - ZY, IZ - ZZ
    px0, py0, pz0 = IX_true + ZX_true, IY_true + ZY_true, IZ_true + ZZ_true
    px1, py1, pz1 = IX_true - ZX_true, IY_true - ZY_true, IZ_true - ZZ_true

    t_ns = np.linspace(0, 500, 80)  # ns
    t_s = t_ns * 1e-9  # seconds for the Bloch model (Hz coefficients)
    c0x, c0y, c0z, c1x, c1y, c1z = _make_synthetic_data(t_s, px0, py0, pz0, px1, py1, pz1)

    result = analyze_cross_resonance(
        t_ns,
        {"x": c0x, "y": c0y, "z": c0z},
        {"x": c1x, "y": c1y, "z": c1z},
    )

    assert isinstance(result, CRHamiltonianResult)

    # 10% tolerance on noise-free data; results are GHz, truths are the
    # Hz-valued Bloch-model inputs.
    tol = 0.10
    for name, true_val_hz in [("IX", IX_true), ("IY", IY_true), ("IZ", IZ_true),
                              ("ZX", ZX_true), ("ZY", ZY_true), ("ZZ", ZZ_true)]:
        val, _ = result.coeffs()[name]
        true_val = true_val_hz * 1e-9
        rel_err = abs(val - true_val) / (abs(true_val) + 1e-18)
        assert rel_err < tol, (
            f"{name}: recovered {val*1e3:.4f} MHz, expected {true_val*1e3:.4f} MHz "
            f"(rel err = {rel_err:.2%})"
        )


@pytest.mark.parametrize(
    ("IX_true", "ZX_true"),
    [
        (1.8e6, 0.0),
        (0.0, 1.4e6),
        (-0.45e6, 1.35e6),
    ],
)
def test_ix_zx_conventions_from_closed_form_x_axis_traces(IX_true, ZX_true):
    """Hand-derived pure-X traces catch IX/ZX sign and decomposition mistakes."""
    t_ns = np.linspace(0, 1200, 140)  # ns (1.2 us)
    t_s = t_ns * 1e-9  # seconds for the closed-form trace (Hz coefficients)
    px0 = IX_true + ZX_true
    px1 = IX_true - ZX_true
    c0x, c0y, c0z = _closed_form_x_axis_trace(t_s, px0)
    c1x, c1y, c1z = _closed_form_x_axis_trace(t_s, px1)

    result = analyze_cross_resonance(
        t_ns,
        {"x": c0x, "y": c0y, "z": c0z},
        {"x": c1x, "y": c1y, "z": c1z},
    )

    npt = np.testing
    npt.assert_allclose(result.IX, IX_true * 1e-9, atol=0.12e6 * 1e-9, rtol=0.08)
    npt.assert_allclose(result.ZX, ZX_true * 1e-9, atol=0.12e6 * 1e-9, rtol=0.08)
    for name in ("IY", "IZ", "ZY", "ZZ"):
        assert abs(getattr(result, name)) < 0.15e6 * 1e-9, f"{name} should stay near zero"


def test_recovery_with_noise():
    """Recover CR coefficients with 5% additive noise at reduced precision."""
    rng = np.random.default_rng(42)
    IX_true, ZX_true = 2.5e6, 2.5e6
    px0, py0, pz0 = IX_true + ZX_true, 0.5e6, 0.1e6
    px1, py1, pz1 = IX_true - ZX_true, -0.5e6, -0.1e6

    t_ns = np.linspace(0, 800, 120)  # ns
    t_s = t_ns * 1e-9  # seconds for the Bloch model (Hz coefficients)
    c0x, c0y, c0z, c1x, c1y, c1z = _make_synthetic_data(
        t_s, px0, py0, pz0, px1, py1, pz1, noise_level=0.05, rng=rng
    )

    result = analyze_cross_resonance(
        t_ns,
        {"x": c0x, "y": c0y, "z": c0z},
        {"x": c1x, "y": c1y, "z": c1z},
    )

    # ZX should be recovered within 20% with noisy data (result GHz, truth Hz)
    zx_val, _ = result.coeffs()["ZX"]
    assert abs(zx_val - ZX_true * 1e-9) / (ZX_true * 1e-9) < 0.20, (
        f"ZX recovery under noise: {zx_val*1e3:.3f} vs {ZX_true*1e-6:.3f} MHz"
    )


def test_zero_amplitude_returns_near_zero():
    """Near-zero drive → all six CR coefficients should be near zero."""
    t_ns = np.linspace(0, 1000, 60)  # ns
    t_s = t_ns * 1e-9  # seconds for the Bloch model (Hz coefficients)
    tiny = 1e3  # 1 kHz drive — effectively zero at MHz scale
    c0x, c0y, c0z, c1x, c1y, c1z = _make_synthetic_data(t_s, tiny, tiny, tiny, -tiny, -tiny, -tiny)

    result = analyze_cross_resonance(
        t_ns,
        {"x": c0x, "y": c0y, "z": c0z},
        {"x": c1x, "y": c1y, "z": c1z},
    )
    for name, (val, _) in result.coeffs().items():
        assert abs(val) < 0.1e6 * 1e-9, f"{name} = {val*1e3:.3f} MHz should be ~0 for near-zero drive"


def test_result_container_structure():
    """CRHamiltonianResult has the right fields and coeffs() method."""
    r = CRHamiltonianResult(
        IX=1e6, IY=2e6, IZ=3e6, ZX=4e6, ZY=5e6, ZZ=6e6,
        IX_err=0.1e6, IY_err=0.1e6, IZ_err=0.1e6,
        ZX_err=0.1e6, ZY_err=0.1e6, ZZ_err=0.1e6,
    )
    coeffs = r.coeffs()
    assert set(coeffs.keys()) == {"IX", "IY", "IZ", "ZX", "ZY", "ZZ"}
    for name, (val, err) in coeffs.items():
        assert isinstance(val, float)
        assert isinstance(err, float)
        assert err >= 0


def test_coeffs_summary_prints(capsys):
    """summary() prints and returns a string."""
    r = CRHamiltonianResult(
        IX=1e6, IY=2e6, IZ=3e6, ZX=4e6, ZY=5e6, ZZ=6e6,
        IX_err=0.1e6, IY_err=0.1e6, IZ_err=0.1e6,
        ZX_err=0.1e6, ZY_err=0.1e6, ZZ_err=0.1e6,
    )
    text = r.summary()
    assert isinstance(text, str)
    assert "MHz" in text
    captured = capsys.readouterr()
    assert "MHz" in captured.out


def test_public_reexport():
    """analyze_cross_resonance and CRHamiltonianResult are importable from quchip."""
    from quchip import CRHamiltonianResult as R2
    from quchip import analyze_cross_resonance as acr2

    assert acr2 is analyze_cross_resonance
    assert R2 is CRHamiltonianResult


def test_analyze_from_quchip_analysis():
    """analyze_cross_resonance is importable from quchip.analysis."""
    from quchip.analysis import analyze_cross_resonance as acr3

    assert acr3 is analyze_cross_resonance


# ---------------------------------------------------------------------------
# Weak-drive susceptibility tests
# ---------------------------------------------------------------------------


def _driven_pair(*, two_control_lines: bool = False, backend: str | None = None):
    control = DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=3, label="control")
    target = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="target")
    drive = ChargeDrive(control, label="control_xy")
    lines = [drive]
    if two_control_lines:
        lines.append(ChargeDrive(control, label="control_xy_2"))
    chip = Chip(
        [control, target],
        [Capacitive(control, target, g=0.004, rwa=True)],
        control_equipment=ControlEquipment(lines),
        rwa=True,
        backend=backend,
    )
    return chip, control, target, drive


def test_cr_susceptibility_is_conditional_dressed_drive_difference() -> None:
    """IX and ZX are the sum and difference of conditioned target-transition elements."""
    chip, control, target, drive = _driven_pair()

    result = analyze_cr_susceptibility(chip, control, target)
    m0 = chip.drive_matrix_elements(({}, {target: 1}), drives=[drive])[drive]
    m1 = chip.drive_matrix_elements(
        ({control: 1}, {control: 1, target: 1}), drives=[drive]
    )[drive]

    assert isinstance(result, CRSusceptibilityResult)
    assert result.m_control_0 == pytest.approx(m0)
    assert result.m_control_1 == pytest.approx(m1)
    assert result.IX_per_amplitude == pytest.approx(m0 + m1)
    assert result.ZX_per_amplitude == pytest.approx(m0 - m1)
    assert result.control == control.label
    assert result.target == target.label
    assert result.drive == drive.label


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

    def sign_flipped_eigensystem():
        eigenvalues, eigenvectors, overlaps, labeling = compute()
        signs = np.where(np.arange(eigenvectors.shape[1]) % 2, -1.0, 1.0)
        return eigenvalues, eigenvectors * signs[None, :], overlaps, labeling

    monkeypatch.setattr(chip._analysis, "_compute_array_labeled", sign_flipped_eigensystem)
    flipped = analyze_cr_susceptibility(chip, control, target, drive=drive)

    for field in ("m_control_0", "m_control_1", "IX_per_amplitude", "ZX_per_amplitude"):
        np.testing.assert_allclose(getattr(flipped, field), getattr(reference, field), rtol=1e-12, atol=1e-12)


def test_cr_susceptibility_resolves_labels_and_explicit_drive() -> None:
    """Control, target, and drive accept labels as well as objects."""
    chip, control, target, drive = _driven_pair()

    by_object = analyze_cr_susceptibility(chip, control, target, drive=drive)
    by_label = analyze_cr_susceptibility(chip, control.label, target.label, drive=drive.label)

    assert by_label.m_control_0 == pytest.approx(by_object.m_control_0)
    assert by_label.m_control_1 == pytest.approx(by_object.m_control_1)


def test_cr_susceptibility_rejects_missing_or_ambiguous_control_line() -> None:
    """Implicit drive resolution requires exactly one line targeting the control."""
    control = DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=3, label="control")
    target = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="target")
    chip = Chip([control, target], [Capacitive(control, target, g=0.004, rwa=True)], rwa=True)
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


def test_cr_susceptibility_public_reexports() -> None:
    """The susceptibility result and analysis are available from both public surfaces."""
    from quchip import CRSusceptibilityResult as TopResult
    from quchip import analyze_cr_susceptibility as top_analysis
    from quchip.analysis import CRSusceptibilityResult as AnalysisResult
    from quchip.analysis import analyze_cr_susceptibility as analysis_function

    assert TopResult is CRSusceptibilityResult
    assert AnalysisResult is CRSusceptibilityResult
    assert top_analysis is analyze_cr_susceptibility
    assert analysis_function is analyze_cr_susceptibility
