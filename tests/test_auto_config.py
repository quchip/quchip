"""Tests for lazy auto-dressing and derived Chip quantities (energy, state, freq, frame_info).

System: DuffingTransmon (5.0 GHz, alpha -0.25 GHz, 4 levels) capacitively coupled (g=0.05 GHz)
to a 10-level Resonator (7.0 GHz).
"""

from __future__ import annotations

import pytest

from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon


OMEGA_Q = 5.0
OMEGA_R = 7.0
ALPHA = -0.25
G = 0.05
Q_LEVELS = 4
R_LEVELS = 10


def _squared_overlap(chip: Chip, left: object, right: object) -> float:
    """Return |⟨left|right⟩|² through the active backend protocol."""
    overlap = chip.backend.overlap(left, right)
    return float(abs(overlap) ** 2)


@pytest.fixture
def dispersive_chip():
    """Build the shared transmon-resonator test system."""
    qubit = DuffingTransmon(
        freq=OMEGA_Q,
        anharmonicity=ALPHA,
        levels=Q_LEVELS,
        label="q",
    )
    resonator = Resonator(freq=OMEGA_R, levels=R_LEVELS, label="r")
    coupling = Capacitive(qubit, resonator, g=G)
    chip = Chip(devices=[qubit, resonator], couplings=[coupling])
    return chip, qubit, resonator


class TestAutoConfig:
    """Tests for auto-dress and derived Chip energy/state APIs."""

    def test_auto_dress_ensure_dressed(self, dispersive_chip) -> None:
        """_ensure_dressed() calls dress() when _dressed_result is None."""
        chip, _, _ = dispersive_chip
        assert chip._analysis._dressed_result is None
        result = chip._ensure_dressed()
        assert result is not None
        assert chip._analysis._dressed_result is not None
        result2 = chip._ensure_dressed()
        assert result2 is result

    def test_energy_auto_dresses(self, dispersive_chip) -> None:
        """chip.energy(q=0) works without a prior dress() call."""
        chip, _, _ = dispersive_chip
        assert chip._analysis._dressed_result is None
        e0 = chip.energy(q=0)
        assert isinstance(e0, float)

    def test_energy_accepts_device_keyed_mapping(self, dispersive_chip) -> None:
        """chip.energy({device: level}) matches the keyword form."""
        chip, qubit, resonator = dispersive_chip

        mapping_value = chip.energy({qubit: 1, resonator: 0})
        keyword_value = chip.energy(q=1, r=0)

        assert mapping_value == pytest.approx(keyword_value)

    def test_energy_matches_perturbation_theory(self, dispersive_chip) -> None:
        """Dispersive shift via energy arithmetic matches perturbation theory."""
        chip, _, _ = dispersive_chip

        # chi = E(q=1,r=1) - E(q=1,r=0) - E(q=0,r=1) + E(q=0,r=0)
        chi_numeric = chip.energy(q=1, r=1) - chip.energy(q=1, r=0) - chip.energy(q=0, r=1) + chip.energy(q=0, r=0)

        delta = OMEGA_Q - OMEGA_R
        chi_pert = (G**2 * ALPHA) / (delta * (delta + ALPHA))
        expected = 2.0 * chi_pert

        assert chi_numeric == pytest.approx(expected, rel=0.15)

    def test_energy_invalid_label_raises(self, dispersive_chip) -> None:
        """energy() with invalid state label raises KeyError."""
        chip, _, _ = dispersive_chip
        # q=5 exceeds Q_LEVELS=4, so label won't exist in dressed_eigenvalues
        with pytest.raises(KeyError, match="Available"):
            chip.energy(q=5, r=0)

    def test_state_returns_dressed_eigenstate(self, dispersive_chip) -> None:
        """chip.state(q=0, r=0) returns dressed eigenstate with high overlap to bare."""
        chip, _, _ = dispersive_chip
        ds = chip.state(q=0, r=0)
        bare_ground = chip.bare_state(q=0, r=0)
        overlap = _squared_overlap(chip, bare_ground, ds)
        assert overlap > 0.9, f"Overlap with bare ground = {overlap}, expected > 0.9"

    def test_state_auto_dresses(self, dispersive_chip) -> None:
        """state() auto-dresses if needed."""
        chip, _, _ = dispersive_chip
        assert chip._analysis._dressed_result is None
        ds = chip.state(q=0, r=0)
        assert ds is not None
        assert chip._analysis._dressed_result is not None

    def test_state_accepts_device_keyed_mapping(self, dispersive_chip) -> None:
        """chip.state({device: level}) matches the keyword form."""
        chip, qubit, resonator = dispersive_chip
        ds = chip.state({qubit: 0, resonator: 0})
        bare_ground = chip.bare_state({qubit: 0, resonator: 0})
        overlap = _squared_overlap(chip, bare_ground, ds)
        assert overlap > 0.9

    def test_dressed_state_overlap_matches_backend_protocol(self, dispersive_chip) -> None:
        """Dressed ground-state assignment remains numerically close to the bare ground."""
        chip, _, _ = dispersive_chip
        dressed = chip.state(q=0, r=0)
        bare = chip.bare_state(q=0, r=0)
        assert _squared_overlap(chip, bare, dressed) == pytest.approx(1.0, abs=0.1)

    def test_freq_returns_dict(self, dispersive_chip) -> None:
        """freq() returns a plain label-keyed dict."""
        chip, qubit, resonator = dispersive_chip
        freqs = chip.freq()
        assert isinstance(freqs, dict)
        assert "q" in freqs
        assert "r" in freqs
        assert isinstance(freqs["q"], float)
        assert isinstance(freqs["r"], float)
        # Near bare frequencies (weak coupling)
        assert abs(freqs["q"] - OMEGA_Q) < 0.05
        assert abs(freqs["r"] - OMEGA_R) < 0.05

    def test_frame_info_flat_labels(self, dispersive_chip) -> None:
        """frame_info() is a flat {label: ω_ref} dict — no nesting."""
        chip, _, _ = dispersive_chip
        info = chip.frame_info()
        assert isinstance(info, dict)
        assert set(info.keys()) == {"q", "r"}
        for val in info.values():
            assert val == pytest.approx(0.0)

    def test_frame_info_matches_stage2_subtraction(self, dispersive_chip) -> None:
        """frame_info() matches the per-device reference (-Σ ω_ref,i n̂_i) stage2 subtracts."""
        from quchip.engine.stage1_frames import resolve_frame

        chip, qubit, resonator = dispersive_chip
        chip.set_frame("rotating")

        # Chip.hamiltonian() is lab-frame; the rotating-frame subtraction happens at solve
        # time via stage2's _build_static_h0, using the same resolver invoked here.
        resolved = resolve_frame(chip, chip.frame)

        info = chip.frame_info()
        assert info == dict(resolved.frequencies)
        assert info["q"] == pytest.approx(qubit.drive_freq)
        assert info["r"] == pytest.approx(resonator.drive_freq)

    def test_resolved_frame_rotating_returns_dressed(self, dispersive_chip) -> None:
        """In rotating mode, resolved frame frequencies match dressed freqs."""
        from quchip.engine.stage1_frames import resolve_frame

        chip, qubit, resonator = dispersive_chip
        chip.dress()
        chip.set_frame("rotating")
        resolved = resolve_frame(chip, chip.frame)
        freqs = resolved.frequencies
        assert freqs["q"] == pytest.approx(qubit.drive_freq)
        assert freqs["r"] == pytest.approx(resonator.drive_freq)
        assert abs(freqs["q"] - OMEGA_Q) > 1e-6
        assert abs(freqs["r"] - OMEGA_R) > 1e-6

    def test_resolved_frame_lab_returns_zeros(self, dispersive_chip) -> None:
        """In lab mode, resolved frame frequencies are zeros."""
        from quchip.engine.stage1_frames import resolve_frame

        chip, _, _ = dispersive_chip
        chip.set_frame("lab")
        freqs = resolve_frame(chip, chip.frame).frequencies
        for val in freqs.values():
            assert val == pytest.approx(0.0)

    def test_is_dressed_property(self, dispersive_chip) -> None:
        """is_dressed is False before dress, True after."""
        chip, _, _ = dispersive_chip
        assert chip.is_dressed is False
        chip.dress()
        assert chip.is_dressed is True
