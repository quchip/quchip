"""Physics-verifying dressed-state tests for Chip.dress/energy/transition_freq.

All expected values are derived analytically from the dispersive model,
not from running implementation code first.

System under test (matches TestDispersiveShift in tests/test_physics.py):
    - Transmon:  w_q = 5.0 GHz, alpha = -0.25 GHz, levels = 4
    - Resonator: w_r = 7.0 GHz, levels = 10
    - Coupling:  g = 0.05 GHz

Perturbative dispersive formula:
    chi_pert = g^2 * alpha / [Delta * (Delta + alpha)] with Delta = w_q - w_r

For these parameters:
    Delta = -2.0
    chi_pert ~ -1.389e-4 GHz

In this codebase, the observable number-splitting between
n=1 and n=0 manifolds is 2*chi_pert.  This is computed via
energy arithmetic: E(q=1,r=1) - E(q=1,r=0) - E(q=0,r=1) + E(q=0,r=0).
"""

from __future__ import annotations

import numpy as np
import pytest
import warnings

from quchip.chip import Chip, DressedResult
from quchip.chip.couplings import Capacitive
from quchip.backend.qutip import QuTiPBackend
from quchip.control import ChargeDrive, ControlEquipment
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon


OMEGA_Q = 5.0
OMEGA_R = 7.0
ALPHA = -0.25
G = 0.05
Q_LEVELS = 4
R_LEVELS = 10


@pytest.fixture
def dispersive_system():
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


class TestDressedStates:
    """Verify dressed-state computation and state assignment surfaces."""

    def test_dress_returns_dressed_result(self, dispersive_system) -> None:
        """dress() returns a populated DressedResult with sorted eigenvalues."""
        chip, _, _ = dispersive_system

        result = chip.dress()

        assert isinstance(result, DressedResult)
        assert isinstance(result.eigenvalues, np.ndarray)
        assert isinstance(result.state_map, dict)
        assert isinstance(result.dressed_eigenvalues, dict)

        assert result.eigenvalues.size > 0
        assert len(result.eigenstates) == result.eigenvalues.size
        assert len(result.state_map) > 0
        assert len(result.dressed_eigenvalues) > 0

        eigenvalues = np.real(result.eigenvalues)
        assert np.all(np.diff(eigenvalues) >= -1e-12), "Dressed eigenvalues are not sorted ascending"

    def test_dressed_frequency_is_chip_derived_not_device_state(self, dispersive_system) -> None:
        """Per-device dressed frequencies are derived from the owning chip, not stored on devices."""
        chip, qubit, resonator = dispersive_system

        assert "_dressed_freq" not in qubit.__dict__
        assert "_dressed_freq" not in resonator.__dict__

        q_freq = chip.freq(qubit)
        r_freq = chip.freq(resonator)

        assert qubit.dressed_freq == pytest.approx(q_freq)
        assert resonator.dressed_freq == pytest.approx(r_freq)

        # Perturbative coupling: dressed frequencies remain close to bare.
        assert abs(qubit.dressed_freq - OMEGA_Q) < 0.05
        assert abs(resonator.dressed_freq - OMEGA_R) < 0.05

    def test_drive_freq_uses_dressed(self, dispersive_system) -> None:
        """drive_freq resolves to the chip-derived dressed frequency."""
        chip, qubit, resonator = dispersive_system

        assert qubit.drive_freq == pytest.approx(chip.freq(qubit))
        assert resonator.drive_freq == pytest.approx(chip.freq(resonator))

        # Shifts are non-zero in this coupled system.
        assert abs(qubit.drive_freq - OMEGA_Q) > 1e-6
        assert abs(resonator.drive_freq - OMEGA_R) > 1e-6

    def test_state_map_covers_low_energy(self, dispersive_system) -> None:
        """state_map must include |0,0>, |1,0>, and |0,1> labels."""
        chip, _, _ = dispersive_system

        result = chip.dress()

        for label in ((0, 0), (1, 0), (0, 1)):
            assert label in result.state_map
            assert label in result.dressed_eigenvalues

    def test_single_device_has_no_hybridized_labels(self) -> None:
        """A single-device chip has no hybridization and no hybridization warning."""
        chip = Chip([Resonator(freq=6.0, levels=4, label="r")])

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = chip.dress()

        assert result.hybridized_labels == ()
        assert result.assignment_overlaps[(0,)] == pytest.approx(1.0)
        assert not [warning for warning in captured if "Strong hybridization detected" in str(warning.message)]

    def test_resonant_system_exposes_hybridization_diagnostics(self) -> None:
        """Resonant coupling produces hybridized labels, overlap fractions, and one warning."""
        r_a = Resonator(freq=6.0, levels=4, label="r_a")
        r_b = Resonator(freq=6.0, levels=4, label="r_b")
        chip = Chip([r_a, r_b], [Capacitive(r_a, r_b, g=0.05)])

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = chip.dress()

        assert result.hybridized_labels
        assert (0, 3) in result.hybridized_labels
        assert (3, 0) in result.hybridized_labels
        assert result.assignment_overlaps[(0, 3)] == pytest.approx(0.375, abs=1e-7)
        assert result.assignment_overlaps[(3, 0)] == pytest.approx(0.375, abs=1e-7)

        hybridization_warnings = [
            warning for warning in captured if "Strong hybridization detected" in str(warning.message)
        ]
        assert len(hybridization_warnings) == 1
        assert "assignment_overlaps" in str(hybridization_warnings[0].message)

    def test_overlap_threshold_controls_hybridization_flags(self) -> None:
        """Relaxing overlap_threshold drops labels from hybridized_labels."""
        r_a = Resonator(freq=6.0, levels=4, label="r_a")
        r_b = Resonator(freq=6.0, levels=4, label="r_b")
        chip = Chip([r_a, r_b], [Capacitive(r_a, r_b, g=0.05)])

        default = chip.dress()
        relaxed = chip.dress(overlap_threshold=0.2, force=True)

        assert (0, 3) in default.hybridized_labels
        assert (3, 0) in default.hybridized_labels
        assert (0, 3) not in relaxed.hybridized_labels
        assert (3, 0) not in relaxed.hybridized_labels

    def test_vectorized_assignment_matches_bruteforce_greedy_overlap(self) -> None:
        """state_map matches a brute-force greedy overlap assignment computed independently."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=3, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.05)])
        result = chip.dress()

        backend = chip.backend
        bare_labels = list(result.bare_labels)
        overlaps: dict[tuple[tuple[int, ...], int], float] = {}
        for bare_label in bare_labels:
            basis_states = [backend.basis(dev.levels, k) for dev, k in zip(chip.devices, bare_label)]
            bare_state = backend.tensor_states(*basis_states)
            for dressed_idx, estate in enumerate(result.eigenstates):
                overlaps[(bare_label, dressed_idx)] = float(abs(backend.overlap(bare_state, estate)) ** 2)

        assigned_bare: set[tuple[int, ...]] = set()
        assigned_dressed: set[int] = set()
        reference_map: dict[tuple[int, ...], int] = {}
        for (bare_label, dressed_idx), overlap in sorted(overlaps.items(), key=lambda item: item[1], reverse=True):
            if bare_label in assigned_bare or dressed_idx in assigned_dressed:
                continue
            reference_map[bare_label] = dressed_idx
            assigned_bare.add(bare_label)
            assigned_dressed.add(dressed_idx)

        assert result.state_map == reference_map


class TestBackendEigensystemData:
    def test_qutip_backend_eigensystem_data_returns_consistent_matrix(self) -> None:
        """QuTiPBackend.eigensystem_data reports eigenvalues, eigenvector matrix, and eigenstates consistently."""
        backend = QuTiPBackend()
        H = backend.number(3)
        data = backend.eigensystem_data(H)

        np.testing.assert_allclose(np.asarray(data.eigenvalues, dtype=float), [0.0, 1.0, 2.0])
        np.testing.assert_allclose(np.abs(np.asarray(data.eigenvector_matrix, dtype=complex)), np.eye(3))
        assert len(data.eigenstates) == 3

    @pytest.mark.optional_backend
    def test_dynamiqs_backend_eigensystem_data_returns_consistent_matrix(self) -> None:
        """DynamiqsBackend.eigensystem_data reports eigenvalues, eigenvector matrix, and eigenstates consistently."""
        pytest.importorskip("dynamiqs")
        from quchip.backend.dynamiqs import DynamiqsBackend

        backend = DynamiqsBackend()
        H = backend.number(3)
        data = backend.eigensystem_data(H)

        np.testing.assert_allclose(np.asarray(data.eigenvalues, dtype=float), [0.0, 1.0, 2.0])
        np.testing.assert_allclose(np.abs(np.asarray(data.eigenvector_matrix, dtype=complex)), np.eye(3))
        assert len(data.eigenstates) == 3


class TestEnergy:
    """Verify energy() extraction against perturbation theory."""

    def test_energy_matches_perturbation_theory(self, dispersive_system) -> None:
        """Dispersive shift via energy arithmetic matches perturbative prediction."""
        chip, _, _ = dispersive_system

        # chi = E(q=1,r=1) - E(q=1,r=0) - E(q=0,r=1) + E(q=0,r=0)
        chi_numeric = chip.energy(q=1, r=1) - chip.energy(q=1, r=0) - chip.energy(q=0, r=1) + chip.energy(q=0, r=0)

        delta = OMEGA_Q - OMEGA_R
        chi_pert = (G**2 * ALPHA) / (delta * (delta + ALPHA))

        # Observable number splitting = 2*chi_pert
        expected = 2.0 * chi_pert

        assert chi_numeric == pytest.approx(expected, rel=0.15)

    def test_energy_auto_dresses(self, dispersive_system) -> None:
        """energy() auto-dresses an undressed chip before evaluation."""
        chip, _, _ = dispersive_system

        assert chip._analysis._dressed_result is None
        e0 = chip.energy(q=0)
        assert isinstance(e0, float)

    def test_energy_symmetric(self, dispersive_system) -> None:
        """Dispersive shift computed both ways should be equal."""
        chip, _, _ = dispersive_system

        chi_qr = chip.energy(q=1, r=1) - chip.energy(q=1, r=0) - chip.energy(q=0, r=1) + chip.energy(q=0, r=0)
        chi_rq = chip.energy(q=1, r=1) - chip.energy(q=0, r=1) - chip.energy(q=1, r=0) + chip.energy(q=0, r=0)

        assert chi_qr == pytest.approx(chi_rq, abs=1e-14)

    def test_dispersive_shift_matches_energy_arithmetic(self, dispersive_system) -> None:
        """dispersive_shift() matches the equivalent energy() arithmetic."""
        chip, qubit, resonator = dispersive_system

        direct = chip.dispersive_shift(qubit, resonator)
        reference = chip.energy(q=1, r=1) - chip.energy(q=1, r=0) - chip.energy(q=0, r=1) + chip.energy(q=0, r=0)

        assert direct == pytest.approx(reference, abs=1e-14)

    def test_dressed_anharmonicity_matches_energy_arithmetic(self, dispersive_system) -> None:
        """dressed_anharmonicity() matches the equivalent energy() arithmetic."""
        chip, qubit, _ = dispersive_system

        direct = chip.dressed_anharmonicity(qubit)
        reference = chip.energy(q=2) - 2.0 * chip.energy(q=1) + chip.energy(q=0)

        assert direct == pytest.approx(reference, abs=1e-14)


def test_static_zz_matches_energy_arithmetic() -> None:
    """static_zz() matches the equivalent energy() arithmetic."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    chip = Chip([q0, q1], [Capacitive(q0, q1, g=0.02)])

    direct = chip.static_zz(q0, q1)
    manual = (
        chip.energy(q0=1, q1=1)
        - chip.energy(q0=1, q1=0)
        - chip.energy(q0=0, q1=1)
        + chip.energy(q0=0, q1=0)
    )

    assert direct == pytest.approx(manual)


class TestTransitionFreq:
    """Verify conditional dressed transition frequencies."""

    def test_transition_freq_vacuum(self, dispersive_system) -> None:
        """Vacuum-conditioned transition stays near bare qubit frequency."""
        chip, qubit, _ = dispersive_system

        chip.dress()
        freq_vacuum = chip.freq(qubit)

        assert freq_vacuum == pytest.approx(qubit.dressed_freq)
        assert abs(freq_vacuum - OMEGA_Q) < 0.05

    def test_transition_freq_conditional(self, dispersive_system) -> None:
        """n=1 conditional transition shift matches perturbative splitting."""
        chip, qubit, resonator = dispersive_system

        chip.dress()

        freq_vacuum = chip.freq(qubit)
        freq_conditional = chip.freq(qubit, when={resonator: 1})
        delta_freq = freq_conditional - freq_vacuum

        delta = OMEGA_Q - OMEGA_R
        chi_pert = (G**2 * ALPHA) / (delta * (delta + ALPHA))
        expected_delta = 2.0 * chi_pert

        assert delta_freq == pytest.approx(expected_delta, rel=0.15)

        chi_energy = chip.energy(q=1, r=1) - chip.energy(q=1, r=0) - chip.energy(q=0, r=1) + chip.energy(q=0, r=0)
        assert delta_freq == pytest.approx(chi_energy, abs=1e-14)

    def test_transition_freq_bool_rejection(self, dispersive_system) -> None:
        """Boolean Fock indices in `when` are explicitly rejected."""
        chip, qubit, resonator = dispersive_system

        chip.dress()

        with pytest.raises(ValueError, match="got bool"):
            chip.freq(qubit, when={resonator: True})

    def test_transition_freq_auto_dresses(self, dispersive_system) -> None:
        """freq() works without a prior dress() call by routing through the array kernel."""
        chip, qubit, _ = dispersive_system

        assert chip._analysis._dressed_result is None
        assert chip._analysis._array_cache is None
        freq = chip.freq(qubit)
        assert isinstance(freq, float)
        # The trace-friendly array path populates _array_cache; the eager
        # dict-based DressedResult is only built on explicit dress() calls.
        assert chip._analysis._array_cache is not None


class TestLookupHelpers:
    def test_dressed_index_and_bare_label_round_trip(self, dispersive_system) -> None:
        """dressed_index() and bare_label() round-trip a bare Fock label."""
        chip, _, _ = dispersive_system
        chip.dress()

        dressed_idx = chip.dressed_index(q=1, r=0)

        assert dressed_idx is not None
        assert chip.bare_label(dressed_idx) == (1, 0)

    def test_state_components_are_normalized(self, dispersive_system) -> None:
        """state_components() weights sum to 1."""
        chip, _, _ = dispersive_system
        chip.dress()

        components = chip.state_components(0, n_components=len(chip._ensure_dressed().bare_labels))

        assert components
        assert sum(components.values()) == pytest.approx(1.0, abs=1e-12)

    def test_operator_in_dressed_basis_preserves_hermiticity(self, dispersive_system) -> None:
        """operator_in_dressed_basis() of a Hermitian operator stays Hermitian after truncation."""
        chip, qubit, _ = dispersive_system
        dressed_op = chip.operator_in_dressed_basis(qubit, "Z", truncate=4)
        dense = np.asarray(chip.backend.to_array(dressed_op), dtype=complex)

        assert dense.shape == (4, 4)
        np.testing.assert_allclose(dense, dense.conj().T, atol=1e-12)


class TestDriveMatrixElements:
    """Verify dressed drive operators against an explicit basis transform."""

    @staticmethod
    def _driven_pair():
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
        q2 = DuffingTransmon(freq=5.3, anharmonicity=-0.24, levels=3, label="q2")
        d1 = ChargeDrive(q1, label="d1")
        d2 = ChargeDrive(q2, label="d2")
        chip = Chip(
            [q1, q2],
            [Capacitive(q1, q2, g=0.03)],
            control_equipment=ControlEquipment([d1, d2]),
        )
        return chip, q1, q2, d1, d2

    def test_device_transition_matches_explicit_final_initial_matrix_element(self) -> None:
        """Device shorthand returns ``<1~|D_j|0~>`` with object and label lookup."""
        chip, q1, q2, d1, d2 = self._driven_pair()

        elements = chip.drive_matrix_elements(q1, drives=[d1, d2.label])

        initial = chip.dressed_index({q1: 0})
        final = chip.dressed_index({q1: 1})
        assert initial is not None and final is not None
        for drive, target in ((d1, q1), (d2, q2)):
            channel = drive.local_channels(target)[0]
            dressed = chip.operator_in_dressed_basis(target, channel.operator)
            explicit = np.asarray(chip.backend.to_array(dressed), dtype=complex)[final, initial]
            assert elements[drive] == pytest.approx(explicit, abs=1e-12)
            assert elements[drive.label] == pytest.approx(explicit, abs=1e-12)

    def test_explicit_transition_supports_arbitrary_bare_state_mappings(self) -> None:
        """Explicit ``(initial, final)`` mappings select arbitrary dressed transitions."""
        chip, q1, q2, d1, _ = self._driven_pair()
        transition = ({q1: 0, q2: 1}, {q1: 1, q2: 1})

        element = chip.drive_matrix_elements(transition, drives=[d1])[d1]

        initial = chip.dressed_index(transition[0])
        final = chip.dressed_index(transition[1])
        assert initial is not None and final is not None
        channel = d1.local_channels(q1)[0]
        dressed = chip.operator_in_dressed_basis(q1, channel.operator)
        explicit = np.asarray(chip.backend.to_array(dressed), dtype=complex)[final, initial]
        assert element == pytest.approx(explicit, abs=1e-12)

    def test_missing_equipment_and_unknown_drive_report_available_lines(self) -> None:
        """Drive resolution failures identify missing equipment and available labels."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        unwired = Chip([q])
        with pytest.raises(ValueError, match="control equipment"):
            unwired.drive_matrix_elements(q)

        chip, q1, _, _, _ = self._driven_pair()
        with pytest.raises(KeyError, match=r"missing.*Available.*d1.*d2"):
            chip.drive_matrix_elements(q1, drives=["missing"])

    def test_multiple_local_channels_are_rejected_as_ambiguous(self) -> None:
        """A drive with multiple local operators requires an explicit projection policy."""
        class TwoChannelDrive(ChargeDrive):
            def local_channels(self, device):
                channel = super().local_channels(device)[0]
                return [channel, channel]

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = TwoChannelDrive(q, label="two_channel")
        chip = Chip([q], control_equipment=ControlEquipment([drive]))

        with pytest.raises(ValueError, match=r"2 local Hamiltonian channels.*exactly one"):
            chip.drive_matrix_elements(q, drives=[drive])


class TestCacheInvalidation:
    """Verify frame changes do NOT invalidate dressed-state cache."""

    def test_set_frame_preserves_dressed_result(self, dispersive_system) -> None:
        """set_frame() should preserve the cached dressed result."""
        chip, _, _ = dispersive_system

        result = chip.dress()
        assert chip._analysis._dressed_result is result

        chip.set_frame("rotating")
        assert chip._ensure_dressed() is result  # Cache preserved

    def test_static_parameter_change_invalidates_dressed_cache(self, dispersive_system) -> None:
        """Mutating a device parameter invalidates the cached dressed result."""
        chip, qubit, _ = dispersive_system

        original = chip.dress()
        qubit.freq += 0.1

        refreshed = chip._ensure_dressed()

        assert refreshed is not original
        assert refreshed.eigenvalues[1] != pytest.approx(original.eigenvalues[1])

    def test_hamiltonian_is_frame_independent(self, dispersive_system) -> None:
        """chip.hamiltonian() always returns lab-frame Hamiltonian."""
        chip, qubit, resonator = dispersive_system

        chip.dress()
        H_lab = chip.hamiltonian()
        chip.set_frame("rotating")
        H_rotating = chip.hamiltonian()

        assert (H_lab - H_rotating).norm() < 1e-12

    def test_chi_removed(self, dispersive_system) -> None:
        """chip.chi() is not part of the public API."""
        chip, qubit, resonator = dispersive_system
        chip.dress()
        assert not hasattr(chip, "chi")

    def test_resolve_frame_rotating_returns_dressed(self, dispersive_system) -> None:
        """resolve_frame(rotating) uses dressed frequencies."""
        from quchip.engine.stage1_frames import resolve_frame

        chip, qubit, resonator = dispersive_system
        chip.dress()
        chip.set_frame("rotating")
        freqs = resolve_frame(chip, chip.frame).frequencies
        # Should use dressed frequencies (dev.drive_freq after dressing)
        assert freqs["q"] == pytest.approx(qubit.drive_freq)
        assert freqs["r"] == pytest.approx(resonator.drive_freq)


def test_effective_subspace_hamiltonian_lowdin_on_bus_coupled_pair() -> None:
    """Loewdin orthonormalization yields the exchange coupling off-diagonally and dressed energies as eigenvalues."""
    q0 = DuffingTransmon(freq=5.00, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.10, anharmonicity=-0.25, levels=3, label="q1")
    bus = Resonator(freq=6.35, levels=4, label="bus")
    chip = Chip([q0, q1, bus], [Capacitive(q0, bus, g=0.075), Capacitive(q1, bus, g=0.075)])

    # Löwdin orthonormalization prevents ~17 MHz of absolute-energy leakage from dwarfing the ~4.3 MHz exchange.
    effective = chip.effective_subspace_hamiltonian(({q0: 1, q1: 0}, {q0: 0, q1: 1}))

    g, d0, d1 = 0.075, 5.00 - 6.35, 5.10 - 6.35
    j_sw = g**2 / 2 * (1 / d0 + 1 / d1)
    assert np.real(effective[0, 1]) == pytest.approx(j_sw, abs=1e-4)

    expected = sorted([chip.energy({q0: 1}), chip.energy({q1: 1})])
    assert np.linalg.eigvalsh(effective) == pytest.approx(expected, abs=1e-12)
