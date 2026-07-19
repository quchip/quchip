"""Chip assembly and frame-spec tests checked against closed-form eigenvalue formulas.

chip.hamiltonian() is always lab-frame; frame behavior resolves at simulation time via resolve_frame.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from quchip.backend.protocol import Backend
from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive, Coupling
from quchip.control.drive import ChargeDrive
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon


# ---------------------------------------------------------------------------
# TestChipHamiltonian — System Hamiltonian construction
# ---------------------------------------------------------------------------


class TestChipHamiltonian:
    """Verify system Hamiltonian eigenvalues against analytical formulas."""

    def test_single_device_hamiltonian(self, backend: Backend) -> None:
        """DuffingTransmon(freq=5.0, α=-0.25, levels=3) eigenvalues match E_n = ω·n + (α/2)·n·(n−1)."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])
        H = chip.hamiltonian()
        evals = np.sort(np.real(backend.eigenenergies(H)))

        expected = np.array([0.0, 5.0, 9.75])
        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_two_device_hamiltonian_no_coupling(self, backend: Backend) -> None:
        """Uncoupled two-device eigenvalues equal the tensor sums of each device's own eigenvalues."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r])
        H = chip.hamiltonian()
        evals = np.sort(np.real(backend.eigenenergies(H)))

        q_evals = [0.0, 5.0, 9.75]
        r_evals = [0.0, 6.0, 12.0, 18.0]
        expected = np.sort([eq + er for eq in q_evals for er in r_evals])
        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_coupled_system_hamiltonian_hermiticity(self, backend: Backend) -> None:
        """Coupled system Hamiltonian must be Hermitian."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        coupling = Capacitive(q, r, g=0.02)
        chip = Chip(devices=[q, r], couplings=[coupling])
        H = chip.hamiltonian()

        H_dag = backend.dag(H)
        diff = (H - H_dag).norm()
        assert diff < 1e-12, f"H is not Hermitian: ||H - H†|| = {diff}"

    def test_coupled_system_hamiltonian_dimension(self) -> None:
        """Coupled system dimension = product of device levels (3 x 4 = 12)."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        coupling = Capacitive(q, r, g=0.02)
        chip = Chip(devices=[q, r], couplings=[coupling])
        H = chip.hamiltonian()

        assert H.shape == (12, 12)

    def test_coupled_system_eigenvalue_perturbation(self, backend: Backend) -> None:
        """Coupling shifts eigenvalues by ~O(g) from the uncoupled tensor-sum values."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        coupling = Capacitive(q, r, g=0.02)
        chip = Chip(devices=[q, r], couplings=[coupling])
        H = chip.hamiltonian()
        evals = np.sort(np.real(backend.eigenenergies(H)))

        q_evals = [0.0, 5.0, 9.75]
        r_evals = [0.0, 6.0, 12.0, 18.0]
        uncoupled = np.sort([eq + er for eq in q_evals for er in r_evals])

        # Ground state should be very close (second-order shift ~g²/Δ)
        assert abs(evals[0] - uncoupled[0]) < 0.01

        for i in range(len(evals)):
            assert abs(evals[i] - uncoupled[i]) < 0.1, (
                f"Eigenvalue {i}: coupled={evals[i]:.6f}, "
                f"uncoupled={uncoupled[i]:.6f}, "
                f"shift={abs(evals[i] - uncoupled[i]):.6f}"
            )

    def test_exact_zero_coupling_emits_no_rwa_vanish_warning(self) -> None:
        """A coupling whose interaction Hamiltonian is exactly zero builds silently, without the RWA-vanish warning."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r], couplings=[Capacitive(q, r, g=0.0)])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chip.hamiltonian()

    def test_nonzero_fully_rwa_rejected_coupling_warns(self) -> None:
        """A nonzero interaction whose every populated band the RWA predicate rejects still warns."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        # a⊗b + a†⊗b† is the two-mode-squeezing term: both populated bands
        # violate the default number-conserving predicate (Δa + Δb == 0).
        squeezing = Coupling(
            q, r, g=0.05,
            interaction=lambda a, b, bk: (
                bk.tensor(a.lowering_operator(), b.lowering_operator())
                + bk.tensor(bk.dag(a.lowering_operator()), bk.dag(b.lowering_operator()))
            ),
        )
        chip = Chip(devices=[q, r], couplings=[squeezing])

        with pytest.warns(UserWarning, match="vanishes entirely under the resolved RWA"):
            chip.hamiltonian()


# ---------------------------------------------------------------------------
# TestRotatingFrame — Frame management
# ---------------------------------------------------------------------------


class TestFrameSpec:
    """Verify frame-spec APIs and frame-resolution behavior."""

    def test_lab_frame_is_default(self) -> None:
        """Chip with no frame args should have frame='lab'."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])
        assert chip.frame == "lab"

    def test_hamiltonian_frame_independent(self, backend: Backend) -> None:
        """hamiltonian() is always lab-frame regardless of frame spec."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])
        H_lab = chip.hamiltonian()
        chip.set_frame("rotating")
        H_rot = chip.hamiltonian()
        assert (H_lab - H_rot).norm() < 1e-12

    def test_invalid_frame_raises(self) -> None:
        """Invalid frame string raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])
        with pytest.raises(ValueError, match="frame string must be"):
            chip.set_frame("invalid")

    def test_float_and_dict_frame_specs(self) -> None:
        """set_frame accepts float and per-device dict specs."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=3, label="r")
        chip = Chip(devices=[q, r])
        chip.set_frame(5.2)
        assert chip.frame == 5.2
        chip.set_frame({"q": 5.0, r: 7.5})
        assert chip.frame["q"] == 5.0
        assert chip.frame["r"] == 7.5


# ---------------------------------------------------------------------------
# TestStateFactory — Tensor-product states
# ---------------------------------------------------------------------------


class TestStateFactory:
    """Verify state factory builds correct tensor-product states."""

    def test_ground_state_default(self, backend: Backend) -> None:
        """bare_state() with no kwargs returns |0,0>, dimension = product of device levels."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r])

        psi = chip.bare_state()
        assert psi.shape[0] == 12

        n_q = backend.embed(q.number_operator(), 0, [3, 4])
        n_r = backend.embed(r.number_operator(), 1, [3, 4])
        assert abs(backend.expect(n_q, psi)) < 1e-12
        assert abs(backend.expect(n_r, psi)) < 1e-12

    def test_excited_state(self, backend: Backend) -> None:
        """bare_state(q=1) for a single-device chip produces |1>, <n̂>=1."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])

        psi = chip.bare_state(q=1)
        n_op = q.number_operator()
        assert abs(backend.expect(n_op, psi) - 1.0) < 1e-12

    def test_tensor_product_state(self, backend: Backend) -> None:
        """bare_state(q=0, r=1) = |0>⊗|1>: <n̂_q>=0, <n̂_r>=1."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r])

        psi = chip.bare_state(q=0, r=1)
        dims = [3, 4]
        n_q = backend.embed(q.number_operator(), 0, dims)
        n_r = backend.embed(r.number_operator(), 1, dims)

        assert abs(backend.expect(n_q, psi)) < 1e-12
        assert abs(backend.expect(n_r, psi) - 1.0) < 1e-12

    def test_bare_state_accepts_device_keyed_mapping(self, backend: Backend) -> None:
        """bare_state({device: level}) matches the label-keyed form."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r])

        psi = chip.bare_state({q: 0, r: 1})
        dims = [3, 4]
        n_q = backend.embed(q.number_operator(), 0, dims)
        n_r = backend.embed(r.number_operator(), 1, dims)

        assert abs(backend.expect(n_q, psi)) < 1e-12
        assert abs(backend.expect(n_r, psi) - 1.0) < 1e-12

    def test_multi_excitation_state(self, backend: Backend) -> None:
        """bare_state(q=2, r=1) = |2>⊗|1>, matching both device excitation numbers."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        chip = Chip(devices=[q, r])

        psi = chip.bare_state(q=2, r=1)
        dims = [3, 4]
        n_q = backend.embed(q.number_operator(), 0, dims)
        n_r = backend.embed(r.number_operator(), 1, dims)

        assert abs(backend.expect(n_q, psi) - 2.0) < 1e-12
        assert abs(backend.expect(n_r, psi) - 1.0) < 1e-12

    def test_invalid_label_raises(self) -> None:
        """Unknown device label raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])

        with pytest.raises(ValueError, match="Device 'nonexistent' not found"):
            chip.state(nonexistent=1)

    def test_fock_index_out_of_range_raises(self) -> None:
        """Fock index >= levels raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])

        with pytest.raises(ValueError, match="exceeds device dimension"):
            chip.state(q=3)

    def test_negative_fock_index_raises(self) -> None:
        """Negative Fock index raises ValueError."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])

        with pytest.raises(ValueError, match="must be >= 0"):
            chip.state(q=-1)

    def test_state_rejects_duplicate_mapping_and_keywords(self) -> None:
        """Providing the same label via mapping and kwargs is ambiguous."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip(devices=[q])

        with pytest.raises(ValueError, match="Duplicate device specification"):
            chip.state({q: 1}, q=1)


# ---------------------------------------------------------------------------
# TestStateOrderShorthand — string-state parsing, superposition primitive
# ---------------------------------------------------------------------------


class TestStateOrderShorthand:
    """`chip.set_state_order(...)` + string shorthand for bare_state / state."""

    def _chip(self) -> Chip:
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        return Chip(devices=[q, r])

    def test_string_shorthand_parses_letters_and_digits(self) -> None:
        """Level-order string shorthand parses letters and digits to the same state as an explicit index mapping."""
        chip = self._chip()
        q, r = chip["q"], chip["r"]
        chip.set_state_order(q, r)
        # "e2" → {q: 1 (e), r: 2}
        expected = chip.bare_state({q: 1, r: 2})
        actual = chip.bare_state("e2")
        import numpy as np
        diff = np.linalg.norm(np.asarray(chip.backend.to_array(expected - actual)))
        assert diff < 1e-12

    def test_string_shorthand_requires_set_state_order(self) -> None:
        """String shorthand raises ValueError unless set_state_order has configured a device order."""
        chip = self._chip()
        with pytest.raises(ValueError, match="set_state_order"):
            chip.bare_state("e0")

    def test_string_shorthand_wrong_length_rejected(self) -> None:
        """String shorthand must supply exactly one symbol per ordered device, or bare_state raises ValueError."""
        chip = self._chip()
        chip.set_state_order("q", "r")
        with pytest.raises(ValueError, match="has .* chars but .* devices"):
            chip.bare_state("e")
        with pytest.raises(ValueError, match="has .* chars but .* devices"):
            chip.bare_state("e11")

    def test_string_shorthand_rejects_unknown_symbols(self) -> None:
        """Unrecognized level symbols in string shorthand raise ValueError."""
        chip = self._chip()
        chip.set_state_order("q", "r")
        with pytest.raises(ValueError, match="Unknown level symbol"):
            chip.bare_state("xz")

    def test_set_state_order_requires_every_device(self) -> None:
        """set_state_order raises ValueError unless every chip device is included in the order."""
        chip = self._chip()
        with pytest.raises(ValueError, match="every device"):
            chip.set_state_order("q")

    def test_set_state_order_custom_levels(self) -> None:
        """Custom level-symbol mapping in set_state_order resolves string shorthand to its level indices."""
        chip = self._chip()
        chip.set_state_order("q", "r", levels={"a": 0, "b": 1, "c": 2})
        import numpy as np
        expected = chip.bare_state({"q": 1, "r": 2})
        actual = chip.bare_state("bc")
        diff = np.linalg.norm(np.asarray(chip.backend.to_array(expected - actual)))
        assert diff < 1e-12


class TestSuperposition:
    """`chip.superposition(...)` primitive."""

    def _chip(self) -> Chip:
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=4, label="r")
        return Chip(devices=[q, r])

    def test_equal_two_component_superposition_is_normalized(self) -> None:
        """Equal-weight two-component superposition normalizes to unit norm."""
        chip = self._chip()
        psi = chip.superposition({"q": 0, "r": 0}, {"q": 1, "r": 0})
        assert abs(chip.backend.norm(psi) - 1.0) < 1e-12

    def test_equal_two_component_matches_manual(self) -> None:
        """Default equal-weight superposition equals the manually summed and renormalized bare states."""
        chip = self._chip()
        a = chip.bare_state({"q": 0, "r": 0})
        b = chip.bare_state({"q": 1, "r": 0})
        manual = a + b
        manual = manual / chip.backend.norm(manual)
        psi = chip.superposition({"q": 0, "r": 0}, {"q": 1, "r": 0})
        import numpy as np
        diff = np.linalg.norm(np.asarray(chip.backend.to_array(psi - manual)))
        assert diff < 1e-12

    def test_weighted_superposition(self) -> None:
        """Weighted superposition with amplitude coefficients summing to unit probability normalizes to unit norm."""
        chip = self._chip()
        import numpy as np
        psi = chip.superposition(
            (np.sqrt(0.3), {"q": 0, "r": 0}),
            (np.sqrt(0.7), {"q": 1, "r": 0}),
        )
        # |c_00|^2 = 0.3, |c_10|^2 = 0.7 ⇒ unit norm
        assert abs(chip.backend.norm(psi) - 1.0) < 1e-12

    def test_accepts_string_shorthand(self) -> None:
        """superposition() built from string-shorthand components matches the equivalent dict-keyed components."""
        chip = self._chip()
        chip.set_state_order("q", "r")
        psi_str = chip.superposition("g0", "e0")
        psi_dict = chip.superposition({"q": 0, "r": 0}, {"q": 1, "r": 0})
        import numpy as np
        diff = np.linalg.norm(np.asarray(chip.backend.to_array(psi_str - psi_dict)))
        assert diff < 1e-12

    def test_single_component_is_just_bare_state(self) -> None:
        """A single-component superposition reduces to the corresponding bare state."""
        chip = self._chip()
        import numpy as np
        psi = chip.superposition({"q": 1, "r": 0})
        expected = chip.bare_state({"q": 1, "r": 0})
        diff = np.linalg.norm(np.asarray(chip.backend.to_array(psi - expected)))
        assert diff < 1e-12

    def test_empty_rejected(self) -> None:
        """superposition() with no components raises ValueError."""
        chip = self._chip()
        with pytest.raises(ValueError, match="at least one"):
            chip.superposition()


# ---------------------------------------------------------------------------
# TestDeviceOperatorCaching — cached_property invalidation
# ---------------------------------------------------------------------------


class TestDeviceOperatorCaching:
    """Verify that sigma_x/y/z are cached and invalidate when levels changes."""

    def test_sigma_cache_invalidates_when_levels_change(self) -> None:
        """sigma_x cache is invalidated when levels changes."""
        q = Resonator(freq=6.0, levels=3, label="r0")
        first = q.sigma_x
        q.levels = 4
        second = q.sigma_x
        assert first.shape != second.shape

    def test_sigma_x_cached_identity(self) -> None:
        """sigma_x returns the same object on repeated access."""
        q = Resonator(freq=6.0, levels=3, label="r0")
        first = q.sigma_x
        second = q.sigma_x
        assert first is second

    def test_sigma_y_cached_identity(self) -> None:
        """sigma_y returns the same object on repeated access."""
        q = Resonator(freq=6.0, levels=3, label="r0")
        first = q.sigma_y
        second = q.sigma_y
        assert first is second

    def test_sigma_z_cached_identity(self) -> None:
        """sigma_z returns the same object on repeated access."""
        q = Resonator(freq=6.0, levels=3, label="r0")
        first = q.sigma_z
        second = q.sigma_z
        assert first is second

    def test_all_three_invalidate_when_levels_change(self) -> None:
        """All three Pauli operators re-compute after levels change."""
        q = Resonator(freq=6.0, levels=3, label="r0")
        _ = q.sigma_x
        _ = q.sigma_y
        _ = q.sigma_z
        q.levels = 5
        assert q.sigma_x.shape == (5, 5)
        assert q.sigma_y.shape == (5, 5)
        assert q.sigma_z.shape == (5, 5)


# ---------------------------------------------------------------------------
# TestSubspaceAccessors — sigma_plus/minus, projector, transition
# ---------------------------------------------------------------------------


class TestSubspaceAccessors:
    """Explicit Fock-basis accessors for qudits and multi-level devices.

    ``sigma_plus`` / ``sigma_minus`` project into ``{|0>, |1>}`` (the
    computational subspace); ``projector(i, j)`` / ``transition(i, j)``
    name the subspace explicitly.
    """

    def test_sigma_plus_equals_projector_1_0(self) -> None:
        """sigma_plus equals the Fock-basis raising projector |1><0|."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        backend = q.sigma_plus
        # |1><0| has a single 1 at (row=1, col=0) in the Fock basis
        import numpy as np
        arr = np.asarray(backend.full() if hasattr(backend, "full") else backend)
        expected = np.zeros((3, 3), dtype=complex)
        expected[1, 0] = 1.0
        assert np.allclose(arr, expected)

    def test_sigma_minus_equals_projector_0_1(self) -> None:
        """sigma_minus equals the Fock-basis lowering projector |0><1|."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        import numpy as np
        arr = np.asarray(q.sigma_minus.full() if hasattr(q.sigma_minus, "full") else q.sigma_minus)
        expected = np.zeros((3, 3), dtype=complex)
        expected[0, 1] = 1.0
        assert np.allclose(arr, expected)

    def test_sigma_plus_minus_rebuild_sigma_x(self) -> None:
        """σ_x = σ_+ + σ_- on the computational subspace."""
        import numpy as np
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        recon = q.sigma_plus + q.sigma_minus
        recon_arr = np.asarray(recon.full() if hasattr(recon, "full") else recon)
        sx_arr = np.asarray(q.sigma_x.full() if hasattr(q.sigma_x, "full") else q.sigma_x)
        assert np.allclose(recon_arr, sx_arr)

    def test_projector_diagonal_is_level_projector(self) -> None:
        """projector(i, i) is the population operator for level |i>."""
        import numpy as np
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
        for i in range(4):
            p = q.projector(i, i)
            arr = np.asarray(p.full() if hasattr(p, "full") else p)
            expected = np.zeros((4, 4), dtype=complex)
            expected[i, i] = 1.0
            assert np.allclose(arr, expected)

    def test_transition_is_symmetric(self) -> None:
        """transition(i, j) == |i><j| + |j><i|."""
        import numpy as np
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        t12 = q.transition(1, 2)
        pij = q.projector(1, 2) + q.projector(2, 1)
        a = np.asarray(t12.full() if hasattr(t12, "full") else t12)
        b = np.asarray(pij.full() if hasattr(pij, "full") else pij)
        assert np.allclose(a, b)

    def test_sigma_plus_cache_invalidates_on_levels_change(self) -> None:
        """sigma_plus cache is invalidated and rebuilt at the new dimension when levels changes."""
        q = Resonator(freq=6.0, levels=3, label="r0")
        first = q.sigma_plus
        q.levels = 5
        second = q.sigma_plus
        assert first.shape != second.shape
        assert second.shape == (5, 5)


# ---------------------------------------------------------------------------
# TestConnectedDrives — Drive connection tracking
# ---------------------------------------------------------------------------


class TestConnectedDrives:
    """Verify that devices track connected drives via _connected_drives list."""

    def test_device_tracks_connected_drives(self) -> None:
        """A drive targeting a device appears in connected_drives."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3)
        d = ChargeDrive(target=q)
        assert d in q.connected_drives
        assert not hasattr(d, "line_name")

    def test_device_allows_multiple_drives_same_type(self) -> None:
        """Multiple drives of the same type can connect to one device."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3)
        d1 = ChargeDrive(target=q)
        d2 = ChargeDrive(target=q)
        assert len(q.connected_drives) == 2
        assert d1.label != d2.label


class TestConstructorLabelUniqueness:
    """Chip.__init__ rejects duplicate labels within each component kind."""

    def test_duplicate_bath_labels_raise(self) -> None:
        """Two baths sharing a label raise at construction."""
        from quchip.chip.baths import Bath

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        with pytest.raises(ValueError, match="[Dd]uplicate bath"):
            Chip(
                [q],
                baths=[
                    Bath("thermal", temperature=20.0, rate=1e-3, label="b"),
                    Bath("thermal", temperature=25.0, rate=1e-3, label="b"),
                ],
            )

    def test_duplicate_drive_labels_via_direct_control_equipment_raise(self) -> None:
        """Two drives sharing a label, wired via a directly-built ControlEquipment, raise at construction."""
        from quchip.control.equipment import ControlEquipment

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        d1 = ChargeDrive(target=q, label="d")
        d2 = ChargeDrive(target=q, label="d")
        with pytest.raises(ValueError, match="[Dd]uplicate drive"):
            Chip([q], control_equipment=ControlEquipment([d1, d2]))

    def test_duplicate_bath_labels_via_set_noise_raise(self) -> None:
        """Two baths sharing a label raise in set_noise before any mutation."""
        from quchip.chip.baths import Bath

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip([q])
        with pytest.raises(ValueError, match="[Dd]uplicate bath"):
            chip.set_noise(
                baths=[
                    Bath("thermal", temperature=20.0, rate=1e-3, label="b"),
                    Bath("thermal", temperature=25.0, rate=1e-3, label="b"),
                ],
            )
        assert chip.baths == ()


# ---------------------------------------------------------------------------
# TestFromArray — chip-scoped operator constructor
# ---------------------------------------------------------------------------


class TestFromArray:
    """Verify Chip.from_array embeds local and full-space operators correctly."""

    def test_from_array_embeds_local_operator_for_device_object(self, backend: Backend) -> None:
        """from_array embeds a local operator for a device object at its tensor slot in the full chip space."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=2, label="r")
        chip = Chip([q, r], backend=backend)

        local = np.diag([0.0, 1.0, 2.0]).astype(complex)
        op = chip.from_array(local, device=q)
        expected = backend.embed(backend.from_array(local, dims=[[3], [3]]), 0, [3, 2])

        assert (op - expected).norm() < 1e-12

    def test_from_array_accepts_string_device_label(self, backend: Backend) -> None:
        """from_array accepts a string device label equivalent to passing the device object."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip([q], backend=backend)

        op = chip.from_array(np.eye(3, dtype=complex), device="q")
        assert (op - backend.identity(3)).norm() < 1e-12

    def test_from_array_validates_full_space_shape(self) -> None:
        """from_array with no device raises ValueError when the array shape mismatches the tensor-product space."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.0, levels=2, label="r")
        chip = Chip([q, r])

        with pytest.raises(ValueError, match="full-space operator shape"):
            chip.from_array(np.eye(3, dtype=complex))
