"""Unit tests for (Δa, Δb) band decomposition of two-body operators.

Every expected value is derived from analytical formulas — no reference
to the function-under-test is used to *generate* ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from quchip.engine.bands import canonical_to_dense_array, decompose_two_body_canonical_bands
from quchip.engine.ir import CanonicalOperator, Carrier, ScalarModulation


# ---------------------------------------------------------------------------
# Helpers — build operators analytically
# ---------------------------------------------------------------------------


def _annihilation(d: int) -> np.ndarray:
    """Lowering operator a for a d-level truncated oscillator."""
    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    return a


def _capacitive_full(d_a: int, d_b: int) -> np.ndarray:
    """Full (a + a†)(b + b†) in product basis, analytically."""
    a = _annihilation(d_a)
    b = _annihilation(d_b)
    x_a = a + a.conj().T  # a + a†
    x_b = b + b.conj().T  # b + b†
    return np.kron(x_a, x_b)


def _rwa_coupling(d_a: int, d_b: int) -> np.ndarray:
    """RWA part a†b + ab† in product basis, analytically."""
    a = _annihilation(d_a)
    b = _annihilation(d_b)
    adag_b = np.kron(a.conj().T, b)
    a_bdag = np.kron(a, b.conj().T)
    return adag_b + a_bdag


def _two_body_canonical(matrix: np.ndarray, d_a: int, d_b: int) -> CanonicalOperator:
    """Wrap a dense product-basis matrix as a two-mode CanonicalOperator."""
    return CanonicalOperator.from_dense(matrix, dims=(d_a, d_b), basis="fock", subsystem_labels=("a", "b"))


def _decompose(matrix: np.ndarray, d_a: int, d_b: int) -> dict[tuple[int, int], np.ndarray]:
    """Decompose via the production canonical path; return dense band matrices."""
    bands = decompose_two_body_canonical_bands(_two_body_canonical(matrix, d_a, d_b), [d_a, d_b])
    return {band: np.asarray(canonical_to_dense_array(c)) for band, c in bands.items()}


# ---------------------------------------------------------------------------
# Test class — decomposition unit tests
# ---------------------------------------------------------------------------


class TestDecomposition:
    """Tests for decompose_two_body_canonical_bands()."""

    # Shared dimensions for the capacitive coupling tests
    DA, DB = 3, 5

    # ----- completeness ---------------------------------------------------
    def test_decomposition_completeness(self):
        """Sum of all (Δa, Δb) sectors must reconstruct the original matrix."""
        full = _capacitive_full(self.DA, self.DB)
        bands = _decompose(full, self.DA, self.DB)

        reconstructed = sum(bands.values())
        np.testing.assert_allclose(
            reconstructed,
            full,
            atol=1e-14,
            err_msg="Reconstructed matrix differs from original",
        )

    # ----- sector keys ----------------------------------------------------
    def test_per_mode_sectors(self):
        """Capacitive coupling (a+a†)(b+b†) populates exactly the four bilinear bands."""
        # w = col - row; a carries weight +1, a† weight -1, so ab, ab†, a†b, a†b†
        # land on (1,1), (1,-1), (-1,1), (-1,-1).
        full = _capacitive_full(self.DA, self.DB)
        bands = _decompose(full, self.DA, self.DB)

        assert set(bands.keys()) == {(1, 1), (1, -1), (-1, 1), (-1, -1)}

    # ----- RWA match ------------------------------------------------------
    def test_rwa_match(self):
        """The two ΔN=0 sectors of (a+a†)(b+b†) sum to a†b + ab†."""
        full = _capacitive_full(self.DA, self.DB)
        bands = _decompose(full, self.DA, self.DB)

        expected_rwa = _rwa_coupling(self.DA, self.DB)
        np.testing.assert_allclose(
            bands[(1, -1)] + bands[(-1, 1)],
            expected_rwa,
            atol=1e-14,
            err_msg="ΔN=0 sectors do not sum to analytical RWA a†b + ab†",
        )

    # ----- identity (trivial) --------------------------------------------
    def test_identity_trivial(self):
        """Identity decomposes to a single (0, 0) sector equal to itself."""
        d_a, d_b = 4, 3
        eye = np.eye(d_a * d_b, dtype=complex)
        bands = _decompose(eye, d_a, d_b)

        assert list(bands.keys()) == [(0, 0)], f"Expected only (0, 0) for identity, got {list(bands.keys())}"
        np.testing.assert_allclose(bands[(0, 0)], eye, atol=1e-15)

    # ----- relative pruning contract ---------------------------------------
    def test_zero_matrix_returns_no_bands(self):
        """An exactly-zero operator produces no bands."""
        zero = np.zeros((6, 6), dtype=complex)
        bands = _decompose(zero, 2, 3)
        assert bands == {}, f"Expected empty dict for the zero operator, got {bands.keys()}"

    def test_uniformly_tiny_operator_keeps_proportionate_bands(self):
        """Scaling every entry down by the same factor prunes no band: the cutoff is relative to the operator's norm."""
        full = _capacitive_full(self.DA, self.DB)
        tiny = 1e-18 * full
        bands = _decompose(tiny, self.DA, self.DB)
        assert set(bands.keys()) == {(1, 1), (1, -1), (-1, 1), (-1, -1)}

    def test_band_far_below_rtol_of_parent_is_dropped(self):
        """A band many orders of magnitude below the parent's norm is pruned; comparable bands survive."""
        full = _capacitive_full(self.DA, self.DB)
        contaminated = full.copy()
        contaminated[0, 0] = 1e-20  # populates the (0, 0) band alone, far below the O(1) dominant bands
        bands = _decompose(contaminated, self.DA, self.DB)
        assert (0, 0) not in bands, f"Expected (0, 0) band to be pruned, got {bands.keys()}"
        assert set(bands.keys()) == {(1, 1), (1, -1), (-1, 1), (-1, -1)}

    # ----- input validation -----------------------------------------------
    def test_bad_dims_length(self):
        """dims with != 2 entries should raise ValueError."""
        with pytest.raises(ValueError, match="exactly 2"):
            decompose_two_body_canonical_bands(_two_body_canonical(np.eye(8, dtype=complex), 2, 4), [2, 2, 2])

    def test_shape_mismatch(self):
        """Canonical shape inconsistent with dims should raise ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            decompose_two_body_canonical_bands(_two_body_canonical(np.eye(6, dtype=complex), 2, 3), [2, 2])

    # ----- Hermiticity of sectors -----------------------------------------
    def test_hermiticity_of_sectors(self):
        """For a Hermitian input, each (Δa, Δb) sector is the adjoint of (−Δa, −Δb)."""
        full = _capacitive_full(self.DA, self.DB)
        bands = _decompose(full, self.DA, self.DB)

        for (da, db), mat in bands.items():
            partner = (-da, -db)
            assert partner in bands, f"{partner} missing (partner of {(da, db)})"
            np.testing.assert_allclose(
                bands[partner],
                mat.conj().T,
                atol=1e-14,
                err_msg=f"{partner} is not the adjoint of {(da, db)}",
            )

    def test_two_body_decomposition_supports_jitted_jax_inputs(self) -> None:
        """Two-body decomposition should remain usable under ``jax.jit``."""
        import jax
        import jax.numpy as jnp

        full_np = _capacitive_full(2, 3)

        @jax.jit
        def reconstruct(op):
            canonical = CanonicalOperator.from_dense(op, dims=(2, 3), basis="fock", subsystem_labels=("a", "b"))
            bands = decompose_two_body_canonical_bands(canonical, [2, 3])
            return sum(canonical_to_dense_array(c) for c in bands.values())

        reconstructed = reconstruct(jnp.asarray(full_np))
        np.testing.assert_allclose(np.asarray(reconstructed), full_np, atol=1e-14)

# ---------------------------------------------------------------------------
# Test class — Hamiltonian structure tests
# ---------------------------------------------------------------------------


class TestHamiltonianStructure:
    """Tests for ΔN decomposition wiring in Chip and engine."""

    def _make_chip(self, rwa=False, frame="rotating"):
        """Build a transmon+resonator chip with Capacitive coupling."""
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.devices.resonator import Resonator
        from quchip.chip.couplings import Capacitive
        from quchip.chip.chip import Chip

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=5, label="r")
        coupling = Capacitive(q, r, g=0.05, rwa=rwa)
        chip = Chip([q, r], [coupling], frame=frame)
        return chip

    def _get_backend(self):
        from quchip.backend.qutip import QuTiPBackend

        return QuTiPBackend()

    def test_rotating_frame_h0_excludes_cr_terms(self):
        """For detuned devices, engine H0 in rotating frame is diagonal (all coupling sectors are time-dependent)."""
        from quchip.engine import build_hamiltonian_description
        from quchip.engine.stage1_frames import resolve_frame

        chip_full = self._make_chip(rwa=False, frame="rotating")
        resolved_full = resolve_frame(chip_full, chip_full.frame)
        desc_full = build_hamiltonian_description(chip_full, [], resolved_frame=resolved_full)
        H0_full = desc_full.static_terms[0].operator.to_dense()

        chip_rwa = self._make_chip(rwa=True, frame="rotating")
        resolved_rwa = resolve_frame(chip_rwa, chip_rwa.frame)
        desc_rwa = build_hamiltonian_description(chip_rwa, [], resolved_frame=resolved_rwa)
        H0_rwa = desc_rwa.static_terms[0].operator.to_dense()

        for name, H0 in (("rwa=False", H0_full), ("rwa=True", H0_rwa)):
            offdiag = H0 - np.diag(np.diag(H0))
            assert np.linalg.norm(offdiag) < 1e-12, f"{name} rotating-frame H0 contains residual coupling terms"

    def test_td_hamiltonian_has_cr_terms(self):
        """A non-RWA rotating-frame description produces extra time-dependent terms for ΔN≠0 sectors."""
        from quchip.engine import build_hamiltonian_description
        from quchip.engine.stage1_frames import resolve_frame

        chip = self._make_chip(rwa=False, frame="rotating")

        resolved = resolve_frame(chip, chip.frame)
        desc = build_hamiltonian_description(chip, [], resolved_frame=resolved)

        # rwa=False coupling produces: swap cos + cross sin (at Δ)
        # + ΔN=+2 + ΔN=-2 counter-rotating terms (at sum freq)
        assert len(desc.dynamic_terms) >= 4, (
            f"Expected swap + cross + ΔN=+2 + ΔN=-2 = 4 dynamic terms, got {len(desc.dynamic_terms)}"
        )
        for i, term in enumerate(desc.dynamic_terms):
            assert isinstance(term.time_dependence, ScalarModulation), (
                f"Dynamic term {i} time dependence should be ScalarModulation, got {type(term.time_dependence)}"
            )
            assert isinstance(term.time_dependence.signal, Carrier), (
                f"Dynamic term {i} signal should be Carrier, got {type(term.time_dependence.signal)}"
            )

    def test_rwa_true_has_swap_cross_terms(self):
        """With rwa=True in per-qubit rotating frame, coupling produces swap/cross td terms at detuning frequency."""
        from quchip.engine import build_hamiltonian_description
        from quchip.engine.stage1_frames import resolve_frame

        chip = self._make_chip(rwa=True, frame="rotating")

        resolved = resolve_frame(chip, chip.frame)
        desc = build_hamiltonian_description(chip, [], resolved_frame=resolved)

        assert len(desc.dynamic_terms) == 2, f"Expected swap + cross = 2 dynamic terms, got {len(desc.dynamic_terms)}"
        assert all(isinstance(term.time_dependence, ScalarModulation) for term in desc.dynamic_terms)
        assert all(isinstance(term.time_dependence.signal, Carrier) for term in desc.dynamic_terms)

    def test_lab_frame_unchanged(self):
        """In lab frame, rwa=False produces no td coupling terms; all sectors stay in H0."""
        from quchip.engine import build_hamiltonian_description
        from quchip.engine.stage1_frames import resolve_frame

        chip = self._make_chip(rwa=False, frame="lab")

        resolved = resolve_frame(chip, chip.frame)
        desc = build_hamiltonian_description(chip, [], resolved_frame=resolved)

        assert len(desc.dynamic_terms) == 0, (
            f"Expected only H0 in lab frame, got {len(desc.dynamic_terms)} dynamic terms"
        )


# ---------------------------------------------------------------------------
# Test class — physics match: lab frame vs rotating frame
# ---------------------------------------------------------------------------


class TestPhysicsMatch:
    """Prove lab-frame and rotating-frame simulations produce matching results."""

    def _get_backend(self):
        from quchip.backend.qutip import QuTiPBackend

        return QuTiPBackend()

    def test_lab_vs_rotating_populations(self):
        """Lab and rotating frame P(|1,0⟩) traces must match within tolerance for a coupled transmon-resonator."""
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.devices.resonator import Resonator
        from quchip.chip.couplings import Capacitive
        from quchip.chip.chip import Chip
        from quchip.engine import simulate

        self._get_backend()

        # Lab frame
        q_lab = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r_lab = Resonator(freq=7.0, levels=5, label="r")
        chip_lab = Chip(
            [q_lab, r_lab],
            [Capacitive(q_lab, r_lab, g=0.05, rwa=False)],
            frame="lab",
        )

        psi0_lab = chip_lab.bare_state(q=1, r=0)
        tlist = np.linspace(0, 200, 501)

        result_lab = simulate(
            chip_lab,
            [],
            tlist,
            initial_state=psi0_lab,
            options={"store_states": True, "nsteps": 10000},
        )

        # Rotating frame
        q_rot = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r_rot = Resonator(freq=7.0, levels=5, label="r")
        chip_rot = Chip(
            [q_rot, r_rot],
            [Capacitive(q_rot, r_rot, g=0.05, rwa=False)],
            frame="rotating",
        )

        psi0_rot = chip_rot.bare_state(q=1, r=0)

        result_rot = simulate(
            chip_rot,
            [],
            tlist,
            initial_state=psi0_rot,
            options={"store_states": True, "nsteps": 10000},
        )

        pops_lab = result_lab.populations
        pops_rot = result_rot.populations

        # The key (1, 0) corresponds to qubit in |1⟩, resonator in |0⟩
        p10_lab = pops_lab[(1, 0)]
        p10_rot = pops_rot[(1, 0)]

        max_diff = np.max(np.abs(p10_lab - p10_rot))
        assert max_diff < 3e-3, (
            f"Lab vs rotating frame P(|1,0⟩) max difference = {max_diff:.2e}, exceeds tolerance 3e-3"
        )

    def test_lab_vs_rotating_lowering_phase_parity_nonrwa(self):
        """Non-RWA lab-vs-rotating parity for complex ⟨a_r⟩(t) (phase-sensitive, not just populations)."""
        # ⟨a_r⟩_rot(t) vs ⟨a_r⟩_lab(t) * exp(+i 2π f_r t) for a coupled transmon-resonator, no drive.
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.devices.resonator import Resonator
        from quchip.chip.couplings import Capacitive
        from quchip.chip.chip import Chip
        from quchip.engine import simulate

        backend = self._get_backend()
        tlist = np.linspace(0, 200, 501)

        # Lab frame
        q_lab = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r_lab = Resonator(freq=7.0, levels=5, label="r")
        chip_lab = Chip(
            [q_lab, r_lab],
            [Capacitive(q_lab, r_lab, g=0.05, rwa=False)],
            frame="lab",
        )
        psi_lab = chip_lab.bare_state(
            q=0,
            r=backend.coherent(r_lab.levels, 0.2),
        )
        result_lab = simulate(
            chip_lab,
            [],
            tlist,
            initial_state=psi_lab,
            e_ops={"r": r_lab.lowering_operator()},
            options={"nsteps": 10000},
        )

        # Rotating frame
        q_rot = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r_rot = Resonator(freq=7.0, levels=5, label="r")
        chip_rot = Chip(
            [q_rot, r_rot],
            [Capacitive(q_rot, r_rot, g=0.05, rwa=False)],
            frame="rotating",
        )
        psi_rot = chip_rot.bare_state(
            q=0,
            r=backend.coherent(r_rot.levels, 0.2),
        )
        result_rot = simulate(
            chip_rot,
            [],
            tlist,
            initial_state=psi_rot,
            e_ops={"r": r_rot.lowering_operator()},
            options={"nsteps": 10000},
        )

        # Both `.values` traces are already in the slowly-varying demodulated frame.
        lab = np.asarray(result_lab._expect_data["r"].values)
        rot = np.asarray(result_rot._expect_data["r"].values)

        max_diff = np.max(np.abs(rot - lab))
        assert max_diff < 2e-4, (
            f"Phase parity mismatch for rwa=False lowering trace: max diff {max_diff:.2e} exceeds 2e-4"
        )


class TestThreeDeviceFrameConsistency:
    """Lab vs rotating frame must match for a driven transmon + readout + filter resonator, rwa={True, False}."""

    FREQ_Q = 5.0
    FREQ_R = 5.5
    FREQ_F = 6.0
    G_QR = 0.04
    G_RF = 0.03
    DURATION = 1000.0
    AMPLITUDE = 0.025

    def _run_in_frame(self, frame: str, initial_q: int, rwa: bool):
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.devices.resonator import Resonator
        from quchip.chip.couplings import Capacitive
        from quchip.chip.chip import Chip
        from quchip.control.drive import ChargeDrive
        from quchip import ControlEquipment
        from quchip.control.envelopes import Gaussian
        from quchip.control.sequence import QuantumSequence

        q = DuffingTransmon(freq=self.FREQ_Q, anharmonicity=-0.3, levels=3, label="q")
        r = Resonator(freq=self.FREQ_R, levels=6, label="r")
        f = Resonator(freq=self.FREQ_F, levels=4, label="f")
        chip = Chip(
            [q, r, f],
            [Capacitive(q, r, g=self.G_QR, rwa=rwa), Capacitive(r, f, g=self.G_RF, rwa=rwa)],
            frame=frame,
        )
        drive_f = ChargeDrive(target=f, label="filter")
        drive_q = ChargeDrive(target=q, label="qubit")
        equip = ControlEquipment(lines=[drive_f, drive_q])
        chip.connect(equip)

        readout_freq = (chip.freq(r, when={q: 0}) + chip.freq(r, when={q: 1})) / 2.0
        tlist = np.linspace(0, self.DURATION, 500)

        seq = QuantumSequence(chip)
        seq.charge(f, envelope=Gaussian(duration=self.DURATION, amplitude=self.AMPLITUDE, sigmas=4), freq=readout_freq)
        result = seq.simulate(
            tlist=tlist,
            initial_state=chip.state(q=initial_q, r=0, f=0),
            options={"nsteps": 50000},
        )
        return result, tlist

    @pytest.mark.parametrize("rwa", [True, False], ids=["rwa", "nonrwa"])
    def test_qubit_ground_populations(self, rwa):
        """Lab vs rotating P(q=0) with qubit in |0⟩."""
        res_lab, _ = self._run_in_frame("lab", initial_q=0, rwa=rwa)
        res_rot, _ = self._run_in_frame("rotating", initial_q=0, rwa=rwa)

        p_lab = res_lab.population("q", 0)
        p_rot = res_rot.population("q", 0)
        max_diff = np.max(np.abs(p_lab - p_rot))

        assert max_diff < 2e-3, (
            f"3-device lab vs rotating P(q=0|init=0) rwa={rwa}: max diff = {max_diff:.2e}, exceeds 2e-3"
        )

    @pytest.mark.parametrize("rwa", [True, False], ids=["rwa", "nonrwa"])
    def test_qubit_excited_populations(self, rwa):
        """Lab vs rotating P(q=1) with qubit in |1⟩."""
        res_lab, _ = self._run_in_frame("lab", initial_q=1, rwa=rwa)
        res_rot, _ = self._run_in_frame("rotating", initial_q=1, rwa=rwa)

        p_lab = res_lab.population("q", 1)
        p_rot = res_rot.population("q", 1)
        max_diff = np.max(np.abs(p_lab - p_rot))

        assert max_diff < 2e-3, (
            f"3-device lab vs rotating P(q=1|init=1) rwa={rwa}: max diff = {max_diff:.2e}, exceeds 2e-3"
        )

    @pytest.mark.parametrize("rwa", [True, False], ids=["rwa", "nonrwa"])
    def test_resonator_lowering_phase_parity(self, rwa):
        """Dict e_ops ⟨a_r⟩ demodulated result must match between lab and rotating frames."""
        # Dict-form e_ops routes through the band decomposition + demod pipeline for the frame transform.
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.devices.resonator import Resonator
        from quchip.chip.couplings import Capacitive
        from quchip.chip.chip import Chip
        from quchip.control.drive import ChargeDrive
        from quchip import ControlEquipment
        from quchip.control.envelopes import Gaussian
        from quchip.control.sequence import QuantumSequence

        tlist = np.linspace(0, self.DURATION, 500)

        results = {}
        for frame in ("lab", "rotating"):
            q = DuffingTransmon(freq=self.FREQ_Q, anharmonicity=-0.3, levels=3, label="q")
            r = Resonator(freq=self.FREQ_R, levels=6, label="r")
            f = Resonator(freq=self.FREQ_F, levels=4, label="f")
            chip = Chip(
                [q, r, f],
                [Capacitive(q, r, g=self.G_QR, rwa=rwa), Capacitive(r, f, g=self.G_RF, rwa=rwa)],
                frame=frame,
            )
            drive_f = ChargeDrive(target=f, label="filter")
            drive_q = ChargeDrive(target=q, label="qubit")
            equip = ControlEquipment(lines=[drive_f, drive_q])
            chip.connect(equip)

            readout_freq = (chip.freq(r, when={q: 0}) + chip.freq(r, when={q: 1})) / 2.0

            seq = QuantumSequence(chip)
            seq.charge(
                f, envelope=Gaussian(duration=self.DURATION, amplitude=self.AMPLITUDE, sigmas=4), freq=readout_freq
            )
            res = seq.simulate(
                tlist=tlist,
                initial_state=chip.state(q=0, r=0, f=0),
                e_ops={"r": r.lowering_operator()},
                options={"nsteps": 50000},
            )
            results[frame] = res

        a_lab = np.asarray(results["lab"].expect_values("r"), dtype=complex)
        a_rot = np.asarray(results["rotating"].expect_values("r"), dtype=complex)

        max_diff = np.max(np.abs(a_rot - a_lab))
        assert max_diff < 5e-3, (
            f"3-device dict e_ops ⟨a_r⟩ lab vs rotating rwa={rwa}: max diff = {max_diff:.2e}, exceeds 5e-3"
        )


# ---------------------------------------------------------------------------
# Test class — concrete-vs-traced zero-band dropping
# ---------------------------------------------------------------------------


class TestConcreteZeroBandDrop:
    """Concrete payloads drop structurally-zero bands; traced payloads keep every band."""
    # Regression for the dynamiqs overhead-ladder gap (benchmarks/overhead, 2026-07-04): drop
    # guards used is_jax_array instead of contains_tracer, so concrete SparseDIA payloads kept
    # explicitly-stored zero diagonals and stage 2 emitted band terms the solver then integrated.

    def test_concrete_jax_dia_payload_drops_zero_bands(self) -> None:
        """A DIA payload with jnp values keeps only genuinely populated (Δa, Δb) bands."""
        # RWA exchange a†b + ab† on 3x3 modes stores two full diagonals (offsets +-2), each
        # interleaving the real exchange band with in-block slots exactly zero; those slots
        # must not materialize as (0, +-2) zero-operator bands.
        import jax.numpy as jnp

        d = 3
        full = _rwa_coupling(d, d)
        offsets = (-2, 2)
        diags = np.zeros((len(offsets), d * d), dtype=complex)
        for k, off in enumerate(offsets):  # dense → full stored diagonals, zeros included
            for col in range(d * d):
                row = col - off
                if 0 <= row < d * d:
                    diags[k, col] = full[row, col]

        canonical = CanonicalOperator.from_dia(
            jnp.asarray(diags), np.asarray(offsets, dtype=int),
            shape=(d * d, d * d), dims=(d, d), basis="fock", subsystem_labels=("a", "b"),
        )
        bands = decompose_two_body_canonical_bands(canonical, [d, d])

        assert set(bands.keys()) == {(1, -1), (-1, 1)}, (
            f"Concrete jnp DIA payload emitted spurious bands: {sorted(bands.keys())}"
        )
        reconstructed = sum(np.asarray(canonical_to_dense_array(c)) for c in bands.values())
        np.testing.assert_allclose(reconstructed, full, atol=1e-14)

    def test_concrete_jax_dense_payload_drops_zero_bands(self) -> None:
        """decompose_bands on a concrete jnp matrix drops the empty weights."""
        import jax.numpy as jnp

        from quchip.engine.bands import decompose_bands

        matrix = jnp.asarray(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=complex))
        bands = decompose_bands(matrix, 2)
        assert set(bands.keys()) == {-1, 1}, f"Expected weights ±1 only, got {sorted(bands.keys())}"

    def test_traced_payload_keeps_every_band(self) -> None:
        """Under jit tracing the band set must stay statically known (all weights kept)."""
        import jax
        import jax.numpy as jnp

        from quchip.engine.bands import decompose_bands

        seen: dict[str, int] = {}

        @jax.jit
        def reconstruct(m):
            bands = decompose_bands(m, 2)
            seen["n_bands"] = len(bands)
            return sum(bands.values())

        matrix = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=complex)
        reconstructed = reconstruct(jnp.asarray(matrix))
        assert seen["n_bands"] == 3, f"Traced payload must keep all 2·dim−1 bands, got {seen['n_bands']}"
        np.testing.assert_allclose(np.asarray(reconstructed), matrix, atol=1e-15)


class TestPruneZeroDiagonals:
    """prune_zero_diagonals drops concretely-zero stored DIA diagonals, tracer-guarded."""
    # Companion to TestConcreteZeroBandDrop: stage 2's coupling fold cancels the lab-frame
    # interaction out of H0 exactly, leaving cancelled offsets stored as explicit zeros.

    @staticmethod
    def _dia(values, offsets):
        from quchip.engine.bands import prune_zero_diagonals

        canonical = CanonicalOperator.from_dia(
            values, np.asarray(offsets, dtype=int),
            shape=(3, 3), dims=(3,), basis="fock", subsystem_labels=("q",),
        )
        return prune_zero_diagonals(canonical)

    def test_drops_concrete_zero_diagonals(self) -> None:
        """Pruning drops a diagonal whose stored values are all concretely zero."""
        import jax.numpy as jnp

        values = jnp.asarray(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=complex))
        pruned = self._dia(values, [0, 1])
        assert tuple(int(x) for x in np.asarray(pruned.offsets)) == (0,)
        np.testing.assert_allclose(np.asarray(pruned.values), [[1.0, 2.0, 3.0]], atol=0)

    def test_all_zero_payload_keeps_one_diagonal(self) -> None:
        """Pruning an all-zero payload still keeps exactly one diagonal, never zero of them."""
        values = np.zeros((2, 3), dtype=complex)
        pruned = self._dia(values, [0, 1])
        assert np.asarray(pruned.offsets).shape == (1,)

    def test_traced_payload_passes_through_untouched(self) -> None:
        """Under jit, a traced payload's diagonal count stays static and is left unpruned."""
        import jax
        import jax.numpy as jnp

        from quchip.engine.bands import prune_zero_diagonals

        seen: dict[str, int] = {}

        @jax.jit
        def run(values):
            canonical = CanonicalOperator.from_dia(
                values, np.asarray([0, 1], dtype=int),
                shape=(3, 3), dims=(3,), basis="fock", subsystem_labels=("q",),
            )
            pruned = prune_zero_diagonals(canonical)
            seen["n_offsets"] = len(np.asarray(pruned.offsets))
            return pruned.values

        run(jnp.asarray(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=complex)))
        assert seen["n_offsets"] == 2, "traced payloads must keep their stored structure static"
