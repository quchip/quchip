"""Validation and advisory-metadata contracts introduced in the engine review pass."""

from __future__ import annotations

import numpy as np
import pytest


class TestTlistValidation:
    """prepare_solve_problem_context rejects a malformed concrete tlist."""

    def test_unsorted_tlist_raises(self):
        """A non-strictly-increasing tlist raises ValueError."""
        from quchip.chip.chip import Chip
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.engine.stage4_problem import prepare_solve_problem_context

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        chip = Chip([q])
        with pytest.raises(ValueError, match="strictly increasing"):
            prepare_solve_problem_context(chip, np.array([0.0, 5.0, 3.0, 20.0]))

    def test_single_point_tlist_raises(self):
        """A tlist with fewer than two points raises ValueError."""
        from quchip.chip.chip import Chip
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.engine.stage4_problem import prepare_solve_problem_context

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        chip = Chip([q])
        with pytest.raises(ValueError, match="at least two points"):
            prepare_solve_problem_context(chip, np.array([0.0]))


class TestDriveWindowValidation:
    """prepare_solve_problem_context rejects a DriveOp window with no positive-measure tlist overlap."""

    def _chip_and_drive(self):
        from quchip.chip.chip import Chip
        from quchip.control.drive import ChargeDrive
        from quchip.control.equipment import ControlEquipment
        from quchip.devices.transmon.duffing import DuffingTransmon

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        drive = ChargeDrive(target=q, label="d0")
        chip = Chip([q], control_equipment=ControlEquipment(lines=[drive]))
        return chip, drive

    def test_window_touching_endpoint_raises(self):
        """A pulse window that only touches tlist's start endpoint raises ValueError."""
        from quchip.control.envelopes import Square
        from quchip.engine.ir import DriveOp
        from quchip.engine.stage4_problem import prepare_solve_problem_context

        chip, drive = self._chip_and_drive()
        op = DriveOp(
            target_label="q0", envelope=Square(duration=20.0, amplitude=0.01),
            freq=5.0, start_time=-20.0, drive_label="d0",
        )
        with pytest.raises(ValueError, match="no positive-measure overlap"):
            prepare_solve_problem_context(chip, np.linspace(0.0, 20.0, 21), drive_ops=[op])

    def test_window_fully_outside_raises(self):
        """A pulse window strictly outside tlist raises ValueError."""
        from quchip.control.envelopes import Square
        from quchip.engine.ir import DriveOp
        from quchip.engine.stage4_problem import prepare_solve_problem_context

        chip, drive = self._chip_and_drive()
        op = DriveOp(
            target_label="q0", envelope=Square(duration=5.0, amplitude=0.01),
            freq=5.0, start_time=100.0, drive_label="d0",
        )
        with pytest.raises(ValueError, match="no positive-measure overlap"):
            prepare_solve_problem_context(chip, np.linspace(0.0, 20.0, 21), drive_ops=[op])


class TestResolveDrivesValidation:
    """_resolve_drives cross-checks a drive's own wiring against its DriveOp."""

    def test_unconnected_drive_raises(self):
        """A DriveOp routed through an unconnected drive line raises ValueError."""
        from quchip.chip.chip import Chip
        from quchip.control.drive import ChargeDrive
        from quchip.control.envelopes import Square
        from quchip.control.equipment import ControlEquipment
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.engine.ir import DriveOp
        from quchip.engine.stage2_assembly import _resolve_drives

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        orphan = ChargeDrive(label="orphan")
        chip = Chip([q], control_equipment=ControlEquipment(lines=[orphan]))
        op = DriveOp(
            target_label="q0", envelope=Square(duration=20.0, amplitude=0.01),
            freq=5.0, start_time=0.0, drive_label="orphan",
        )
        with pytest.raises(ValueError, match="not connected"):
            _resolve_drives(chip, [op])

    def test_mismatched_target_raises(self):
        """A DriveOp targeting a label other than its drive's wired target raises ValueError."""
        from quchip.chip.chip import Chip
        from quchip.control.drive import ChargeDrive
        from quchip.control.envelopes import Square
        from quchip.control.equipment import ControlEquipment
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.engine.ir import DriveOp
        from quchip.engine.stage2_assembly import _resolve_drives

        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="q1")
        drive = ChargeDrive(target=q0, label="d0")
        chip = Chip([q0, q1], control_equipment=ControlEquipment(lines=[drive]))
        op = DriveOp(
            target_label="q1", envelope=Square(duration=20.0, amplitude=0.01),
            freq=5.0, start_time=0.0, drive_label="d0",
        )
        with pytest.raises(ValueError, match="wired to target"):
            _resolve_drives(chip, [op])

    def test_target_kind_mismatch_raises(self):
        """A drive's target_kind disagreeing with the map its DriveOp's target resolved from raises ValueError."""
        from quchip.control.envelopes import Square
        from quchip.engine.ir import DriveOp
        from quchip.engine.stage2_assembly import _resolve_drives

        # Real Chip namespaces cannot produce this mismatch; doubles isolate inconsistent drive bookkeeping.
        class _MismatchedKindDrive:
            label = "d0"
            target_kind = "edge"

            @property
            def target_label(self):
                return "q0"

        class _Equipment:
            lines = [_MismatchedKindDrive()]

        class _FakeChip:
            device_map = {"q0": object()}
            coupling_map: dict = {}
            control_equipment = _Equipment()

        op = DriveOp(
            target_label="q0", envelope=Square(duration=20.0, amplitude=0.01),
            freq=5.0, start_time=0.0, drive_label="d0",
        )
        with pytest.raises(ValueError, match="target_kind"):
            _resolve_drives(_FakeChip(), [op])


class TestWeightZeroRwaDrop:
    """A SINGLE_TONE band at weight 0 is dropped structurally under RWA, with an audit record."""

    def test_predicate_fires_only_for_single_tone_weight_zero_under_rwa(self):
        """_is_dropped_weight_zero_single_tone keys only on modulation, weight, and RWA."""
        from quchip.control.signal_spec import DriveModulation
        from quchip.engine.stage2_assembly import _is_dropped_weight_zero_single_tone

        assert _is_dropped_weight_zero_single_tone(DriveModulation.SINGLE_TONE, 0, True)
        assert not _is_dropped_weight_zero_single_tone(DriveModulation.SINGLE_TONE, 0, False)
        assert not _is_dropped_weight_zero_single_tone(DriveModulation.SINGLE_TONE, 1, True)
        assert not _is_dropped_weight_zero_single_tone(DriveModulation.DIRECT_REAL, 0, True)

    def test_dropped_term_records_band_weights_zero(self):
        """The audit record for a weight-0 SINGLE_TONE drop carries band_weights=(0,) and abs(drive_freq)."""
        from quchip.engine.stage2_assembly import _weight_zero_dropped_term

        record = _weight_zero_dropped_term(source="d0", device_label="q0", drive_freq=-5.0)
        assert record.band_weights == (0,)
        assert record.frequency == pytest.approx(5.0)
        assert record.source == "d0"

    def test_dropped_term_raises_when_drive_freq_is_none(self):
        """A weight-0 SINGLE_TONE band with no drive_freq raises rather than silently omitting frequency."""
        from quchip.engine.stage2_assembly import _weight_zero_dropped_term

        with pytest.raises(ValueError, match="drive_freq"):
            _weight_zero_dropped_term(source="d0", device_label="q0", drive_freq=None)

    def test_single_tone_coefficient_raises_for_bypassed_weight_zero_drop(self):
        """Reaching _single_tone_coefficient with a weight-0 RWA band signals a bypassed structural drop."""
        from quchip.engine.ir import Constant
        from quchip.engine.stage2_assembly import BandContext, _single_tone_coefficient

        band = BandContext(weight=0, device_frame_freq=0.0, drive_freq=5.0, rwa=True)
        with pytest.raises(ValueError, match="dropped structurally"):
            _single_tone_coefficient(Constant(1.0 + 0j), band)


class TestRelativeBandPruning:
    """decompose_bands drops a band by relative Frobenius norm, not an absolute cutoff."""

    def test_eigh_noise_scale_band_is_dropped(self):
        """A band at eigh's ~1e-15 roundoff floor is dropped relative to an O(1) parent operator."""
        from quchip.engine.bands import decompose_bands

        matrix = np.diag([1.0, 2.0, 3.0]).astype(complex)
        matrix[0, 1] = 2.6e-15  # ChargeBasisTransmon-scale eigh diagonalization noise.
        bands = decompose_bands(matrix, 3)
        assert 1 not in bands

    def test_genuinely_small_band_above_relative_cutoff_survives(self):
        """A band far above the relative cutoff survives even though it is numerically small."""
        from quchip.engine.bands import decompose_bands

        matrix = np.diag([1.0, 2.0, 3.0]).astype(complex)
        matrix[0, 1] = 1e-6  # >> 1e-12 * ||matrix||_F
        bands = decompose_bands(matrix, 3)
        assert 1 in bands
        np.testing.assert_allclose(bands[1][0, 1], 1e-6)

    def test_operator_entirely_below_absolute_floor_still_yields_its_band(self):
        """A DIA operator whose entries are all below the absolute 1e-15 floor still yields its band."""
        from quchip.engine.ir import CanonicalOperator
        from quchip.engine.bands import decompose_canonical_bands

        # Entry extraction must drop only exact zeros; an absolute cutoff erases this operator before band pruning.
        dim = 3
        diag_values = np.zeros((1, dim), dtype=complex)
        diag_values[0, 1] = 1e-18
        diag_values[0, 2] = 1e-18
        canonical = CanonicalOperator.from_dia(
            diag_values, np.array([1], dtype=int), shape=(dim, dim),
            dims=(dim,), basis="fock", subsystem_labels=("q0",),
        )
        bands = decompose_canonical_bands(canonical, dim)
        assert 1 in bands


class TestSolverHintsMaxStep:
    """_solver_hint_metadata exposes max_step_ns from the narrowest concrete Window."""

    def _term_with_window(self, start, stop):
        from quchip.engine.ir import CanonicalOperator, Constant, DynamicTerm, ScalarModulation, Window

        op = CanonicalOperator.from_dense(
            np.eye(2, dtype=complex), dims=(2,), basis="fock", subsystem_labels=("q0",)
        )
        window = Window(child=Constant(1.0 + 0j), start=start, stop=stop)
        return DynamicTerm(operator=op, time_dependence=ScalarModulation(signal=window), origin="drive")

    def test_max_step_ns_is_half_shortest_window(self):
        """max_step_ns equals half the narrowest positive concrete Window width across dynamic terms."""
        from quchip.engine.solver_hints import _solver_hint_metadata

        wide = self._term_with_window(0.0, 50.0)
        narrow = self._term_with_window(10.0, 30.0)  # width 20
        metadata = _solver_hint_metadata(None, (wide, narrow))
        assert metadata["max_step_ns"] == pytest.approx(10.0)

    def test_max_step_ns_omitted_when_window_bound_traced(self):
        """A traced window bound anywhere in the term set omits max_step_ns entirely."""
        import jax
        import jax.numpy as jnp

        from quchip.engine.solver_hints import _solver_hint_metadata

        @jax.jit
        def check(duration):
            traced_term = self._term_with_window(0.0, duration)
            concrete_term = self._term_with_window(10.0, 30.0)
            metadata = _solver_hint_metadata(None, (traced_term, concrete_term))
            assert "max_step_ns" not in metadata
            return duration

        check(jnp.asarray(50.0))


class TestBatchMetadataAggregation:
    """_aggregate_batch_metadata combines advisory hints across every element in a batch."""

    def test_max_step_ns_aggregates_by_minimum(self):
        """max_step_ns takes the minimum across batch elements."""
        from quchip.engine.ir import HamiltonianDescription
        from quchip.engine.stage4_problem import _aggregate_batch_metadata

        wide = HamiltonianDescription(static_terms=(), dynamic_terms=(), metadata={"max_step_ns": 10.0})
        narrow = HamiltonianDescription(static_terms=(), dynamic_terms=(), metadata={"max_step_ns": 2.5})
        metadata = _aggregate_batch_metadata([wide, narrow])
        assert metadata["max_step_ns"] == pytest.approx(2.5)

    def test_max_step_ns_omitted_when_any_element_lacks_it(self):
        """A single element missing max_step_ns (e.g. from tracing) omits it for the whole batch."""
        from quchip.engine.ir import HamiltonianDescription
        from quchip.engine.stage4_problem import _aggregate_batch_metadata

        has_hint = HamiltonianDescription(static_terms=(), dynamic_terms=(), metadata={"max_step_ns": 10.0})
        missing_hint = HamiltonianDescription(static_terms=(), dynamic_terms=(), metadata={})
        metadata = _aggregate_batch_metadata([has_hint, missing_hint])
        assert "max_step_ns" not in metadata

    def test_carrier_and_spectral_bounds_aggregate_by_maximum(self):
        """max_carrier_freq_ghz and spectral_bound_ghz take the maximum across batch elements."""
        from quchip.engine.ir import HamiltonianDescription
        from quchip.engine.stage4_problem import _aggregate_batch_metadata

        a = HamiltonianDescription(
            static_terms=(), dynamic_terms=(),
            metadata={"max_carrier_freq_ghz": 5.0, "spectral_bound_ghz": 1.0},
        )
        b = HamiltonianDescription(
            static_terms=(), dynamic_terms=(),
            metadata={"max_carrier_freq_ghz": 7.5, "spectral_bound_ghz": 0.5},
        )
        metadata = _aggregate_batch_metadata([a, b])
        assert metadata["max_carrier_freq_ghz"] == pytest.approx(7.5)
        assert metadata["spectral_bound_ghz"] == pytest.approx(1.0)


class TestBatchedDroppedTermsRetention:
    """BatchedHamiltonianDescription.element() restores each element's own dropped_terms."""

    def test_element_restores_dropped_terms(self):
        """dropped_terms set on a single-element batch reappear on the reconstructed element."""
        from quchip.engine.ir import BatchedHamiltonianDescription, DroppedTerm

        record = DroppedTerm(source="d0", operator="drive band w=+0 on q0", reason="test", band_weights=(0,))
        batched = BatchedHamiltonianDescription(
            batch_size=1,
            static_terms=(),
            dynamic_operators=(),
            dynamic_origins=(),
            dynamic_tags=(),
            dynamic_signals=(),
            dropped_terms_by_element=((record,),),
        )
        element = batched.element(0)
        assert element.dropped_terms == (record,)

    def test_element_restores_its_own_frequency_not_another_elements(self):
        """Two elements with different dropped-term frequencies each restore their own, not the reference's."""
        from quchip.engine.ir import BatchedHamiltonianDescription, DroppedTerm

        record_a = DroppedTerm(
            source="d0", operator="drive band w=+0 on q0", reason="test", band_weights=(0,), frequency=5.0
        )
        record_b = DroppedTerm(
            source="d0", operator="drive band w=+0 on q0", reason="test", band_weights=(0,), frequency=6.0
        )
        batched = BatchedHamiltonianDescription(
            batch_size=2,
            static_terms=(),
            dynamic_operators=(),
            dynamic_origins=(),
            dynamic_tags=(),
            dynamic_signals=(),
            dropped_terms_by_element=((record_a,), (record_b,)),
        )
        assert batched.element(0).dropped_terms[0].frequency == pytest.approx(5.0)
        assert batched.element(1).dropped_terms[0].frequency == pytest.approx(6.0)

    def test_length_mismatch_raises(self):
        """dropped_terms_by_element whose length disagrees with batch_size raises ValueError."""
        from quchip.engine.ir import BatchedHamiltonianDescription, DroppedTerm

        record = DroppedTerm(source="d0", operator="drive band w=+0 on q0", reason="test", band_weights=(0,))
        with pytest.raises(ValueError, match="dropped_terms_by_element"):
            BatchedHamiltonianDescription(
                batch_size=2,
                static_terms=(),
                dynamic_operators=(),
                dynamic_origins=(),
                dynamic_tags=(),
                dynamic_signals=(),
                dropped_terms_by_element=((record,),),
            )
