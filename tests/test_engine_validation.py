"""Validation and advisory-metadata contracts introduced in the engine review pass."""

from __future__ import annotations

from quchip.approximations import RWA

from dataclasses import replace

import numpy as np
import pytest


class TestTlistValidation:
    """prepare_solve_problem_context rejects a malformed concrete tlist."""

    def test_unsorted_tlist_raises(self):
        """A non-strictly-increasing tlist raises ValueError."""
        from quchip.chip.chip import Chip
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.engine.problem import prepare_solve_problem_context

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        chip = Chip([q])
        with pytest.raises(ValueError, match="strictly increasing"):
            prepare_solve_problem_context(chip, np.array([0.0, 5.0, 3.0, 20.0]))

    def test_single_point_tlist_raises(self):
        """A tlist with fewer than two points raises ValueError."""
        from quchip.chip.chip import Chip
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.engine.problem import prepare_solve_problem_context

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
        from quchip.engine.problem import prepare_solve_problem_context

        chip, drive = self._chip_and_drive()
        op = DriveOp(
            target_label="q0",
            envelope=Square(duration=20.0, amplitude=0.01),
            freq=5.0,
            start_time=-20.0,
            drive_label="d0",
        )
        with pytest.raises(ValueError, match="no positive-measure overlap"):
            prepare_solve_problem_context(chip, np.linspace(0.0, 20.0, 21), drive_ops=[op])

    def test_window_fully_outside_raises(self):
        """A pulse window strictly outside tlist raises ValueError."""
        from quchip.control.envelopes import Square
        from quchip.engine.ir import DriveOp
        from quchip.engine.problem import prepare_solve_problem_context

        chip, drive = self._chip_and_drive()
        op = DriveOp(
            target_label="q0",
            envelope=Square(duration=5.0, amplitude=0.01),
            freq=5.0,
            start_time=100.0,
            drive_label="d0",
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
        from quchip.engine.assembly import _resolve_drives

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        orphan = ChargeDrive(label="orphan")
        chip = Chip([q], control_equipment=ControlEquipment(lines=[orphan]))
        op = DriveOp(
            target_label="q0",
            envelope=Square(duration=20.0, amplitude=0.01),
            freq=5.0,
            start_time=0.0,
            drive_label="orphan",
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
        from quchip.engine.assembly import _resolve_drives

        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="q1")
        drive = ChargeDrive(target=q0, label="d0")
        chip = Chip([q0, q1], control_equipment=ControlEquipment(lines=[drive]))
        op = DriveOp(
            target_label="q1",
            envelope=Square(duration=20.0, amplitude=0.01),
            freq=5.0,
            start_time=0.0,
            drive_label="d0",
        )
        with pytest.raises(ValueError, match="wired to target"):
            _resolve_drives(chip, [op])

    def test_definition_target_mismatch_raises(self):
        """A drive definition disagreeing with its resolved target raises ValueError."""
        from quchip.control.envelopes import Square
        from quchip.engine.ir import DriveOp
        from quchip.engine.assembly import _resolve_drives

        # Real Chip namespaces cannot produce this mismatch; doubles isolate inconsistent drive bookkeeping.
        from quchip.control.drive import CouplingDrive

        class _MismatchedKindDrive(CouplingDrive):
            def hamiltonian(self, target, signal):
                raise AssertionError("resolution should fail before Hamiltonian authorship")

        class _Equipment:
            lines = [_MismatchedKindDrive("q0", label="d0")]

        class _FakeChip:
            device_map = {"q0": object()}
            coupling_map: dict = {}
            control_equipment = _Equipment()

        op = DriveOp(
            target_label="q0",
            envelope=Square(duration=20.0, amplitude=0.01),
            freq=5.0,
            start_time=0.0,
            drive_label="d0",
        )
        with pytest.raises(ValueError, match="declares target"):
            _resolve_drives(_FakeChip(), [op])


def test_symbolic_drive_channel_reaches_engine_without_custom_dispatch():
    """A drive extension may return the shared symbolic expression directly."""
    from quchip.chip.chip import Chip
    from quchip.control.drive import FluxDrive
    from quchip.control.envelopes import Square
    from quchip.control.equipment import ControlEquipment
    from quchip.declarative.ops import LocalOps
    from quchip.devices.transmon.duffing import DuffingTransmon
    from quchip.engine.ir import DriveOp
    from quchip.engine.frames import resolve_frame
    from quchip.engine.assembly import build_engine_result

    class SymbolicFluxDrive(FluxDrive):
        def operator(self, device):
            return LocalOps(device.label, device.local_space()).n

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    drive = SymbolicFluxDrive(q, label="flux")
    chip = Chip([q], control_equipment=ControlEquipment([drive]))
    drive_op = DriveOp(
        target_label="q",
        envelope=Square(duration=10.0, amplitude=0.01),
        freq=None,
        start_time=0.0,
        drive_label="flux",
    )
    result = build_engine_result(
        chip,
        [drive_op],
        resolved_frame=resolve_frame(chip, chip.frame),
    )
    assert result.dynamic_terms
    expression = result.hamiltonian()
    np.testing.assert_allclose(
        expression.matrix(t=2.0, backend=chip.backend),
        result.hamiltonian().matrix(t=2.0, backend=chip.backend),
    )
    assert r"f_{drive,0}\!\left(t\right)" in result.latex()


class TestWeightZeroRwaDrop:
    """A carrier-driven weight-zero band has an explicit audit record."""

    def test_dropped_term_records_band_weights_zero(self):
        """The audit record for a carrier-driven weight-zero drop preserves its frequency."""
        from quchip.engine.assembly import _weight_zero_dropped_term

        record = _weight_zero_dropped_term(source="d0", device_label="q0", drive_freq=-5.0)
        assert record.band_weights == (0,)
        assert record.frequency == pytest.approx(5.0)
        assert record.source == "d0"

    def test_dropped_term_raises_when_carrier_is_none(self):
        from quchip.engine.assembly import _weight_zero_dropped_term

        with pytest.raises(ValueError, match="carrier frequency"):
            _weight_zero_dropped_term(source="d0", device_label="q0", drive_freq=None)

    def test_drive_coefficient_rejects_a_bypassed_weight_zero_drop(self):
        from quchip.engine.ir import Constant, RealPart
        from quchip.engine.approximations import resolve_drive_program

        with pytest.raises(ValueError, match="weight-zero"):
            resolve_drive_program(
                RWA(),
                RealPart(Constant(1.0 + 0j)),
                weight=0,
                frame_frequency=0.0,
                has_carrier=True,
            )


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
            diag_values,
            np.array([1], dtype=int),
            shape=(dim, dim),
            dims=(dim,),
            basis="fock",
            subsystem_labels=("q0",),
        )
        bands = decompose_canonical_bands(canonical, dim)
        assert 1 in bands


class TestSolverHintsMaxStep:
    """_solver_hint_metadata exposes max_step_ns from the narrowest concrete Window."""

    def _term_with_window(self, start, stop):
        from quchip.engine.ir import CanonicalOperator, Constant, DynamicTerm, ScalarModulation, Window

        op = CanonicalOperator.from_dense(np.eye(2, dtype=complex), dims=(2,), basis="fock", subsystem_labels=("q0",))
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
        from quchip.engine.ir import EngineResult
        from quchip.engine.ir import _aggregate_batch_metadata

        wide = EngineResult(static_terms=(), dynamic_terms=(), metadata={"max_step_ns": 10.0})
        narrow = EngineResult(static_terms=(), dynamic_terms=(), metadata={"max_step_ns": 2.5})
        metadata = _aggregate_batch_metadata([wide, narrow])
        assert metadata["max_step_ns"] == pytest.approx(2.5)

    def test_max_step_ns_omitted_when_any_element_lacks_it(self):
        """A single element missing max_step_ns (e.g. from tracing) omits it for the whole batch."""
        from quchip.engine.ir import EngineResult
        from quchip.engine.ir import _aggregate_batch_metadata

        has_hint = EngineResult(static_terms=(), dynamic_terms=(), metadata={"max_step_ns": 10.0})
        missing_hint = EngineResult(static_terms=(), dynamic_terms=(), metadata={})
        metadata = _aggregate_batch_metadata([has_hint, missing_hint])
        assert "max_step_ns" not in metadata

    def test_carrier_and_spectral_bounds_aggregate_by_maximum(self):
        """max_carrier_freq_ghz and spectral_bound_ghz take the maximum across batch elements."""
        from quchip.engine.ir import EngineResult
        from quchip.engine.ir import _aggregate_batch_metadata

        a = EngineResult(
            static_terms=(),
            dynamic_terms=(),
            metadata={"max_carrier_freq_ghz": 5.0, "spectral_bound_ghz": 1.0},
        )
        b = EngineResult(
            static_terms=(),
            dynamic_terms=(),
            metadata={"max_carrier_freq_ghz": 7.5, "spectral_bound_ghz": 0.5},
        )
        metadata = _aggregate_batch_metadata([a, b])
        assert metadata["max_carrier_freq_ghz"] == pytest.approx(7.5)
        assert metadata["spectral_bound_ghz"] == pytest.approx(1.0)


class TestSolveBatchPointRetention:
    """SolveBatch keeps each point's complete problem snapshot."""

    def test_batch_rejects_structural_dimension_changes(self):
        from quchip.engine.ir import EngineResult, SolveBatch, SolveProblem

        first = SolveProblem(
            chip=None,
            engine_result=EngineResult(static_terms=(), dynamic_terms=(), dims=(2,)),
            initial_state=None,
            tlist=(0.0, 1.0),
        )
        second = replace(first, engine_result=replace(first.engine_result, dims=(3,)))
        with pytest.raises(ValueError, match="Structural settings"):
            SolveBatch(chip=None, problems=(first, second))
        different_grid = replace(first, tlist=(0.0, 0.5, 1.0))
        with pytest.raises(ValueError, match="one time grid"):
            SolveBatch(chip=None, problems=(first, different_grid))

    def test_element_restores_dropped_terms(self):
        """dropped_terms set on a single-element batch reappear on the reconstructed element."""
        from quchip.engine.ir import DroppedTerm, EngineResult, SolveBatch, SolveProblem

        record = DroppedTerm(source="d0", operator="drive band w=+0 on q0", reason="test", band_weights=(0,))
        problem = SolveProblem(
            chip=None,
            engine_result=EngineResult(static_terms=(), dynamic_terms=(), dropped_terms=(record,)),
            initial_state=None,
            tlist=(0.0, 1.0),
        )
        batch = SolveBatch(chip=None, problems=(problem,))
        element = batch.element(0)
        assert element.engine_result.dropped_terms == (record,)

    def test_element_restores_its_own_frequency_not_another_elements(self):
        """Two elements with different dropped-term frequencies each restore their own, not the reference's."""
        from quchip.engine.ir import DroppedTerm, EngineResult, SolveBatch, SolveProblem

        record_a = DroppedTerm(
            source="d0", operator="drive band w=+0 on q0", reason="test", band_weights=(0,), frequency=5.0
        )
        record_b = DroppedTerm(
            source="d0", operator="drive band w=+0 on q0", reason="test", band_weights=(0,), frequency=6.0
        )
        problem_a = SolveProblem(
            chip=None,
            engine_result=EngineResult(static_terms=(), dynamic_terms=(), dropped_terms=(record_a,)),
            initial_state=None,
            tlist=(0.0, 1.0),
        )
        problem_b = replace(
            problem_a,
            engine_result=replace(problem_a.engine_result, dropped_terms=(record_b,)),
        )
        batch = SolveBatch(chip=None, problems=(problem_a, problem_b))
        assert batch.element(0).engine_result.dropped_terms[0].frequency == pytest.approx(5.0)
        assert batch.element(1).engine_result.dropped_terms[0].frequency == pytest.approx(6.0)
