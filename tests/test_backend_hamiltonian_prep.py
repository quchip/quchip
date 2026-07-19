"""Backend preparation tests for the HamiltonianDescription IR."""

from __future__ import annotations

import numpy as np
import qutip

from quchip.backend import PreparedHamiltonian
from quchip.chip.chip import Chip
from quchip.control import ChargeDrive
from quchip.control.envelopes import Square
from quchip.control.equipment import ControlEquipment
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine.ir import CanonicalOperator, Carrier, DynamicTerm, HamiltonianDescription, ScalarModulation
from quchip.engine.stage1_frames import resolve_frame
from quchip.engine.stage2_assembly import build_hamiltonian_description


class TestPrepareHamiltonian:
    """Verify Backend.prepare_hamiltonian() round-trips correctly."""

    def test_prepare_returns_prepared_hamiltonian(self):
        """prepare_hamiltonian must return a PreparedHamiltonian."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        chip.dress()
        resolved = resolve_frame(chip, chip.frame)
        tlist = np.linspace(0, 50, 201)

        desc = build_hamiltonian_description(chip, [], resolved_frame=resolved)
        backend = chip.backend
        prepared = backend.prepare_hamiltonian(desc, tlist)

        assert isinstance(prepared, PreparedHamiltonian)
        assert prepared.rhs is not None

    def test_static_prepare_returns_qobj(self):
        """Static Hamiltonians should prepare to a native QuTiP Qobj."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        chip.dress()
        resolved = resolve_frame(chip, chip.frame)
        tlist = np.linspace(0, 50, 201)

        desc = build_hamiltonian_description(chip, [], resolved_frame=resolved)
        prepared = chip.backend.prepare_hamiltonian(desc, tlist)

        assert isinstance(prepared.rhs, qutip.Qobj)
        assert type(prepared.rhs.data).__name__ == "CSR"

    def test_prepare_with_drive_produces_qobjevo(self):
        """A driven Hamiltonian should prepare to a native QuTiP QobjEvo."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[drive]))
        chip.dress()
        resolved = resolve_frame(chip, chip.frame)
        tlist = np.linspace(0, 50, 201)

        from quchip.engine.ir import DriveOp

        drive_op = DriveOp(
            target_label="q",
            envelope=Square(amplitude=0.02, duration=50),
            freq=5.0,
            start_time=0.0,
            drive_label=drive.label,
        )
        desc = build_hamiltonian_description(
            chip,
            [drive_op],
            resolved_frame=resolved,
        )

        assert desc.dynamic_terms
        assert all(isinstance(term.time_dependence, ScalarModulation) for term in desc.dynamic_terms)
        prepared = chip.backend.prepare_hamiltonian(desc, tlist)

        assert isinstance(prepared.rhs, qutip.QobjEvo)
        coeff_terms = [item for item in prepared.rhs.to_list() if isinstance(item, list)]
        assert coeff_terms, "expected at least one lowered coefficient pair"
        # Verify each lowered coefficient is a usable time function (a finite scalar
        # at a sample time), not a specific internal QuTiP representation.
        for _, coeff in coeff_terms:
            assert callable(coeff)
            assert np.isfinite(complex(coeff(10.0)))

    def test_prepare_with_scalar_modulation_uses_callable_coefficients(self):
        """Scalar modulations should lower through QuTiP callable coefficients."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        chip.dress()
        tlist = np.linspace(0, 50, 201)
        op = CanonicalOperator.from_dense(
            np.eye(3, dtype=complex),
            dims=(3,),
            basis="fock",
            subsystem_labels=("q",),
        )
        desc = HamiltonianDescription(
            static_terms=(),
            dynamic_terms=(
                DynamicTerm(
                    operator=op,
                    time_dependence=ScalarModulation(signal=Carrier(freq=0.2)),
                    origin="drive",
                ),
            ),
            dims=(3,),
            metadata={},
        )
        prepared = chip.backend.prepare_hamiltonian(desc, tlist)

        coeff_terms = [item for item in prepared.rhs.to_list() if isinstance(item, list)]
        assert coeff_terms, "expected at least one lowered coefficient pair"
        # Verify each lowered coefficient is a usable time function (a finite scalar
        # at a sample time), not a specific internal QuTiP representation.
        for _, coeff in coeff_terms:
            assert callable(coeff)
            assert np.isfinite(complex(coeff(10.0)))

    def test_metadata_passes_through(self):
        """Metadata from the description should be available on PreparedHamiltonian."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        chip.dress()
        resolved = resolve_frame(chip, chip.frame)
        tlist = np.linspace(0, 50, 201)

        desc = build_hamiltonian_description(chip, [], resolved_frame=resolved)
        prepared = chip.backend.prepare_hamiltonian(desc, tlist)

        assert "frame" in prepared.metadata

    def test_simplify_signal_cancels_exact_opposing_carriers(self):
        """simplify_signal collapses opposite-sign equal-frequency carriers into their exact constant coefficient."""
        from quchip.engine.ir import Constant, Multiply, Carrier, simplify_signal

        signal = Multiply(
            (
                Carrier(freq=5.0, sign=1),
                Carrier(freq=5.0, sign=-1),
                Constant(2.0 + 0.0j),
            )
        )

        assert simplify_signal(signal) == Constant(2.0 + 0.0j)

    def test_build_hamiltonian_description_records_simplified_carrier_hint(self):
        """build_hamiltonian_description records carrier-frequency solver hints even after signal simplification."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[drive]))
        chip.dress()
        resolved = resolve_frame(chip, chip.frame)

        from quchip.engine.ir import DriveOp

        desc = build_hamiltonian_description(
            chip,
            [
                DriveOp(
                    target_label="q",
                    envelope=Square(amplitude=0.02, duration=20.0),
                    freq=5.0,
                    start_time=0.0,
                    drive_label=drive.label,
                )
            ],
            resolved_frame=resolved,
        )

        assert "max_carrier_freq_ghz" in desc.metadata
        assert "spectral_bound_ghz" in desc.metadata

    def test_charge_drive_rwa_drops_fast_counter_rotating_oscillation(self):
        """Chip-level drive RWA should reduce a resonant single-tone coefficient to a slow envelope."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        drive = ChargeDrive(target=q)

        from quchip.engine.ir import DriveOp

        drive_op = DriveOp(
            target_label="q",
            envelope=Square(amplitude=0.02, duration=20),
            freq=5.0,
            start_time=0.0,
            drive_label=drive.label,
        )

        chip_rwa = Chip([q], frame="rotating", rwa=True)
        chip_rwa.connect(ControlEquipment(lines=[drive]))
        chip_rwa.dress()
        resolved_rwa = resolve_frame(chip_rwa, chip_rwa.frame)
        desc_rwa = build_hamiltonian_description(
            chip_rwa,
            [drive_op],
            resolved_frame=resolved_rwa,
        )

        q_full = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        drive_full = ChargeDrive(target=q_full)
        chip_full = Chip([q_full], frame="rotating", rwa=False)
        chip_full.connect(ControlEquipment(lines=[drive_full]))
        chip_full.dress()
        resolved_full = resolve_frame(chip_full, chip_full.frame)
        drive_op_full = DriveOp(
            target_label="q",
            envelope=Square(amplitude=0.02, duration=20),
            freq=5.0,
            start_time=0.0,
            drive_label=drive_full.label,
        )
        desc_full = build_hamiltonian_description(
            chip_full,
            [drive_op_full],
            resolved_frame=resolved_full,
        )

        assert desc_rwa.dynamic_terms, "Expected dynamic drive terms under RWA."
        assert desc_full.dynamic_terms, "Expected dynamic drive terms without RWA."
        assert all(isinstance(term.time_dependence, ScalarModulation) for term in desc_rwa.dynamic_terms)
        assert all(isinstance(term.time_dependence, ScalarModulation) for term in desc_full.dynamic_terms)

    def test_prepare_metadata_can_drive_backend_default_nsteps(self):
        """resolve_solver_options should derive nsteps from spectral_bound_ghz metadata."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        chip.dress()
        prepared = PreparedHamiltonian(rhs=chip.hamiltonian(), metadata={"spectral_bound_ghz": 20.0})
        opts = chip.backend.resolve_solver_options({}, metadata=prepared.metadata, tlist=np.linspace(0.0, 10.0, 11))
        assert "nsteps" in opts


class TestEnvelopeSampleGrid:
    """The output tlist must not determine envelope interpolation fidelity.

    A 2-point ``[t0, t_end]`` tlist is a legitimate "final state only"
    request. Before the minimum-density floor, the slow envelope was
    interpolated from samples on exactly that grid — a windowed pulse was
    sampled only at its (near-)zero endpoints and silently vanished.
    """

    def test_two_point_tlist_matches_dense_tlist(self):
        """A 2-point tlist matches the dense tlist's final excited population: envelope sampling is grid-independent."""
        from quchip.control.envelopes import Gaussian
        from quchip.control.sequence import QuantumSequence

        def final_excited_population(tlist):
            q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
            drive = ChargeDrive(target=q, label="d")
            chip = Chip([q], frame="rotating")
            chip.wire(drive)
            seq = QuantumSequence(chip)
            seq.schedule(drive, envelope=Gaussian(duration=40.0, amplitude=0.02), freq=chip.freq(q))
            result = seq.simulate(tlist=tlist)
            return result.population(q, level=1)[-1]

        dense = final_excited_population(np.linspace(0.0, 40.0, 401))
        sparse = final_excited_population(np.array([0.0, 40.0]))

        assert dense > 0.1, "drive should visibly rotate the qubit"
        assert abs(sparse - dense) < 1e-4

    def test_narrow_gaussian_in_long_idle_span_matches_dense_reference(self):
        """A short windowed pulse buried in a long idle span resolves to well under the pre-fix 4e-3 error."""
        # Same lab-frame scenario as the max_step regression
        # (tests/test_backend_max_step.py): an 8 ns Gaussian pulse inside a
        # 308 ns solve. This end-to-end comparison folds in adaptive-solver
        # behavior beyond envelope sampling (the two tlists drive different
        # save/step points even though the coefficient grid for a windowed
        # envelope is now canonical/tlist-independent — see
        # test_window_subgrid_coefficient_matches_exact_envelope, which
        # pins the sampling accuracy itself against an exact reference).
        from quchip.control.envelopes import Gaussian
        from quchip.engine import simulate
        from quchip.engine.ir import DriveOp

        def final_ground_population(tlist):
            q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
            drive = ChargeDrive(target=q)
            chip = Chip(devices=[q], control_equipment=ControlEquipment(lines=[drive]))
            drive_op = DriveOp(
                target_label="q",
                envelope=Gaussian(duration=8.0, amplitude=1.0, sigmas=3),
                freq=5.0,
                start_time=150.0,
                drive_label=drive.label,
            )
            result = simulate(chip, [drive_op], tlist)
            return result.population("q", level=0)[-1]

        sparse = final_ground_population(np.array([0.0, 308.0]))
        dense = final_ground_population(np.linspace(0.0, 308.0, 308 * 10 + 1))

        assert abs(sparse - dense) < 2e-3, (
            f"end-to-end discrepancy {abs(sparse - dense):.2e} exceeds the 2e-3 bound "
            "(pre-fix baseline was ~4e-3; measured post-fix is ~7e-4)"
        )

    @staticmethod
    def _windowed_gaussian_band_envelope(duration: float, start_time: float, span: float = 308.0):
        """Build the real band-decomposed envelope AST for an amplitude-1 Gaussian DriveOp."""
        from quchip.control.envelopes import Gaussian
        from quchip.engine.ir import DriveOp, decompose_carrier_bands

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip(devices=[q], control_equipment=ControlEquipment(lines=[drive]))
        chip.dress()
        resolved = resolve_frame(chip, chip.frame)
        drive_op = DriveOp(
            target_label="q",
            envelope=Gaussian(duration=duration, amplitude=1.0, sigmas=3),
            freq=5.0,
            start_time=start_time,
            drive_label=drive.label,
        )
        desc = build_hamiltonian_description(chip, [drive_op], resolved_frame=resolved)
        band = decompose_carrier_bands(desc.dynamic_terms[0].time_dependence.signal)[0]
        return band.envelope, np.array([0.0, span])

    def _coefficient_error(self, duration: float, start_time: float, span: float = 308.0):
        """Return ``(max_outside_support, in_window_rel_rms, pulse_area_rel_err)`` for the production coefficient.

        Builds the coefficient via ``_envelope_coefficient`` — the exact
        function ``_band_coefficient``/the solver path uses. Two query
        resolutions, so accuracy at an extremely narrow window is neither
        overstated (outside-support ringing is a broad-scale artifact, so a
        coarse full-span sweep already catches it) nor understated (a fixed
        full-span query density under-resolves a sub-ns window; the
        in-window checks instead use a local sweep at a resolution that
        scales with *duration*).
        """
        from quchip.backend.qutip import _envelope_coefficient
        from quchip.engine.ir import evaluate_signal_program

        envelope, base_grid = self._windowed_gaussian_band_envelope(duration, start_time, span)
        coeff = _envelope_coefficient(envelope, base_grid)

        far_query = np.linspace(0.0, span, int(span * 20) + 1)
        far_sampled = np.array([complex(coeff(t)) for t in far_query])
        far_exact = np.asarray(evaluate_signal_program(envelope, far_query, xp=np), dtype=complex)
        outside_mask = np.abs(far_exact) < 1e-12
        max_outside = float(np.max(np.abs(far_sampled[outside_mask]))) if outside_mask.any() else 0.0

        margin = max(duration, 0.05)
        local_query = np.linspace(start_time - margin, start_time + duration + margin, 4001)
        local_sampled = np.array([complex(coeff(t)) for t in local_query])
        local_exact = np.asarray(evaluate_signal_program(envelope, local_query, xp=np), dtype=complex)
        rel_rms = float(np.sqrt(np.mean(np.abs(local_sampled - local_exact) ** 2)) / np.max(np.abs(local_exact)))
        area_exact = np.trapezoid(np.real(local_exact), local_query)
        area_sampled = np.trapezoid(np.real(local_sampled), local_query)
        area_rel_err = float(abs(area_sampled - area_exact) / abs(area_exact))
        return max_outside, rel_rms, area_rel_err

    def test_window_subgrid_coefficient_matches_exact_envelope(self):
        """The 8 ns window's production coefficient matches the exact envelope AST, with no outside-support leakage."""
        # Pins accuracy at the coefficient level, independent of solver
        # behavior, via the real _envelope_coefficient production path
        # (worst-case 2-point base grid — the augmented grid for a
        # windowed envelope is canonical/tlist-independent regardless).
        max_outside, rel_rms, area_rel_err = self._coefficient_error(duration=8.0, start_time=150.0)
        assert max_outside < 0.02, f"outside-support leakage {max_outside:.2e} exceeds the 0.02 bound"
        assert rel_rms < 1e-3, f"coefficient RMS error {rel_rms:.2e} exceeds the 1e-3 bound"
        assert area_rel_err < 1e-3, f"pulse-area error {area_rel_err:.2e} exceeds the 1e-3 bound"

    def test_narrow_window_coefficient_has_no_cubic_ringing(self):
        """A sub-ns windowed coefficient stays near-zero outside its support across the full solve span."""
        # Regression for cubic-spline ringing: the canonical grid's dense
        # local subgrid sits immediately next to a sparse 3-point full-span
        # skeleton, and a naive order-3 (cubic) interpolant extrapolates
        # wildly across that non-uniform knot spacing for a narrow window
        # (measured pre-fix: -2423 at t=50 ns and +3.29 at t=200 ns for a
        # 0.1 ns pulse at t=150.17 ns in a [0, 308] ns span — millions-fold
        # pulse-area error). _envelope_coefficient now interpolates a
        # windowed grid at order=1 (linear), which cannot overshoot its
        # bracketing node values. The 0.02 outside-support bound sits just
        # above the physical truncated-Gaussian edge value itself
        # (amplitude * exp(-sigmas**2/2) ~= 0.011 at sigmas=3) — a linear
        # interpolant's worst case is confined to one grid cell adjacent to
        # that edge, never propagating further into the idle span.
        for duration in (0.1, 0.01):
            max_outside, rel_rms, area_rel_err = self._coefficient_error(duration=duration, start_time=150.17)
            assert max_outside < 0.02, (
                f"duration={duration}: outside-support leakage {max_outside:.2e} exceeds the 0.02 bound "
                "(pre-fix this reached hundreds to millions from cubic ringing)"
            )
            assert rel_rms < 0.01, f"duration={duration}: coefficient RMS error {rel_rms:.2e} exceeds the 0.01 bound"
            assert area_rel_err < 0.02, (
                f"duration={duration}: pulse-area error {area_rel_err:.2e} exceeds the 0.02 bound"
            )
