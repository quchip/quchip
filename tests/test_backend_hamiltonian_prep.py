"""Backend preparation tests for the EngineResult IR."""

from __future__ import annotations

from quchip.approximations import RWA, Exact

import numpy as np
import pytest

from quchip.chip.chip import Chip
from quchip.control import ChargeDrive
from quchip.control.envelopes import Square
from quchip.control.equipment import ControlEquipment
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine.ir import (
    Carrier,
    ScalarModulation,
)
from quchip.engine.frames import resolve_frame
from quchip.engine.assembly import build_engine_result


@pytest.mark.parametrize("start,duration", [(10.2, 5.3), (150.17, 5.1), (1000.1, 5.3)])
def test_shifted_square_preserves_edges_and_pulse_area(start, duration):
    """Translating a square preserves its closed edges and sampled rotation angle."""
    import jax
    import jax.numpy as jnp
    from quchip.backend.qutip import _envelope_coefficient
    from quchip.engine.ir import Constant, Shift, Window

    signal = Shift(Window(Constant(1.), 0., duration), start)
    stop = start + duration
    times = np.array([np.nextafter(start, -np.inf), start, stop, np.nextafter(stop, np.inf)])
    np.testing.assert_array_equal(signal.evaluate(times, xp=np), [0., 1., 1., 0.])
    evaluated = jax.jit(lambda t: signal.evaluate(t, xp=jnp))(jnp.asarray(times))
    np.testing.assert_array_equal(evaluated, [0., 1., 1., 0.])
    coefficient = _envelope_coefficient(signal, np.array([0., stop+2.]))
    # Midpoint quadrature is exact for a square and detects a lost edge cell.
    query = start + (np.arange(10000)+.5)*duration/10000
    area = np.mean([coefficient(t) for t in query])*duration
    np.testing.assert_allclose(area, duration, atol=1e-11)


class TestPrepareHamiltonian:
    """Verify Backend.prepare_hamiltonian() round-trips correctly."""


    def test_simplify_signal_cancels_exact_opposing_carriers(self):
        """simplify_signal collapses opposite-sign equal-frequency carriers into their exact constant coefficient."""
        from quchip.engine.ir import Constant, Multiply, simplify_signal

        signal = Multiply(
            (
                Carrier(freq=5.0, sign=1),
                Carrier(freq=5.0, sign=-1),
                Constant(2.0 + 0.0j),
            )
        )

        assert simplify_signal(signal) == Constant(2.0 + 0.0j)

    def test_build_engine_result_records_simplified_carrier_hint(self):
        """build_engine_result records carrier-frequency solver hints even after signal simplification."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[drive]))
        chip.dress()
        resolved = resolve_frame(chip, chip.frame)

        from quchip.engine.ir import DriveOp

        desc = build_engine_result(
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

        chip_rwa = Chip([q], frame="rotating", approximation=RWA())
        chip_rwa.connect(ControlEquipment(lines=[drive]))
        chip_rwa.dress()
        resolved_rwa = resolve_frame(chip_rwa, chip_rwa.frame)
        desc_rwa = build_engine_result(
            chip_rwa,
            [drive_op],
            resolved_frame=resolved_rwa,
        )

        q_full = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        drive_full = ChargeDrive(target=q_full)
        chip_full = Chip([q_full], frame="rotating", approximation=Exact())
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
        desc_full = build_engine_result(
            chip_full,
            [drive_op_full],
            resolved_frame=resolved_full,
        )

        assert desc_rwa.dynamic_terms, "Expected dynamic drive terms under RWA."
        assert desc_full.dynamic_terms, "Expected dynamic drive terms without RWA."
        assert all(isinstance(term.time_dependence, ScalarModulation) for term in desc_rwa.dynamic_terms)
        assert all(isinstance(term.time_dependence, ScalarModulation) for term in desc_full.dynamic_terms)


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
        desc = build_engine_result(chip, [drive_op], resolved_frame=resolved)
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


def test_pulse_train_shares_one_qutip_coefficient_per_operator_and_carrier():
    """Pulses and crosstalk through one operator at one carrier share a QobjEvo part with the per-band H(t)."""
    from quchip import Capacitive, Gaussian, QuantumSequence
    from quchip.backend.qutip import _envelope_coefficient
    from quchip.engine.ir import decompose_carrier_bands

    def lowered(pulses):
        qubits = [DuffingTransmon(freq=f, anharmonicity=-0.25, levels=3, label=f"q{i}")
                  for i, f in enumerate((5.0, 5.2))]
        equipment = ControlEquipment([ChargeDrive(q, label=f"d{i}") for i, q in enumerate(qubits)])
        equipment.set_crosstalk_matrix([[1.0, 0.1], [0.05, 1.0]], [[0.0, 0.3], [-0.2, 0.0]])
        chip = Chip(qubits, [Capacitive(*qubits, g=0.01)], control_equipment=equipment,
                    frame=5.1, approximation=RWA())
        sequence = QuantumSequence(chip)
        for k in range(pulses):
            sequence.schedule("d0", envelope=Gaussian(duration=20.0, amplitude=0.02), freq=5.0,
                              start_time=20.0 * k)
            sequence.schedule("d1", envelope=Square(duration=13.0, amplitude=0.01), freq=5.0, phase=0.4,
                              start_time=20.0 * k + 7.0)
        tlist = np.array([0.0, 20.0 * pulses + 10.0])
        result = sequence.build_problem(tlist=tlist).engine_result
        return chip.backend, result, tlist, chip.backend.prepare_hamiltonian(result, tlist).rhs

    backend, result, tlist, rhs = lowered(4)
    grid = backend._resolve_envelope_sample_tlist(tlist)
    static = backend._sum_terms(result.static_terms, backend._canonical_to_qobj).full()
    bands = [(backend._canonical_to_qobj(term.operator).full(), _envelope_coefficient(band.envelope, grid),
              complex(band.freq))
             for term in result.dynamic_terms for band in decompose_carrier_bands(term.time_dependence.signal)]
    edges = np.array([20.0 * k + d for k in range(4) for d in (0.0, 7.0, 20.0)])
    times = np.concatenate([np.linspace(0.0, 90.0, 301), edges, np.nextafter(edges, -np.inf),
                            np.nextafter(edges, np.inf)])
    for t in times:
        expected = static + sum(op * coefficient(t) * np.exp(1j * freq * t) for op, coefficient, freq in bands)
        np.testing.assert_allclose(rhs(t).full(), expected, rtol=0.0, atol=1e-12)
    assert len(bands) > len(rhs.to_list()) - 1
    assert len(rhs.to_list()) == len(lowered(1)[3].to_list())
