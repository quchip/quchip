"""Tests for the typed SolveProblem dispatch path.

Verifies build_problem(), solve_problem(), Chip.solve(), and
Chip.solve_many() produce correct results that match the
``simulate()`` convenience path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from quchip.backend import reset_default_backend, set_default_backend
from quchip.chip.chip import Chip
from quchip.control.sequence import QuantumSequence
from quchip.control.drive import ChargeDrive
from quchip.control.envelopes import Square
from quchip.control.equipment import ControlEquipment
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.devices.transmon.charge_basis import ChargeBasisTransmon
from quchip.engine import build_problem, simulate, solve_problem
from quchip.engine.ir import DriveOp, SolveProblem


class TestBuildSolveProblem:
    """Verify build_problem assembles correct SolveProblem."""

    def test_built_request_keeps_dimensions_and_observable_context_after_source_edits(self):
        """An old solve uses captured dimensions and labels after the authoring device changes."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q], frame="rotating")
        problem = build_problem(chip, [], np.linspace(0.0, 2.0, 5), initial_state={q: 1}, e_ops={q: q.sigma_z})
        q.levels = 4
        q.freq = 5.2
        result = solve_problem(problem)

        assert tuple(result.dims) == (3,)
        assert result.reduced_state(2.0, "q").shape == (3, 3)
        npt.assert_allclose(result.expect("q"), -1.0, atol=1e-12)

    def test_built_request_copies_time_and_native_state_buffers(self):
        """Mutating caller-owned time and ket buffers cannot change a built solve."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q], frame="rotating")
        times = np.linspace(0.0, 2.0, 5)
        state = chip.bare_state({q: 1})
        problem = build_problem(chip, [], times, initial_state=state)
        times[:] = np.linspace(0.0, 4.0, 5)
        state.data = chip.bare_state({q: 0}).data

        result = solve_problem(problem)
        npt.assert_array_equal(result.times, np.linspace(0.0, 2.0, 5))
        npt.assert_allclose(result.population("q", 1), 1.0, atol=1e-12)

    @pytest.mark.parametrize("source", ["envelope", "transform"])
    def test_built_request_captures_mutable_signal_payloads(self, source):
        """Source edits change future builds but cannot retune an existing request."""
        from quchip.control.signal import SignalTransform

        class BufferGain(SignalTransform):
            def __init__(self, factor):
                self.factor = factor

            def apply(self, signals):
                return {key: signal.scaled(self.factor) for key, signal in signals.items()}

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
        chip = Chip([q], frame="rotating", backend="qutip")
        factor = np.array(1.0)
        chip.wire(ChargeDrive(q), signal_chain=[BufferGain(factor)])
        envelope = Square(duration=10.0, amplitude=0.01)
        seq = QuantumSequence(chip)
        seq.charge(q, envelope=envelope, freq=chip.freq(q))
        times = np.linspace(0.0, 10.0, 21)
        problem = seq.build_problem(times)
        before = solve_problem(problem).population("q", 1)
        assert before[-1] > 0.09
        if source == "envelope":
            envelope.amplitude = 0.0
        else:
            factor[...] = 0.0
        after = solve_problem(problem).population("q", 1)
        fresh = solve_problem(seq.build_problem(times)).population("q", 1)
        npt.assert_allclose(after, before, atol=1e-12)
        npt.assert_allclose(fresh, 0.0, atol=1e-12)

    def test_built_request_keeps_its_backend_after_the_default_changes(self):
        """Backend dispatch belongs to the built request rather than a later global default."""
        from dataclasses import replace

        from quchip.backend.qutip import QuTiPBackend
        from quchip.engine import solve_many

        class LaterBackend(QuTiPBackend):
            def solve_problem(self, problem):
                raise AssertionError("old request dispatched through the new default")

            def is_native_state(self, state):
                raise AssertionError("old request rematerialized through the new default")

        reset_default_backend()
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        problem = build_problem(chip, [], np.linspace(0.0, 2.0, 5))
        try:
            set_default_backend(LaterBackend())
            result = solve_problem(problem)
            npt.assert_allclose(result.population("q", 0), 1.0, atol=1e-12)
            variant = replace(problem, solver="sesolve")
            assert variant.backend is problem.backend
            for result in solve_many([problem, variant], progress=False):
                npt.assert_allclose(result.population("q", 0), 1.0, atol=1e-12)
        finally:
            reset_default_backend()

    def test_batch_rejects_mixed_captured_backends(self):
        """A shared batch cannot silently run a point through another point's backend."""
        from dataclasses import replace

        from quchip.backend.qutip import QuTiPBackend
        from quchip.engine import solve_many
        from quchip.engine.ir import SolveBatch

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        problem = build_problem(chip, [], np.linspace(0.0, 2.0, 5))
        other = replace(problem, backend=QuTiPBackend())

        with pytest.raises(ValueError, match="captured backend"):
            SolveBatch(chip=chip, problems=(problem, other))
        results = solve_many([problem, other], progress=False)
        assert results[0]._backend is problem.backend
        assert results[1]._backend is other.backend

    def test_options_dict_is_copied(self):
        """SolveProblem captures nested option containers and their NumPy buffers."""
        original = {"normalize_output": True, "nested": [{"values": np.array([1.0, 2.0])}]}
        problem = SolveProblem(
            chip=None,
            engine_result=None,
            initial_state=None,
            tlist=np.linspace(0, 50, 201),
            options=original,
        )
        original["normalize_output"] = False
        original["nested"][0]["values"][:] = 0.0
        assert problem.options["normalize_output"] is True
        npt.assert_array_equal(problem.options["nested"][0]["values"], [1.0, 2.0])


class TestSolveProblemDispatch:
    """Verify solve_problem matches ``simulate()`` results."""

    def test_solve_problem_rabi(self):
        """solve_problem produces same Rabi oscillation as simulate."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))

        envelope = Square(duration=50.0, amplitude=0.02)
        drive_op = DriveOp(
            target_label="q",
            envelope=envelope,
            freq=5.0,
            start_time=0.0,
            drive_label=drive.label,
        )
        tlist = np.linspace(0, 50, 201)

        # Convenience path
        result_reference = simulate(chip, [drive_op], tlist)
        p1_reference = result_reference.population("q", 1)

        # Typed path
        problem = build_problem(chip, [drive_op], tlist)
        result_typed = solve_problem(problem)
        p1_typed = result_typed.population("q", 1)

        npt.assert_allclose(p1_typed, p1_reference, atol=1e-6)


class TestChipSolve:
    """Verify Chip.solve() dispatches correctly."""

    def test_chip_solve_rejects_wrong_chip(self):
        """Chip.solve() rejects problems built for a different chip."""
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip1 = Chip([q1])

        q2 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip2 = Chip([q2])

        tlist = np.linspace(0, 50, 201)
        problem = build_problem(chip1, [], tlist)

        with pytest.raises(ValueError, match="different chip"):
            chip2.solve(problem)

    def test_chip_solve_rejects_non_problem(self):
        """Chip.solve() rejects non-SolveProblem input."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])

        with pytest.raises(TypeError, match="SolveProblem"):
            chip.solve({"not": "a problem"})

    def test_solve_problem_uses_built_snapshot_even_after_chip_mutation(self):
        """A built SolveProblem is a frozen snapshot and remains solvable after chip mutations."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        drive_op = DriveOp(
            target_label="q",
            envelope=Square(amplitude=0.02, duration=50),
            freq=5.0,
            start_time=0.0,
            drive_label=drive.label,
        )
        tlist = np.linspace(0, 50, 201)
        problem = build_problem(chip, [drive_op], tlist)
        q.freq += 0.1  # mutate chip after build
        result = solve_problem(problem)
        assert result.solver in {"sesolve", "mesolve"}


class TestQuantumSequenceBuildProblem:
    """Verify QuantumSequence.build_problem() matches run()-time assembly."""

    def test_build_problem_uses_run_default_tlist(self):
        """build_problem() with no tlist uses the same default tlist as run()."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=50.0, amplitude=0.02), freq=5.0)

        problem = sequence.build_problem()

        result = sequence.simulate(states="none")
        npt.assert_array_equal(problem.tlist, result.times)
        assert problem.tlist[0] == 0.0
        assert problem.tlist[-1] == 50.0

    def test_run_matches_chip_solve_of_built_problem(self):
        """sequence.simulate() matches chip.solve() of the sequence's own built problem."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=50.0, amplitude=0.02), freq=5.0)

        tlist = np.linspace(0.0, 50.0, 201)
        problem = sequence.build_problem(tlist=tlist)
        result_solve = chip.solve(problem)
        result_run = sequence.simulate(tlist=tlist)

        npt.assert_allclose(result_run.population("q", 1), result_solve.population("q", 1), atol=1e-6)

    def test_virtual_z_matches_explicit_pulse_phase(self):
        """A vz() phase kick produces the same final state as an equal explicit pulse phase."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        tlist = np.linspace(0.0, 20.0, 201)

        explicit = QuantumSequence(chip)
        explicit.schedule(
            drive,
            envelope=Square(duration=20.0, amplitude=0.02),
            freq=5.0,
            phase=np.pi / 2.0,
        )
        explicit_result = explicit.simulate(tlist=tlist, states="all")

        virtual = QuantumSequence(chip)
        virtual.vz("q", np.pi / 2.0)
        virtual.schedule(
            drive,
            envelope=Square(duration=20.0, amplitude=0.02),
            freq=5.0,
        )
        virtual_result = virtual.simulate(tlist=tlist, states="all")

        overlap = chip.backend.overlap(explicit_result.final_state, virtual_result.final_state)
        npt.assert_allclose(np.abs(complex(overlap)), 1.0, atol=1e-6)

    def test_solve_many_matches_separate_runs(self, monkeypatch: pytest.MonkeyPatch):
        """chip.solve_many() over a batch of problems matches solving each one separately."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))

        original_batched_sesolve = chip.backend.batched_sesolve
        monkeypatch.setattr(
            chip.backend,
            "batched_sesolve",
            lambda problems, *, progress=True: original_batched_sesolve(
                problems,
                n_jobs=1,
                progress=progress,
            ),
        )

        tlist = np.linspace(0.0, 50.0, 201)
        states = [chip.state(q=0), chip.state(q=1)]

        problems = []
        run_results = []
        for initial_state in states:
            sequence = QuantumSequence(chip)
            sequence.schedule(drive, envelope=Square(duration=50.0, amplitude=0.02), freq=5.0)
            problems.append(sequence.build_problem(tlist=tlist, initial_state=initial_state))
            run_results.append(sequence.simulate(tlist=tlist, initial_state=initial_state))

        batch_results = chip.solve_many(problems, progress=False)

        for batch_result, run_result in zip(batch_results, run_results):
            npt.assert_allclose(
                batch_result.population("q", 1),
                run_result.population("q", 1),
                atol=1e-6,
            )


    def test_simulate_batch_matches_manual_batch(self, monkeypatch: pytest.MonkeyPatch):
        """sequence.simulate_batch() matches a manually built batch solved via chip.solve_many()."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))

        original_batched_sesolve = chip.backend.batched_sesolve
        monkeypatch.setattr(
            chip.backend,
            "batched_sesolve",
            lambda problems, *, progress=True: original_batched_sesolve(
                problems,
                n_jobs=1,
                progress=progress,
            ),
        )

        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=20.0, amplitude=0.02), freq=5.0)
        tlist = np.linspace(0.0, 20.0, 81)
        state_specs = [{"q": 0}, {"q": 1}]
        state_axis = sequence.vary("initial_state", state_specs, name="state")

        manual_results = chip.solve_many(
            sequence.build_batch(state_axis, tlist=tlist),
            progress=False,
        )
        batched_results = sequence.simulate_batch(state_axis, tlist=tlist, progress=False)

        assert batched_results.shape == (2,)
        assert batched_results.axes == (("state", state_specs),)

        for manual_result, batched_result in zip(manual_results, batched_results):
            npt.assert_allclose(
                manual_result.population("q", 1),
                batched_result.population("q", 1),
                atol=1e-6,
            )


    def test_build_batch_uses_the_semantic_ground_state_by_default(self):
        """A construction sweep rebuilds each Hamiltonian and its semantic ground state."""
        q = ChargeBasisTransmon(
            E_C=0.25,
            E_J=12.0,
            num_basis=7,
            basis="eigen",
            levels=3,
            label="q",
        )
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        pulse = sequence.schedule(drive, envelope=Square(duration=20.0, amplitude=0.02), freq=5.0)
        amp = pulse.vary("amplitude", [0.01, 0.02], name="amp")
        ej = sequence.vary("q.E_J", [10.0, 12.0], name="EJ")

        batch = sequence.build_batch(
            amp,
            ej,
            tlist=np.linspace(0.0, 20.0, 81),
        )

        assert len(batch) == 4
        expected = np.asarray([1.0, 0.0, 0.0])
        for problem in batch:
            state = chip.backend.to_array(problem.initial_state).reshape(-1)
            np.testing.assert_allclose(state, expected)
        h_low = batch[0].engine_result.hamiltonian().matrix(t=0.0)
        h_high = batch[1].engine_result.hamiltonian().matrix(t=0.0)
        assert not np.allclose(chip.backend.to_array(h_low), chip.backend.to_array(h_high))
        results = chip.solve_many(batch, progress=False)
        assert results.shape == (2, 2)


class TestSolverSelection:
    """The solver follows the state as well as the collapse terms."""

    @staticmethod
    def _mixed_ancilla_chip() -> tuple[Chip, Any]:
        from quchip import Resonator
        from quchip.chip.couplings import CrossKerr

        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
        ancilla = Resonator(freq=6.0, levels=2, label="a")
        chip = Chip([qubit, ancilla], couplings=[CrossKerr(qubit, ancilla, chi=0.01)], frame="rotating")
        ground = chip.backend.as_density_matrix(chip.bare_state({qubit: 1, ancilla: 0}))
        excited = chip.backend.as_density_matrix(chip.bare_state({qubit: 1, ancilla: 1}))
        return chip, 0.5 * ground + 0.5 * excited

    def test_density_matrix_without_loss_runs_mesolve(self):
        """A mixed initial state must evolve as U rho U^dagger even without collapse terms."""
        chip, rho = self._mixed_ancilla_chip()
        result = QuantumSequence(chip).simulate(
            np.linspace(0.0, 40.0, 41), initial_state=rho, partition=False
        )
        final = chip.backend.to_array(result.final_state)
        assert result.solver == "mesolve"
        npt.assert_allclose(np.trace(final), 1.0, atol=1e-8)
        npt.assert_allclose(final, final.conj().T, atol=1e-8)
        npt.assert_allclose(np.trace(final @ final).real, 0.5, atol=1e-8)

    def test_foreign_array_states_route_by_shape(self):
        """A problem carrying a native array state still routes before the solve boundary coerces it."""
        from dataclasses import replace

        chip, rho = self._mixed_ancilla_chip()
        problem = build_problem(chip, [], np.linspace(0.0, 4.0, 5))
        ket = np.asarray(chip.backend.to_array(problem.initial_state))
        assert replace(problem, initial_state=ket).solver_name(chip.backend) == "sesolve"
        flat = replace(problem, initial_state=ket.reshape(-1), solver="sesolve")
        assert flat.solver_name(chip.backend) == "sesolve"
        assert solve_problem(flat).solver == "sesolve"
        assert chip.backend.as_density_matrix(ket.reshape(-1)).shape == (ket.size, ket.size)
        mixed = build_problem(chip, [], np.linspace(0.0, 4.0, 5), initial_state=rho)
        mixed = replace(mixed, initial_state=np.asarray(chip.backend.to_array(mixed.initial_state)))
        assert mixed.solver_name(chip.backend) == "mesolve"
        assert solve_problem(mixed).solver == "mesolve"

    def test_flat_kets_solve_on_dynamiqs(self):
        """dynamiqs promotes a flat native ket to a column before its solvers see it."""
        pytest.importorskip("dynamiqs")
        from dataclasses import replace

        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip([qubit], frame="rotating", backend="dynamiqs")
        problem = build_problem(chip, [], np.linspace(0.0, 4.0, 5))
        flat = np.asarray(chip.backend.to_array(problem.initial_state)).reshape(-1)
        assert chip.backend.is_ket(flat)
        assert np.asarray(chip.backend.to_array(chip.backend.as_density_matrix(flat))).shape == (flat.size, flat.size)
        result = solve_problem(replace(problem, initial_state=flat, solver="sesolve"))
        assert result.solver == "sesolve"
        npt.assert_allclose(np.asarray(result.population("q", 0)), 1.0, atol=1e-8)

    def test_explicit_sesolve_rejects_a_density_matrix(self):
        chip, rho = self._mixed_ancilla_chip()
        with pytest.raises(RuntimeError) as excinfo:
            QuantumSequence(chip).simulate(
                np.linspace(0.0, 4.0, 5), initial_state=rho, solver="sesolve", partition=False
            )
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "sesolve evolves kets only" in str(excinfo.value.__cause__)
