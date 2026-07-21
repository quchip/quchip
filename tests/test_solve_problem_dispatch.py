"""Tests for the typed SolveProblem dispatch path.

Verifies build_problem(), solve_problem(), Chip.solve(), and
Chip.solve_many() produce correct results that match the
``simulate()`` convenience path.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip.backend import reset_default_backend, set_default_backend
from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.sequence import QuantumSequence
from quchip.control.drive import ChargeDrive
from quchip.control.envelopes import Square
from quchip.control.equipment import ControlEquipment
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine import build_problem, simulate, solve_problem
from quchip.engine.ir import DriveOp, SolveProblem


class _NoisyChargeDrive(ChargeDrive):
    def collapse_operators(self, device):
        return [0.1 * device.number_operator()]


class _NoisyCapacitive(Capacitive):
    def collapse_operators(self, chip):
        _ = chip
        return [0.05 * self.interaction_hamiltonian()]


class TestBuildSolveProblem:
    """Verify build_problem assembles correct SolveProblem."""

    def test_returns_solve_problem(self):
        """build_problem must return a SolveProblem instance."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        ChargeDrive(target=q)
        chip = Chip([q])
        tlist = np.linspace(0, 50, 201)

        problem = build_problem(chip, [], tlist)
        assert isinstance(problem, SolveProblem)

    def test_chip_reference_preserved(self):
        """SolveProblem.chip must be the same chip instance."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        tlist = np.linspace(0, 50, 201)

        problem = build_problem(chip, [], tlist)
        assert problem.chip is chip

    def test_backend_rejected_in_options(self):
        """SolveProblem rejects 'backend' key in options."""
        with pytest.raises(ValueError, match="must not contain 'backend'"):
            SolveProblem(
                chip=None,
                hamiltonian=None,
                initial_state=None,
                tlist=np.linspace(0, 50, 201),
                options={"backend": "something"},
            )

    def test_options_dict_is_copied(self):
        """SolveProblem copies the options dict so external mutation is isolated."""
        original = {"store_states": True}
        problem = SolveProblem(
            chip=None,
            hamiltonian=None,
            initial_state=None,
            tlist=np.linspace(0, 50, 201),
            options=original,
        )
        original["store_states"] = False
        assert problem.options["store_states"] is True

    def test_problem_with_drive_ops(self):
        """SolveProblem should include drive Hamiltonian terms."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[drive]))
        tlist = np.linspace(0, 50, 201)
        drive_op = DriveOp(
            target_label="q",
            envelope=Square(amplitude=0.02, duration=50),
            freq=5.0,
            start_time=0.0,
            drive_label=drive.label,
        )

        problem = build_problem(chip, [drive_op], tlist)
        assert problem.hamiltonian is not None
        assert len(problem.hamiltonian.dynamic_terms) > 0

    def test_problem_collects_drive_level_collapse_operators(self):
        """build_problem() collects a drive's own collapse_operators() into problem.c_ops."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        drive = _NoisyChargeDrive(target=q)
        chip = Chip([q])
        chip.wire(drive)

        problem = build_problem(chip, [], np.linspace(0.0, 10.0, 11))

        assert len(problem.c_ops) == 1

    def test_problem_collects_coupling_level_collapse_operators(self):
        """build_problem() collects a coupling's own collapse_operators() into problem.c_ops."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        r = DuffingTransmon(freq=5.4, anharmonicity=-0.2, levels=3, label="r")
        coupling = _NoisyCapacitive(q, r, g=0.01)
        chip = Chip([q, r], [coupling])

        problem = build_problem(chip, [], np.linspace(0.0, 10.0, 11))

        assert len(problem.c_ops) == 1


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

    def test_chip_solve(self):
        """Chip.solve() produces results matching simulate."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        tlist = np.linspace(0, 50, 201)

        problem = build_problem(chip, [], tlist)
        result = chip.solve(problem)
        assert result is not None
        assert result.populations is not None

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

    def test_chip_solve_accepts_problem_after_chip_mutation(self):
        """solve() treats SolveProblem as a frozen snapshot — chip mutations after build are allowed."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        problem = build_problem(chip, [], np.linspace(0.0, 10.0, 11))

        q.freq += 0.1

        result = chip.solve(problem)
        assert result.solver in {"sesolve", "mesolve"}

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

    def test_build_problem_returns_solve_problem(self):
        """sequence.build_problem() returns a SolveProblem referencing the sequence's chip."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=50.0, amplitude=0.02), freq=5.0)

        problem = sequence.build_problem()

        assert isinstance(problem, SolveProblem)
        assert problem.chip is chip
        assert len(sequence.scheduled_ops) == 1

    def test_build_problem_uses_run_default_tlist(self):
        """build_problem() with no tlist uses the same default tlist as run()."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=50.0, amplitude=0.02), freq=5.0)

        problem = sequence.build_problem()

        expected_tlist = np.linspace(0.0, 50.0, 500)
        npt.assert_allclose(problem.tlist, expected_tlist)

    def test_build_problem_preserves_inputs(self):
        """build_problem() preserves tlist, initial_state, solver, options, and e_ops verbatim."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=50.0, amplitude=0.02), freq=5.0)

        tlist = np.linspace(0.0, 50.0, 101)
        initial_state = chip.state(q=0)
        e_ops = chip.e_ops(q="X")
        options = {"progress_bar": "tqdm", "store_states": False}

        problem = sequence.build_problem(
            tlist=tlist,
            solver="sesolve",
            options=options,
            e_ops=e_ops,
            initial_state=initial_state,
        )

        npt.assert_allclose(problem.tlist, tlist)
        assert problem.initial_state is initial_state
        assert problem.solver == "sesolve"
        assert problem.e_ops is not None
        assert problem.e_ops_meta is not None
        assert len(problem.e_ops) == len(problem.e_ops_meta)
        assert all(meta.key == "q" for meta in problem.e_ops_meta)
        assert problem.options["progress_bar"] == "tqdm"
        assert problem.options["store_states"] is False
        assert problem.options["store_final_state"] is True

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
        explicit_result = explicit.simulate(tlist=tlist, options={"store_states": True, "store_final_state": True})

        virtual = QuantumSequence(chip)
        virtual.vz("q", np.pi / 2.0)
        virtual.schedule(
            drive,
            envelope=Square(duration=20.0, amplitude=0.02),
            freq=5.0,
        )
        virtual_result = virtual.simulate(tlist=tlist, options={"store_states": True, "store_final_state": True})

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

    def test_build_batch_accepts_mapping_initial_states(self):
        """build_batch() accepts a mix of dict and device-object initial-state specs per axis."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=20.0, amplitude=0.02), freq=5.0)
        state_axis = sequence.vary("initial_state", [{"q": 0}, {q: 1}], name="state")

        problems = sequence.build_batch(
            state_axis,
            tlist=np.linspace(0.0, 20.0, 81),
        )

        assert len(problems) == 2
        assert all(problem.chip is chip for problem in problems)
        assert problems[0].initial_state is not problems[1].initial_state

    def test_build_batch_reuses_problem_scaffold(self):
        """build_batch() shares the Hamiltonian operator skeleton, tlist, and frame across elements."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=20.0, amplitude=0.02), freq=5.0)
        tlist = np.linspace(0.0, 20.0, 81)
        e_ops = chip.e_ops(q=["X"])
        state_axis = sequence.vary("initial_state", [{"q": 0}, {"q": 1}], name="state")

        problems = sequence.build_batch(
            state_axis,
            tlist=tlist,
            e_ops=e_ops,
        )

        assert len(problems) == 2
        # Element-level HamiltonianDescriptions are materialised on demand; identity equality
        # of static_terms holds against the SolveBatch and across elements, not just within one.
        batch = problems.batch
        assert batch.hamiltonian.static_terms is problems[0].hamiltonian.static_terms
        assert problems[0].hamiltonian.static_terms is problems[1].hamiltonian.static_terms
        for slot in range(len(problems[0].hamiltonian.dynamic_terms)):
            assert (
                problems[0].hamiltonian.dynamic_terms[slot].operator
                is problems[1].hamiltonian.dynamic_terms[slot].operator
            )
        assert problems[0].tlist is problems[1].tlist
        assert problems[0].resolved_frame is problems[1].resolved_frame
        assert problems[0].e_ops is problems[1].e_ops
        assert problems[0].initial_state is not problems[1].initial_state

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

    def test_build_batch_compiles_template_once_for_homogeneous_sweep(self, monkeypatch: pytest.MonkeyPatch):
        """build_batch() compiles the Hamiltonian template once and instantiates it per element."""
        import quchip.control.sequence as sequence_module

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        pulse = sequence.schedule(drive, envelope=Square(duration=20.0, amplitude=0.02), freq=5.0)
        amp = pulse.vary("amplitude", [0.01, 0.02], name="amp")
        freq = pulse.vary("freq", [4.9, 5.1], name="freq")

        compile_calls = 0
        instantiate_calls = 0

        original_compile = sequence_module.compile_hamiltonian_template
        original_instantiate = sequence_module.instantiate_hamiltonian_description

        def counted_compile(*args, **kwargs):
            nonlocal compile_calls
            compile_calls += 1
            return original_compile(*args, **kwargs)

        def counted_instantiate(*args, **kwargs):
            nonlocal instantiate_calls
            instantiate_calls += 1
            return original_instantiate(*args, **kwargs)

        monkeypatch.setattr(sequence_module, "compile_hamiltonian_template", counted_compile)
        monkeypatch.setattr(sequence_module, "instantiate_hamiltonian_description", counted_instantiate)

        batch = sequence.build_batch(
            amp,
            freq,
            tlist=np.linspace(0.0, 20.0, 81),
            initial_state=chip.state(q=0),
        )

        assert len(batch) == 4
        assert compile_calls == 1
        assert instantiate_calls == 5

    def test_build_batch_delay_axis_shifts_later_pulses(self, monkeypatch: pytest.MonkeyPatch):
        """Sweeping a delay's duration in build_batch() shifts every later pulse's start time."""
        import quchip.control.sequence as sequence_module

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=2.0, amplitude=0.01), freq=5.0)
        wait = sequence.delay(q, 3.0)
        sequence.schedule(drive, envelope=Square(duration=1.0, amplitude=0.02), freq=5.0)

        captured_start_times: list[tuple[float, ...]] = []
        original_instantiate = sequence_module.instantiate_hamiltonian_description

        def capture_start_times(template, drive_ops, chip):
            captured_start_times.append(tuple(float(op.start_time) for op in drive_ops))
            return original_instantiate(template, drive_ops, chip)

        monkeypatch.setattr(sequence_module, "instantiate_hamiltonian_description", capture_start_times)

        batch = sequence.build_batch(
            wait.vary("duration", [5.0, 9.0], name="tau"),
            tlist=np.linspace(0.0, 12.0, 49),
            initial_state=chip.state(q=0),
        )

        assert len(batch) == 2
        assert (0.0, 7.0) in captured_start_times
        assert (0.0, 11.0) in captured_start_times

    def test_build_batch_computes_default_initial_state_once(self, monkeypatch: pytest.MonkeyPatch):
        """build_batch() computes the shared default initial state once, not once per element."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        pulse = sequence.schedule(drive, envelope=Square(duration=20.0, amplitude=0.02), freq=5.0)
        amp = pulse.vary("amplitude", [0.01, 0.02], name="amp")
        freq = pulse.vary("freq", [4.9, 5.1], name="freq")

        original_state = chip.state
        state_calls = 0

        def counted_state(*args, **kwargs):
            nonlocal state_calls
            state_calls += 1
            return original_state(*args, **kwargs)

        monkeypatch.setattr(chip, "state", counted_state)

        batch = sequence.build_batch(
            amp,
            freq,
            tlist=np.linspace(0.0, 20.0, 81),
        )

        assert len(batch) == 4
        assert state_calls == 1


class TestChipSolveMany:
    """Verify Chip.solve_many() uses batched dispatch."""

    def test_solve_many_returns_results(self, monkeypatch: pytest.MonkeyPatch):
        """Chip.solve_many() returns one result per problem."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        tlist = np.linspace(0, 50, 201)

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

        problems = [build_problem(chip, [], tlist) for _ in range(3)]
        results = chip.solve_many(problems, progress=False)

        assert len(results) == 3
        assert results.shape == (3,)
        assert results.axes == (("batch", (0, 1, 2)),)
        assert results[{"batch": 2}] is results[2]
        for r in results:
            assert r.populations is not None

    def test_solve_many_parallelizes_heterogeneous_qutip_problems(self, monkeypatch: pytest.MonkeyPatch):
        """Heterogeneous QuTiP problems reach one thresholded executor dispatch."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip([q], frame="rotating")
        problems = [
            build_problem(chip, [], np.linspace(0.0, duration, 21 + index))
            for index, duration in enumerate(np.linspace(10.0, 80.0, 8))
        ]

        mapped_items: list[list[SolveProblem]] = []
        requested_jobs: list[int] = []

        class InlineExecutor:
            def map(self, task, items):
                batch = list(items)
                mapped_items.append(batch)
                return map(task, batch)

        def get_executor(n_jobs):
            requested_jobs.append(n_jobs)
            return InlineExecutor()

        monkeypatch.setattr(chip.backend, "_get_executor", get_executor)

        results = chip.solve_many(problems, progress=False)

        assert requested_jobs == [-1]
        assert mapped_items == [problems]
        assert len(results) == 8

    def test_solve_many_retains_structural_dispatch_below_qutip_threshold(self, monkeypatch: pytest.MonkeyPatch):
        """Small heterogeneous QuTiP lists retain structural-group dispatch."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        chip = Chip([q], frame="rotating")
        problems = [
            build_problem(chip, [], np.linspace(0.0, duration, 21 + index))
            for index, duration in enumerate((10.0, 20.0, 30.0))
        ]

        original_solve_batch = chip.backend.solve_batch
        solve_batch_sizes: list[int] = []

        def counted_solve_batch(batch, *, progress=True):
            solve_batch_sizes.append(batch.batch_size)
            return original_solve_batch(batch, progress=progress)

        monkeypatch.setattr(chip.backend, "solve_batch", counted_solve_batch)

        results = chip.solve_many(problems, progress=False)

        assert solve_batch_sizes == [1, 1, 1]
        assert len(results) == 3

    @pytest.mark.optional_backend
    def test_solve_many_groups_homogeneous_subbatches(self, monkeypatch: pytest.MonkeyPatch):
        """Two build_batch() results concatenated into a list should reach solve_batch in two groups."""
        pytest.importorskip("dynamiqs")
        reset_default_backend()
        set_default_backend("dynamiqs")
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=20.0, amplitude=0.02), freq=5.0)
        e_ops = chip.e_ops(q=["X"])

        state_axis = sequence.vary("initial_state", [{"q": 0}, {"q": 1}], name="state")

        problems = list(sequence.build_batch(
            state_axis,
            tlist=np.linspace(0.0, 20.0, 81),
            e_ops=e_ops,
        )) + list(sequence.build_batch(
            state_axis,
            tlist=np.linspace(0.0, 30.0, 121),
            e_ops=e_ops,
        ))

        original_solve_batch = chip.backend.solve_batch
        solve_batch_sizes: list[int] = []

        def counted_solve_batch(batch, *, progress=True):
            solve_batch_sizes.append(batch.batch_size)
            return original_solve_batch(batch, progress=progress)

        monkeypatch.setattr(chip.backend, "solve_batch", counted_solve_batch)

        results = chip.solve_many(problems, progress=False)

        assert len(results) == 4
        assert sorted(solve_batch_sizes) == [2, 2]
        reset_default_backend()

    @pytest.mark.optional_backend
    def test_build_batch_chevron_avoids_prepare_per_point_on_dynamiqs(self, monkeypatch: pytest.MonkeyPatch):
        """A chevron sweep should produce a single vmapped solve_batch call on dynamiqs."""
        pytest.importorskip("dynamiqs")
        reset_default_backend()
        set_default_backend("dynamiqs")
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], frame="rotating")
        chip.connect(ControlEquipment(lines=[drive]))
        sequence = QuantumSequence(chip)
        pulse = sequence.schedule(drive, envelope=Square(duration=20.0, amplitude=0.02), freq=5.0)
        amp = pulse.vary("amplitude", [0.01, 0.02], name="amp")
        freq = pulse.vary("freq", [4.9, 5.1], name="freq")

        original_prepare_hamiltonian = chip.backend.prepare_hamiltonian
        prepare_calls: list[tuple[int, int]] = []

        def counted_prepare_hamiltonian(description, tlist):
            prepare_calls.append((id(description), id(tlist)))
            return original_prepare_hamiltonian(description, tlist)

        original_solve_batch = chip.backend.solve_batch
        solve_batch_sizes: list[int] = []

        def counted_solve_batch(batch, *, progress=True):
            solve_batch_sizes.append(batch.batch_size)
            return original_solve_batch(batch, progress=progress)

        monkeypatch.setattr(chip.backend, "prepare_hamiltonian", counted_prepare_hamiltonian)
        monkeypatch.setattr(chip.backend, "solve_batch", counted_solve_batch)

        problems = sequence.build_batch(
            amp,
            freq,
            tlist=np.linspace(0.0, 20.0, 81),
            e_ops=chip.e_ops(q=["X"]),
            initial_state=chip.state(q=0),
        )
        results = chip.solve_many(problems, progress=False)

        assert len(results) == 4
        assert solve_batch_sizes == [4]
        # prepare_hamiltonian is the single-element entry point; the new
        # fast path prepares once per batch via prepare_batch, so
        # prepare_hamiltonian must not be called at all.
        assert len(prepare_calls) == 0
        reset_default_backend()

    def test_solve_many_rejects_wrong_chip(self):
        """solve_many rejects problems from a different chip."""
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip1 = Chip([q1])

        q2 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip2 = Chip([q2])

        tlist = np.linspace(0, 50, 201)
        problems = [build_problem(chip1, [], tlist)]

        with pytest.raises(ValueError, match="different chip"):
            chip2.solve_many(problems, progress=False)

    def test_solve_many_accepts_problem_after_chip_mutation(self):
        """solve_many() treats SolveProblem as a frozen snapshot — chip mutations after build are allowed."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        chip = Chip([q])
        problem = build_problem(chip, [], np.linspace(0.0, 10.0, 11))

        q.freq += 0.1

        results = chip.solve_many([problem], progress=False)
        assert len(results) == 1
