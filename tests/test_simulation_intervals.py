"""The solve interval belongs to the requested clock, independently of the schedule."""

import numpy as np
import pytest

from quchip import Chip, ControlEquipment, DuffingTransmon, Exact, FluxDrive, QuantumSequence, Square


def _options(backend):
    if backend == "dynamiqs":
        import dynamiqs as dq

        return {"method": dq.method.Tsit5(rtol=1e-9, atol=1e-11)}
    return {"rtol": 1e-9, "atol": 1e-11}


def _sequence(backend="qutip"):
    q = DuffingTransmon(freq=0.2, anharmonicity=-0.02, levels=2, label="q")
    q.reference_freq = 0.0
    drive = FluxDrive(q, label="flux")
    chip = Chip([q], backend=backend, frame="lab", approximation=Exact(),
                control_equipment=ControlEquipment([drive]))
    return chip, drive, QuantumSequence(chip)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_partial_interval_preserves_initial_time_and_carrier_phase(backend):
    chip, drive, sequence = _sequence(backend)
    sequence.schedule(drive, envelope=Square(duration=1.0, amplitude=0.4))
    sequence.schedule(drive, envelope=Square(duration=4.0, amplitude=0.07),
                      start_time=2.0, freq=0.13, phase=0.2)
    sequence.schedule(drive, envelope=Square(duration=1.0, amplitude=0.4), start_time=8.0)
    times = np.array([3.0, 3.17, 3.9, 4.2])
    initial = (chip.backend.basis(2, 0) + chip.backend.basis(2, 1)) / np.sqrt(2)
    result = sequence.simulate(times, initial_state=initial, partition=False, options=_options(backend))
    np.testing.assert_array_equal(result.times, times)
    phase = 2 * np.pi * 0.2 * (times - times[0]) + (0.07 / 0.13) * (
        np.sin(2 * np.pi * 0.13 * times - 0.2) - np.sin(2 * np.pi * 0.13 * times[0] - 0.2)
    )
    coherences = [chip.backend.to_array(chip.backend.state_to_dm(state))[0, 1] for state in result.states]
    np.testing.assert_allclose(coherences, 0.5 * np.exp(1j * phase), atol=2e-6)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("start", [0.0, 6.0])
def test_pulse_touching_interval_has_no_evolution(backend, start):
    chip, drive, sequence = _sequence(backend)
    sequence.schedule(drive, envelope=Square(duration=2.0, amplitude=0.4), start_time=start)
    initial = (chip.backend.basis(2, 0) + chip.backend.basis(2, 1)) / np.sqrt(2)
    result = sequence.simulate([2.0, 6.0], initial_state=initial, partition=False, options=_options(backend))
    coherence = chip.backend.to_array(chip.backend.state_to_dm(result.final_state))[0, 1]
    np.testing.assert_allclose(coherence, 0.5 * np.exp(2j * np.pi * 0.2 * 4), atol=2e-6)


@pytest.mark.parametrize("entrypoint", ["build_problem", "build_batch", "simulate", "simulate_batch"])
def test_empty_schedule_requires_an_interval(entrypoint):
    _, _, sequence = _sequence()
    with pytest.raises(ValueError, match="duration.*tlist"):
        getattr(sequence, entrypoint)()


@pytest.mark.parametrize("duration", [0.0, -1.0, np.inf, np.nan])
def test_invalid_duration_rejected(duration):
    _, _, sequence = _sequence()
    with pytest.raises(ValueError, match="finite.*positive"):
        sequence.build_problem(duration=duration)


def test_duration_is_exclusive_and_cannot_cut_schedule():
    _, drive, sequence = _sequence()
    sequence.schedule(drive, envelope=Square(duration=2.0, amplitude=0.1))
    with pytest.raises(ValueError, match="duration.*tlist"):
        sequence.build_problem([0.0, 1.0], duration=3.0)
    with pytest.raises(ValueError, match="schedule.*2"):
        sequence.build_problem(duration=1.0)
    problem = sequence.build_problem(duration=3.0)
    assert problem.tlist[0] == 0.0
    assert problem.tlist[-1] == 3.0


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_unscheduled_duration_includes_idle_dynamics(backend):
    chip, _, sequence = _sequence(backend)
    initial = (chip.backend.basis(2, 0) + chip.backend.basis(2, 1)) / np.sqrt(2)
    result = sequence.simulate(duration=1.25, initial_state=initial)
    coherence = chip.backend.to_array(chip.backend.state_to_dm(result.final_state))[0, 1]
    np.testing.assert_allclose(coherence, 0.5j, atol=2e-6)
    assert result.times[0] == 0.0
    assert result.times[-1] == 1.25


def test_duration_sweep_preserves_each_interval():
    _, drive, sequence = _sequence()
    pulse = sequence.schedule(drive, envelope=Square(duration=2.0, amplitude=0.1))
    axis = pulse.vary("duration", [1.0, 3.0])
    batch = sequence.build_batch(axis)
    assert [float(point.tlist[-1]) for point in batch.problems] == [1.0, 3.0]
    extended = sequence.build_batch(axis, duration=4.0)
    assert [float(point.tlist[-1]) for point in extended.problems] == [4.0, 4.0]
    with pytest.raises(ValueError, match="schedule.*3"):
        sequence.build_batch(axis, duration=2.0)
    partial = sequence.build_batch(axis, tlist=[1.5, 2.5])
    for point in partial.problems:
        np.testing.assert_array_equal(point.tlist, [1.5, 2.5])


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_duration_batch_solves_each_actual_interval(backend):
    chip, drive, sequence = _sequence(backend)
    pulse = sequence.schedule(drive, envelope=Square(duration=2.0, amplitude=0.1))
    initial = (chip.backend.basis(2, 0) + chip.backend.basis(2, 1)) / np.sqrt(2)
    durations = [1.0, 3.0]
    result = sequence.simulate_batch(pulse.vary("duration", durations), initial_state=initial,
                                    states="final", progress=False,
                                    options=_options(backend))
    for point, duration in zip(result, durations):
        assert point.times[0] == 0.0
        assert point.times[-1] == duration
        coherence = chip.backend.to_array(chip.backend.state_to_dm(point.final_state))[0, 1]
        np.testing.assert_allclose(coherence, 0.5 * np.exp(2j * np.pi * 0.3 * duration), atol=2e-6)


def test_batch_duration_validates_requested_points_not_unused_base_schedule():
    _, drive, sequence = _sequence()
    pulse = sequence.schedule(drive, envelope=Square(duration=2.0, amplitude=0.1))
    batch = sequence.build_batch(pulse.vary("duration", [1.0]), duration=1.5)
    assert batch.problems[0].tlist[-1] == 1.5


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_backend_option_cannot_override_the_initial_time(backend):
    _, _, sequence = _sequence(backend)
    with pytest.raises(ValueError, match=r"initial-state time is tlist\[0\]"):
        sequence.build_problem([2.0, 4.0], options={"t0": 0.0})
