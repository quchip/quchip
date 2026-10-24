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


@pytest.mark.parametrize("start", [0.0, 6.0])
def test_pulse_touching_interval_has_no_evolution(start):
    chip, drive, sequence = _sequence("qutip")
    sequence.schedule(drive, envelope=Square(duration=2.0, amplitude=0.4), start_time=start)
    initial = (chip.backend.basis(2, 0) + chip.backend.basis(2, 1)) / np.sqrt(2)
    result = sequence.simulate([2.0, 6.0], initial_state=initial, partition=False, options=_options("qutip"))
    coherence = chip.backend.to_array(chip.backend.state_to_dm(result.final_state))[0, 1]
    np.testing.assert_allclose(coherence, 0.5 * np.exp(2j * np.pi * 0.2 * 4), atol=2e-6)


@pytest.mark.parametrize("duration", [0.0, np.inf, np.nan])
def test_invalid_duration_rejected(duration):
    _, _, sequence = _sequence()
    with pytest.raises(ValueError, match="finite.*positive"):
        sequence.build_problem(duration=duration)


def test_unscheduled_duration_includes_idle_dynamics():
    chip, _, sequence = _sequence("qutip")
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
def test_backend_option_cannot_override_the_initial_time(backend):
    _, _, sequence = _sequence(backend)
    with pytest.raises(ValueError, match=r"initial-state time is tlist\[0\]"):
        sequence.build_problem([2.0, 4.0], options={"t0": 0.0})
