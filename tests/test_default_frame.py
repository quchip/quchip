"""A chip without a frame: lab-frame statics, rotating-frame solves, and the warnings that name the choice."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from quchip import (
    Bath,
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Gaussian,
    PortNetwork,
    QuantumSequence,
    Resonator,
    Square,
    eliminate,
)
from quchip.engine.frames import IncompatibleFrameError
from quchip.engine.problem import ROTATING_DEFAULT_WARNING

TIMES = np.linspace(0.0, 40.0, 5)
AUTO_FALLBACK = re.escape(
    "The chip sets no frame, and the rotating frame does not fit it, so this solve uses the frame that frame='auto' "
    "plans, with "
)
LAB_FALLBACK = re.escape(
    "The chip sets no frame, and neither the rotating frame nor the frame that frame='auto' plans fits it, so this "
    "solve uses the lab frame."
)
UNRELIABLE = re.escape(
    "The chip sets no frame, and the rotating frame of a solve needs the dressed 0 -> 1 frequency of each device, but "
    "the dressed label assignment is unreliable for 'a' and 'b' ('a': overlap="
)


def _pair(frame=None, bath=False, coupled=True):
    """Return a driven transmon and resonator, optionally with a collective bath that cannot rotate."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=2e4)
    r = Resonator(freq=7.0, levels=3, label="r")
    baths = [Bath("collective_decay", targets=[q, r], rate=1e-3, label="loss")] if bath else None
    couplings = [Capacitive(q, r, g=0.05, label="qr")] if coupled else []
    chip = Chip([q, r], couplings, frame=frame, baths=baths)
    chip.wire(ChargeDrive(q, label="xy"))
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Gaussian(duration=20.0, amplitude=0.02), freq=float(chip.freq(q)))
    return chip, sequence


def _reduced_bus_pair(frame=None):
    """Return two lossy transmons after the elimination of their bus, which carries each jump onto both."""
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, T1=6e4, label="q1")
    q2 = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, T1=6e4, label="q2")
    bus = Resonator(freq=6.4, levels=3, label="b")
    couplings = [Capacitive(q1, bus, g=0.07, label="q1_b"), Capacitive(q2, bus, g=0.07, label="q2_b")]
    reduced = eliminate(Chip([q1, q2, bus], couplings, frame=frame), "b").chip
    return reduced, QuantumSequence(reduced)


def _hanger(frame=None):
    """Return two readout resonators side-coupled to one hanger feedline, driven through its input."""
    resonators = [Resonator(freq=freq, levels=3, label=f"r{index}") for index, freq in enumerate((7.0, 7.2), 1)]
    line = PortNetwork(label="feedline")
    rate = 2 * np.pi * 1e-3
    right = [line.port(f"{mode.label}_right", target=mode, rate=rate / 2) for mode in resonators]
    left = [line.port(f"{mode.label}_left", target=mode, rate=rate / 2) for mode in resonators]
    line.cascade(*right)
    line.cascade(*reversed(left))
    feed = line.expose("in", input=right[0], output=left[0])
    line.expose("out", input=left[-1], output=right[-1])
    chip = Chip(resonators, port_network=line, frame=frame)
    sequence = QuantumSequence(chip)
    sequence.schedule(feed.input, envelope=Square(duration=30.0, amplitude=0.01), freq=7.0)
    return chip, sequence


def _resonant_pair(reference_freq=None):
    """Return two identical coupled transmons, whose dressed 0 -> 1 states split evenly over both."""
    devices = [DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label=label) for label in ("a", "b")]
    for device in devices:
        device.reference_freq = reference_freq
    return Chip(devices, [Capacitive(*devices, g=0.01, label="ab")])


@pytest.mark.unit
def test_a_chip_without_a_frame_keeps_lab_frame_statics() -> None:
    """Statics resolve in the lab frame, and frame_info() reports the rotating frame of later solves and warns so."""
    chip, _ = _pair()
    assert chip.frame is None
    assert chip.resolve().resolved_frame.mode == "lab"
    with pytest.warns(UserWarning) as record:
        assert chip.frame_info() == pytest.approx({"q": chip.freq("q"), "r": chip.freq("r")})
    assert [str(warning.message) for warning in record] == [ROTATING_DEFAULT_WARNING.replace("this solve", "a solve")]
    assert record[0].filename == __file__
    assert "Frame    : None (lab for statics, rotating for solves)" in chip.describe()
    chip.set_frame("lab")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert chip.frame == "lab" and chip.frame_info() == {"q": 0.0, "r": 0.0}
    chip.set_frame(None)
    assert chip.frame is None


@pytest.mark.unit
def test_a_solve_without_a_frame_uses_the_rotating_frame_and_warns_once_per_line() -> None:
    """The warning names the rotating frame at the user's line, and an explicit frame stays silent."""
    chip, sequence = _pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(2):
            problem = sequence.build_problem(TIMES)
    assert problem.resolved_frame.mode == "rotating"
    assert problem.resolved_frame.frequencies == pytest.approx({"q": chip.freq("q"), "r": chip.freq("r")})
    assert [str(warning.message) for warning in caught] == [ROTATING_DEFAULT_WARNING]
    assert caught[0].filename == __file__
    assert chip.frame is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for frame in ("rotating", "lab"):
            assert sequence.build_problem(TIMES, frame=frame).resolved_frame.mode == frame
        chip.set_frame("lab")
        assert sequence.build_problem(TIMES).resolved_frame.mode == "lab"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("build", "source", "devices"),
    [
        (lambda frame=None: _pair(frame, bath=True), "'collective_decay' has operator bands", ("q", "r")),
        (_reduced_bus_pair, "Effective channel 'q1'/'thermal_emission'", ("q1", "q2")),
        (_hanger, "PortNetwork cannot statically compose exposure 'in'", ("r1", "r2")),
    ],
    ids=["collective-bath", "reduced-bus-pair", "hanger-feedline"],
)
def test_a_solve_without_a_frame_uses_the_auto_frame_when_an_operator_cannot_rotate(build, source, devices) -> None:
    """Operators that need one phase on several devices take the planned frame, and an explicit rotating one raises."""
    chip, sequence = build()
    assert chip.resolve().resolved_frame.mode == "lab"
    with pytest.warns(UserWarning, match=AUTO_FALLBACK) as record:
        problem = sequence.build_problem(TIMES)
    frequencies = problem.resolved_frame.frequencies
    assert problem.resolved_frame.mode == "auto" and frequencies[devices[0]] == frequencies[devices[1]]
    assert len(record) == 1 and source in str(record[0].message)
    # frame_info() reports the same frame and reason for a solve without pulses.
    with pytest.warns(UserWarning, match=AUTO_FALLBACK.replace("this", "a")) as info:
        assert chip.frame_info() == pytest.approx(dict(frequencies))
    assert len(info) == 1 and source in str(info[0].message)
    assert f"plans, with {devices[0]!r} and {devices[1]!r} at {float(frequencies[devices[0]]):.6g} GHz. " in str(
        record[0].message
    )
    with pytest.raises(IncompatibleFrameError, match=re.escape(source)):
        build(frame="rotating")[1].build_problem(TIMES)


@pytest.mark.unit
def test_a_solve_without_a_frame_uses_the_lab_frame_when_no_planned_frame_fits() -> None:
    """A collective bath on uncoupled detuned devices fits neither the rotating nor the planned frame."""
    chip, sequence = _pair(bath=True, coupled=False)
    with pytest.warns(UserWarning, match=LAB_FALLBACK) as record:
        problem = sequence.build_problem(TIMES)
    assert problem.resolved_frame.mode == "lab"
    assert len(record) == 1 and "'collective_decay' has operator bands with different phases" in str(record[0].message)
    with pytest.warns(UserWarning, match=LAB_FALLBACK.replace("this", "a")) as info:
        assert chip.frame_info() == {"q": 0.0, "r": 0.0}
    assert "'collective_decay' has operator bands with different phases" in str(info[0].message)
    with pytest.raises(IncompatibleFrameError, match="has operator bands with different phases"):
        sequence.build_problem(TIMES, frame="auto")


@pytest.mark.unit
def test_a_solve_without_dissipation_keeps_the_rotating_frame() -> None:
    """Channels that a solve leaves out need not fit its frame, in a single solve and in both batch paths."""
    _, sequence = _pair(bath=True)
    axes = [sequence.vary("q.freq", [5.0, 5.1]), sequence.vary("pulse.0.amplitude", [0.01, 0.02])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problems = [sequence.build_problem(TIMES, dissipation=False)]
        for axis in axes:
            problems += sequence.build_batch(axis, tlist=TIMES, dissipation=False).problems
        problems.append(_reduced_bus_pair()[1].build_problem(TIMES, dissipation=False))
    assert [problem.resolved_frame.mode for problem in problems] == ["rotating"] * 6
    assert [str(warning.message) for warning in caught] == [ROTATING_DEFAULT_WARNING] * 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sequence.build_problem(TIMES, frame="rotating", dissipation=False).resolved_frame.mode == "rotating"
        for build in (lambda frame: _pair(frame, bath=True), _reduced_bus_pair):
            assert build(frame="rotating")[1].build_problem(TIMES, dissipation=False).resolved_frame.mode == "rotating"
    # A dissipative solve on the same chip still checks its channels.
    with pytest.warns(UserWarning, match=AUTO_FALLBACK):
        assert sequence.build_problem(TIMES).resolved_frame.mode == "auto"


@pytest.mark.e2e
def test_subspace_propagator_keeps_the_rotating_frame_on_a_chip_with_a_collective_bath() -> None:
    """The propagator leaves the channels out, so a collective bath changes neither its frame nor its matrix."""
    propagators = []
    for bath in (True, False):
        _, sequence = _pair(bath=bath)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            propagators.append(sequence.subspace_propagator("q", duration=20.0))
        messages = [str(warning.message) for warning in caught]
        assert len(messages) == 2 and messages[0] == ROTATING_DEFAULT_WARNING
        assert messages[1].startswith("subspace_propagator() solves without the collapse channels")
    with_bath, without_bath = propagators
    assert [result.frame.mode for result in with_bath.batch] == ["rotating", "rotating"]
    assert "hidden.loss.collective_decay" in with_bath.omitted_channels
    np.testing.assert_allclose(with_bath.matrix, without_bath.matrix, rtol=0, atol=1e-12)


@pytest.mark.unit
def test_a_solve_without_a_frame_raises_when_a_reference_has_no_reliable_assignment() -> None:
    """A resonant pair has no dressed 0 -> 1 reference per device, unless each device sets reference_freq."""
    chip = _resonant_pair()
    for call in (QuantumSequence(chip).build_problem, lambda _: chip.frame_info()):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=UNRELIABLE) as error:
                call(TIMES)
        assert not isinstance(error.value, IncompatibleFrameError)
        assert "margin=0). Choose a frame with Chip(frame='lab') or chip.set_frame('lab')." in str(error.value)
        assert "frame='auto' may also work: it gives exchange-coupled devices one frequency." in str(error.value)
    with pytest.warns(UserWarning, match="so this solve uses the rotating frame"):
        problem = QuantumSequence(_resonant_pair(reference_freq=5.0)).build_problem(TIMES)
    assert problem.resolved_frame.frequencies == {"a": 5.0, "b": 5.0}


@pytest.mark.unit
def test_an_explicit_rotating_frame_warns_when_a_reference_has_no_reliable_assignment() -> None:
    """frame='rotating' keeps its frame on a resonant pair, warns with the same check, and names frame='auto'."""
    chip = _resonant_pair()
    chip.set_frame("rotating")
    with pytest.warns(UserWarning) as record:
        problem = QuantumSequence(chip).build_problem(TIMES)
    messages = [str(warning.message) for warning in record]
    assert len(messages) == 1 and record[0].filename == __file__
    assert "the dressed label assignment is unreliable for 'a' and 'b' ('a': overlap=" in messages[0]
    assert "frame='auto' gives exchange-coupled devices one frequency." in messages[0]
    assert problem.resolved_frame.mode == "rotating"
    for reliable in (_resonant_pair(reference_freq=5.0), _pair(frame="rotating")[0]):
        reliable.set_frame("rotating")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            QuantumSequence(reliable).build_problem(TIMES)


@pytest.mark.unit
def test_a_batch_without_a_frame_checks_every_point() -> None:
    """One point that needs the lab frame moves the whole batch to it, and one unreliable point raises."""
    _, sequence = _pair(bath=True, coupled=False)
    # A zero rate leaves no collective jump, so the first point alone fits the rotating frame.
    with pytest.warns(UserWarning, match=LAB_FALLBACK) as record:
        batch = sequence.build_batch(sequence.vary("bath.loss.rate", [0.0, 1e-3]), tlist=TIMES)
    assert [problem.resolved_frame.mode for problem in batch.problems] == ["lab", "lab"]
    assert len(record) == 1 and "'loss'/'collective_decay' has operator bands" in str(record[0].message)
    sequence = QuantumSequence(_resonant_pair())
    with pytest.raises(ValueError, match=UNRELIABLE):
        sequence.build_batch(sequence.vary("b.freq", [5.2, 5.0]), tlist=TIMES)


@pytest.mark.unit
@pytest.mark.parametrize("bath", [False, True])
def test_batch_points_share_one_frame_and_warn_once(bath: bool) -> None:
    """A parameter axis builds every point in one frame mode, warns once, and resolves each point's references."""
    chip, sequence = _pair(bath=bath)
    with pytest.warns(UserWarning, match="The chip sets no frame") as record:
        batch = sequence.build_batch(sequence.vary("q.freq", [5.0, 5.1]), tlist=TIMES)
    assert len(record) == 1
    frames = [problem.resolved_frame for problem in batch.problems]
    if bath:
        assert [frame.mode for frame in frames] == ["auto", "auto"]
    else:
        assert [frame.mode for frame in frames] == ["rotating", "rotating"]
        moved = chip.with_params({"q.freq": 5.1})
        assert frames[1].frequencies == pytest.approx({"q": moved.freq("q"), "r": moved.freq("r")})


@pytest.mark.unit
def test_a_carrier_axis_in_the_planned_frame_plans_each_point() -> None:
    """Each point of a carrier sweep gets its own plan, as on a chip with frame="auto"."""
    carriers = [6.9, 7.0, 7.15]
    _, sequence = _hanger()
    with pytest.warns(UserWarning, match=AUTO_FALLBACK + "frequencies that differ between the batch points") as record:
        batch = sequence.build_batch(sequence.vary("pulse.0.freq", carriers), tlist=TIMES)
    assert len(record) == 1
    _, planned = _hanger("auto")
    reference = planned.build_batch(planned.vary("pulse.0.freq", carriers), tlist=TIMES)
    frames = [problem.resolved_frame.frequencies for problem in batch.problems]
    assert frames == [problem.resolved_frame.frequencies for problem in reference.problems]
    assert frames[0] == pytest.approx({"r1": 6.9, "r2": 6.9})


def _strongly_driven_pair():
    """Return exchange-coupled modes with a collective decay, under strong tones that untie their planned frame."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=7.0, levels=2, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.05)], baths=[Bath("collective_decay", targets=[q, r], rate=1e-3)])
    chip.wire(ChargeDrive(q, label="dq"), ChargeDrive(r, label="dr"))
    sequence = QuantumSequence(chip)
    sequence.schedule("dq", envelope=Square(duration=20.0, amplitude=1.0), freq=5.0)
    sequence.schedule("dr", envelope=Square(duration=20.0, amplitude=1.0), freq=7.0)
    return chip, sequence


@pytest.mark.unit
def test_pulses_can_move_a_solve_from_the_planned_frame_to_the_lab_frame() -> None:
    """frame_info() reports the solve without pulses, and a batch checks its own points, not the sequence as written."""
    chip, sequence = _strongly_driven_pair()
    with pytest.warns(UserWarning, match=AUTO_FALLBACK.replace("this", "a")):
        assert chip.frame_info() == pytest.approx({"q": 4.99875078, "r": 4.99875078})
    with pytest.warns(UserWarning, match=LAB_FALLBACK):
        assert sequence.build_problem(TIMES).resolved_frame.mode == "lab"
    amplitudes = [sequence.vary(f"pulse.{index}.amplitude", [1e-4, 2e-4]) for index in (0, 1)]
    with pytest.warns(UserWarning, match=AUTO_FALLBACK) as record:
        batch = sequence.build_batch(*amplitudes, tlist=TIMES)
    assert len(record) == 1
    assert [problem.resolved_frame.mode for problem in batch.problems] == ["auto"] * 4
    with pytest.warns(UserWarning, match=AUTO_FALLBACK):
        single = sequence.with_params({"pulse.0.amplitude": 1e-4, "pulse.1.amplitude": 1e-4}).build_problem(TIMES)
    assert batch.problems[0].resolved_frame.frequencies == single.resolved_frame.frequencies


@pytest.mark.e2e
def test_a_point_solved_in_the_batch_frame_starts_in_the_state_of_its_own_solve() -> None:
    """A ket at a nonzero start time is a vector of the frame of the point's own solve, also in a lab-frame batch."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=7.0, levels=2, label="r")
    chip = Chip([q, r], baths=[Bath("collective_decay", targets=[q, r], rate=1e-3, label="loss")])
    sequence = QuantumSequence(chip)
    ket = np.array([1, 0, 1, 0], dtype=complex).reshape(-1, 1) / np.sqrt(2)
    times = np.array([0.05, 0.15])
    # At a zero rate the bath has no jump, so that point alone fits the rotating frame.
    with pytest.warns(UserWarning, match=LAB_FALLBACK):
        batch = sequence.simulate_batch(sequence.vary("bath.loss.rate", [0.0, 1e-3]), tlist=times, initial_state=ket)
    with pytest.warns(UserWarning, match="so this solve uses the rotating frame"):
        single = sequence.with_params({"bath.loss.rate": 0.0}).simulate(times, initial_state=ket)
    assert (batch[0].frame.mode, single.frame.mode) == ("lab", "rotating")
    for time in times:
        np.testing.assert_allclose(batch[0].state(time, dm=True, frame="lab").full(),
                                   single.state(time, dm=True, frame="lab").full(), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("build", "frame"),
    [
        (_pair, "rotating"),
        (lambda frame=None: _pair(frame, bath=True), "auto"),
        (lambda frame=None: _pair(frame, bath=True, coupled=False), "lab"),
        (_reduced_bus_pair, "auto"),
        (_hanger, "auto"),
    ],
    ids=["rotating", "collective-bath", "uncoupled-bath", "reduced-bus-pair", "hanger-feedline"],
)
def test_a_solve_without_a_frame_equals_the_explicit_frame_it_chose(build, frame: str) -> None:
    """The default solve reproduces the explicit-frame solve, and lab-frame states follow from its states."""
    results = []
    for chip, sequence in (build(), build(frame=frame)):
        label = chip.devices[0].label
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The chip sets no frame")
            results.append(sequence.simulate(tlist=TIMES, initial_state=chip.state({label: 1})))
    default, explicit = results
    assert default.frame.mode == frame
    np.testing.assert_array_equal(np.asarray(default.population(label, 1)), np.asarray(explicit.population(label, 1)))
    assert float(default.population(label, 1)[-1]) > 0.5
    np.testing.assert_allclose(
        np.asarray(default.state(frame="lab").full()), np.asarray(explicit.state(frame="lab").full()), atol=1e-12
    )


def test_the_frame_that_the_error_names_solves_a_resonant_pair() -> None:
    """frame='auto' turns both identical transmons at one frequency, so their idle exchange stays static."""
    chip = _resonant_pair()
    chip.set_frame("auto")
    times = np.linspace(0.0, 100.0, 21)
    start = chip.compact_state({"a": 1}, dressed=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = QuantumSequence(chip).simulate(tlist=times, initial_state=start)
    assert result.frame.mode == "auto"
    # Under RWA, one excitation swaps between the transmons as cos²(2πgt).
    exchange = np.cos(2 * np.pi * 0.01 * times) ** 2
    np.testing.assert_allclose(np.asarray(result.population("a", 1)).real, exchange, rtol=0, atol=1e-10)


def test_a_partitioned_solve_without_a_frame_warns_once_and_rotates_every_component() -> None:
    """Components of a frameless chip solve in the frame of the joint choice, with one warning."""
    devices = [
        DuffingTransmon(freq=freq, anharmonicity=-0.25, levels=3, label=f"q{index}")
        for index, freq in enumerate((5.0, 5.3))
    ]
    chip = Chip(devices)
    chip.wire(*(ChargeDrive(device, label=f"d{index}") for index, device in enumerate(devices)))
    sequence = QuantumSequence(chip)
    for index, device in enumerate(devices):
        sequence.schedule(f"d{index}", envelope=Gaussian(duration=20.0, amplitude=0.02), freq=float(chip.freq(device)))
    with pytest.warns(UserWarning, match="The chip sets no frame") as record:
        result = sequence.simulate(tlist=TIMES)
    assert len(record) == 1 and record[0].filename == __file__
    assert [component.frame.mode for component in result.components] == ["rotating", "rotating"]
    chip.set_frame("rotating")
    explicit = sequence.simulate(tlist=TIMES)
    for label in ("q0", "q1"):
        np.testing.assert_array_equal(
            np.asarray(result.population(label, 1)), np.asarray(explicit.population(label, 1))
        )
