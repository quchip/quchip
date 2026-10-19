"""frame="auto" plans one rotating frame per tone cluster and reports what stays oscillating."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quchip import Chip, Coupling, PortNetwork, QuantumSequence, Resonator
from quchip.chip.couplings import Capacitive
from quchip.control.drive import ChargeDrive
from quchip.control.drives_two_photon import TwoPhotonDrive
from quchip.control.envelopes import Gaussian, Square
from quchip.control.equipment import ControlEquipment
from quchip.control.field import CoherentInput
from quchip.control.signal import Crosstalk, Gain
from quchip.devices.kerr_cavity import KerrCavity
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.approximations import Exact, RWA
from quchip.engine.frames import FramePlan
from quchip.engine.input_output import resolve_stationary_engine


def _pair(frame: Any = "auto", backend: str | None = None) -> tuple[Chip, ChargeDrive, ChargeDrive]:
    first = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
    first.reference_freq = 5.0
    second = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, label="q2")
    second.reference_freq = 5.4
    drive_1, drive_2 = ChargeDrive(target=first), ChargeDrive(target=second)
    chip = Chip([first, second], couplings=[Capacitive(first, second, g=0.01)], frame=frame, backend=backend)
    chip.connect(ControlEquipment(lines=[drive_1, drive_2]))
    return chip, drive_1, drive_2


def test_auto_frame_pins_driven_device_and_shares_its_exchange_cluster() -> None:
    """One tone frames the driven qubit and every device it exchanges excitations with."""
    chip, drive_1, _ = _pair()
    sequence = QuantumSequence(chip)
    sequence.schedule(drive_1, envelope=Square(duration=40.0, amplitude=0.01), freq=5.1)

    engine = sequence.resolve()
    plan = engine.resolved_frame.plan

    assert isinstance(plan, FramePlan)
    assert engine.resolved_frame.mode == "auto"
    assert plan.frequencies == {"q1": 5.1, "q2": 5.1}
    assert plan.clusters == (("q1", "q2"),)
    assert plan.residuals == ()
    assert {term.origin for term in engine.dynamic_terms} == {"drive"}


def test_auto_frame_pins_undriven_cluster_to_a_reference() -> None:
    """Without tones, exchange-coupled modes rotate together at one member's reference frequency."""
    chip, _, _ = _pair()
    engine = chip.resolve()
    plan = engine.resolved_frame.plan

    assert plan is not None
    assert plan.frequencies == {"q1": 5.0, "q2": 5.0}
    assert plan.pins == (("q1", 5.0),)
    assert not engine.dynamic_terms


def test_auto_frame_keeps_isolated_modes_at_their_own_reference() -> None:
    """Uncoupled devices keep their reference frames, as the rotating frame already does."""
    first = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="a")
    first.reference_freq = 5.0
    second = Resonator(freq=7.0, levels=3, label="b")
    second.reference_freq = 7.0
    plan = Chip([first, second], frame="auto").resolve().resolved_frame.plan
    assert plan is not None
    assert plan.frequencies == {"a": 5.0, "b": 7.0}
    assert plan.clusters == (("a",), ("b",))


@pytest.mark.parametrize("approximation", [RWA(), Exact()])
def test_auto_frame_makes_two_photon_drive_static_at_half_its_frequency(approximation: Any) -> None:
    """A two-photon tone at f pins its cavity to f/2 so a^2 + a^dag^2 stops oscillating."""
    cavity = KerrCavity(freq=5.0, kerr=0.02, levels=8, label="cav")
    cavity.reference_freq = 5.0
    pump = TwoPhotonDrive(target=cavity)
    chip = Chip([cavity], frame="auto", approximation=approximation)
    chip.connect(ControlEquipment(lines=[pump]))
    sequence = QuantumSequence(chip)
    sequence.schedule(pump, envelope=Square(duration=30.0, amplitude=0.05), freq=10.2)

    engine = sequence.resolve()
    assert engine.resolved_frame.plan is not None
    assert engine.resolved_frame.plan.frequencies == {"cav": 5.1}
    assert {term.origin for term in engine.dynamic_terms} == {"drive"}


def test_auto_frame_conflict_keeps_the_costlier_tone_and_reports_the_residual() -> None:
    """Two tones on one cluster: the frame follows the larger integrated drive, the other keeps oscillating."""

    def plan_for(strong_first: bool) -> FramePlan:
        chip, drive_1, drive_2 = _pair()
        sequence = QuantumSequence(chip)
        big, small = (0.05, 0.005) if strong_first else (0.005, 0.05)
        sequence.schedule(drive_1, envelope=Square(duration=40.0, amplitude=big), freq=5.0)
        sequence.schedule(drive_2, envelope=Square(duration=40.0, amplitude=small), freq=5.4)
        plan = sequence.resolve().resolved_frame.plan
        assert plan is not None
        return plan

    plan = plan_for(True)
    assert plan.frequencies == {"q1": 5.0, "q2": 5.0}
    assert len(plan.residuals) == 1
    residual = plan.residuals[0]
    assert residual.devices == ("q2",)
    assert residual.frequency == pytest.approx(-0.4)
    assert residual.weight is not None and residual.weight > 0

    flipped = plan_for(False)
    assert flipped.frequencies == {"q1": 5.4, "q2": 5.4}
    assert flipped.residuals[0].devices == ("q1",)


def test_auto_frame_prefers_the_set_with_the_largest_total_weight() -> None:
    """Two resonant drives outweigh a weak exchange coupling, which is left rotating at the detuning."""
    chip, drive_1, drive_2 = _pair()
    sequence = QuantumSequence(chip)
    sequence.schedule(drive_1, envelope=Square(duration=40.0, amplitude=0.03), freq=5.0)
    sequence.schedule(drive_2, envelope=Square(duration=40.0, amplitude=0.025), freq=5.4)
    plan = sequence.resolve().resolved_frame.plan
    assert plan is not None
    assert plan.frequencies == {"q1": 5.0, "q2": 5.4}
    assert plan.clusters == (("q1",), ("q2",))
    (residual,) = plan.residuals
    assert residual.devices == ("q1", "q2") and residual.frequency == pytest.approx(-0.4)


def test_strict_planning_keeps_the_stationary_route_errors() -> None:
    """Port tones that disagree across an exchange cluster still fail with the stationary message."""
    first = Resonator(freq=5.0, levels=3, label="a")
    first.reference_freq = 5.0
    second = Resonator(freq=5.4, levels=3, label="b")
    second.reference_freq = 5.4
    network = PortNetwork(label="line")
    network.port("pa", target=first, rate=0.02)
    network.port("pb", target=second, rate=0.02)
    chip = Chip([first, second], couplings=[Capacitive(first, second, g=0.01)], port_network=network)

    engine = resolve_stationary_engine(chip, (("pa", 5.2), ("pb", 5.2)))
    assert engine.resolved_frame.frequencies == {"a": 5.2, "b": 5.2}
    with pytest.raises(ValueError, match="Exchange-connected devices"):
        resolve_stationary_engine(chip, (("pa", 5.0), ("pb", 5.4)))
    with pytest.raises(ValueError, match="Ports address"):
        resolve_stationary_engine(chip, (("pa", 5.0), ("pa", 5.4)))


@pytest.mark.parametrize("approximation", [RWA(), Exact()])
def test_auto_frame_traced_tone_frequency_stays_traced(approximation: Any) -> None:
    """A jitted tone frequency flows into the planned frame without concretization."""
    jax = pytest.importorskip("jax")
    chip, drive_1, _ = _pair(backend="dynamiqs")

    def frame_of(freq: Any) -> Any:
        sequence = QuantumSequence(chip)
        sequence.schedule(drive_1, envelope=Square(duration=40.0, amplitude=0.01), freq=freq)
        plan = sequence.resolve(approximation=approximation).resolved_frame.plan
        assert plan is not None
        return plan.frequencies["q2"]

    assert float(jax.jit(frame_of)(jax.numpy.asarray(5.1))) == pytest.approx(5.1)
    assert float(jax.grad(frame_of)(jax.numpy.asarray(5.1))) == pytest.approx(1.0)


def test_describe_reports_the_planned_frame() -> None:
    """describe() shows the chosen frame per device and any residual oscillation."""
    chip, drive_1, drive_2 = _pair()
    description = chip.describe()
    assert "Frame    : auto" in description
    assert "q1       : 5 GHz" in description.split("Approx.")[0]
    sequence = QuantumSequence(chip)
    sequence.schedule(drive_1, envelope=Square(duration=40.0, amplitude=0.05), freq=5.0)
    sequence.schedule(drive_2, envelope=Square(duration=40.0, amplitude=0.005), freq=5.4)
    frame_block = sequence.describe().split("\nFrame\n")[1]
    assert "q1       : 5 GHz" in frame_block and "q2       : 5 GHz" in frame_block
    assert "tone     : charge_0 → q1 on q1 at 5 GHz" in frame_block
    assert "residual" in frame_block and "q2" in frame_block and "-0.4 GHz" in frame_block


@pytest.mark.validation
@pytest.mark.parametrize("workflow", ["rabi", "coupled_pair", "dispersive_readout", "two_photon"])
def test_auto_frame_reproduces_current_default_observables(workflow: str) -> None:
    """Flip readiness: the planned frame gives the same demodulated observables as the explicit frame."""
    populations: dict[str, np.ndarray] = {}
    for label, frame in (("explicit", None), ("auto", "auto")):
        if workflow == "coupled_pair":
            chip, drive_1, _ = _pair(frame or {"q1": 5.1, "q2": 5.1})
            sequence = QuantumSequence(chip)
            sequence.schedule(drive_1, envelope=Square(duration=40.0, amplitude=0.01), freq=5.1)
            times = np.linspace(0.0, 40.0, 81)
            result = sequence.simulate(times, e_ops={"q2": chip["q2"].lowering_operator()})
            field = np.asarray(result.expect("q2"))
            populations[label] = np.concatenate([result.population("q1", 1), field.real, field.imag])
        elif workflow == "rabi":
            qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
            qubit.reference_freq = 5.0
            drive = ChargeDrive(target=qubit)
            chip = Chip([qubit], frame=frame or "rotating")
            chip.connect(ControlEquipment(lines=[drive]))
            sequence = QuantumSequence(chip)
            sequence.schedule(drive, envelope=Square(duration=50.0, amplitude=0.02), freq=5.0)
            times = np.linspace(0.0, 50.0, 101)
            result = sequence.simulate(times, e_ops={"q": qubit.lowering_operator()})
            transverse = np.asarray(result.expect("q"))
            populations[label] = np.concatenate([result.population("q", 1), transverse.real, transverse.imag])
        elif workflow == "dispersive_readout":
            qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
            qubit.reference_freq = 5.0
            resonator = Resonator(freq=7.0, levels=4, label="r")
            resonator.reference_freq = 7.0
            network = PortNetwork(label="line")
            port = network.port("readout", target=resonator, rate=0.05)
            network.expose("line", at=port)
            chip = Chip(
                [qubit, resonator],
                couplings=[Capacitive(qubit, resonator, g=0.05)],
                port_network=network,
                frame=frame or {"q": 7.0, "r": 7.0},
            )
            sequence = QuantumSequence(chip)
            sequence.schedule(CoherentInput("line"), envelope=Square(duration=40.0, amplitude=0.02), freq=7.0)
            times = np.linspace(0.0, 40.0, 81)
            result = sequence.simulate(times, e_ops={"r": resonator.lowering_operator()})
            field = np.asarray(result.expect("r"))
            populations[label] = np.concatenate([field.real, field.imag, result.population("q", 1)])
        else:
            cavity = KerrCavity(freq=5.0, kerr=0.02, levels=8, label="cav")
            cavity.reference_freq = 5.0
            pump = TwoPhotonDrive(target=cavity)
            chip = Chip([cavity], frame=frame or {"cav": 5.1})
            chip.connect(ControlEquipment(lines=[pump]))
            sequence = QuantumSequence(chip)
            sequence.schedule(pump, envelope=Square(duration=30.0, amplitude=0.05), freq=10.2)
            times = np.linspace(0.0, 30.0, 61)
            populations[label] = np.asarray(sequence.simulate(times).population("cav", 2))
    np.testing.assert_allclose(populations["auto"], populations["explicit"], atol=1e-8)


def _two_lines(signal_chain: list[Any] | None = None) -> tuple[Chip, ChargeDrive, ChargeDrive]:
    """Two independent transmons, one charge line each, with an optional equipment chain."""
    first = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
    first.reference_freq = 5.0
    second = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, label="q2")
    second.reference_freq = 5.4
    line_1, line_2 = ChargeDrive(target=first, label="line_1"), ChargeDrive(target=second, label="line_2")
    chip = Chip([first, second], frame="auto")
    chip.connect(ControlEquipment(lines=[line_1, line_2], signal_chain=signal_chain))
    return chip, line_1, line_2


def test_crosstalk_victims_receive_the_leaked_tone() -> None:
    """A tone leaking onto q2's line frames q2 at the source frequency; without leakage q2 keeps its reference."""

    def frame_of(signal_chain: list[Any] | None) -> dict[str, Any]:
        chip, line_1, _ = _two_lines(signal_chain)
        sequence = QuantumSequence(chip)
        sequence.schedule(line_1, envelope=Square(duration=40.0, amplitude=0.02), freq=5.0)
        plan = sequence.resolve().resolved_frame.plan
        assert plan is not None
        return dict(plan.frequencies)

    assert frame_of(None) == {"q1": 5.0, "q2": 5.4}
    leaked = frame_of([Crosstalk(source="line_1", victim="line_2", beta=0.5)])
    assert leaked == {"q1": 5.0, "q2": 5.0}


def test_equipment_gain_changes_which_tone_wins() -> None:
    """Two lines on one qubit: a 3x gain on the weaker line makes its tone the heavier one."""

    def frame_of(signal_chain: list[Any] | None) -> Any:
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        qubit.reference_freq = 5.0
        strong, weak = ChargeDrive(target=qubit, label="strong"), ChargeDrive(target=qubit, label="weak")
        chip = Chip([qubit], frame="auto")
        chip.connect(ControlEquipment(lines=[strong, weak], signal_chain=signal_chain))
        sequence = QuantumSequence(chip)
        sequence.schedule(strong, envelope=Square(duration=40.0, amplitude=0.02), freq=5.0)
        sequence.schedule(weak, envelope=Square(duration=40.0, amplitude=0.01), freq=5.2)
        plan = sequence.resolve().resolved_frame.plan
        assert plan is not None
        return plan.frequencies["q"]

    assert frame_of(None) == 5.0
    assert frame_of([Gain(line="weak", factor=3.0)]) == 5.2


def test_sequential_resolutions_do_not_reuse_a_stale_plan() -> None:
    """Two plans with equal frequencies but different residuals are distinct cache entries."""
    chip, d1, d2 = _pair()
    first = QuantumSequence(chip)
    first.schedule(d1, envelope=Square(duration=40.0, amplitude=0.05), freq=5.0)
    second = QuantumSequence(chip)
    second.schedule(d1, envelope=Square(duration=40.0, amplitude=0.05), freq=5.0)
    second.schedule(d2, envelope=Square(duration=40.0, amplitude=0.005), freq=5.4)
    assert first.resolve().resolved_frame.plan.residuals == ()
    assert len(second.resolve().resolved_frame.plan.residuals) == 1
    assert first.resolve().resolved_frame.plan.residuals == ()


def test_resolve_plans_with_the_requested_approximation() -> None:
    """chip.resolve(frame="auto", approximation=Exact()) plans with Exact, not the chip default."""
    chip, _, _ = _pair()
    default = chip.resolve().resolved_frame.plan
    exact = chip.resolve(approximation=Exact()).resolved_frame.plan
    assert default is not None and exact is not None
    assert default.frequencies == {"q1": 5.0, "q2": 5.0}
    assert exact.frequencies == default.frequencies
    assert exact.pins == (("q1", 5.0),)
    (residual,) = exact.residuals
    assert residual.frequency == 10.0 and residual.weight is None
    assert chip.resolve(frame="lab", approximation=Exact()).resolved_frame.frequencies == {"q1": 0.0, "q2": 0.0}


def _exact_pi_pulse(frame: Any = "auto") -> QuantumSequence:
    """Build the coupled-transmon pulse whose old automatic frame changed with the solve window."""
    first = DuffingTransmon(freq=4.912, anharmonicity=-0.214, levels=3, label="q1")
    second = DuffingTransmon(freq=5.268, anharmonicity=-0.206, levels=3, label="q2")
    chip = Chip([first, second], [Capacitive(first, second, g=0.004, label="c12")],
                frame=frame, approximation=Exact())
    line = ChargeDrive(target=first, label="d1")
    chip.connect(ControlEquipment(lines=[line]))
    sequence = QuantumSequence(chip)
    sequence.schedule(line, envelope=Gaussian(duration=24.0, sigmas=2, amplitude=0.0348), freq=4.9119)
    return sequence


@pytest.mark.parametrize("exchange_first", [False, True])
def test_exact_auto_frame_follows_tone_and_exchange_across_solve_windows(
    monkeypatch: pytest.MonkeyPatch, exchange_first: bool,
) -> None:
    """Counter-rotating partners cannot displace tones or exchange bands as their weights grow."""
    if exchange_first:
        from quchip.engine import assembly

        records = assembly.coupling_band_records
        monkeypatch.setattr(assembly, "coupling_band_records",
                            lambda *args: sorted(records(*args), key=lambda record: sum(record.charges) != 0))
    sequence = _exact_pi_pulse()
    for stop in (24.0, 300.0, 500.0, 2000.0):
        problem = sequence.build_problem(np.array([0.0, stop]))
        plan = problem.resolved_frame.plan
        assert plan is not None
        assert plan.frequencies == {"q1": 4.9119, "q2": 4.9119}
        assert len(plan.tones) == 1 and plan.tones[0].source == "d1 → q1"
        assert plan.pins == () and plan.clusters == (("q1", "q2"),)
        (residual,) = plan.residuals
        assert residual.source == "c12"
        assert residual.frequency == pytest.approx(9.8238)
        # Each adjoint band has largest entry 2g; their weights add.
        assert residual.weight == pytest.approx(2 * (2 * 0.004) ** 2 * stop)


@pytest.mark.parametrize("reference", [0.0, 5.0])
def test_non_exchange_couplings_follow_reference_pins(reference: float) -> None:
    """A retained a²b† band stays residual unless independently chosen references make it static."""
    first = Resonator(freq=5.0, levels=3, label="a")
    second = Resonator(freq=5.4, levels=3, label="b")
    first.reference_freq = second.reference_freq = reference

    def conversion(a: Any, b: Any, backend: Any) -> Any:
        lower = a.lowering_operator()
        band = backend.tensor(backend.matmul(lower, lower), backend.dag(b.lowering_operator()))
        return band + backend.dag(band)

    chip = Chip([first, second], [Coupling(first, second, g=0.01, interaction=conversion)], frame="auto",
                approximation=RWA(keep_bands={(2, -1), (-2, 1)}))
    engine = chip.resolve()
    plan = engine.resolved_frame.plan
    assert plan is not None
    assert plan.frequencies == {"a": reference, "b": reference}
    assert plan.pins == (("a", reference), ("b", reference))
    if reference == 0.0:
        assert plan.residuals == () and not engine.dynamic_terms
    else:
        assert len(plan.residuals) == 1 and plan.residuals[0].frequency == reference
        assert engine.dynamic_terms


def test_exact_auto_hamiltonian_is_the_rotated_complete_lab_hamiltonian() -> None:
    """Automatic planning retains every Exact term with the unitary's phases and frame subtraction."""
    sequence = _exact_pi_pulse()
    automatic = sequence.resolve()
    lab = sequence.resolve(frame="lab")
    offsets = 4.9119 * np.add.outer(np.arange(3), np.arange(3)).ravel()
    for time in (0.0, 7.3, 32.0):
        unitary = np.exp(-2j * np.pi * time * offsets)
        matrix = np.asarray(lab.hamiltonian().matrix(t=time))
        expected = unitary.conj()[:, None] * matrix * unitary[None, :] - np.diag(offsets)
        # Both paths are finite matrix algebra in GHz; only roundoff differs.
        np.testing.assert_allclose(automatic.hamiltonian().matrix(t=time), expected, rtol=0, atol=1e-12)


@pytest.mark.validation
def test_exact_auto_population_matches_a_converged_lab_reference() -> None:
    """The automatic frame resolves a long driven Exact solve at least as well as the explicit rotating frame."""
    times = np.linspace(0.0, 500.0, 301)

    def population(frame: str, options: dict[str, float] | None = None) -> float:
        result = _exact_pi_pulse(frame).simulate(times, states="final", options=options)
        return float(abs(np.asarray(result.final_state.full())[3, 0]) ** 2)

    reference = population("lab", {"atol": 1e-14, "rtol": 1e-12})
    refined = population("lab", {"atol": 1e-15, "rtol": 1e-13})
    reference_error = abs(reference - refined)
    # Tightening both lab-frame tolerances resolves a 1e-9 population budget,
    # below 1% of the explicit rotating frame's ~2e-7 default integration error.
    assert reference_error < 1e-9
    auto_error = abs(population("auto") - refined)
    rotating_error = abs(population("rotating") - refined)
    assert auto_error <= rotating_error + 2 * reference_error


def test_build_problem_accepts_a_frame_and_uses_the_solve_window() -> None:
    """build_problem(frame=...) overrides the chip frame; the tlist span sets coupling weights."""
    chip, d1, d2 = _pair(frame="lab")
    sequence = QuantumSequence(chip)
    sequence.schedule(d1, envelope=Square(duration=40.0, amplitude=0.02), freq=5.0)
    sequence.schedule(d2, envelope=Square(duration=40.0, amplitude=0.02), freq=5.4)
    problem = sequence.build_problem(np.linspace(0.0, 40.0, 41), frame="auto")
    assert problem.resolved_frame.plan is not None
    assert problem.resolved_frame.plan.frequencies == {"q1": 5.0, "q2": 5.4}
    longer = sequence.build_problem(np.linspace(0.0, 4000.0, 41), frame="auto")
    assert longer.resolved_frame.plan is not None
    assert longer.resolved_frame.plan.frequencies == {"q1": 5.0, "q2": 5.0}


def test_batches_replan_when_a_pulse_frequency_axis_moves_the_frame() -> None:
    """A pulse-frequency axis gets one frame per point, matching the explicit-frame result at each."""
    chip, d1, _ = _pair()
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(d1, envelope=Square(duration=40.0, amplitude=0.01), freq=5.0)
    axis = pulse.vary("freq", [5.0, 5.2], name="freq")
    times = np.linspace(0.0, 40.0, 41)
    batch = sequence.build_batch(axis, tlist=times)
    frames = [problem.resolved_frame.plan.frequencies for problem in batch.problems]
    assert frames == [{"q1": 5.0, "q2": 5.0}, {"q1": 5.2, "q2": 5.2}]
    swept = sequence.simulate_batch(axis, tlist=times, progress=False)
    for index, freq in enumerate((5.0, 5.2)):
        explicit_chip, e1, _ = _pair({"q1": freq, "q2": freq})
        explicit = QuantumSequence(explicit_chip)
        explicit.schedule(e1, envelope=Square(duration=40.0, amplitude=0.01), freq=freq)
        np.testing.assert_allclose(
            np.asarray(swept[index].population("q1", 1)),
            np.asarray(explicit.simulate(times).population("q1", 1)),
            atol=1e-8,
        )


def test_batches_keep_one_frame_when_the_axis_does_not_move_it() -> None:
    """A phase axis leaves the planned frame alone, so every point shares the reference frame object."""
    chip, d1, _ = _pair()
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(d1, envelope=Square(duration=40.0, amplitude=0.01), freq=5.0)
    axis = pulse.vary("phase", [0.0, 0.5], name="phase")
    batch = sequence.build_batch(axis, tlist=np.linspace(0.0, 40.0, 41))
    first, second = (problem.resolved_frame for problem in batch.problems)
    assert first is second and first.plan is not None
    assert first.plan.frequencies == {"q1": 5.0, "q2": 5.0}


def test_strict_planning_leaves_non_exchange_bands_to_the_dynamic_terms_error() -> None:
    """Under Exact(), a counter-rotating coupling with one tone hits the dynamic-Hamiltonian message."""
    first = Resonator(freq=5.0, levels=3, label="a")
    first.reference_freq = 5.0
    second = Resonator(freq=5.4, levels=3, label="b")
    second.reference_freq = 5.4
    network = PortNetwork(label="line")
    network.port("pa", target=first, rate=0.02)
    chip = Chip(
        [first, second], couplings=[Capacitive(first, second, g=0.01)], port_network=network, approximation=Exact()
    )
    with pytest.raises(ValueError, match="dynamic Hamiltonian terms"):
        resolve_stationary_engine(chip, (("pa", 5.2),))


def test_coherent_tones_weigh_their_scheduled_windows() -> None:
    """Two coherent tones on one port: the longer pulse wins, whichever is scheduled first."""

    def frame_of(first_duration: float, second_duration: float) -> Any:
        resonator = Resonator(freq=7.0, levels=3, label="r")
        resonator.reference_freq = 7.0
        network = PortNetwork(label="line")
        port = network.port("readout", target=resonator, rate=0.05)
        network.expose("line", at=port)
        chip = Chip([resonator], port_network=network, frame="auto")
        sequence = QuantumSequence(chip)
        sequence.schedule(CoherentInput("line"), envelope=Square(duration=first_duration, amplitude=0.02), freq=7.0)
        sequence.schedule(CoherentInput("line"), envelope=Square(duration=second_duration, amplitude=0.02), freq=7.2)
        plan = sequence.build_problem(np.linspace(0.0, 100.0, 11)).resolved_frame.plan
        assert plan is not None
        return plan.frequencies["r"]

    assert frame_of(40.0, 10.0) == 7.0
    assert frame_of(10.0, 40.0) == 7.2


def test_a_traced_contribution_leaves_the_merged_weight_unknown() -> None:
    """A tone merged with a traced-amplitude twin has unknown weight and loses to a known one."""
    jax = pytest.importorskip("jax")
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    qubit.reference_freq = 5.0
    drive = ChargeDrive(target=qubit)
    chip = Chip([qubit], frame="auto", backend="dynamiqs")
    chip.connect(ControlEquipment(lines=[drive]))

    def frame_of(amplitude: Any) -> Any:
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=amplitude), freq=5.0)
        sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=0.05), freq=5.0)
        sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=0.001), freq=5.2)
        plan = sequence.resolve().resolved_frame.plan
        assert plan is not None
        return plan.frequencies["q"]

    assert float(jax.jit(frame_of)(jax.numpy.asarray(0.05))) == pytest.approx(5.2)
    assert frame_of(0.05) == 5.0


def test_planning_treats_nearly_equal_tones_as_distinct() -> None:
    """Frequencies compare exactly, as on the stationary route: a 1e-12 GHz offset is a residual."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    qubit.reference_freq = 5.0
    drive = ChargeDrive(target=qubit)
    chip = Chip([qubit], frame="auto")
    chip.connect(ControlEquipment(lines=[drive]))
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=0.02), freq=5.0)
    sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=0.01), freq=5.0 + 1e-12)
    plan = sequence.resolve().resolved_frame.plan
    assert plan is not None
    assert plan.frequencies["q"] == 5.0
    assert len(plan.residuals) == 1 and plan.residuals[0].frequency == pytest.approx(-1e-12, abs=1e-15)


def test_equal_or_unknown_weights_fall_back_to_declaration_order() -> None:
    """Two equally strong conflicting tones keep the first; so do two traced (unknown-weight) ones."""
    jax = pytest.importorskip("jax")
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    qubit.reference_freq = 5.0
    drive = ChargeDrive(target=qubit)
    chip = Chip([qubit], frame="auto", backend="dynamiqs")
    chip.connect(ControlEquipment(lines=[drive]))

    def frame_of(first: Any, second: Any) -> Any:
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=first), freq=5.2)
        sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=second), freq=5.0)
        plan = sequence.resolve().resolved_frame.plan
        assert plan is not None
        return plan.frequencies["q"]

    assert frame_of(0.02, 0.02) == 5.2
    assert float(jax.jit(frame_of)(jax.numpy.asarray(0.02), jax.numpy.asarray(0.02))) == pytest.approx(5.2)


def test_eigen_basis_plans_the_same_frame_under_jit() -> None:
    """With basis="eigen" the stronger tone declared second still wins inside jax.jit, as it does eagerly."""
    jax = pytest.importorskip("jax")
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    qubit.reference_freq = 5.0
    drive = ChargeDrive(target=qubit)
    chip = Chip([qubit], frame="auto", backend="dynamiqs", basis="eigen")
    chip.connect(ControlEquipment(lines=[drive]))

    def frame_of(scale: Any) -> Any:
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=0.01), freq=5.0)
        sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=0.05), freq=5.2)
        plan = sequence.resolve().resolved_frame.plan
        assert plan is not None
        return plan.frequencies["q"] * scale

    assert frame_of(1.0) == 5.2
    assert float(jax.jit(frame_of)(jax.numpy.asarray(1.0))) == pytest.approx(5.2)


def test_weights_integrate_each_pulse_over_its_own_window() -> None:
    """A short pulse deep inside a long solve keeps its full energy; a solve window clips what it excludes."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    qubit.reference_freq = 5.0
    drive = ChargeDrive(target=qubit)
    chip = Chip([qubit], frame="auto")
    chip.connect(ControlEquipment(lines=[drive]))
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Square(duration=40.0, amplitude=0.01), freq=5.0)
    sequence.schedule(drive, envelope=Square(duration=0.5, amplitude=0.1), freq=5.2, start_time=500.0)
    plan = sequence.build_problem(np.linspace(0.0, 1000.0, 11)).resolved_frame.plan
    assert plan is not None and plan.frequencies["q"] == 5.2

    later = QuantumSequence(chip)
    later.schedule(drive, envelope=Square(duration=40.0, amplitude=0.02), freq=5.0)
    later.schedule(drive, envelope=Square(duration=40.0, amplitude=0.02), freq=5.2)
    whole = later.build_problem(np.linspace(0.0, 80.0, 9)).resolved_frame.plan
    tail = later.build_problem(np.linspace(30.0, 80.0, 6)).resolved_frame.plan
    assert whole is not None and whole.frequencies["q"] == 5.0
    assert tail is not None and tail.frequencies["q"] == 5.2
