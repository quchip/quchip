"""Gate sets: ideals and rules, knobs, the calibration residual, the cache, and circuits."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import expm

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Exact,
    Gaussian,
    GaussianDRAG,
    QuantumSequence,
    Resonator,
)
from quchip.gateset import GateSet, Knob, NativeGate, drag_sx
from quchip.gateset.cache import covariance, split_pieces
from quchip.gateset.calibration import _Residual, average_fidelity, error_generator
from quchip.gateset.ideals import IDEALS, RULES, embed, ideal_unitary
from quchip.gateset.native import ResolvedKnob, bind_pulses, resolve_knob
from quchip.results.propagator import project_computational


def two_transmons(df: float = 0.2, approximation=None, **noise):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0", **noise)
    q1 = DuffingTransmon(freq=5.0 + df, anharmonicity=-0.33, levels=3, label="q1", **noise)
    extra = {} if approximation is None else {"approximation": approximation}
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003, label="J")], **extra)
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    return chip


def same_up_to_phase(left, right):
    overlap = np.trace(np.asarray(right).conj().T @ np.asarray(left))
    return np.abs(left - np.exp(1j * np.angle(overlap)) * right).max()


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(RULES))
def test_built_in_rules_reproduce_their_ideals(name):
    """Each built-in expansion multiplies out to its ideal up to a global phase."""
    count, expansion = RULES[name]
    params = (0.7,) if name in ("RX", "RY") else ()
    natives = {"SX": IDEALS["SX"], "ZX90": IDEALS["ZX90"]}

    def unitary(body, n):
        result = np.eye(2 ** n, dtype=complex)
        for item in body:
            gate, slots, *rest = item
            slots = slots if isinstance(slots, tuple) else (slots,)
            if gate == "RZ":
                matrix = ideal_unitary("RZ", rest)
            elif gate in natives:
                matrix = natives[gate]
            else:
                sub_count, sub = RULES[gate]
                matrix = unitary(sub(*rest), sub_count)
            result = embed(matrix, list(slots), n) @ result
        return result

    expected = ideal_unitary(name, params)
    assert same_up_to_phase(unitary(expansion(*params), count), expected) < 1e-12


@pytest.mark.unit
def test_embedding_follows_the_first_qubit_most_significant_order():
    """A gate on qubits (2, 0) of three acts like the matching permuted Kronecker product."""
    cx = IDEALS["CX"]
    x = IDEALS["X"]
    projector = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    expected = np.kron(np.kron(np.eye(2), np.eye(2)), projector[0]) + np.kron(np.kron(x, np.eye(2)), projector[1])
    assert np.allclose(embed(cx, [2, 0], 3), expected)
    with pytest.raises(ValueError, match="not unitary"):
        ideal_unitary(np.array([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(KeyError, match="No registered ideal"):
        ideal_unitary("FOO")


@pytest.mark.unit
def test_residual_norm_is_the_pedersen_infidelity_plus_a_phase_term():
    """|r|² = 1 - F_avg + (d - |Tr M|)² / (d (d + 1)) at γ = arg Tr M, with leakage."""
    rng = np.random.default_rng(7)
    ideal = IDEALS["CX"]
    generator = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    generator = 1e-2 * (generator + generator.conj().T)
    full = expm(-1j * generator)
    embedded = np.zeros((6, 4), dtype=complex)
    embedded[:4, :4] = ideal
    image = full @ embedded
    block, leaked = image[:4], np.vstack([np.zeros((4, 4)), image[4:]])
    knob = ResolvedKnob("x", (("pulse.0.amplitude", 1.0),), (0.0,), 0.0, 1.0, -np.inf, np.inf, True)
    residual = _Residual(lambda x: (block, leaked, 0.0), ideal, np.zeros((4, 0)), [knob])
    product = ideal.conj().T @ block
    gamma = float(np.angle(np.trace(product)))
    r = residual(np.array([gamma]))
    expected = 1 - average_fidelity(block, ideal) + (4 - abs(np.trace(product))) ** 2 / 20
    assert abs(float(r @ r) - expected) < 1e-14
    assert np.sum(np.abs(leaked) ** 2) > 1e-5


@pytest.mark.unit
def test_error_generator_and_distance_of_a_known_over_rotation():
    """An SX over-rotated by ε about X has generator coefficient ε/2 on X and distance sin(ε/2)."""
    epsilon = 3e-3
    over = expm(-0.5j * epsilon * IDEALS["X"])
    generator, distance = error_generator(IDEALS["SX"].conj().T @ over @ IDEALS["SX"])
    assert abs(generator["X"] - epsilon / 2) < 1e-12
    assert abs(generator["Y"]) < 1e-12 and abs(generator["Z"]) < 1e-12
    assert abs(distance - np.sin(epsilon / 2)) < 1e-12
    assert abs(1 - average_fidelity(over @ IDEALS["SX"], IDEALS["SX"]) - (2 / 3) * np.sin(epsilon / 2) ** 2) < 1e-15


@pytest.mark.unit
def test_knobs_resolve_handles_paths_and_ties():
    """Handles after a leading vz resolve to their entry; ties move every value by its factor."""
    chip = two_transmons()
    seq = QuantumSequence(chip)
    seq.vz("q0", 0.1)
    first = seq.schedule("xy0", envelope=GaussianDRAG(duration=20.0, sigmas=4.0, amplitude=0.04, beta=0.2),
                         freq=5.0)
    second = seq.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=5.0, phase=np.pi)
    amplitude = resolve_knob("amplitude", Knob(first, "amplitude"), seq)
    assert amplitude.targets == (("pulse.1.amplitude", 1.0),)
    assert amplitude.scale == pytest.approx(0.04) and amplitude.lower == -np.inf
    detuning = resolve_knob("detuning", Knob(first, "detuning"), seq)
    assert detuning.initial == 0.0 and detuning.scale == 1e-3
    duration = resolve_knob("duration", Knob("pulse.1.duration"), seq)
    assert duration.lower == 0.0 and duration.scale == 20.0
    tie = resolve_knob("phase", Knob([((first, "phase"), 1.0), ((second, "phase"), -1.0)]), seq)
    bound = bind_pulses(seq, tie.bindings(0.3))
    assert bound._chip is chip
    assert bound.parameters["pulse.1.phase"] == pytest.approx(0.3)
    assert bound.parameters["pulse.2.phase"] == pytest.approx(np.pi - 0.3)
    with pytest.raises(ValueError, match="pulse fields"):
        resolve_knob("g", Knob("J.g"), seq)
    with pytest.raises(KeyError, match="Pulse paths"):
        resolve_knob("x", Knob("pulse.0.amplitude"), seq)
    other = QuantumSequence(chip)
    with pytest.raises(ValueError, match="another sequence"):
        resolve_knob("x", Knob(first, "amplitude"), other)
    seq.schedule("xy0", envelope=Gaussian(duration=10.0, amplitude=0.01), freq=5.0)
    seq._entries.insert(0, seq._entries.pop())
    with pytest.raises(RuntimeError, match="no longer matches"):
        resolve_knob("x", Knob(first, "amplitude"), seq)


@pytest.mark.unit
def test_gate_set_validates_natives_and_instructions():
    """Templates stay on the chip, in the window, and in gate-qubit frames; unknown gates raise."""
    chip = two_transmons()
    gates = GateSet(chip, ["q0", "q1"])
    gates.add(drag_sx(chip, "q0"))
    with pytest.raises(ValueError, match="already has"):
        gates.add(drag_sx(chip, "q0"))
    foreign = QuantumSequence(chip)
    foreign.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=5.2, frame="q1")
    with pytest.raises(ValueError, match="frame"):
        gates.add(NativeGate("Y90", ("q0",), foreign, ideal=IDEALS["SX"]))
    late = QuantumSequence(chip)
    late.schedule("xy1", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=5.2)
    with pytest.raises(ValueError, match="outside"):
        gates.add(NativeGate("SX", ("q1",), late, duration=10.0))
    with pytest.raises(ValueError, match="acts on 2 qubits"):
        NativeGate("CX", ("q0",), late)
    with pytest.raises(ValueError, match="free_z"):
        NativeGate("SX", ("q1",), late, free_z=("q0",))
    with pytest.raises(ValueError, match="cr_duration"):
        GateSet.standard(chip, ["q0", "q1"], pairs=[("q0", "q1")])
    with pytest.raises(KeyError, match="No native or rule"):
        gates._expand([("ZX90", ("q0", "q1"))])
    expanded = gates._expand([("H", "q0"), ("RZ", "q0", 0.2), ("IDLE", "q0", 5.0)])
    assert [item[0] for item in expanded] == ["RZ", "native", "RZ", "RZ", "IDLE"]
    assert np.allclose(gates.ideal([("RZ", "q1", 0.4)]), np.kron(np.eye(2), ideal_unitary("RZ", (0.4,))))


@pytest.mark.unit
def test_covariance_check_is_structural():
    """RWA pulses with one carrier pass; two carriers, Exact couplings, and frames fail."""
    from quchip.engine.problem import build_solve_problem

    chip = two_transmons()
    seq = QuantumSequence(chip)
    seq.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=chip.freq("q0"), detuning=0.002)
    seq.schedule("xy1", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=chip.freq("q1"))
    ops = list(seq.scheduled_ops)
    problem = build_solve_problem(chip, ops, np.array([0.0, 20.0]), frame="lab", states="final",
                                  dissipation=False, _product_space=True)
    assert split_pieces(ops) == [(0.0, 20.0, [0, 1])]
    assert covariance(problem, ops, [0], (0.0, 20.0))[0] is False
    single = QuantumSequence(chip)
    single.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=chip.freq("q0"), detuning=0.002)
    ops = list(single.scheduled_ops)
    problem = build_solve_problem(chip, ops, np.array([0.0, 20.0]), frame="lab", states="final",
                                  dissipation=False, _product_space=True)
    assert covariance(problem, ops, [0], (0.0, 20.0)) == (True, "", chip.freq("q0"), "q0")
    exact = two_transmons(approximation=Exact())
    problem = build_solve_problem(exact, ops, np.array([0.0, 20.0]), frame="lab", states="final",
                                  dissipation=False, _product_space=True)
    covariant, reason, _, _ = covariance(problem, ops, [0], (0.0, 20.0))
    assert not covariant and "excitation number" in reason


def calibrated_sx(df=0.2, duration=30.0, **noise):
    chip = two_transmons(df, **noise)
    gates = GateSet(chip, ["q0", "q1"])
    gates.add(drag_sx(chip, "q0", duration=duration))
    gates.add(drag_sx(chip, "q1", duration=duration))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate()
    return chip, gates


def test_drag_sx_calibrates_to_the_leakage_floor():
    """A 30 ns ±4σ DRAG SX reaches 1 - F_avg < 1e-8 from the physics seeds, with β near 1/(4π|α|)."""
    _, gates = calibrated_sx()
    calibration = gates.calibration("SX", "q0")
    assert calibration.infidelity < 1e-8
    assert calibration.status in ("ok", "solver_floor")
    assert calibration.knobs["beta"].value == pytest.approx(1 / (4 * np.pi * 0.33), rel=0.05)
    assert abs(calibration.knobs["detuning"].value) < 1e-4
    assert calibration.solves < 120
    assert set(calibration.spectators) == {"q1"} and calibration.placement_sensitivity < 1e-9
    assert calibration.repetitions[1] == pytest.approx(calibration.infidelity, rel=0.2, abs=1e-9)
    text = gates.describe()
    assert "SX" in text and "q0" in text


def test_cached_circuit_matches_the_direct_schedule_solve():
    """unitary() composes placed, frame-shifted gates and idles to the direct propagator of schedule()."""
    _, gates = calibrated_sx()
    circuit = [("SX", "q0"), ("RZ", "q1", 0.3), ("IDLE", "q0", 7.3), ("H", "q1"), ("RX", "q0", 0.5),
               ("SX", "q1"), ("RZ", "q0", -1.1), ("SX", "q0")]
    composed = gates.unitary(circuit, policy="serial")
    direct = gates.schedule(circuit, policy="serial").propagator(frame="rotating", options=gates.options,
                                                                 dissipation=False)
    assert composed.solves == 0
    assert np.abs(composed.full - np.asarray(direct.matrix)).max() < 2e-5
    # Static ZZ acts for the whole 250 ns circuit, so the circuit is not ideal.
    assert composed.fidelity() > 0.99
    overlapping = gates.unitary(circuit)
    assert overlapping.solves > 0
    direct = gates.schedule(circuit).propagator(frame="rotating", options=gates.options, dissipation=False)
    assert np.abs(overlapping.full - np.asarray(direct.matrix)).max() < 2e-5


def test_simulate_maps_the_final_state_into_the_computational_frame():
    """The simulated state equals the composed unitary on |00>, and T1 and T2 lower the fidelity."""
    _, gates = calibrated_sx(T1=5000.0, T2=4000.0)
    circuit = [("H", "q0"), ("RZ", "q0", 0.4), ("SX", "q1")]
    run = gates.simulate(circuit, "00", dissipation=False)
    composed = gates.unitary(circuit)
    assert np.abs(run.state - composed.computational[:, 0]).max() < 1e-6
    assert sum(run.probabilities.values()) == pytest.approx(1.0 - run.leakage, abs=1e-12)
    pure = np.outer(np.eye(4)[0], np.eye(4)[0])
    closed = gates.simulate(circuit, pure, dissipation=False)
    assert np.abs(closed.state - np.outer(run.state, run.state.conj())).max() < 1e-6
    ideal = gates.ideal(circuit)[:, 0]
    lossy = gates.simulate(circuit, "00")
    loss = abs(ideal.conj() @ run.state) ** 2 - np.real(ideal.conj() @ lossy.state @ ideal)
    duration = gates.schedule(circuit).total_duration
    assert 0.2 * duration / 5000.0 < loss < duration / 4000.0


def test_auto_cutoff_keeps_the_gate_and_warns():
    """With two readout resonators (225 states), the gate set picks ExcitationCutoff(3) and warns."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.33, levels=3, label="q1")
    r0 = Resonator(freq=7.0, levels=5, label="r0")
    r1 = Resonator(freq=7.2, levels=5, label="r1")
    chip = Chip([q0, q1, r0, r1], couplings=[Capacitive(q0, q1, g=0.003, label="J"),
                                             Capacitive(q0, r0, g=0.08, label="g0"),
                                             Capacitive(q1, r1, g=0.08, label="g1")])
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    gates = GateSet(chip, ["q0", "q1"])
    gates.add(drag_sx(chip, "q0", duration=30.0))
    with pytest.warns(UserWarning, match=r"ExcitationCutoff\(max_total=3\)"):
        gates.calibrate()
    assert gates.cutoff == 3
    calibration = gates.calibration("SX", "q0")
    assert calibration.infidelity < 1e-6
    assert getattr(calibration, "cutoff_difference") < 1e-6
    reference, _ = GateSet(chip, ["q0", "q1"], space=None)._block(calibration.sequence, ("q0",), 30.0)
    cut, _ = gates._block(calibration.sequence, ("q0",), 30.0)
    assert np.abs(np.asarray(reference.matrix) - np.asarray(cut.matrix)).max() < 1e-6


def test_exact_chip_places_gates_by_direct_solves():
    """Without RWA, the cache flags its pieces and solves each placement, which keeps circuits exact."""
    chip = two_transmons(approximation=Exact())
    gates = GateSet(chip, ["q0", "q1"], frame="lab", options={"atol": 1e-10, "rtol": 1e-8, "nsteps": 10**7})
    seq = QuantumSequence(chip)
    m = chip.drive_matrix_elements("q0")["xy0"]
    pulse = seq.schedule("xy0", envelope=GaussianDRAG(duration=20.0, sigmas=4.0, amplitude=0.04, beta=0.24),
                         freq=chip.freq("q0"), phase=-np.angle(m))
    gates.add(NativeGate("SX", ("q0",), seq, knobs={"amplitude": Knob(pulse, "amplitude")}, free_z=("q0",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate(max_nfev=3)
    entry = gates.entry("SX", "q0")
    assert not entry.covariant and "excitation number" in entry.pieces[0].reason
    circuit = [("SX", "q0"), ("RZ", "q0", 0.3), ("SX", "q0")]
    composed = gates.unitary(circuit)
    assert composed.solves in (1, 2)  # the gate at t = 0 may already be placed by the calibration report
    direct = gates.schedule(circuit).propagator(frame="lab", options=gates.options, dissipation=False)
    columns, energies = gates._states(gates.qubits)
    block = project_computational(np.asarray(direct.matrix) @ columns, columns, energies, 0.0, composed.duration,
                                  gates.qubits)
    correction = gates._frame_correction(gates.qubits, composed.frame_offsets)
    # Lab-frame solves at rtol 1e-8 set the agreement.
    assert np.abs(composed.computational - correction[:, None] * np.asarray(block.matrix)).max() < 2e-5


@pytest.mark.validation
def test_echoed_cross_resonance_calibrates_a_cx():
    """GateSet.standard() calibrates an echoed ZX90, and the CX it implies reaches 1 - F < 1e-3."""
    chip = two_transmons(df=0.1)
    gates = GateSet.standard(chip, ["q0", "q1"], pairs=[("q0", "q1")], sx_duration=40.0, cr_duration=400.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate()
    calibration = gates.calibration("ZX90", ("q0", "q1"))
    assert calibration.infidelity < 1e-3
    assert calibration.dependencies and gates.entry("ZX90", ("q0", "q1")).covariant
    assert gates.unitary([("CX", ("q0", "q1"))]).fidelity() > 1 - 2e-3
