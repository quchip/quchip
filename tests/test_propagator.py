"""Lab-frame propagators of scheduled sequences and sequence composition with extend()."""

from __future__ import annotations

import numpy as np
import pytest
import qutip
from scipy.linalg import expm
from scipy.special import erf

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    ExcitationCutoff,
    Gaussian,
    GaussianDRAG,
    QuantumSequence,
    Qubit,
    Resonator,
)
from quchip.gateset.blocks import computational_states, project_computational

TIGHT = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 10**7}


def two_transmons(**noise):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0", **noise)
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q1", **noise)
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003, label="J")])
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    return chip


def drag_pulse(seq, chip, line="xy0", qubit="q0", detuning=0.001):
    m = chip.drive_matrix_elements(qubit)[line]
    return seq.schedule(line, envelope=GaussianDRAG(duration=20.0, sigmas=4.0, amplitude=0.04, beta=0.24),
                        freq=chip.freq(qubit), phase=-np.angle(m), detuning=detuning)


def ket(state):
    return np.asarray(state.full()).reshape(-1)


def computational_block(propagator, qubits):
    """Project a full propagator on dressed computational states, as the gate set does."""
    columns, energies = computational_states(propagator._problem, tuple(qubits), propagator.levels)
    image = np.asarray(propagator.matrix) @ np.asarray(columns)
    return project_computational(image, np.asarray(columns), np.asarray(energies), propagator.t0, propagator.t1,
                                 tuple(qubits))


def test_unitary_matches_lab_frame_simulation_and_is_frame_independent():
    """A rotating-frame propagator applied to a dressed state equals a lab-frame solve from that state."""
    chip = two_transmons()
    seq = QuantumSequence(chip)
    drag_pulse(seq, chip)
    rotating = seq.propagator(frame="rotating", options=TIGHT)
    assert rotating.kind == "unitary" and rotating.matrix.shape == (9, 9)
    assert rotating.unitarity_defect < 1e-6
    start = ket(chip.state({"q0": 0, "q1": 1}))
    result = seq.simulate(initial_state=chip.state({"q0": 0, "q1": 1}), options=TIGHT, frame="lab", states="final")
    assert np.abs(ket(result.final_state) - np.asarray(rotating.matrix) @ start).max() < 1e-6
    lab = seq.propagator(frame="lab", options=TIGHT)
    assert np.abs(np.asarray(lab.matrix) - np.asarray(rotating.matrix)).max() < 1e-4
    assert np.abs(np.asarray(rotating.solver_frame()) - np.asarray(rotating.matrix)).max() > 1e-2


def test_columns_give_the_matching_columns_of_the_full_propagator():
    """Evolving only the computational inputs gives the matching columns of the full propagator."""
    chip = two_transmons()
    seq = QuantumSequence(chip)
    drag_pulse(seq, chip)
    full = seq.propagator(frame="rotating", options=TIGHT)
    columns, _ = computational_states(full._problem, ("q0", "q1"), full.levels)
    block = seq.propagator(frame="rotating", options=TIGHT, columns=np.asarray(columns))
    assert block.matrix.shape == (9, 4)
    assert np.abs(np.asarray(block.matrix) - np.asarray(full.matrix) @ np.asarray(columns)).max() < 1e-8


def test_gate_set_block_matches_the_subspace_propagator():
    """One propagator solve, projected as the gate set does, gives the dressed matrix of subspace_propagator()."""
    chip = two_transmons()
    seq = QuantumSequence(chip)
    drag_pulse(seq, chip)
    times = np.array([3.0, 43.0])
    options = {"atol": 1e-14, "rtol": 1e-13, "nsteps": 10**8}
    record = seq.subspace_propagator(["q0", "q1"], tlist=times, options=options, dissipation=False, max_workers=1)
    block = computational_block(seq.propagator(times, options=options), ["q0", "q1"])
    ground = np.exp(2j * np.pi * record.ground_energy * (times[-1] - times[0]))
    assert np.abs(np.asarray(record.matrix) - ground * np.asarray(block.matrix)).max() < 1e-8
    assert np.abs(np.asarray(record.leakage) - np.asarray(block.leakage)).max() < 1e-10


def test_excitation_cutoff_propagator_converges_to_the_product_space():
    """Under ExcitationCutoff(3) the solve keeps 18 of 45 labels and reproduces the computational block."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q1")
    r0 = Resonator(freq=7.0, levels=5, label="r0")
    chip = Chip([q0, q1, r0], couplings=[Capacitive(q0, q1, g=0.003, label="J"), Capacitive(q0, r0, g=0.08, label="g")])
    chip.wire(ChargeDrive(q0, label="xy0"))
    blocks = {}
    for cutoff in (None, 3):
        model = chip if cutoff is None else chip.with_space(ExcitationCutoff(max_total=cutoff))
        seq = QuantumSequence(model)
        drag_pulse(seq, model, detuning=None)
        propagator = seq.propagator(frame="rotating", options=TIGHT)
        blocks[cutoff] = (propagator.matrix.shape[0], computational_block(propagator, ["q0", "q1"]))
    assert blocks[None][0] == 45 and blocks[3][0] == 18
    assert np.abs(np.asarray(blocks[None][1].matrix) - np.asarray(blocks[3][1].matrix)).max() < 1e-6


def test_superoperator_matches_a_density_matrix_solve():
    """With T1 and T2, the propagator acts on column-stacked density matrices like mesolve."""
    chip = two_transmons(T1=20000.0, T2=15000.0)
    seq = QuantumSequence(chip)
    drag_pulse(seq, chip)
    channel = seq.propagator(frame="rotating", options=TIGHT)
    assert channel.kind == "superoperator" and channel.matrix.shape == (81, 81)
    assert channel.unitarity_defect is None
    state = ket(chip.state({"q0": 1})) + ket(chip.state({}))
    rho = np.outer(state, state.conj()) / 2
    result = seq.simulate(initial_state=qutip.Qobj(rho, dims=[[3, 3], [3, 3]]), options=TIGHT, frame="lab",
                          states="final")
    final = (np.asarray(channel.matrix) @ rho.reshape(-1, order="F")).reshape(9, 9, order="F")
    assert np.abs(final - np.asarray(result.final_state.full())).max() < 1e-6
    with pytest.raises(ValueError, match="closed system"):
        seq.propagator(columns=np.eye(9)[:, :2])


def test_extend_appends_aligned_copies_with_their_frame_shifts():
    """extend() starts a native after the channels it uses and keeps its vz."""
    chip = two_transmons()
    native = QuantumSequence(chip)
    drag_pulse(native, chip, detuning=None)
    native.vz("q0", 0.4)
    composed = QuantumSequence(chip)
    composed.schedule("xy1", envelope=Gaussian(duration=30.0, sigmas=4.0, amplitude=0.02), freq=chip.freq("q1"))
    composed.extend(native)
    composed.extend(native)
    by_hand = QuantumSequence(chip)
    by_hand.schedule("xy1", envelope=Gaussian(duration=30.0, sigmas=4.0, amplitude=0.02), freq=chip.freq("q1"))
    drag_pulse(by_hand, chip, detuning=None)
    by_hand.vz("q0", 0.4)
    drag_pulse(by_hand, chip, detuning=None)
    by_hand.vz("q0", 0.4)
    starts = [(op.start_time, op.phase_offset) for op in composed.scheduled_ops]
    assert starts == [(op.start_time, op.phase_offset) for op in by_hand.scheduled_ops]
    assert composed.total_duration == 40.0
    left = composed.propagator(frame="rotating", options=TIGHT, tlist=[0.0, 50.0])
    right = by_hand.propagator(frame="rotating", options=TIGHT, tlist=[0.0, 50.0])
    assert np.abs(np.asarray(left.matrix) - np.asarray(right.matrix)).max() < 1e-10
    # A device that the native only shifts keeps its cursor; the shift still applies in program order.
    shifting = QuantumSequence(chip)
    drag_pulse(shifting, chip, detuning=None)
    shifting.vz("q1", 0.2)
    parallel = QuantumSequence(chip)
    parallel.extend(shifting)
    parallel.schedule("xy1", envelope=Gaussian(duration=30.0, sigmas=4.0, amplitude=0.02), freq=chip.freq("q1"))
    last = list(parallel.scheduled_ops)[-1]
    assert (last.start_time, last.phase_offset) == (0.0, pytest.approx(0.2))


@pytest.mark.unit
def test_extend_and_propagator_reject_invalid_requests():
    """extend() needs the same chip and cursor timing; propagator() needs two times."""
    chip = two_transmons()
    other_chip = two_transmons()
    seq = QuantumSequence(chip)
    with pytest.raises(ValueError, match="same chip"):
        seq.extend(QuantumSequence(other_chip))
    timed = QuantumSequence(chip)
    timed.schedule("xy0", envelope=Gaussian(duration=10.0, amplitude=0.01), freq=5.0, start_time=5.0)
    with pytest.raises(ValueError, match="cursor-timed"):
        seq.extend(timed)
    with pytest.raises(ValueError, match="itself"):
        seq.extend(seq)
    with pytest.raises(ValueError, match="start and an end"):
        timed.propagator(tlist=[1.0])


@pytest.mark.optional_backend
def test_dynamiqs_propagator_matches_qutip():
    """dynamiqs evaluates envelopes exactly; QuTiP's sampled envelopes agree to its interpolation error."""
    pytest.importorskip("dynamiqs")
    chip = two_transmons()
    seq = QuantumSequence(chip)
    drag_pulse(seq, chip)
    reference = seq.propagator(frame="rotating", options=TIGHT)
    full = seq.propagator(frame="rotating", backend="dynamiqs")
    assert np.abs(np.asarray(full.matrix) - np.asarray(reference.matrix)).max() < 5e-6
    columns, _ = computational_states(reference._problem, ("q0", "q1"), reference.levels)
    block = seq.propagator(frame="rotating", backend="dynamiqs", columns=np.asarray(columns))
    assert np.abs(np.asarray(block.matrix) - np.asarray(full.matrix) @ np.asarray(columns)).max() < 1e-8


@pytest.mark.validation
def test_two_level_rotation_matches_the_pulse_area():
    """A resonant Gaussian on an ideal qubit gives exp(-iθX/2) with θ = 2π|m|A∫E dt."""
    q = Qubit(freq=5.0, label="q")
    chip = Chip([q])
    chip.wire(ChargeDrive(q, label="xy"))
    m = chip.drive_matrix_elements("q")["xy"]
    envelope = Gaussian(duration=40.0, sigmas=4.0, amplitude=0.01)
    seq = QuantumSequence(chip)
    seq.schedule("xy", envelope=envelope, freq=chip.freq("q"), phase=-np.angle(m))
    block = computational_block(
        seq.propagator(frame="rotating", options={"atol": 1e-13, "rtol": 1e-11, "nsteps": 10**7}), ["q"])
    sigma = 40.0 / 8.0
    area = 0.01 * sigma * np.sqrt(2 * np.pi) * erf(20.0 / (np.sqrt(2) * sigma))
    theta = 2 * np.pi * abs(m) * area
    ideal = expm(-0.5j * theta * np.array([[0, 1], [1, 0]]))
    assert block.fidelity(ideal) > 1 - 1e-9
