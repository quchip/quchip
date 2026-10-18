"""Results record their integration frame and map states and targets between it and the lab frame."""

import numpy as np
import pytest
from scipy.linalg import expm

from quchip import (
    RWA,
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Exact,
    ExcitationCutoff,
    QuantumSequence,
    Resonator,
    Square,
)
from quchip.declarative import DeviceModel, parameter
from quchip.declarative.parameters import Scalar
from quchip.engine.frames import FrameMap

BACKENDS = ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)]


class TiltedMode(DeviceModel):
    """A mode whose native basis is not its energy basis."""

    freq: Scalar = parameter(default=4.0)
    mixing: Scalar = parameter(default=0.3)

    def local_hamiltonian(self, op, p):
        return p.freq * op.n + p.mixing * op.sigma_x + 0.2 * op.sigma_y


def tight(backend):
    """Solver tolerances far below the checked differences."""
    if backend == "dynamiqs":
        dq = pytest.importorskip("dynamiqs")
        return {"method": dq.method.Tsit5(rtol=1e-10, atol=1e-12)}
    return {"atol": 1e-12, "rtol": 1e-10}


def array(chip, state):
    return np.asarray(chip.backend.to_array(state))


def native_pair(frame, backend="qutip"):
    q = TiltedMode(levels=3, label="q")
    t = DuffingTransmon(freq=4.6, anharmonicity=-0.25, levels=3, label="t", T1=2e3)
    return Chip([q, t], [Capacitive(q, t, g=0.02)], basis="native", frame=frame, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("approximation", [RWA(), Exact()], ids=["RWA", "Exact"])
def test_an_idle_dressed_state_keeps_overlap_one_in_every_frame(backend, approximation):
    """Two transmons 50 MHz apart: the label target and the lab-frame ket read 1; a plain ket keeps its meaning."""
    q0 = DuffingTransmon(freq=5.00, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.05, anharmonicity=-0.25, levels=3, label="q1")
    times = np.linspace(0.0, 20.0, 201)
    label = {"q0": 0, "q1": 1}
    for frame in ("lab", 5.0, "rotating"):
        chip = Chip([q0, q1], [Capacitive(q0, q1, g=0.01)], approximation=approximation, frame=frame,
                    backend=backend)
        dressed = chip.state(label)
        result = QuantumSequence(chip).simulate(times, initial_state=dressed, options=tight(backend))
        frequencies = result.frame.frequencies

        # Lab-frame phases turn at 5 GHz, so the lab solve sets the tolerance.
        np.testing.assert_allclose(result.overlap(label, dressed=True), 1.0, atol=1e-7)
        np.testing.assert_allclose(result.overlap(dressed, frame="lab"), 1.0, atol=1e-7)
        # The label's amplitude turns at its frame energy minus its dressed energy.
        expected = np.exp(-2j * np.pi * (chip.energy(label) - frequencies["q1"]) * times)
        np.testing.assert_allclose(result.amplitude_array(label, dressed=True), expected, atol=1e-7)
        if isinstance(approximation, RWA):
            # A fixed integration-frame ket sees the components of c|01> + s|10> turn apart.
            s2 = abs(array(chip, dressed).reshape(3, 3)[1, 0]) ** 2
            beat = np.cos(2 * np.pi * (frequencies["q1"] - frequencies["q0"]) * times)
            np.testing.assert_allclose(result.overlap(dressed), 1 - 2 * s2 * (1 - s2) * (1 - beat), atol=1e-7)


def test_a_bare_plus_target_keeps_its_rotating_frame_values():
    """An idle qubit in its own rotating frame stays |+>; the same vector in the lab frame beats at its frequency."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating")
    plus = (chip.bare_state(q=0) + chip.bare_state(q=1)) / np.sqrt(2)
    times = np.linspace(0.0, 1.0, 101)
    result = QuantumSequence(chip).simulate(times, initial_state=plus, options=tight("qutip"))

    np.testing.assert_allclose(result.overlap(plus), 1.0, atol=1e-8)
    np.testing.assert_allclose(result.amplitude_array(plus), 1.0, atol=1e-8)
    np.testing.assert_allclose(result.overlap(plus, frame="lab"), np.cos(np.pi * 5.0 * times) ** 2, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lab_frame_states_match_a_lab_frame_solve(backend):
    """With a native basis that is not the energy basis, mapped states agree with a solve in the lab frame."""
    times = np.linspace(0.0, 9.7, 98)
    results = {}
    for frame in ("lab", "rotating", {"q": 3.7, "t": 4.4}):
        chip = native_pair(frame, backend)
        psi = (chip.bare_state(q=0, t=0) + chip.bare_state(q=1, t=0) + chip.bare_state(q=0, t=1)) / np.sqrt(3)
        sequence = QuantumSequence(chip)
        options = tight(backend)
        results[str(frame)] = (sequence.simulate(times, initial_state=psi, dissipation=False, options=options),
                               sequence.simulate(times, initial_state=psi, options=options))
    lab_ket, lab_rho = results.pop("lab")
    # The lab-frame solve itself is accurate to about 3e-7 with these tolerances.
    for ket, rho in results.values():
        for time in times[[20, 74]]:
            np.testing.assert_allclose(array(chip, ket.state_at(time, frame="lab")),
                                       array(chip, lab_ket.state_at(time)), atol=1e-6)
            np.testing.assert_allclose(array(chip, rho.dm_at(time, frame="lab")),
                                       array(chip, lab_rho.dm_at(time)), atol=1e-6)
            np.testing.assert_allclose(array(chip, rho.reduced_state(time, "q", frame="lab")),
                                       array(chip, lab_rho.reduced_state(time, "q")), atol=1e-6)
        np.testing.assert_allclose(array(chip, ket.state(frame="lab")), array(chip, lab_ket.final_state), atol=1e-6)
        assert not np.allclose(array(chip, ket.final_state), array(chip, lab_ket.final_state), atol=1e-3)


def test_dressed_label_targets_on_cutoff_and_batch_results():
    """Compact results read their dressed targets, and batch finals equal the last point of each history."""
    q0 = DuffingTransmon(freq=5.00, anharmonicity=-0.25, levels=3, label="q0", T1=1e4)
    q1 = DuffingTransmon(freq=5.05, anharmonicity=-0.25, levels=3, label="q1", T1=1e4)
    r = Resonator(freq=6.5, levels=4, label="r")
    chip = Chip([q0, q1, r], [Capacitive(q0, q1, g=0.01), Capacitive(q1, r, g=0.05)], approximation=RWA(),
                frame="rotating").with_space(ExcitationCutoff(2))
    label = {"q0": 0, "q1": 1}
    times = np.linspace(0.0, 20.0, 101)
    start = chip.compact_state(label, dressed=True)
    closed = QuantumSequence(chip).simulate(times, initial_state=start, dissipation=False, options=tight("qutip"))
    assert closed.stats["excitation_reduction"]["exact"]
    np.testing.assert_allclose(closed.overlap(label, dressed=True), 1.0, atol=1e-8)

    qa = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="qa")
    qb = DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="qb")
    chip = Chip([qa, qb], [Capacitive(qa, qb, g=0.02)], frame="rotating")
    xy = ChargeDrive(qa, label="xy")
    chip.wire(xy)
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(xy, envelope=Square(duration=20.0, amplitude=0.01), freq=chip.freq(qa))
    batch = sequence.simulate_batch(pulse.vary("amplitude", np.linspace(0.005, 0.03, 4)), duration=20.0,
                                    progress=False, options=tight("qutip"))
    ket = chip.state(qa=1)
    finals = batch.final_overlap_magnitudes([{"qa": 1}] * 4, dressed=True)
    amplitudes = batch.final_amplitudes([ket] * 4, frame="lab")

    for point, final, amplitude in zip(batch.results, finals, amplitudes):
        assert final == pytest.approx(point.overlap({"qa": 1}, dressed=True)[-1], abs=1e-12)
        assert amplitude == pytest.approx(point.amplitude_array(ket, frame="lab")[-1], abs=1e-12)
    np.testing.assert_allclose(finals, np.abs(amplitudes) ** 2, atol=1e-12)
    assert np.ptp(finals) > 0.5


def test_every_result_reports_its_frame():
    """Pulse, trajectory, batch, steady-state and partitioned results name their integration frame."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=1e4)
    p = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, label="p", T1=1e4)
    chip = Chip([q, p], frame={"q": 4.9, "p": 5.3})
    xy = ChargeDrive(q, label="xy")
    chip.wire(xy)
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(xy, envelope=Square(duration=5.0, amplitude=0.01), freq=5.0)
    times = np.linspace(0.0, 5.0, 11)
    results = {
        "pulse": sequence.simulate(times, partition=False),
        "trajectories": sequence.simulate(times, solver="mcsolve", partition=False,
                                          run_args={"ntraj": 2, "seeds": 1}, initial_state={"q": 1}),
        "steady state": chip.steadystate(),
        "batch point": sequence.simulate_batch(pulse.vary("amplitude", [0.01, 0.02]), tlist=times,
                                               progress=False).results[0],
    }
    results["trajectory average"] = results["trajectories"].average()
    for name, result in results.items():
        assert result.frame.frequencies == {"q": 4.9, "p": 5.3}, name
        assert result.frame.mode == "dict", name

    partitioned = sequence.simulate(times, partition=True)
    assert len(partitioned.components) == 2
    assert partitioned.frame.frequencies == {"q": 4.9, "p": 5.3}
    with pytest.warns(UserWarning, match="joint state of a partitioned result"):
        joint = partitioned.state_at(5.0, frame="lab")
    np.testing.assert_allclose(array(chip, joint), array(chip, results["pulse"].state_at(5.0, frame="lab")),
                               atol=1e-6)


@pytest.mark.unit
def test_frame_map_matches_the_exponential_of_the_frame_generator():
    """U(t)† = exp(i 2π t Σ f n) with n in the native basis, for kets, stacks and density matrices."""
    chip = native_pair({"q": 3.7, "t": 4.4})
    resolution = chip.resolve()
    frame_map = FrameMap.create(resolution.resolved_frame, ("q", "t"), resolution.bases, np)
    levels = [record.energy_vectors @ np.diag(np.arange(3)) @ record.energy_vectors.conj().T
              for record in (resolution.bases["q"], resolution.bases["t"])]
    generator = 3.7 * np.kron(levels[0], np.eye(3)) + 4.4 * np.kron(np.eye(3), levels[1])
    rng = np.random.default_rng(3)
    psi = rng.normal(size=9) + 1j * rng.normal(size=9)
    rho = np.outer(psi, psi.conj()) / np.vdot(psi, psi)
    times = np.array([0.3, 1.7])
    rotations = [expm(2j * np.pi * time * generator) for time in times]

    np.testing.assert_allclose(frame_map.kets(psi, times), [rotation @ psi for rotation in rotations], atol=1e-12)
    np.testing.assert_allclose(frame_map.operators(rho, 1.7), rotations[1] @ rho @ rotations[1].conj().T,
                               atol=1e-12)
    np.testing.assert_allclose(frame_map.operators(frame_map.operators(rho, 1.7), 1.7, to_lab=True), rho,
                               atol=1e-12)
    assert frame_map.label_energy((1, 2)) == pytest.approx(3.7 + 2 * 4.4)


@pytest.mark.unit
def test_frame_arguments_and_label_targets_reject_ambiguous_requests():
    """Unknown frames, labels in the lab frame and dressed kets raise; a later chip change keeps the targets."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    p = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=2, label="p")
    chip = Chip([q, p], [Capacitive(q, p, g=0.01, label="c")], frame="rotating")
    ket = chip.state(q=1)
    result = QuantumSequence(chip).simulate(np.linspace(0.0, 1.0, 3), initial_state=ket, partition=False)

    with pytest.raises(ValueError, match='frame must be "integration" or "lab"'):
        result.state_at(1.0, frame="rotating")
    with pytest.raises(ValueError, match="frame= applies to ket targets"):
        result.overlap({"q": 1}, frame="lab")
    with pytest.raises(ValueError, match="dressed=True applies to label targets"):
        result.overlap(ket, dressed=True)
    with pytest.raises(ValueError, match="needs integer energy levels below"):
        result.overlap({"q": 2})
    chip.couplings[0].g = 0.03
    assert abs(chip.state(q=1).full().ravel() @ ket.full().ravel().conj()) < 0.999
    np.testing.assert_allclose(result.overlap({"q": 1}, dressed=True), 1.0, atol=1e-8)
