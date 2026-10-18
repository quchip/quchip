"""A sequence returns its propagator on the dressed states of a product of levels, with leakage."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from quchip import (
    RWA,
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Exact,
    Gaussian,
    QuantumSequence,
    Square,
    analyze_static_zz,
)
from quchip.results.propagator import PropagatorResult

# A lab-frame solve accumulates 2π·500 rad over 100 ns, so its phase error needs rtol = 1e-12.
TIGHT = {"atol": 1e-13, "rtol": 1e-12}
BACKENDS = ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)]


def tight(backend):
    """Solver tolerances far below the checked differences."""
    if backend == "dynamiqs":
        dq = pytest.importorskip("dynamiqs")
        return {"method": dq.method.Tsit5(rtol=1e-12, atol=1e-13)}
    return TIGHT


def idle_pair(frame="rotating", backend=None):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, label="q1")
    return Chip([q0, q1], [Capacitive(q0, q1, g=0.005, label="c")], approximation=RWA(), frame=frame,
                backend=backend)


def driven_transmon(levels=2, frame="rotating", **kwargs):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=levels, label="q", **kwargs)
    chip = Chip([q], approximation=RWA(), frame=frame)
    chip.wire(ChargeDrive(q, label="xy"))
    return chip


@pytest.mark.unit
def test_average_fidelity_counts_leakage():
    """F = (Tr M†M + |Tr V†M|²) / (d(d + 1)): 1 for the target, 1/3 for an orthogonal gate, lower with leakage."""
    def record(matrix):
        return PropagatorResult(matrix=np.asarray(matrix, dtype=complex), labels=(), devices=(), leakage=None,
                                frame="dressed", frequencies={}, ground_energy=0.0, times=(0.0, 1.0),
                                omitted_channels=(), batch=None)

    np.testing.assert_allclose(record(np.eye(2)).fidelity(np.eye(2)), 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(record([[0, 1], [1, 0]]).fidelity(np.eye(2)), 1 / 3, rtol=0, atol=1e-15)
    kept = np.sqrt(0.9)
    leaky = record(np.diag([1.0, kept]))
    np.testing.assert_allclose(leaky.fidelity(np.eye(2)), (1.9 + (1 + kept) ** 2) / 6, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="shape"):
        leaky.fidelity(np.eye(3))


@pytest.mark.unit
def test_invalid_selections_raise_before_solving():
    """An unknown frame, a repeated device, or a repeated level raises."""
    sequence = QuantumSequence(idle_pair())
    with pytest.raises(ValueError, match="frame must be one of"):
        sequence.subspace_propagator("q0", frame="rotating")
    with pytest.raises(ValueError, match="each device once"):
        sequence.subspace_propagator(["q0", "q0"])
    with pytest.raises(ValueError, match="each level once"):
        sequence.subspace_propagator("q0", levels=(0, 0))


@pytest.mark.parametrize("frame", ["rotating", "lab"])
def test_idle_pair_gives_the_conditional_phase_in_the_dressed_frame(frame):
    """#193: 100 ns of idle time gives diag(1, 1, 1, exp(−2πiζT)) at any start time, in either solve frame."""
    chip = idle_pair(frame)
    zeta = float(analyze_static_zz(chip, "q0", "q1").zz)
    expected = np.diag([1, 1, 1, np.exp(-2j * np.pi * zeta * 100.0)])
    for start in (0.0, 3.7):
        record = QuantumSequence(chip).subspace_propagator(
            ["q0", "q1"], tlist=np.array([start, start + 100.0]), options=TIGHT)
        assert record.labels == ((0, 0), (0, 1), (1, 0), (1, 1))
        assert record.devices == ("q0", "q1") and record.frame == "dressed"
        np.testing.assert_allclose(record.times, (start, start + 100.0))
        np.testing.assert_allclose(record.matrix, expected, rtol=0, atol=1e-7)
        np.testing.assert_allclose(record.leakage, 0.0, rtol=0, atol=1e-10)
        np.testing.assert_allclose([record.frequencies["q0"], record.frequencies["q1"]],
                                   [chip.freq("q0"), chip.freq("q1")], rtol=0, atol=1e-12)


def test_frames_differ_only_by_the_label_energies():
    """The lab and integration matrices are the dressed matrix with each frame's label phases."""
    chip = driven_transmon(levels=3)
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Gaussian(duration=12.0, sigmas=3.0, amplitude=0.06), freq=chip.freq("q"))
    times = np.array([2.0, 16.0])
    records = {frame: sequence.subspace_propagator("q", frame=frame, tlist=times, options=TIGHT)
               for frame in ("dressed", "lab", "integration")}
    dressed = np.asarray(records["dressed"].matrix)
    for frame in ("lab", "integration"):
        record = records[frame]
        energies = np.array([record.ground_energy + record.frequencies["q"] * level for (level,) in record.labels])
        reference = np.array([records["dressed"].ground_energy + records["dressed"].frequencies["q"] * level
                              for (level,) in record.labels])
        shift = energies - reference
        phases = np.exp(2j * np.pi * times[1] * shift)[:, None] * np.exp(-2j * np.pi * times[0] * shift)[None, :]
        np.testing.assert_allclose(record.matrix, phases * dressed, rtol=0, atol=1e-12)
        np.testing.assert_allclose(record.leakage, records["dressed"].leakage, rtol=0, atol=1e-14)
    assert records["lab"].frequencies == {"q": 0.0}
    assert records["integration"].frequencies["q"] == pytest.approx(chip.frame_info()["q"])


def test_resonant_pulse_rotation_axes():
    """On a two-level transmon, a real π pulse gives iY, because the charge drive has m = −i; phase π/2 gives −iX."""
    chip = driven_transmon()
    targets = {0.0: np.array([[0, 1], [-1, 0]]), np.pi / 2: np.array([[0, -1j], [-1j, 0]])}
    for phase, target in targets.items():
        sequence = QuantumSequence(chip)
        sequence.schedule("xy", envelope=Square(duration=20.0, amplitude=0.025), freq=chip.freq("q"), phase=phase)
        record = sequence.subspace_propagator("q", tlist=np.array([0.0, 20.0]), options=TIGHT)
        np.testing.assert_allclose(record.matrix, target, rtol=0, atol=1e-8)
        np.testing.assert_allclose(record.fidelity(target), 1.0, rtol=0, atol=1e-12)


def test_leakage_is_the_population_of_the_other_dressed_levels():
    """A fast pulse leaks to level 2; the leakage of each input equals its lab-frame overlap with |2̃⟩."""
    chip = driven_transmon(levels=3)
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Gaussian(duration=4.0, sigmas=2.0, amplitude=0.25), freq=chip.freq("q"))
    record = sequence.subspace_propagator("q", tlist=np.array([0.0, 4.0]), options=TIGHT)
    leaked = chip.backend.to_array(chip.state({"q": 2})).reshape(-1)
    overlaps = [abs(np.vdot(leaked, np.asarray(chip.backend.to_array(result.state(frame="lab"))).reshape(-1))) ** 2
                for result in record.batch]
    assert min(overlaps) > 1e-3
    np.testing.assert_allclose(record.leakage, overlaps, rtol=0, atol=1e-10)


def test_collapse_channels_are_left_out_with_a_warning():
    """A chip with T1 gives the propagator of its Hamiltonian, and the record names the omitted channel."""
    chip = driven_transmon(T1=10.0)
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Square(duration=20.0, amplitude=0.025), freq=chip.freq("q"))
    with pytest.warns(UserWarning, match="without the collapse channels"):
        record = sequence.subspace_propagator("q", tlist=np.array([0.0, 20.0]), options=TIGHT)
    assert len(record.omitted_channels) == 1
    np.testing.assert_allclose(record.matrix, [[0, 1], [-1, 0]], rtol=0, atol=1e-8)


def lab_hamiltonian(f, alpha, g, levels):
    """H/h of two Duffing transmons with the full charge coupling g Q₀Q₁, Q = i(a − a†), in GHz."""
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)
    n = np.diag(np.arange(levels))
    eye = np.eye(levels)
    charge = 1j * (a - a.T)
    local = [fi * n + ai / 2 * n @ (n - eye) for fi, ai in zip(f, alpha)]
    h0 = np.kron(local[0], eye) + np.kron(eye, local[1]) + g * np.kron(charge, charge)
    return h0, np.kron(charge, eye)


@pytest.mark.validation
@pytest.mark.parametrize("backend", BACKENDS)
def test_driven_gate_matches_an_independent_lab_frame_propagator(backend):
    """A π/2 pulse on q0 that starts at 3.7 ns, under Exact(), against a SciPy lab-frame propagator of the
    same Hamiltonian, projected on its own dressed eigenvectors, entry by entry."""
    f, alpha, g, levels = (5.0, 5.4), (-0.25, -0.25), 0.01, 3
    q0 = DuffingTransmon(freq=f[0], anharmonicity=alpha[0], levels=levels, label="q0")
    q1 = DuffingTransmon(freq=f[1], anharmonicity=alpha[1], levels=levels, label="q1")
    chip = Chip([q0, q1], [Capacitive(q0, q1, g=g)], approximation=Exact(), frame="rotating", backend=backend)
    chip.wire(ChargeDrive(q0, label="xy"))
    start, duration, phase = 3.7, 20.0, 0.4
    envelope = Gaussian(duration=duration, sigmas=3.0)
    amplitude = float(chip.rotation_amplitude(q0, "xy", envelope, np.pi / 2))
    carrier = chip.freq(q0)
    sequence = QuantumSequence(chip)
    sequence.delay(q0, start)
    sequence.schedule("xy", envelope=envelope.with_params({"amplitude": amplitude}), freq=carrier, phase=phase)
    record = sequence.subspace_propagator([q0, q1], tlist=np.array([start, start + duration]),
                                          options=tight(backend))

    h0, charge = lab_hamiltonian(f, alpha, g, levels)
    energies, vectors = np.linalg.eigh(h0)
    labels = [(i, j) for i in range(levels) for j in range(levels)]
    dressed = {}
    for column in range(len(energies)):
        row = int(np.argmax(np.abs(vectors[:, column])))
        dressed[labels[row]] = (energies[column], vectors[:, column] * np.exp(-1j * np.angle(vectors[row, column])))

    def field(t):
        sigma = duration / 6.0
        shape = amplitude * np.exp(-((t - start - duration / 2) ** 2) / (2 * sigma**2))
        return np.real(shape * np.exp(1j * phase) * np.exp(-2j * np.pi * carrier * t))

    def rhs(t, psi):
        return -2j * np.pi * ((h0 + field(t) * charge) @ psi)

    ground = dressed[(0, 0)][0]
    frequencies = {0: dressed[(1, 0)][0] - ground, 1: dressed[(0, 1)][0] - ground}
    selected = [(0, 0), (0, 1), (1, 0), (1, 1)]
    frame_energy = {k: ground + frequencies[0] * k[0] + frequencies[1] * k[1] for k in selected}
    expected = np.zeros((4, 4), dtype=complex)
    for j, initial in enumerate(selected):
        final = solve_ivp(rhs, (start, start + duration), dressed[initial][1].astype(complex), method="DOP853",
                          rtol=1e-12, atol=1e-13).y[:, -1]
        for i, label in enumerate(selected):
            expected[i, j] = (np.exp(2j * np.pi * (start + duration) * frame_energy[label])
                              * np.vdot(dressed[label][1], final) * np.exp(-2j * np.pi * start * frame_energy[initial]))
    assert abs(expected[2, 0]) > 0.6
    np.testing.assert_allclose([record.frequencies["q0"], record.frequencies["q1"]],
                               [frequencies[0], frequencies[1]], rtol=0, atol=1e-11)
    # QuTiP interpolates the envelope linearly between samples, which moves the entries by about 3e-7.
    np.testing.assert_allclose(record.matrix, expected, rtol=0, atol=1e-6 if backend == "qutip" else 1e-9)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_propagator_entry_gradient_matches_finite_differences():
    """On dynamiqs, the derivative of a gate entry with respect to the pulse amplitude matches central differences."""
    import jax

    chip = driven_transmon(levels=3)
    chip = Chip(chip.devices, approximation=RWA(), frame="rotating", backend="dynamiqs",
                control_equipment=chip.control_equipment)
    options = tight("dynamiqs")

    def entry(amplitude):
        sequence = QuantumSequence(chip)
        sequence.schedule("xy", envelope=Gaussian(duration=12.0, sigmas=3.0, amplitude=amplitude), freq=5.0)
        record = sequence.subspace_propagator("q", tlist=np.array([0.0, 12.0]), options=options)
        return record.matrix[1, 0].real

    amplitude, step = 0.05, 1e-5
    derivative = jax.grad(entry)(amplitude)
    finite = (entry(amplitude + step) - entry(amplitude - step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite, rtol=1e-6, atol=1e-9)
