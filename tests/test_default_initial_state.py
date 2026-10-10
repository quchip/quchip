"""A solve without an initial state starts in the ground state of the Hamiltonian it simulates."""

import numpy as np
import pytest

from quchip import (
    RWA,
    Capacitive,
    ChargeBasisTransmon,
    ChargeDrive,
    Chip,
    CrossKerr,
    DuffingTransmon,
    Exact,
    QuantumSequence,
    Resonator,
    Square,
    eliminate,
)

# Start-state overlaps are read at tlist[0], before any integration step.
_ROUNDOFF = 1e-12
# Bare transmon and resonator frequencies (GHz); the vacuum couples to |11> across their sum.
_QUBIT, _ANHARMONICITY, _RESONATOR = 5.0, -0.25, 7.0
# Every band a capacitive coupling populates, so this RWA keeps the whole coupling.
_ALL_CAPACITIVE_BANDS = RWA(keep_bands={(0, 0), (1, -1), (-1, 1), (1, 1), (-1, -1)})


def _pair(g, *, approximation=Exact(), frame="lab", backend=None, coupling=Capacitive):
    q = DuffingTransmon(freq=_QUBIT, anharmonicity=_ANHARMONICITY, levels=3, label="q")
    r = Resonator(freq=_RESONATOR, levels=3, label="r")
    link = Capacitive(q, r, g=g, label="c") if coupling is Capacitive else CrossKerr(q, r, chi=g, label="c")
    chip = Chip([q, r], couplings=[link], frame=frame, approximation=approximation, backend=backend)
    return chip, q, r


def _ket(chip, state):
    return chip.backend.to_array(state).reshape(-1)


def _drift(result, device):
    populations = np.real(np.asarray(result.population(device, 0)))
    return float(np.max(np.abs(populations - populations[0])))


def _start_fidelity(result, target):
    return float(np.real(np.asarray(result.overlap(target))[0]))


@pytest.mark.parametrize(
    ("approximation", "coupling", "dressed"),
    [
        (RWA(), Capacitive, False),
        (Exact(), Capacitive, True),
        (_ALL_CAPACITIVE_BANDS, Capacitive, True),
        (Exact(), CrossKerr, False),
    ],
    ids=["rwa", "exact", "rwa-all-bands", "exact-cross-kerr"],
)
def test_default_start_is_ground_of_simulated_hamiltonian(approximation, coupling, dressed):
    """The default start is chip.state's dressed ground when retained bands move |00>, else the bare product."""
    chip, q, r = _pair(0.2, approximation=approximation, coupling=coupling)
    result = QuantumSequence(chip).simulate(tlist=np.linspace(0.0, 50.0, 101))
    ground = chip.state({q: 0, r: 0})
    assert _start_fidelity(result, ground) == pytest.approx(1.0, abs=_ROUNDOFF)
    if not dressed:
        assert _start_fidelity(result, chip.bare_state({q: 0, r: 0})) == pytest.approx(1.0, abs=_ROUNDOFF)
    # Each integrator step applies a function of the constant lab-frame generator, so an
    # eigenstate keeps its populations up to its norm; a bare start under Exact moves
    # P(q=0) by about 1e-3, a few times (g/Σ)² ≈ 3e-4.
    assert _drift(result, q) < 1e-10


@pytest.mark.parametrize("kind", ["duffing", "native-charge"])
def test_default_start_is_stationary_in_rotating_frame_from_later_start(kind):
    """A rotating-frame Exact solve starting at t0 = 10 ns keeps the default start's populations."""
    if kind == "duffing":
        q = DuffingTransmon(freq=_QUBIT, anharmonicity=_ANHARMONICITY, levels=3, label="q")
    else:
        q = ChargeBasisTransmon(E_C=0.25, E_J=12.5, num_basis=7, basis="native", label="q")
    r = Resonator(freq=_RESONATOR, levels=3, label="r")
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.2)], frame="rotating", approximation=Exact())
    result = QuantumSequence(chip).simulate(
        tlist=np.linspace(10.0, 15.0, 51), options={"atol": 1e-12, "rtol": 1e-10}
    )
    # Populations are conserved to the integrator tolerance (rtol 1e-10); omitting the
    # frame rotation at t0 leaves an oscillation of order 1e-5 to 1e-4.
    assert _drift(result, q) < 1e-8


def test_parameter_batch_redresses_default_start_per_point():
    """Each point of a coupling sweep starts in the dressed ground of its own chip."""
    chip, q, r = _pair(0.05)
    sequence = QuantumSequence(chip)
    couplings = [0.05, 0.3]
    results = sequence.simulate_batch(sequence.vary("c.g", couplings), tlist=np.array([0.0, 1.0]), progress=False)
    for g, result in zip(couplings, results, strict=True):
        ground = chip.with_params({"c.g": g}).state({q: 0, r: 0})
        assert _start_fidelity(result, ground) == pytest.approx(1.0, abs=_ROUNDOFF)


def test_pulse_batch_shares_the_single_solve_default_start():
    """Every point of a pulse-amplitude sweep starts where the single rotating-frame solve does."""
    chip, q, _ = _pair(0.2, frame="rotating")
    drive = ChargeDrive(q, label="d")
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(drive, envelope=Square(duration=10.0, amplitude=0.01), freq=_QUBIT)
    times = np.array([7.3, 20.0])
    single = _ket(chip, sequence.build_problem(tlist=times).initial_state)
    batch = sequence.build_batch(pulse.vary("amplitude", [0.0, 0.02]), tlist=times)
    for index in range(2):
        np.testing.assert_allclose(_ket(chip, batch.element(index).initial_state), single, rtol=0, atol=_ROUNDOFF)


def test_partitioned_default_start_matches_joint_ground():
    """Independent components start in their own dressed grounds, which multiply to the joint ground."""
    qa, qb = (DuffingTransmon(freq=_QUBIT, anharmonicity=_ANHARMONICITY, levels=3, label=f"q{t}") for t in "ab")
    ra, rb = (Resonator(freq=_RESONATOR, levels=3, label=f"r{t}") for t in "ab")
    chip = Chip([qa, ra, qb, rb], couplings=[Capacitive(qa, ra, g=0.1), Capacitive(qb, rb, g=0.3)],
                approximation=Exact())
    sequence = QuantumSequence(chip)
    times = np.array([0.0, 1.0])
    partitioned = sequence.simulate(tlist=times)
    joint = sequence.simulate(tlist=times, partition=False)
    assert _start_fidelity(joint, chip.state({qa: 0, ra: 0, qb: 0, rb: 0})) == pytest.approx(1.0, abs=_ROUNDOFF)
    for device in (qa, ra, qb, rb):
        np.testing.assert_allclose(
            np.real(np.asarray(partitioned.population(device, 0)))[0],
            np.real(np.asarray(joint.population(device, 0)))[0],
            rtol=0, atol=_ROUNDOFF,
        )


def test_retained_effective_terms_dress_the_rwa_default_start():
    """Explicitly retained counter-rotating bands dress the default start after exact coupler elimination."""
    qubits = [
        DuffingTransmon(freq=_QUBIT + 0.13 * i, anharmonicity=_ANHARMONICITY, levels=3, label=f"q{i}")
        for i in range(2)
    ]
    coupler = DuffingTransmon(freq=6.5, anharmonicity=-0.2, levels=3, label="cp")
    chip = Chip([*qubits, coupler], couplings=[Capacitive(q, coupler, g=0.1) for q in qubits],
                approximation=_ALL_CAPACITIVE_BANDS)
    reduced = eliminate(chip, coupler, method="exact").chip
    result = QuantumSequence(reduced).simulate(tlist=np.linspace(0.0, 50.0, 101))
    # The effective terms couple |00> to two-excitation states by about 5 MHz, so a bare start drifts by about 1e-6.
    assert _start_fidelity(result, reduced.bare_state()) < 1 - 1e-8
    assert _drift(result, reduced.devices[0]) < 1e-10


@pytest.mark.optional_backend
def test_traced_default_start_matches_concrete_start():
    """Under jax.jit on dynamiqs the rotating-frame default start follows a traced coupling and start time."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    def start(g, start_time, backend):
        chip, _, _ = _pair(g, frame="rotating", backend=backend)
        times = (jnp if backend else np).stack([start_time, start_time + 1.0])
        return _ket(chip, QuantumSequence(chip).build_problem(tlist=times).initial_state)

    traced = jax.jit(lambda g, start_time: start(g, start_time, "dynamiqs"))(0.2, 7.3)
    # Both starts fix their phase by the bare overlap, so they agree as vectors.
    np.testing.assert_allclose(np.asarray(traced), start(0.2, 7.3, None), rtol=0, atol=_ROUNDOFF)


def _perturbative_leakage(g):
    """Return the g² and g⁴ Rayleigh–Schrödinger terms of 1 - |<00|G>|²."""
    # V = g i(a - a†) i(b - b†) couples |00> only to |11>, at energy Σ; |11> couples
    # onward to |20>, |02> (element √2 g) and |22> (element 2g).
    total = _QUBIT + _RESONATOR
    energies = {"20": 2 * _QUBIT + _ANHARMONICITY, "02": 2 * _RESONATOR,
                "22": 2 * _QUBIT + _ANHARMONICITY + 2 * _RESONATOR}
    weights = {"20": 2.0, "02": 2.0, "22": 4.0}
    first = sum(weights[k] / energies[k] for k in energies)
    second = sum(weights[k] / energies[k] ** 2 for k in energies)
    return (g / total) ** 2, g**4 * (2 * first / total**3 + second / total**2 - 3 / total**4)


def _leakage(g, backend=None, spectators=0):
    q = DuffingTransmon(freq=_QUBIT, anharmonicity=_ANHARMONICITY, levels=3, label="q")
    r = Resonator(freq=_RESONATOR, levels=3, label="r")
    others = [DuffingTransmon(freq=4.0, anharmonicity=-0.3, levels=2, label=f"s{i}") for i in range(spectators)]
    chip = Chip([q, r, *others], couplings=[Capacitive(q, r, g=g)], approximation=Exact(), backend=backend)
    start = _ket(chip, QuantumSequence(chip).build_problem(tlist=np.array([0.0, 1.0])).initial_state)
    xp = chip.backend.array_module
    return 1.0 - xp.abs(xp.vdot(_ket(chip, chip.bare_state({q: 0, r: 0})), start)) ** 2


@pytest.mark.validation
def test_default_ground_leakage_follows_perturbation_theory():
    """The default start's weight outside |00> leaves a g⁶ residual after its g² and g⁴ terms."""
    def residual(g):
        return _leakage(g) - sum(_perturbative_leakage(g))

    # Halving g divides a g⁶ residual by 64; a wrong g² or g⁴ term leaves a ratio near 4 or 16.
    # The g⁸ term shifts the ratio by a relative O((2g/9.75 GHz)²) ≈ 2e-3, where 2g is the largest
    # matrix element above |11>; roundoff in the smaller residual (≈ 2.5e-11) adds below 1e-4.
    assert residual(0.2) / residual(0.1) == pytest.approx(64, rel=0.02)


@pytest.mark.validation
def test_default_ground_matches_independent_diagonalization_in_rotating_frame():
    """The default start equals U(t0)† applied to an independently diagonalized lab-frame ground."""
    q = DuffingTransmon(freq=_QUBIT, anharmonicity=_ANHARMONICITY, levels=4, label="q")
    r = Resonator(freq=_RESONATOR, levels=5, label="r")
    g = 0.3
    chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)], frame="rotating", approximation=Exact())
    start_time = 7.3
    problem = QuantumSequence(chip).build_problem(tlist=np.array([start_time, start_time + 1.0]))

    nq, nr = np.arange(4), np.arange(5)
    a, b = np.diag(np.sqrt(nq[1:]), 1), np.diag(np.sqrt(nr[1:]), 1)
    hamiltonian = (
        np.kron(np.diag(_QUBIT * nq + 0.5 * _ANHARMONICITY * nq * (nq - 1)), np.eye(5))
        + np.kron(np.eye(4), np.diag(_RESONATOR * nr))
        + g * np.kron(1j * (a - a.T), 1j * (b - b.T))
    )
    _, vectors = np.linalg.eigh(hamiltonian)
    ground = vectors[:, 0]  # every bare energy is nonnegative and g << Σ, so |00>'s partner is lowest
    frequencies = problem.resolved_frame.frequencies
    phases = np.exp(2j * np.pi * start_time * np.add.outer(frequencies["q"] * nq, frequencies["r"] * nr))
    expected = phases.reshape(-1) * ground
    # Both eigenvectors are accurate to about ε‖H‖/gap ≈ 2e-15 (‖H‖ ≈ 40 GHz, gap ≈ 5 GHz),
    # and the phase arguments (up to about 1.3e3 rad) to about 3e-13 rad.
    assert abs(np.vdot(expected, _ket(chip, problem.initial_state))) ** 2 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_default_ground_gradient_matches_finite_difference_beside_tied_spectators():
    """With two identical uncoupled spectators, jax.grad of the default start's leakage matches a central difference."""
    pytest.importorskip("dynamiqs")
    import jax

    g, step = 0.05, 1e-3
    # The spectators tie excited levels exactly, where differentiating eigh itself gives NaN.
    gradient = float(jax.grad(lambda x: _leakage(x, "dynamiqs", spectators=2))(g))
    reference = (_leakage(g + step) - _leakage(g - step)) / (2 * step)
    # The central difference errs by step² |L'''| / 6, with L''' ≈ 24 L₄ / g³ from the g⁴ term (about 1.4e-10);
    # leakage roundoff divided by 2 step stays below 1e-12.
    fourth = _perturbative_leakage(g)[1]
    assert abs(gradient - reference) <= 2 * step**2 / 6 * 24 * fourth / g**3 + 1e-12
