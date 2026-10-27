"""A sequence returns its channel with collapse channels on the dressed states of a product of levels."""

import warnings

import numpy as np
import pytest
import qutip

from quchip import RWA, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence
from quchip.results.channel import ChannelResult
from quchip.results.propagator import PropagatorResult
from quchip.utils.warning_categories import QuchipWarning

TIGHT = {"atol": 1e-12, "rtol": 1e-10}
T1, T2 = 20e3, 15e3
GAMMA_1 = 1 / T1
GAMMA_PHI = 1 / T2 - 1 / (2 * T1)  # the 0-1 coherence decays at γ₁/2 + γ_φ
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
# On the charge drive, a real pulse turns about +Y: a π/2 pulse aims at exp(iπY/4) (see test_subspace_propagator).
SY = (np.eye(2) + 1j * Y) / np.sqrt(2)


def transmon(levels=3, t1=None, t2=None, backend=None):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=levels, label="q", T1=t1, T2=t2)
    chip = Chip([q], approximation=RWA(), frame="rotating", backend=backend)
    chip.wire(ChargeDrive(q, label="xy"))
    return chip


def half_pi(chip, duration=20.0, amplitude=None):
    envelope = Gaussian(duration=duration, sigmas=3.0)
    if amplitude is None:
        amplitude = float(chip.rotation_amplitude("q", "xy", envelope, np.pi / 2))
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=envelope.with_params({"amplitude": amplitude}), freq=5.0)
    return sequence


def record(superoperator, labels=((0,), (1,)), devices=("q",)):
    superoperator = np.asarray(superoperator, dtype=complex)
    return ChannelResult(superoperator=superoperator, labels=labels, devices=devices, leakage=None, frame="dressed",
                         frequencies={}, times=(0.0, 1.0), channels=(), batch=None)


def conjugation(unitary):
    return np.kron(np.conj(unitary), unitary)


def vec(matrix):
    return np.asarray(matrix).reshape(-1, order="F")


@pytest.mark.unit
def test_frame_tilt_counts_in_quchips_frame_but_not_in_the_spectral_score():
    """Rz(a)·U·Rz(a) on a π/2 gate: 1 − F_avg = 1 − (d + |Tr U†V|²)/(d(d + 1)), while the spectrum hardly moves."""
    angle = 8.66e-3
    rz = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
    gate = rz @ SY @ rz
    channel = record(conjugation(gate))
    expected = 1 - (2 + abs(np.trace(SY.conj().T @ gate)) ** 2) / 6
    assert expected > 2e-5
    np.testing.assert_allclose(channel.infidelity(SY), expected, rtol=0, atol=1e-15)
    # The frame tilt leaves the eigenvalues unchanged to first order in the angle.
    assert channel.eigenvalue_infidelity(SY) < 1e-3 * expected
    np.testing.assert_allclose(channel.average_leakage(), 0.0, rtol=0, atol=1e-15)


@pytest.mark.unit
def test_leaky_closed_channel_matches_the_propagator_fidelity():
    """Without collapse channels, the channel infidelity equals 1 − PropagatorResult.fidelity()."""
    rng = np.random.default_rng(5)
    matrix = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0] @ np.diag([1.0, np.sqrt(0.9)])
    propagator = PropagatorResult(matrix=matrix, labels=(), devices=(), leakage=None, frame="dressed", frequencies={},
                                  ground_energy=0.0, times=(0.0, 1.0), omitted_channels=(), batch=None)
    channel = record(conjugation(matrix))
    np.testing.assert_allclose(channel.infidelity(SY), 1 - propagator.fidelity(SY), rtol=0, atol=1e-15)
    np.testing.assert_allclose(channel.average_leakage(), 0.05, rtol=0, atol=1e-15)


@pytest.mark.unit
def test_pauli_rates_of_a_pauli_channel_with_loss():
    """A Pauli channel times a loss factor gives back its rates, which sum to f(I…I) = 1 − L, and its twirl."""
    rates = {"I": 0.9, "X": 0.05, "Y": 0.03, "Z": 0.02}
    kept = 0.99
    paulis = {"I": np.eye(2), "X": X, "Y": Y, "Z": Z}
    superoperator = kept * sum(rate * conjugation(paulis[name]) for name, rate in rates.items())
    channel = record(superoperator)
    found = channel.error(np.eye(2)).pauli_rates()
    np.testing.assert_allclose([found[name] for name in rates], [kept * rate for rate in rates.values()],
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(sum(found.values()), channel.pauli_fidelities()["I"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(channel.average_leakage(), 1 - kept, rtol=0, atol=1e-15)
    np.testing.assert_allclose(channel.twirl().superoperator, superoperator, rtol=0, atol=1e-15)
    # The error map after the gate gives Λ back when composed with the gate.
    gate = record(superoperator @ conjugation(SY))
    np.testing.assert_allclose(gate.error(SY).superoperator, superoperator, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="level above 1"):
        channel.seepage()
    with pytest.raises(ValueError, match="shape"):
        channel.infidelity(np.eye(3))


@pytest.mark.unit
def test_leakage_levels_keep_the_computational_block():
    """With a labeled level 2, the infidelity and Pauli rates use levels 0 and 1, and seepage reads level 2."""
    superoperator = np.zeros((9, 9), dtype=complex)
    superoperator[0, 0] = 1.0  # |0⟩⟨0| stays
    superoperator[4, 4] = 0.8  # |1⟩⟨1| keeps 0.8 ...
    superoperator[8, 4] = 0.2  # ... and leaks 0.2 to |2⟩⟨2|
    superoperator[4, 8] = 0.3  # |2⟩⟨2| seeps 0.3 into |1⟩⟨1|
    superoperator[8, 8] = 0.7
    channel = record(superoperator, labels=((0,), (1,), (2,)))
    np.testing.assert_allclose(channel.average_leakage(), 0.1, rtol=0, atol=1e-15)
    np.testing.assert_allclose(channel.seepage(), 0.3, rtol=0, atol=1e-15)
    error = channel.error(np.eye(2))
    assert error.labels == ((0,), (1,))
    np.testing.assert_allclose(error.leakage, [0.0, 0.2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(error.pauli_fidelities()["I"], 0.9, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="levels \\(0, 1\\)"):
        channel.ptm()


def test_channel_matches_direct_solves_and_the_closed_propagator():
    """Random inputs solved directly in the lab frame match the channel; without T1 and T2 it is conj(M) ⊗ M.

    The direct solves start at t₀ = 0, where the rotating frame is the lab frame, and use lab-frame dressed
    states from ``chip.state()``. The chip without collapse channels warns and points to subspace_propagator().
    """
    times = np.array([0.0, 20.0])
    chip = transmon(t1=T1, t2=T2)
    channel = half_pi(chip).subspace_channel("q", frame="lab", tlist=times, options=TIGHT)
    assert channel.channels == ("hidden.q.thermal_emission", "hidden.q.pure_dephasing")
    assert len(channel.batch) == 4 and channel.frame == "lab"
    backend = chip.backend
    dressed = np.stack([backend.to_array(chip.state({"q": level})).reshape(-1) for level in (0, 1)], axis=1)
    rng = np.random.default_rng(3)
    for _ in range(2):
        psi = rng.normal(size=2) + 1j * rng.normal(size=2)
        rho = np.outer(psi, psi.conj()) / np.vdot(psi, psi).real
        initial = qutip.Qobj(dressed @ rho @ dressed.conj().T, dims=[[3], [3]])
        result = half_pi(chip).simulate(tlist=times, initial_state=initial, options=TIGHT, states="final",
                                        partition=False)
        final = backend.to_array(result.state(frame="lab"))
        expected = dressed.conj().T @ final @ dressed
        np.testing.assert_allclose(channel.superoperator @ vec(rho), vec(expected), rtol=0, atol=1e-9)
    # The Choi matrix is Hermitian and positive.
    choi = channel.superoperator.reshape(2, 2, 2, 2, order="F").transpose(0, 2, 1, 3).reshape(4, 4)
    np.testing.assert_allclose(choi, choi.conj().T, rtol=0, atol=1e-9)
    assert np.linalg.eigvalsh((choi + choi.conj().T) / 2).min() > -1e-9
    # No input gains trace: the adjoint map gives Λ†(I) ≤ I.
    adjoint = (channel.superoperator.conj().T @ vec(np.eye(2))).reshape(2, 2, order="F")
    assert np.linalg.eigvalsh((adjoint + adjoint.conj().T) / 2).max() < 1 + 1e-9

    closed = transmon()
    propagator = np.asarray(half_pi(closed).subspace_propagator("q", tlist=times, options=TIGHT).matrix)
    with pytest.warns(QuchipWarning, match="no collapse channels"):
        unitary = half_pi(closed).subspace_channel("q", tlist=times, options=TIGHT)
    assert unitary.channels == ()
    np.testing.assert_allclose(unitary.superoperator, conjugation(propagator), rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="subspace_channel"):
        half_pi(closed).subspace_propagator("q", tlist=times, dissipation=True)


def test_decoherence_infidelity_matches_the_first_order_formula():
    """A π/2 gate with T1 and T2 against its closed propagator: 1 − F_avg = T(γ₁ + γ_φ)/3 to first order.

    Abad et al., PRL 129, 150504 (2022): 1 − F_avg = d/(2(d + 1)) · T Σ (γ₁ + γ_φ) for d = 2. The two-level
    transmon has no transient leakage, so the next order, of relative size γT, sets the tolerance.
    """
    duration = 20.0
    times = np.array([0.0, duration])
    target = np.asarray(half_pi(transmon(levels=2), duration).subspace_propagator(
        "q", tlist=times, options=TIGHT).matrix)
    channel = half_pi(transmon(levels=2, t1=T1, t2=T2), duration).subspace_channel("q", tlist=times, options=TIGHT)
    rate = GAMMA_1 + GAMMA_PHI
    np.testing.assert_allclose(channel.infidelity(target), duration * rate / 3, rtol=duration * rate, atol=0)


def test_idle_transmon_gives_the_exact_relaxation_channel_at_a_late_start():
    """An idle transmon from t₀ = 3.713 ns over D: the exact amplitude-damping and dephasing channel in each frame.

    ρ₁₁ → e^(−γ₁D) ρ₁₁ and ρ₀₁ → e^(−(γ₁/2 + γ_φ)D) ρ₀₁, times e^(2πifD) in the lab frame. The twirled rates
    are p_X = p_Y = (1 − e^(−γ₁D))/4 and p_Z = (1 + e^(−γ₁D) − 2e^(−(γ₁/2 + γ_φ)D))/4. Level 2 decays to
    level 1 at 2γ₁ and nothing reaches it, so it seeps 1 − e^(−2γ₁D) and the average leakage vanishes.
    """
    start, duration = 3.713, 200.037  # endpoint phases that are not real
    times = np.array([start, start + duration])
    chip = transmon(t1=T1, t2=T2)
    decay, coherence = np.exp(-GAMMA_1 * duration), np.exp(-(GAMMA_1 / 2 + GAMMA_PHI) * duration)
    channels = {}
    for frame, phase in (("dressed", 1.0), ("lab", np.exp(2j * np.pi * chip.freq("q") * duration))):
        levels = (0, 1, 2) if frame == "dressed" else (0, 1)
        channel = QuantumSequence(chip).subspace_channel("q", levels=levels, frame=frame, tlist=times, options=TIGHT,
                                                         max_workers=1)
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0], expected[0, 3], expected[3, 3] = 1.0, 1 - decay, decay
        expected[2, 2], expected[1, 1] = coherence * phase, coherence * np.conj(phase)
        block = channel.superoperator if frame == "lab" else channel.superoperator[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])]
        np.testing.assert_allclose(block, expected, rtol=0, atol=1e-9)
        channels[frame] = channel
    dressed = channels["dressed"]
    rates = dressed.error(np.eye(2)).pauli_rates()
    np.testing.assert_allclose([rates["X"], rates["Y"], rates["Z"]],
                               [(1 - decay) / 4, (1 - decay) / 4, (1 + decay - 2 * coherence) / 4], rtol=0, atol=1e-9)
    np.testing.assert_allclose(dressed.seepage(), 1 - np.exp(-2 * GAMMA_1 * duration), rtol=0, atol=1e-9)
    np.testing.assert_allclose(dressed.average_leakage(), 0.0, rtol=0, atol=1e-9)


@pytest.mark.validation
def test_pauli_frame_randomized_solves_give_the_twirled_error_map():
    """Direct solves with a random Pauli before the gate and its image after it average to the twirled error map.

    With Λ = E ∘ U and a Clifford U, (1/4) Σ_P P' Λ(P ρ P) P'† with P' = U P U† is the twirl of E.
    """
    times = np.array([0.0, 20.0])
    chip = transmon(t1=T1, t2=T2)
    channel = half_pi(chip).subspace_channel("q", frame="lab", tlist=times, options=TIGHT)
    lab = np.diag([1.0, np.exp(-2j * np.pi * chip.freq("q") * times[1])])  # dressed frame → lab frame at T
    target = lab @ SY
    twirled = channel.error(target).twirl()
    backend = chip.backend
    dressed = np.stack([backend.to_array(chip.state({"q": level})).reshape(-1) for level in (0, 1)], axis=1)
    psi = np.array([0.6, 0.8j])
    rho = np.outer(psi, psi.conj())
    average = np.zeros((2, 2), dtype=complex)
    for pauli in (np.eye(2), X, Y, Z):
        initial = qutip.Qobj(dressed @ pauli @ rho @ pauli @ dressed.conj().T, dims=[[3], [3]])
        result = half_pi(chip).simulate(tlist=times, initial_state=initial, options=TIGHT, states="final",
                                        partition=False)
        final = dressed.conj().T @ backend.to_array(result.state(frame="lab")) @ dressed
        after = target @ pauli @ target.conj().T
        average += after @ final @ after.conj().T / 4
    expected = twirled.superoperator @ vec(target @ rho @ target.conj().T)
    untwirled = channel.superoperator @ vec(rho)
    assert np.abs(untwirled - vec(average)).max() > 1e-4
    np.testing.assert_allclose(vec(average), expected, rtol=0, atol=1e-9)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_dynamiqs_channel_and_its_gradient():
    """dynamiqs gives the QuTiP channel, and d(1 − F_avg)/d(amplitude) matches central differences."""
    import jax

    dq = pytest.importorskip("dynamiqs")
    options = {"method": dq.method.Tsit5(rtol=1e-10, atol=1e-12)}
    times = np.array([0.0, 20.0])
    reference = half_pi(transmon(t1=T1, t2=T2)).subspace_channel("q", tlist=times, options=TIGHT)
    chip = transmon(t1=T1, t2=T2, backend="dynamiqs")
    amplitude = float(chip.rotation_amplitude("q", "xy", Gaussian(duration=20.0, sigmas=3.0), np.pi / 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuchipWarning)
        channel = half_pi(chip, amplitude=amplitude).subspace_channel("q", tlist=times, options=options)
    np.testing.assert_allclose(channel.superoperator, reference.superoperator, rtol=0, atol=1e-8)

    def infidelity(value):
        return half_pi(chip, amplitude=value).subspace_channel("q", tlist=times, options=options).infidelity(SY)

    step = 1e-5 * amplitude
    derivative = jax.grad(infidelity)(amplitude)
    finite = (infidelity(amplitude + step) - infidelity(amplitude - step)) / (2 * step)
    assert abs(finite) > 1e-3
    np.testing.assert_allclose(derivative, finite, rtol=1e-5, atol=0)
