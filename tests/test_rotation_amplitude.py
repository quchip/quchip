"""The amplitude for a rotation angle follows θ = 2π|Σ m H||A a| on a resonant two-level transition."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from quchip import (RWA, ChargeDrive, Chip, Crosstalk, Delay, DuffingTransmon, Gaussian, GaussianDRAG, PhaseDrive,
                    QuantumSequence, Square)
from quchip.control.signal import Gain, SignalTransform
from quchip.extensions.envelopes import CosineEnvelope

TIGHT = {"atol": 1e-12, "rtol": 1e-10}


def transmon(levels=2, backend=None):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=levels, label="q")
    chip = Chip([q], approximation=RWA(), frame="rotating", backend=backend)
    chip.wire(ChargeDrive(q, label="xy"))
    return chip


def excited_population(chip, envelope, options=TIGHT):
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=envelope, freq=chip.freq("q"))
    duration = float(envelope.duration)
    result = sequence.simulate(np.array([0.0, duration]), options=options)
    return float(np.asarray(result.population("q", 1))[-1])


@pytest.mark.parametrize("envelope", [
    Square(duration=20.0), Gaussian(duration=20.0, sigmas=3.0), GaussianDRAG(duration=20.0, sigmas=3.0, beta=0.0),
], ids=lambda envelope: type(envelope).__name__)
def test_amplitude_for_pi_and_half_pi(envelope):
    """#197: the amplitude for θ = π inverts the transmon, and θ = π/2 gives P1 = 1/2."""
    chip = transmon()
    for angle, population in ((np.pi, 1.0), (np.pi / 2, 0.5)):
        amplitude = float(chip.rotation_amplitude("q", "xy", envelope, angle))
        # QuTiP interpolates a windowed envelope linearly between samples, about 1e-7 in P1 here.
        np.testing.assert_allclose(excited_population(chip, envelope.with_params({"amplitude": amplitude})),
                                   population, rtol=0, atol=1e-6)
    # |m| = 1, so a π pulse needs A a = 1/2; the Gaussian area per unit amplitude is σ√(2π) erf(3/√2).
    if isinstance(envelope, Square):
        np.testing.assert_allclose(chip.rotation_amplitude("q", "xy", envelope), 0.5 / 20.0, rtol=1e-12)


@pytest.mark.unit
def test_unsupported_transforms_and_envelopes_raise():
    """A transform without a declared transfer, an envelope without amplitude, or a forbidden transition raises."""
    chip = transmon(levels=3)
    with pytest.raises(ValueError, match="cannot drive this transition"):
        chip.rotation_amplitude(({"q": 0}, {"q": 2}), "xy", Square(duration=20.0))

    class Fixed(CosineEnvelope):
        def parameter_values(self):
            return {"duration": self.duration}

    with pytest.raises(TypeError, match="no amplitude field"):
        chip.rotation_amplitude("q", "xy", Fixed(duration=20.0))

    class Clip(SignalTransform):
        def apply(self, signals):
            return signals

        def referenced_lines(self):
            return ("xy",)

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    clipped = Chip([q], approximation=RWA())
    clipped.wire(ChargeDrive(q, label="xy"), signal_chain=[Clip()])
    with pytest.raises(ValueError, match="Clip acts on the signal of line 'xy'.*linear_time_invariant"):
        clipped.rotation_amplitude("q", "xy", Square(duration=20.0))


def two_lines(*transforms, second=ChargeDrive):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], approximation=RWA(), frame="rotating")
    chip.wire(ChargeDrive(q, label="xy"), second(q, label="xy2"), signal_chain=list(transforms))
    return chip


@pytest.mark.unit
def test_the_seed_passes_gain_and_adds_crosstalk_paths_as_phasors():
    """#218: a gain divides the seed by |g|. With m = i on a charge line and m = 1 on a phase line, a crosstalk
    transfer of -i cancels the drive (i - i = 0) and i doubles it; conjugating the transfer would swap the two."""
    bare = float(two_lines().rotation_amplitude("q", "xy", Square(duration=20.0)))
    gained = two_lines(Gain("xy", 0.5j)).rotation_amplitude("q", "xy", Square(duration=20.0))
    np.testing.assert_allclose(gained, bare / 0.5, rtol=1e-12)
    np.testing.assert_allclose(two_lines(Gain("xy2", 3.0)).rotation_amplitude("q", "xy", Square(duration=20.0)), bare,
                               rtol=1e-12)
    with pytest.raises(ValueError, match="cannot drive this transition"):
        two_lines(Crosstalk("xy", "xy2", beta=1.0, theta=-np.pi / 2), second=PhaseDrive).rotation_amplitude(
            "q", "xy", Square(duration=20.0))
    doubled = two_lines(Crosstalk("xy", "xy2", beta=1.0, theta=np.pi / 2), second=PhaseDrive)
    np.testing.assert_allclose(doubled.rotation_amplitude("q", "xy", Square(duration=20.0)), bare / 2, rtol=1e-12)


def test_a_seed_through_a_delay_and_crosstalk_inverts_the_transmon():
    """A delayed line and a delayed, phase-turned crosstalk path drive one transmon; the seed's π pulse inverts it
    to the accuracy of the bare-line seed on three levels (P1 = 0.98985)."""
    envelope = Gaussian(duration=20.0)
    chip = two_lines(Delay("xy", 0.07), Crosstalk("xy", "xy2", beta=0.3, theta=0.4, delay=0.13))
    amplitude = float(chip.rotation_amplitude("q", "xy", envelope))
    bare = float(two_lines().rotation_amplitude("q", "xy", envelope))
    # The crosstalk path adds 0.3 e^{i(0.4 + 2π f 0.13)} to the direct transfer of one.
    phase = 0.4 + 2 * np.pi * float(chip.freq("q")) * 0.13
    np.testing.assert_allclose(amplitude, bare / abs(1 + 0.3 * np.exp(1j * phase)), rtol=1e-9)
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=envelope.with_params({"amplitude": amplitude}), freq=chip.freq("q"))
    result = sequence.simulate(np.array([0.0, 20.5]), options=TIGHT)
    assert abs(float(np.asarray(result.population("q", 1))[-1]) - 0.98985) < 5e-3


@pytest.mark.validation
@pytest.mark.optional_backend
def test_drag_quadrature_changes_the_rotation_as_an_independent_integration_predicts():
    """With β = 1 ns the π amplitude no longer inverts the transition. P1 matches a SciPy integration of
    H̃/ħ = π(s m σ₊ + s* m* σ₋), with m = i and s = I + iβ dI/dt. dynamiqs evaluates the envelope exactly."""
    dq = pytest.importorskip("dynamiqs")
    chip = transmon(backend="dynamiqs")
    duration, beta = 20.0, 1.0
    amplitude = float(chip.rotation_amplitude("q", "xy", Gaussian(duration=duration, sigmas=3.0)))
    envelope = GaussianDRAG(duration=duration, sigmas=3.0, beta=beta, amplitude=amplitude)
    m = complex(np.asarray(chip.drive_matrix_elements("q")["xy"]))

    def rhs(t, psi):
        s = complex(np.asarray(envelope.value(np.asarray(t))))
        # Rows are |0⟩ and |1⟩, and σ₊ = |1⟩⟨0|.
        h = np.pi * np.array([[0, np.conj(s * m)], [s * m, 0]])
        return -1j * (h @ psi)

    final = solve_ivp(rhs, (0.0, duration), np.array([1, 0], dtype=complex), method="DOP853",
                      rtol=1e-12, atol=1e-13).y[:, -1]
    expected = abs(final[1]) ** 2
    assert expected < 0.95
    population = excited_population(chip, envelope, {"method": dq.method.Tsit5(rtol=1e-12, atol=1e-13)})
    np.testing.assert_allclose(population, expected, rtol=0, atol=1e-9)
