"""The amplitude for a rotation angle follows θ = 2π|m||A a| on a resonant two-level transition."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from quchip import RWA, ChargeDrive, Chip, DuffingTransmon, Gaussian, GaussianDRAG, QuantumSequence, Square
from quchip.control.signal import Gain
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
def test_unsupported_lines_and_envelopes_raise():
    """A line with a signal-chain transform, an envelope without amplitude, or a forbidden transition raises."""
    chip = transmon(levels=3)
    with pytest.raises(ValueError, match="cannot drive this transition"):
        chip.rotation_amplitude(({"q": 0}, {"q": 2}), "xy", Square(duration=20.0))

    class Fixed(CosineEnvelope):
        def parameter_values(self):
            return {"duration": self.duration}

    with pytest.raises(TypeError, match="no amplitude field"):
        chip.rotation_amplitude("q", "xy", Fixed(duration=20.0))
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    gained = Chip([q], approximation=RWA())
    line = ChargeDrive(q, label="xy")
    gained.wire(line, signal_chain=[Gain(line, 2.0)])
    with pytest.raises(ValueError, match="signal-chain transforms"):
        gained.rotation_amplitude("q", "xy", Square(duration=20.0))


@pytest.mark.validation
@pytest.mark.optional_backend
def test_drag_quadrature_changes_the_rotation_as_an_independent_integration_predicts():
    """With β = 1 ns the π amplitude no longer inverts the transition. P1 matches a SciPy integration of
    H̃/ħ = π(s m σ₊ + s* m* σ₋), with m = −i and s = I + iβ dI/dt. dynamiqs evaluates the envelope exactly."""
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
