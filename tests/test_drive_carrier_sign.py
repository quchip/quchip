"""RWA keeps the co-rotating sideband for either carrier sign; ChargePhaseDrive drives at a positive carrier."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import RWA, ChargeDrive, Chip, DuffingTransmon, Exact, Gaussian, QuantumSequence, Qubit, Square
from quchip.engine.approximations import resolve_drive_program
from quchip.engine.ir import Carrier, Constant, Multiply, RealPart, evaluate_signal_program
from quchip.extensions import ChargePhaseDrive

TIMES = np.array([0.0, 0.03, 0.11, 0.4])


def _kept(freq, xp=np):
    """Return the RWA signal of Re(c e^{-2πi f t}) on a raising band in a 5 GHz frame."""
    program = RealPart(Multiply((Constant(0.3 + 0.1j), Carrier(freq=2 * np.pi * freq, sign=-1))))
    resolved = resolve_drive_program(RWA(), program, weight=1, frame_frequency=5.0, has_carrier=True)
    return evaluate_signal_program(resolved, xp.asarray(TIMES), xp=xp)


@pytest.mark.unit
@pytest.mark.parametrize("freq, expected", [
    (5.0, np.full(TIMES.shape, (0.3 - 0.1j) / 2)),
    (-5.0, np.full(TIMES.shape, (0.3 + 0.1j) / 2)),
    (0.0, (0.3 - 0.1j) / 2 * np.exp(-2j * np.pi * 5.0 * TIMES)),
])
def test_rwa_keeps_the_co_rotating_band_for_a_concrete_or_traced_carrier(freq, expected):
    """cos(-ωt + φ) = cos(ωt - φ): a negative carrier drives with the conjugate phase. A zero carrier keeps one half."""
    np.testing.assert_allclose(_kept(freq), expected, atol=1e-15)
    np.testing.assert_allclose(jax.jit(lambda f: _kept(f, jnp))(freq), expected, atol=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize("freq, partners", [(5.0, [10.0, 10.0]), (-5.0, [10.0, 10.0]), (0.0, [5.0, 5.0])])
def test_rwa_records_the_dropped_partner_at_the_carrier_plus_the_transition(freq, partners):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", approximation=RWA())
    chip.wire(ChargeDrive(q, label="xy"))
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Gaussian(duration=20.0, amplitude=0.02), freq=freq)
    dropped = [float(term.frame_frequency) for term in sequence.resolve().dropped_terms if "partner" in term.operator]
    assert dropped == pytest.approx(partners)


def _final_population(drive_class, freq, amplitude, approximation):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", approximation=approximation)
    chip.wire(drive_class(q, label="xy"))
    sequence = QuantumSequence(chip)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sequence.schedule("xy", envelope=Gaussian(duration=20.0, amplitude=amplitude), freq=freq)
        result = sequence.simulate(tlist=[0.0, 20.0], states="final")
    return float(np.real(result.population("q", 1)[-1]))


@pytest.mark.parametrize("approximation", [RWA(), Exact()], ids=["rwa", "exact"])
def test_charge_phase_drive_takes_half_the_charge_drive_pi_amplitude_at_a_positive_carrier(approximation):
    """The I/Q drive is i(s a† - s* a): the resonant sideband at twice ChargeDrive's rate, and nothing at -f."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating")
    chip.wire(ChargeDrive(q, label="xy"))
    pi = float(chip.rotation_amplitude(q, "xy", Gaussian(duration=20.0)))
    reference = _final_population(ChargeDrive, 5.0, pi, approximation)
    assert reference > 0.98
    assert _final_population(ChargePhaseDrive, "auto", pi / 2, approximation) == pytest.approx(reference, abs=1e-5)
    assert _final_population(ChargePhaseDrive, -5.0, pi / 2, approximation) < 1e-6
    if isinstance(approximation, RWA):
        assert _final_population(ChargeDrive, -5.0, pi, approximation) == pytest.approx(reference, abs=1e-6)


def test_negative_carrier_rabi_oscillation_matches_the_analytic_two_level_curve():
    """On a two-level qubit at -f under RWA, H = (A/2) Q gives P1 = sin^2(π A t)."""
    q = Qubit(freq=5.0, label="q")
    chip = Chip([q], frame="rotating", approximation=RWA())
    chip.wire(ChargeDrive(q, label="xy"))
    amplitude, times = 0.01, np.linspace(0.0, 60.0, 13)
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Square(duration=60.0, amplitude=amplitude), freq=-5.0, phase=0.7)
    result = sequence.simulate(tlist=times, options={"atol": 1e-12, "rtol": 1e-10})
    np.testing.assert_allclose(np.real(result.population("q", 1)), np.sin(np.pi * amplitude * times) ** 2, atol=1e-8)
