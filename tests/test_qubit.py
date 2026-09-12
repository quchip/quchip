"""Physical contracts for the ideal two-level device."""

import numpy as np
import pytest

from quchip import ChargeDrive, Chip, Qubit, QuantumSequence, Square


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_qubit_frequency_frame_and_round_trip(backend):
    """A qubit retains its transition and noise when rebound and serialized."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    q = Qubit(freq=5.0, T1=100.0, T2=80.0, label="q")
    chip = Chip([q], backend=backend, frame="rotating")
    assert float(chip.freq(q)) == pytest.approx(5.0)
    np.testing.assert_allclose(chip.hamiltonian().matrix(), 0, atol=1e-12)
    restored = Chip.from_dict(chip.with_params({"q.freq": 5.2}).to_dict())
    assert isinstance(restored["q"], Qubit)
    assert restored["q"].T2 == 80.0
    assert float(restored.freq("q")) == pytest.approx(5.2)
    assert q.freq == 5.0
    assert q.truncation_boundary() is None


def test_qubit_noise_matches_relaxation_and_coherence_times():
    """The declared T1 and total T2 give their analytic population and coherence decays."""
    q = Qubit(freq=5.0, T1=100.0, T2=80.0, label="q")
    chip = Chip([q], frame="rotating")
    plus = (chip.bare_state({q: 0}) + chip.bare_state({q: 1})) / np.sqrt(2)
    t = np.linspace(0, 200, 101)
    result = QuantumSequence(chip).simulate(
        tlist=t, initial_state=plus, e_ops=chip.e_ops(q=[q.sigma_x, q.sigma_z]),
        options={"atol": 1e-10, "rtol": 1e-10},
    )
    np.testing.assert_allclose(result.expect(q, 0), np.exp(-t / 80), atol=2e-9)
    np.testing.assert_allclose(result.expect(q, 1), 1 - np.exp(-t / 100), atol=2e-9)


def test_qubit_supports_a_resonant_pi_pulse():
    """The standard charge drive inverts an ideal qubit with a resonant pi-area pulse."""
    q = Qubit(freq=5.0, label="q")
    chip = Chip([q], frame="rotating")
    drive = ChargeDrive(q, label="xy")
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Square(duration=20, amplitude=0.025), freq=q.freq)
    result = sequence.simulate(tlist=[0, 20], initial_state={q: 0})
    assert result.population(q, 1)[-1] == pytest.approx(1, abs=1e-7)


@pytest.mark.parametrize("levels", [3, 4, 5])
def test_qubit_rejects_extra_levels(levels):
    """Construction and mutation cannot turn a qubit into an oscillator."""
    with pytest.raises(ValueError, match="levels=2"):
        Qubit(freq=5.0, levels=levels)
    q = Qubit(freq=5.0, label="q")
    with pytest.raises(ValueError, match="levels=2"):
        q.levels = levels
    assert q.levels == 2
