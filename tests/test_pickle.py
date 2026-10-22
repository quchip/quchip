"""Pickle round trips of chips, sequences, and their results."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from quchip import (
    VNA,
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Gaussian,
    PortNetwork,
    QuantumSequence,
    Resonator,
)


def _round_trip(value):
    return pickle.loads(pickle.dumps(value))


@pytest.fixture
def chip() -> Chip:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=50_000.0)
    r = Resonator(freq=7.0, levels=4, label="r")
    line = PortNetwork(label="line")
    line.expose("feed", at=line.port("feed", target=r, rate=2 * np.pi * 0.002))
    chip = Chip([q, r], [Capacitive(q, r, g=0.07, label="qr")], port_network=line, frame="rotating")
    xy = ChargeDrive(q, label="xy")
    chip.wire(xy)
    return chip


@pytest.fixture
def sequence(chip) -> QuantumSequence:
    seq = QuantumSequence(chip)
    seq.schedule("xy", envelope=Gaussian(duration=20.0, sigmas=4.0, amplitude=0.05), freq=chip.freq("q"))
    return seq


def test_chip_pickles_after_dressed_queries(chip) -> None:
    """A chip with filled caches pickles and gives the same dressed values after loading."""
    expected = (chip.freq("q"), chip.dispersive_shift("q", "r"))
    loaded = _round_trip(chip)
    assert [d.label for d in loaded.devices] == ["q", "r"]
    assert loaded.port_network.label == "line"
    np.testing.assert_allclose(loaded.freq("q"), expected[0], rtol=1e-12)
    np.testing.assert_allclose(loaded.dispersive_shift("q", "r"), expected[1], rtol=1e-10)


def test_sequence_pickles_and_simulates_the_same(sequence) -> None:
    """A loaded sequence keeps its schedule and simulates the same populations."""
    expected = sequence.simulate(duration=30.0).population("q", 1)
    loaded = _round_trip(sequence)
    np.testing.assert_allclose(loaded.simulate(duration=30.0).population("q", 1), expected, atol=1e-10)


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
def test_simulation_result_pickles(sequence, backend) -> None:
    """A simulation result loads with the same populations and final state on each backend."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    result = sequence.simulate(duration=30.0, backend=backend)
    loaded = _round_trip(result)
    np.testing.assert_allclose(loaded.population("q", 1), result.population("q", 1), atol=1e-12)
    np.testing.assert_allclose(loaded._backend.to_array(loaded.final_state),
                               result._backend.to_array(result.final_state), atol=1e-12)


def test_sparameter_result_pickles(chip) -> None:
    """S-parameter arrays and deferred diagnostics keep their values after loading."""
    result = VNA(chip).sweep(np.linspace(6.99, 7.01, 11))
    loaded = _round_trip(result)
    np.testing.assert_allclose(loaded.matrix, result.matrix, atol=1e-14)
    np.testing.assert_allclose(loaded.conjugate_matrix, result.conjugate_matrix, atol=1e-14)
    assert loaded.ports == result.ports
    for before, after in zip(result.diagnostics, loaded.diagnostics, strict=True):
        assert set(after) == set(before)
        np.testing.assert_allclose(after["condition_number"], before["condition_number"], rtol=1e-12)
        np.testing.assert_allclose(after["residual"], before["residual"], atol=1e-14)


def test_resonance_result_pickles(chip) -> None:
    """A loaded resonance result keeps its poles, residues, and pole-sum scattering."""
    result = VNA(chip).resonances()
    loaded = _round_trip(result)
    for name in ("frequencies", "linewidths", "poles", "residues", "composition"):
        np.testing.assert_allclose(getattr(loaded, name), getattr(result, name), atol=1e-12)
    assert loaded.mode_labels == result.mode_labels and loaded.ports == result.ports
    frequencies = np.linspace(6.99, 7.01, 5)
    np.testing.assert_allclose(loaded.scattering(frequencies), result.scattering(frequencies), atol=1e-12)
