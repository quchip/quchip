"""Explicit engine approximation strategies."""

from __future__ import annotations

import numpy as np
import pytest

import quchip
from quchip import Capacitive, Chip, DuffingTransmon, Exact, QuantumSequence, RWA
from quchip.engine.assembly import build_engine_result


def _coupled_chip(approximation):
    first = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="a")
    second = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=2, label="b")
    coupling = Capacitive(first, second, g=0.03, label="ab")
    return Chip(
        [first, second],
        couplings=[coupling],
        frame="lab",
        approximation=approximation,
    )


def test_exact_and_rwa_are_explicit_chip_strategies():
    exact = _coupled_chip(Exact())
    reduced = _coupled_chip(RWA())

    exact_matrix = np.asarray(exact.resolve().hamiltonian().matrix(t=0.0))
    reduced_result = reduced.resolve()
    reduced_matrix = np.asarray(reduced_result.hamiltonian().matrix(t=0.0))

    assert exact.approximation == Exact()
    assert reduced.approximation == RWA()
    assert exact_matrix[0, 3] == pytest.approx(0.03)
    assert reduced_matrix[0, 3] == pytest.approx(0.0)
    assert {term.band_weights for term in reduced_result.dropped_terms} == {
        (-1, -1),
        (1, 1),
    }


def test_resolution_override_does_not_mutate_chip_default():
    chip = _coupled_chip(RWA())

    exact = chip.resolve(approximation=Exact())
    default = chip.resolve()

    assert exact.approximation == Exact()
    assert default.approximation == RWA()
    assert chip.approximation == RWA()
    assert np.asarray(exact.hamiltonian().matrix(t=0.0))[0, 3] == pytest.approx(0.03)
    assert np.asarray(default.hamiltonian().matrix(t=0.0))[0, 3] == pytest.approx(0.0)


def test_sequence_problem_override_uses_one_strategy_without_mutation():
    chip = _coupled_chip(RWA())
    sequence = QuantumSequence(chip)

    exact = sequence.build_problem(
        tlist=np.asarray([0.0, 1.0]),
        approximation=Exact(),
    )
    default = sequence.build_problem(tlist=np.asarray([0.0, 1.0]))

    assert exact.engine_result.approximation == Exact()
    assert default.engine_result.approximation == RWA()
    assert chip.approximation == RWA()


def test_only_explicit_exact_and_rwa_strategies_are_public():
    first = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="a")
    second = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=2, label="b")

    with pytest.raises(TypeError, match="rwa"):
        Chip([first, second], rwa=True)
    with pytest.raises(TypeError, match="rwa"):
        Capacitive(first, second, g=0.03, rwa=True)

    coupling = Capacitive(first, second, g=0.03)
    chip = Chip([first, second], couplings=[coupling])
    assert quchip.Exact is Exact
    assert quchip.RWA is RWA
    assert not hasattr(quchip, "SpectralRWA")
    assert not hasattr(quchip, "BandRWA")

    resolved_frame = chip.resolve().resolved_frame
    with pytest.raises(TypeError, match="approximation"):
        build_engine_result(chip, [], resolved_frame=resolved_frame, approximation=True)


@pytest.mark.parametrize("field", ["rwa", "approximation"])
def test_old_boolean_serialization_is_rejected(field):
    chip = _coupled_chip(RWA())
    payload = chip.to_dict()
    payload[field] = True

    with pytest.raises(TypeError, match=field):
        Chip.from_dict(payload)


def test_serialization_requires_explicit_approximation_strategy():
    payload = _coupled_chip(RWA()).to_dict()
    del payload["approximation"]

    with pytest.raises(TypeError, match="approximation"):
        Chip.from_dict(payload)


def test_custom_strategy_serialization_fails_closed():
    class CustomApproximation(Exact):
        pass

    chip = _coupled_chip(CustomApproximation())

    with pytest.raises(TypeError, match="stable approximation serialization"):
        chip.to_dict()


def test_rwa_explicit_bands_replace_the_default_selection_and_round_trip():
    strategy = RWA(keep_bands={(1, 1)})
    chip = _coupled_chip(strategy)

    matrix = np.asarray(chip.resolve().hamiltonian().matrix(t=0.0))

    assert matrix[0, 3] == pytest.approx(0.03)
    assert matrix[1, 2] == pytest.approx(0.0)
    assert Chip.from_dict(chip.to_dict()).approximation == strategy

    drive = quchip.ChargeDrive(chip.devices[0])
    driven = Chip(
        [chip.devices[0]],
        approximation=RWA(keep_bands={(0,)}),
        control_equipment=quchip.ControlEquipment([drive]),
    )
    sequence = QuantumSequence(driven)
    sequence.schedule(drive, envelope=quchip.Square(duration=1.0, amplitude=0.01), freq=5.0)
    assert not [term for term in sequence.resolve().dynamic_terms if term.origin == "drive"]


def test_rwa_rejects_invalid_explicit_band_sets():
    with pytest.raises(TypeError, match="integer tuples"):
        RWA(keep_bands={(0, 1.0)})
    with pytest.raises(ValueError, match="at least one"):
        RWA(keep_bands=set())
