"""Explicit engine approximation strategies."""

from __future__ import annotations

import numpy as np
import pytest

import quchip
from quchip import Capacitive, Chip, DuffingTransmon, Exact, QuantumSequence, RWA


pytestmark = pytest.mark.unit


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


def test_resolution_override_does_not_mutate_chip_default():
    chip = _coupled_chip(RWA())

    exact = chip.resolve(approximation=Exact())
    default = chip.resolve()

    assert exact.approximation == Exact()
    assert default.approximation == RWA()
    assert chip.approximation == RWA()
    # The charge operators i(a - a†) give the counter-rotating element -g.
    assert np.asarray(exact.hamiltonian().matrix(t=0.0))[0, 3] == pytest.approx(-0.03)
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


def test_rwa_explicit_bands_replace_the_default_selection_and_round_trip():
    strategy = RWA(keep_bands={(1, 1)})
    chip = _coupled_chip(strategy)

    matrix = np.asarray(chip.resolve().hamiltonian().matrix(t=0.0))

    assert matrix[0, 3] == pytest.approx(-0.03)
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


def test_excitation_number_conservation_follows_the_retained_bands():
    """Only approximations whose retained bands all have zero total weight conserve excitation number."""
    assert RWA().conserves_excitation_number()
    assert RWA(keep_bands={(1, -1), (-1, 1)}).conserves_excitation_number()
    assert not RWA(keep_bands={(1, -1), (1, 1)}).conserves_excitation_number()
    assert not Exact().conserves_excitation_number()
