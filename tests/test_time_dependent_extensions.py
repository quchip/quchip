"""Executable reference models for component-owned time dependence."""

from __future__ import annotations

from quchip.approximations import Exact, RWA

import numpy as np

from quchip import Chip, DuffingTransmon
from quchip.declarative.models import _symbolic_parameters
from quchip.declarative.ops import LocalOps
from quchip.engine.ir import evaluate_signal_program
from quchip.extensions import FrequencyModulatedMode, ModulatedCapacitive
from quchip.utils.constants import TWO_PI


def test_frequency_modulated_mode_resolves_device_time_term() -> None:
    mode = FrequencyModulatedMode(
        frequency=5.0,
        modulation_amplitude=0.2,
        modulation_frequency=0.25,
        levels=3,
        label="m",
    )

    result = Chip([mode], frame="lab", approximation=Exact()).resolve()

    device_terms = [term for term in result.dynamic_terms if term.origin == "device"]
    assert len(device_terms) == 1
    assert device_terms[0].operator.subsystem_labels == ("m",)


def test_time_term_coefficient_uses_symbolic_model_fields() -> None:
    mode = FrequencyModulatedMode(
        frequency=5.0,
        modulation_amplitude=0.2,
        modulation_frequency=0.25,
        levels=3,
        label="m",
    )
    op = LocalOps(label=mode.label, space=mode.local_space(), device=mode)

    authored = mode.time_terms(op, _symbolic_parameters(mode))[0]

    assert authored.coefficient.amplitude.parameter_paths() == ("m.modulation_amplitude",)
    assert authored.coefficient.frequency.parameter_paths() == ("m.modulation_frequency",)

    result = Chip([mode], frame="lab", approximation=Exact()).resolve()
    term = next(term for term in result.dynamic_terms if term.origin == "device")
    assert evaluate_signal_program(term.time_dependence.signal, 0.0, xp=np) == 0.2


def test_frequency_modulated_mode_hamiltonian_matches_analytic_diagonal() -> None:
    mode = FrequencyModulatedMode(5.0, 0.2, 0.25, levels=3, label="m")
    result = Chip([mode], frame="lab", approximation=Exact()).resolve()
    time = 0.5
    expected = np.diag(np.arange(3) * (5.0 + 0.2 * np.cos(2.0 * np.pi * 0.25 * time)))

    np.testing.assert_allclose(result.hamiltonian().matrix(t=time), expected, atol=1e-11)


def test_frequency_modulated_mode_uses_fixed_projected_basis() -> None:
    mode = FrequencyModulatedMode(5.0, 0.2, 0.25, levels=3, label="m")

    result = Chip([mode], basis="eigen", frame="lab", approximation=Exact()).resolve()

    assert result.bases["m"].resolved_dim == 3
    np.testing.assert_allclose(result.bases["m"].vectors, np.eye(3), atol=1e-12)


def test_modulated_capacitive_resolves_without_a_drive_line() -> None:
    first = DuffingTransmon(5.0, -0.25, levels=3, label="a")
    second = DuffingTransmon(5.2, -0.25, levels=3, label="b")
    coupling = ModulatedCapacitive(
        first,
        second,
        static_strength=0.01,
        modulation_amplitude=0.02,
        modulation_frequency=0.2,
        label="mc",
    )
    chip = Chip([first, second], [coupling], frame="lab", approximation=Exact())

    result = chip.resolve()

    coupling_terms = [term for term in result.dynamic_terms if term.origin == "coupling"]
    assert coupling_terms
    assert chip.control_equipment is None


def _dynamic_hamiltonian(result: object, time: float) -> np.ndarray:
    terms = [term for term in result.dynamic_terms if term.origin == "coupling"]
    return sum(
        (
            np.asarray(term.operator.to_dense())
            / TWO_PI
            * evaluate_signal_program(term.time_dependence.signal, time, xp=np)
        )
        for term in terms
    )


def test_modulated_capacitive_rwa_keeps_only_exchange_bands() -> None:
    first = DuffingTransmon(5.0, -0.25, levels=2, label="a")
    second = DuffingTransmon(5.2, -0.25, levels=2, label="b")
    coupling = ModulatedCapacitive(
        first,
        second,
        static_strength=0.0,
        modulation_amplitude=0.02,
        modulation_frequency=0.2,
        label="mc",
    )
    lowering = np.asarray([[0.0, 1.0], [0.0, 0.0]])
    raising = lowering.T

    full = Chip([first, second], [coupling], frame="lab", approximation=Exact()).resolve()
    rotating_wave = Chip([first, second], [coupling], frame="lab", approximation=RWA()).resolve()

    np.testing.assert_allclose(
        _dynamic_hamiltonian(full, 0.0),
        0.02 * np.kron(lowering + raising, lowering + raising),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _dynamic_hamiltonian(rotating_wave, 0.0),
        0.02 * (np.kron(lowering, raising) + np.kron(raising, lowering)),
        atol=1e-12,
    )
    assert {term.band_weights for term in rotating_wave.dropped_terms} >= {
        (-1, -1),
        (1, 1),
    }
