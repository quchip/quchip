"""Physics sentinels for component-owned time-dependent models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import Chip, QuantumSequence
from quchip.extensions import FrequencyModulatedMode


def _level_one_energy(amplitude: float, frequency: float, time: float, phase: float) -> jax.Array:
    mode = FrequencyModulatedMode(
        frequency=0.4,
        modulation_amplitude=amplitude,
        modulation_frequency=frequency,
        modulation_phase=phase,
        levels=2,
        label="m",
    )
    chip = Chip([mode], frame="lab", backend="dynamiqs")
    return jnp.real(chip.hamiltonian().matrix(t=time, backend=chip.backend)[1, 1])


def test_frequency_coefficient_gradients_match_closed_form_and_finite_difference() -> None:
    pytest.importorskip("dynamiqs")
    amplitude = 0.07
    frequency = 0.13
    time = 0.83
    phase = 0.21
    expected_amplitude = np.cos(2.0 * np.pi * frequency * time + phase)
    expected_frequency = (
        -amplitude
        * 2.0
        * np.pi
        * time
        * np.sin(2.0 * np.pi * frequency * time + phase)
    )

    grad_amplitude = jax.grad(_level_one_energy, argnums=0)(amplitude, frequency, time, phase)
    grad_frequency = jax.grad(_level_one_energy, argnums=1)(amplitude, frequency, time, phase)
    step = 1e-5
    finite_difference = (
        _level_one_energy(amplitude, frequency + step, time, phase)
        - _level_one_energy(amplitude, frequency - step, time, phase)
    ) / (2.0 * step)

    assert grad_amplitude == pytest.approx(expected_amplitude, rel=1e-10, abs=1e-10)
    assert grad_frequency == pytest.approx(expected_frequency, rel=1e-10, abs=1e-10)
    assert grad_frequency == pytest.approx(float(finite_difference), rel=1e-7, abs=1e-8)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_commuting_device_time_evolution_matches_exact_phase(backend: str) -> None:
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    frequency = 0.4
    amplitude = 0.03
    modulation_frequency = 0.07
    phase = 0.2
    duration = 3.0
    mode = FrequencyModulatedMode(
        frequency=frequency,
        modulation_amplitude=amplitude,
        modulation_frequency=modulation_frequency,
        modulation_phase=phase,
        levels=2,
        label="m",
    )
    chip = Chip([mode], frame="lab", backend=backend)
    initial = chip.superposition({mode: 0}, {mode: 1})

    result = QuantumSequence(chip).simulate(
        tlist=np.linspace(0.0, duration, 121),
        initial_state=initial,
        options={"atol": 1e-11, "rtol": 1e-10, "nsteps": 1_000_000},
        check_truncation=False,
    )

    accumulated_phase = 2.0 * np.pi * (
        frequency * duration
        + amplitude
        / (2.0 * np.pi * modulation_frequency)
        * (
            np.sin(2.0 * np.pi * modulation_frequency * duration + phase)
            - np.sin(phase)
        )
    )
    state = np.asarray(chip.backend.to_array(result.final_state)).reshape(-1)
    relative_phase = state[1] / state[0]
    expected = np.exp(-1j * accumulated_phase)
    assert complex(relative_phase) == pytest.approx(expected, rel=5e-6, abs=5e-7)
