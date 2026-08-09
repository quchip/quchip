from __future__ import annotations

import pytest

from quchip import Capacitive, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence, Resonator


def _chip() -> Chip:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    return Chip([q, r], [Capacitive(q, r, g=0.02, label="qr")])


def test_chip_parameters_and_settings_separate_values_from_structure() -> None:
    chip = _chip()

    assert dict(chip.parameters) == {
        "q.freq": 5.0,
        "q.anharmonicity": -0.2,
        "r.freq": 7.0,
        "qr.g": 0.02,
    }
    assert chip.settings["devices"] == (("q", "DuffingTransmon", 3), ("r", "Resonator", 4))
    with pytest.raises(TypeError):
        chip.parameters["q.freq"] = 5.1


def test_chip_with_params_is_immutable_and_rebinds_multiple_component_kinds() -> None:
    chip = _chip()

    rebound = chip.with_params({"q.freq": 5.1, "qr.g": 0.03})

    assert chip.parameters["q.freq"] == 5.0
    assert chip.parameters["qr.g"] == 0.02
    assert rebound.parameters["q.freq"] == 5.1
    assert rebound.parameters["qr.g"] == 0.03
    with pytest.raises(KeyError, match="Available"):
        chip.with_params({"q.missing": 1.0})


def test_sequence_parameters_rebind_chip_and_pulse_values_directly() -> None:
    chip = _chip()
    drive = ChargeDrive(chip["q"], label="xy")
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3.0),
        freq=5.0,
    )

    rebound = sequence.with_params({"q.freq": 5.1, "pulse.0.amplitude": 0.04})

    assert sequence.parameters["q.freq"] == 5.0
    assert sequence.parameters["pulse.0.amplitude"] == 0.02
    assert rebound.parameters["q.freq"] == 5.1
    assert rebound.parameters["pulse.0.amplitude"] == 0.04
    assert rebound.settings["entries"] == ("PulseEntry",)


def test_chip_with_params_is_differentiable_on_dynamiqs() -> None:
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip.backend.dynamiqs import DynamiqsBackend

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    chip = Chip([q], backend=DynamiqsBackend())

    def first_transition(freq):
        rebound = chip.with_params({"q.freq": freq})
        energies = jnp.linalg.eigvalsh(rebound.hamiltonian().matrix(backend=chip.backend))
        return energies[1] - energies[0]

    assert jax.grad(first_transition)(5.0) == pytest.approx(1.0)
