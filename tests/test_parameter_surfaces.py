from __future__ import annotations

import numpy as np
import pytest

from quchip import (
    Bath,
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    Crosstalk,
    DuffingTransmon,
    Gaussian,
    QuantumSequence,
    Resonator,
    Square,
)


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


def test_sequence_parameters_rebind_chip_and_pulse_values_directly(monkeypatch) -> None:
    chip = _chip()
    drive = ChargeDrive(chip["q"], label="xy")
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Square(duration=20.0, amplitude=0.02, phase=0.1),
        freq=5.0,
    )

    rebound = sequence.with_params({"q.freq": 5.1, "pulse.0.amplitude": 0.04})

    assert sequence.parameters["q.freq"] == 5.0
    assert sequence.parameters["pulse.0.amplitude"] == 0.02
    assert rebound.parameters["q.freq"] == 5.1
    assert rebound.parameters["pulse.0.amplitude"] == 0.04
    assert rebound.settings["entries"] == ("PulseEntry",)
    assert sequence.vary("q.freq", [4.9, 5.1]).field == "q.freq"
    pulse_axis = sequence.vary("pulse.0.amplitude", [0.01, 0.02])
    assert pulse_axis.field == "pulse.0.amplitude"
    batch = sequence.build_batch(pulse_axis, tlist=np.linspace(0.0, 20.0, 21))
    assert batch.params_at(1)["pulse.0.amplitude"] == 0.02

    captured: list[tuple[float, float]] = []
    materialize = sequence._materialize_drive_ops

    def capture(overrides=None):
        ops = materialize(overrides)
        captured.extend((float(op.envelope.phase), float(op.phase_offset)) for op in ops)
        return ops

    monkeypatch.setattr(sequence, "_materialize_drive_ops", capture)
    envelope_phase = sequence.vary("pulse.0.envelope.phase", [0.2, 0.3])
    sequence.build_batch(envelope_phase, tlist=np.linspace(0.0, 20.0, 21))
    assert (0.3, 0.0) in captured
    with pytest.raises(ValueError, match="Available"):
        sequence.vary("basis", ["native", "eigen"])


def test_sequence_reserves_scheduled_pulse_parameter_namespace() -> None:
    device = DuffingTransmon(freq=5.0, anharmonicity=-0.2, label="pulse.3")

    with pytest.raises(ValueError, match="reserved"):
        QuantumSequence(Chip([device]))


def test_sequence_hamiltonian_is_the_resolved_result_view() -> None:
    chip = _chip()
    chip.set_frame("rotating")
    drive = ChargeDrive(chip["q"], label="xy")
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3.0),
        freq=5.0,
    )

    result = sequence.resolve()
    lab_result = sequence.resolve(frame="lab")
    assert chip.frame == "rotating"
    assert not np.allclose(
        result.hamiltonian().matrix(t=10.0, backend=chip.backend),
        lab_result.hamiltonian().matrix(t=10.0, backend=chip.backend),
    )
    np.testing.assert_allclose(
        sequence.hamiltonian().matrix(t=10.0, backend=chip.backend),
        result.hamiltonian().matrix(t=10.0, backend=chip.backend),
    )


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

    from quchip import ChargeBasisTransmon

    circuit = ChargeBasisTransmon(
        E_C=0.25,
        E_J=12.0,
        num_basis=7,
        basis="eigen",
        levels=3,
        label="circuit",
    )
    line = ChargeDrive(circuit, label="xy")
    circuit_chip = Chip([circuit], frame="rotating", backend=DynamiqsBackend())
    circuit_chip.wire(line)
    sequence = QuantumSequence(circuit_chip)
    sequence.schedule(line, envelope=Square(duration=4.0, amplitude=0.02), freq=4.65)
    tlist = jnp.linspace(0.0, 4.0, 12)
    observable = circuit.number_operator()

    def driven_batch_loss(E_J):
        axis = sequence.vary("circuit.E_J", jnp.asarray([E_J - 0.1, E_J + 0.1]))
        results = sequence.simulate_batch(
            axis,
            tlist=tlist,
            e_ops={circuit: observable},
            progress=False,
            check_truncation=False,
        )
        population = results.population(circuit, level=1, reduce="last")
        number = results.expect(circuit, reduce="last")
        return jnp.sum(population + 0.01 * jnp.real(number))

    gradient = jax.jit(jax.grad(driven_batch_loss))(12.0)
    assert jnp.isfinite(gradient)
    assert abs(float(gradient)) > 1e-8


def test_chip_parameter_inventory_includes_drive_control_and_bath_owners() -> None:
    class LossyDrive(ChargeDrive):
        _parameter_names = ("loss_rate",)

        def __init__(self, target, *, loss_rate, label):
            super().__init__(target, label=label)
            self.loss_rate = loss_rate

        def collapse_contributions(self, device):
            return [(device.lowering_operator(), self.loss_rate)]

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.2, levels=2, label="q1")
    source = LossyDrive(q0, loss_rate=0.01, label="source")
    victim = ChargeDrive(q1, label="victim")
    equipment = ControlEquipment(
        [source, victim],
        signal_chain=[Crosstalk(source, victim, beta=0.02, theta=0.1, delay=0.5)],
    )
    bath = Bath("thermal", targets=[q0], temperature=20.0, rate=0.005, label="cold")
    chip = Chip([q0, q1], control_equipment=equipment, baths=[bath])

    assert chip.parameters["drive.source.loss_rate"] == 0.01
    assert chip.parameters["control.0.beta"] == 0.02
    assert chip.parameters["bath.cold.temperature"] == 20.0
    rebound = chip.with_params(
        {
            "drive.source.loss_rate": 0.03,
            "control.0.beta": 0.04,
            "bath.cold.temperature": 25.0,
        }
    )

    assert chip.parameters["drive.source.loss_rate"] == 0.01
    assert rebound.parameters["drive.source.loss_rate"] == 0.03
    assert rebound.parameters["control.0.beta"] == 0.04
    assert rebound.parameters["bath.cold.temperature"] == 25.0
    from quchip.engine import build_problem

    engine_result = build_problem(rebound, [], np.asarray([0.0, 1.0])).engine_result
    paths = {path for term in engine_result.collapse_terms for path in term.parameter_paths}
    assert {"drive.source.loss_rate", "bath.cold.temperature", "bath.cold.rate"} <= paths


def test_active_noise_is_rebindable_and_retained_in_engine_result() -> None:
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip.backend.dynamiqs import DynamiqsBackend
    from quchip.engine import build_problem

    q = DuffingTransmon(
        freq=5.0,
        anharmonicity=-0.2,
        levels=3,
        label="q",
        T1=100.0,
    )
    chip = Chip([q], backend=DynamiqsBackend())

    assert chip.parameters["q.T1"] == 100.0
    assert "q.T2" not in chip.parameters

    def decay_rate(T1):
        rebound = chip.with_params({"q.T1": T1})
        problem = build_problem(rebound, [], jnp.asarray([0.0, 1.0]))
        term = problem.engine_result.collapse_terms[0]
        matrix = term.operator.to_dense()
        return term.rate * jnp.real(matrix[0, 1] * jnp.conj(matrix[0, 1]))

    assert jax.grad(decay_rate)(100.0) == pytest.approx(-1e-4)
    term = build_problem(chip, [], jnp.asarray([0.0, 1.0])).engine_result.collapse_terms[0]
    assert term.parameter_paths == ("q.T1",)
    assert term.latex() == r"\hat L_{q,thermal_emission}\!\left(T_{1,q}\right)"

    sequence = QuantumSequence(chip)
    decay = sequence.vary("q.T1", jnp.asarray([80.0, 120.0]), name="T1")
    results = sequence.simulate_batch(
        decay,
        tlist=jnp.linspace(0.0, 2.0, 8),
        initial_state={"q": 1},
        progress=False,
        check_truncation=False,
    )
    excited = results.population("q", level=1, reduce="last")
    assert excited[0] < excited[1]
