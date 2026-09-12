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
        "q.T1": None,
        "q.T2": None,
        "q.thermal_occupation": None,
        "r.freq": 7.0,
        "r.T1": None,
        "r.T2": None,
        "r.thermal_occupation": None,
        "r.internal_quality_factor": None,
        "qr.g": 0.02,
    }
    assert chip.settings["devices"] == (("q", "DuffingTransmon", 3), ("r", "Resonator", 4))
    with pytest.raises(TypeError):
        chip.parameters["q.freq"] = 5.1


def test_inactive_noise_parameters_can_be_discovered_and_activated():
    chip = _chip()
    assert chip.parameters["q.T1"] is None
    assert chip.parameters["q.T2"] is None
    active = chip.with_params({"q.T1": 100.0, "q.T2": 150.0})
    assert chip["q"].T1 is None
    assert active["q"].T1 == 100.0
    assert active["q"].T2 == 150.0
    assert active.resolve().collapse_terms


def test_inactive_noise_activation_precedes_tracing():
    import jax

    chip = _chip()
    with pytest.raises(ValueError, match="Activate.*before tracing"):
        jax.grad(lambda t1: chip.with_params({"q.T1": t1})["q"].T1)(100.0)
    active = chip.with_params({"q.T1": 100.0})
    assert jax.jit(jax.grad(lambda t1: 1 / active.with_params({"q.T1": t1})["q"].T1))(100.0) == pytest.approx(-1e-4)
    with pytest.raises(ValueError, match="deactivate.*before tracing"):
        jax.grad(lambda freq: active.with_params({"q.T1": None, "q.freq": freq})["q"].freq)(5.0)



@pytest.mark.parametrize("batch", [False, True])
def test_envelope_rebinding_checks_joint_constraints(batch):
    from quchip import GaussianEdge

    chip = _chip()
    drive = ChargeDrive(chip["q"], label="xy")
    chip.wire(drive)
    seq = QuantumSequence(chip)
    pulse = seq.schedule(drive, envelope=GaussianEdge(duration=20.0, edge_duration=5.0))
    with pytest.raises(ValueError, match="edge_duration"):
        if batch:
            seq.build_batch(pulse.vary("edge_duration", [12.0]), tlist=np.linspace(0, 20, 21))
        else:
            seq.with_params({"pulse.0.edge_duration": 12.0})
    changed = seq.with_params({"pulse.0.edge_duration": 12.0, "pulse.0.duration": 30.0})
    assert changed.parameters["pulse.0.edge_duration"] == 12.0
    assert changed.parameters["pulse.0.duration"] == 30.0
    assert seq.parameters["pulse.0.duration"] == 20.0


def test_chip_with_params_is_immutable_and_rebinds_multiple_component_kinds() -> None:
    chip = _chip()

    bindings = {"q.freq": np.array(5.1), "qr.g": np.array(0.03)}
    rebound = chip.with_params(bindings)
    for value in bindings.values():
        value[...] = 9.0

    assert chip.parameters["q.freq"] == 5.0
    assert chip.parameters["qr.g"] == 0.02
    assert rebound.parameters["q.freq"] == 5.1
    assert rebound.parameters["qr.g"] == 0.03
    with pytest.raises(KeyError, match="Available"):
        chip.with_params({"q.missing": 1.0})


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("before,after", [((100.0, 150.0), (20.0, 30.0)), ((20.0, 30.0), (100.0, 150.0))])
def test_joint_noise_rebinding_validates_the_final_state(before, after, reverse) -> None:
    """Both orders of a physical T1/T2 update produce the same decay rates."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, T1=before[0], T2=before[1], label="q")
    chip = Chip([q])
    pairs = [("q.T1", after[0]), ("q.T2", after[1])]
    rebound = chip.with_params(dict(reversed(pairs) if reverse else pairs))

    assert (q.T1, q.T2) == before
    assert (rebound["q"].T1, rebound["q"].T2) == after
    assert rebound["q"]._dephasing_rate(*after) == pytest.approx(1 / after[1] - 1 / (2 * after[0]))


def test_rejected_device_parameter_group_preserves_values_and_cached_physics() -> None:
    """A later invalid field cannot leave earlier writes or cache invalidations behind."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, T1=100.0, T2=150.0, label="q")
    chip = Chip([q])
    before = chip.resolve()
    version = q.state_version

    with pytest.raises(ValueError, match="T2"):
        q.set_parameter_values({"freq": 6.0, "T2": 250.0})

    assert q.freq == 5.0
    assert q.T2 == 150.0
    assert q.state_version == version
    assert chip.resolve() is before


def test_rebinding_runs_custom_device_joint_validation() -> None:
    """Custom device constraints apply to a complete candidate just as at construction."""
    from quchip.declarative import DeviceModel, parameter

    class OrderedDevice(DeviceModel):
        low: float = parameter(default=1.0)
        high: float = parameter(default=2.0)

        def local_hamiltonian(self, op, p):
            return p.low * op.n

        def validate(self):
            if self.low >= self.high:
                raise ValueError("low must be below high")

    chip = Chip([OrderedDevice(label="q")])
    with pytest.raises(ValueError, match="below"):
        chip.with_params({"q.low": 3.0})
    rebound = chip.with_params({"q.low": 3.0, "q.high": 4.0})
    assert rebound["q"].low == 3.0
    assert chip["q"].low == 1.0


def test_grouped_rebinding_preserves_slotted_extension_parameters() -> None:
    """Grouped writes commit numerical values stored in extension slots."""
    from quchip.devices.base import BaseDevice

    class SlottedDevice(BaseDevice):
        __slots__ = ("freq",)
        tunable_param_names = ("freq",)

        def __init__(self):
            super().__init__(levels=3, label="q")
            self.freq = 5.0

        def unresolved_hamiltonian(self):
            return self.freq * self.number_operator()

    q = SlottedDevice()
    q.set_parameter_values({"freq": 6.0})
    assert q.freq == 6.0


def test_sequence_parameters_rebind_chip_and_pulse_values_directly(monkeypatch) -> None:
    chip = _chip()
    drive = ChargeDrive(chip["q"], label="xy")
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Square(duration=20.0, amplitude=0.02),
        freq=5.0,
        phase=0.1,
    )

    bindings = {"q.freq": np.array(5.1), "pulse.0.amplitude": np.array(0.04),
                "pulse.0.freq": np.array(5.2), "pulse.0.phase": np.array(0.2),
                "pulse.0.start_time": np.array(1.0)}
    rebound = sequence.with_params(bindings)
    for value in bindings.values():
        value[...] = 9.0
    assert rebound.parameters["pulse.0.freq"] == 5.2
    assert rebound.parameters["pulse.0.phase"] == 0.2
    assert rebound.parameters["pulse.0.start_time"] == 1.0

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

    captured: list[float] = []
    materialize = sequence._materialize_drive_ops

    def capture(overrides=None):
        ops = materialize(overrides)
        captured.extend(float(op.phase_offset) for op in ops)
        return ops

    monkeypatch.setattr(sequence, "_materialize_drive_ops", capture)
    pulse_phase = sequence.vary("pulse.0.phase", [0.2, 0.3])
    sequence.build_batch(pulse_phase, tlist=np.linspace(0.0, 20.0, 21))
    assert 0.3 in captured
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
            )
        population = results.population(circuit, level=1, reduce="last")
        number = results.expect(circuit, reduce="last")
        return jnp.sum(population + 0.01 * jnp.real(number))

    gradient = jax.jit(jax.grad(driven_batch_loss))(12.0)
    assert jnp.isfinite(gradient)
    assert abs(float(gradient)) > 1e-8


def test_transform_relations_validate_after_joint_rebinding() -> None:
    """Joint transform updates validate the final candidate in either order."""
    from quchip.control import SignalTransform
    from quchip.declarative import parameter

    class BoundedGain(SignalTransform):
        lower: float = parameter()
        upper: float = parameter()

        def validate(self):
            if self.lower >= self.upper:
                raise ValueError("lower must be below upper")

        def apply(self, signals):
            return signals

    chip = Chip([Resonator(freq=5., levels=2)],
                control_equipment=ControlEquipment([], signal_chain=[BoundedGain(lower=0., upper=1.)]))
    pairs = [("control.0.lower", 2.), ("control.0.upper", 3.)]
    for updates in (pairs, pairs[::-1]):
        result = chip.with_params(dict(updates))
        assert result.parameters["control.0.lower"] == 2.
        assert chip.parameters["control.0.lower"] == 0.
    with pytest.raises(ValueError, match="lower"):
        chip.with_params({"control.0.lower": 2.})


def test_chip_parameter_inventory_includes_drive_control_and_bath_owners() -> None:
    from quchip.declarative import CollapseChannel, Scalar, parameter

    class LossyDrive(ChargeDrive):
        loss_rate: Scalar = parameter(nonnegative=True, unit="1/ns", noise=True)

        def dissipation(self, device, op, p):
            return (
                CollapseChannel(op.a, p.loss_rate, "loss"),
            )

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

    first = build_problem(chip, [], np.asarray([0.0, 1.0])).engine_result
    source.loss_rate = 0.03
    second = build_problem(chip, [], np.asarray([0.0, 1.0])).engine_result
    first_rate = next(term.rate for term in first.collapse_terms if term.source == "source")
    second_rate = next(term.rate for term in second.collapse_terms if term.source == "source")
    assert float(first_rate) == pytest.approx(0.01)
    assert float(second_rate) == pytest.approx(0.03)


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
    assert chip.parameters["q.T2"] is None

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
        )
    excited = results.population("q", level=1, reduce="last")
    assert excited[0] < excited[1]
