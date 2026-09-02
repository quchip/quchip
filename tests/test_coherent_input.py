from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from quchip import (
    ChargeDrive,
    Chip,
    CoherentInput,
    ControlEndpoint,
    ControlEquipment,
    Delay,
    DuffingTransmon,
    PortNetwork,
    QuantumSequence,
    Resonator,
    Square,
)
from quchip import Crosstalk, Gain
from quchip.control.drive import BaseDrive
from quchip.engine.ir import CoherentOp, evaluate_signal_program


def _one_port_chip(*, rate: float = 0.04, delay: float = 0.0) -> tuple[Chip, Resonator]:
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=rate)
    if delay:
        network.expose("feedline", input=port.input, output=port.output, delay=delay)
    chip = Chip([resonator], port_network=network, frame="lab")
    return chip, resonator


def _dynamic_matrix(engine: object, time: float) -> np.ndarray:
    terms = engine.dynamic_terms  # type: ignore[attr-defined]
    matrix = np.zeros(terms[0].operator.shape, dtype=complex)
    for term in terms:
        coefficient = evaluate_signal_program(term.time_dependence.signal, time)
        matrix = matrix + np.asarray(term.operator.to_dense()) * coefficient
    return matrix


def test_control_endpoint_is_structural_and_coherent_input_is_not_a_drive() -> None:
    """Drives and coherent sources share the endpoint protocol without sharing a type."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    drive = ChargeDrive(target=resonator, label="charge")
    source = CoherentInput("feedline", label="probe")

    assert isinstance(drive, ControlEndpoint)
    assert isinstance(source, ControlEndpoint)
    assert not isinstance(source, BaseDrive)
    assert source.target_label == "feedline"


def test_sequence_schedules_coherent_input_through_the_common_endpoint_grammar() -> None:
    """Coherent sources use the common sequence scheduling grammar."""
    chip, _ = _one_port_chip()
    source = CoherentInput("coupler", label="probe")
    sequence = QuantumSequence(chip)

    sequence.schedule(
        source,
        envelope=Square(duration=2.0, amplitude=0.3),
        freq=5.0,
        phase=0.2,
    )

    (operation,) = sequence.scheduled_ops
    assert isinstance(operation, CoherentOp)
    assert operation.coherent_input is source
    assert operation.exposure == "coupler"
    assert sequence.channel_cursors == {("coupler", "probe"): 2.0}


def test_coherent_input_binds_to_engine_result_without_mutating_resolved_slh() -> None:
    """Solve-time fields add drive terms without entering immutable resolved SLH."""
    chip, _ = _one_port_chip(rate=0.04)
    input_free = chip.resolve().slh
    source = CoherentInput("coupler", label="probe")
    sequence = QuantumSequence(chip)
    sequence.schedule(source, envelope=Square(duration=2.0, amplitude=0.3))

    engine = sequence.build_problem(tlist=np.linspace(0.0, 2.0, 21)).engine_result

    np.testing.assert_allclose(engine.slh.S, input_free.S)
    np.testing.assert_allclose(engine.slh.L[0].to_dense(), input_free.L[0].to_dense())
    assert input_free.H.dynamic_terms == ()
    assert engine.slh.H.dynamic_terms == ()
    assert len(engine.collapse_terms) == len(input_free.channels) == 1
    assert len(engine.coherent_inputs) == 1
    assert engine.coherent_inputs[0].exposure == "coupler"
    assert evaluate_signal_program(engine.coherent_inputs[0].beta, 1.0) == pytest.approx(0.3)

    lowering = np.asarray(input_free.L[0].to_dense())
    expected = 1j * (0.3 * lowering - 0.3 * lowering.conj().T)
    np.testing.assert_allclose(_dynamic_matrix(engine, 1.0), expected)


def test_beta_uses_the_existing_complex_signal_convention_without_rescaling() -> None:
    """Incident beta follows the existing amplitude, phase, and carrier convention."""
    chip, _ = _one_port_chip()
    amplitude = 0.4
    phase = 0.3
    frequency = 0.75
    sequence = QuantumSequence(chip)
    sequence.schedule(
        CoherentInput("coupler"),
        envelope=Square(duration=1.0, amplitude=amplitude),
        freq=frequency,
        phase=phase,
    )
    engine = sequence.build_problem(tlist=np.linspace(0.0, 1.0, 11)).engine_result

    time = 0.2
    expected = amplitude * np.exp(1j * phase) * np.exp(-1j * 2.0 * np.pi * frequency * time)
    assert evaluate_signal_program(engine.coherent_inputs[0].beta, time) == pytest.approx(expected)


def test_scattering_routes_incident_beta_before_it_drives_the_system() -> None:
    """Boundary scattering routes incident beta before Hamiltonian coupling."""
    left = Resonator(freq=5.0, levels=2, label="left")
    right = Resonator(freq=6.0, levels=2, label="right")
    network = PortNetwork(scattering=[[0.0, 1.0], [1.0, 0.0]], label="swap")
    network.port("left", target=left, rate=0.04)
    network.port("right", target=right, rate=0.09)
    chip = Chip([left, right], port_network=network, frame="lab")
    sequence = QuantumSequence(chip)
    sequence.schedule(
        CoherentInput("left"),
        envelope=Square(duration=1.0, amplitude=0.2),
    )

    engine = sequence.build_problem(tlist=np.linspace(0.0, 1.0, 11)).engine_result

    left_l, right_l = (np.asarray(operator.to_dense()) for operator in engine.slh.L[:2])
    expected = 1j * (0.2 * right_l - 0.2 * right_l.conj().T)
    np.testing.assert_allclose(_dynamic_matrix(engine, 0.5), expected)
    assert not np.allclose(expected, 1j * (0.2 * left_l - 0.2 * left_l.conj().T))


def test_complex_scattering_phase_enters_c_equals_s_beta_with_correct_conjugation() -> None:
    """Complex scattering enters the coherent Hamiltonian with SLH conjugation."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(scattering=[[1j]], label="phase")
    network.port("feedline", target=resonator, rate=0.04)
    chip = Chip([resonator], port_network=network, frame="lab")
    sequence = QuantumSequence(chip)
    sequence.schedule(
        CoherentInput("feedline"),
        envelope=Square(duration=1.0, amplitude=0.2),
    )
    engine = sequence.build_problem(tlist=np.linspace(0.0, 1.0, 11)).engine_result

    coupling = np.asarray(engine.slh.L[0].to_dense())
    c = 1j * 0.2
    expected = 1j * (np.conj(c) * coupling - c * coupling.conj().T)
    np.testing.assert_allclose(_dynamic_matrix(engine, 0.5), expected)


def test_resonant_coherent_input_is_static_in_the_matching_rotating_frame() -> None:
    """A resonant coherent field is static in its matching rotating frame."""
    resonator = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="line")
    network.port("feedline", target=resonator, rate=0.04)
    chip = Chip([resonator], port_network=network, frame="rotating")
    sequence = QuantumSequence(chip)
    sequence.schedule(
        CoherentInput("feedline"),
        envelope=Square(duration=1.0, amplitude=0.2),
        freq=5.0,
    )
    engine = sequence.build_problem(tlist=np.linspace(0.0, 1.0, 11)).engine_result

    np.testing.assert_allclose(_dynamic_matrix(engine, 0.2), _dynamic_matrix(engine, 0.7))


def test_exposure_delay_shifts_incident_beta_at_the_system_boundary() -> None:
    """An exposure delay shifts the incident field at the system reference plane."""
    chip, _ = _one_port_chip(delay=0.25)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        CoherentInput("feedline"),
        envelope=Square(duration=1.0, amplitude=0.2),
    )
    engine = sequence.build_problem(tlist=np.linspace(0.0, 1.5, 16)).engine_result

    beta = engine.coherent_inputs[0].beta
    assert evaluate_signal_program(beta, 0.1) == pytest.approx(0.0)
    assert evaluate_signal_program(beta, 0.5) == pytest.approx(0.2)


def test_control_equipment_delay_rejects_a_coherent_input() -> None:
    """Field delay belongs to network exposure rather than control equipment."""
    with pytest.raises(TypeError, match="reference-plane delay"):
        Delay(CoherentInput("feedline"), 0.2)


def test_control_equipment_rejects_field_inputs_and_field_transforms() -> None:
    """Control equipment and transforms reject field-source endpoints."""
    source = CoherentInput("feedline")
    with pytest.raises(TypeError, match="BaseDrive"):
        ControlEquipment([source])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="PortNetwork"):
        Gain(source, 0.5)
    with pytest.raises(TypeError, match="PortNetwork"):
        Crosstalk(source, "charge", beta=0.1)


def test_unknown_exposure_is_rejected_at_schedule_time() -> None:
    """Scheduling rejects a coherent source bound to an unknown exposure."""
    chip, _ = _one_port_chip()

    with pytest.raises(ValueError, match="Unknown coherent-input exposure"):
        QuantumSequence(chip).schedule(
            CoherentInput("missing"),
            envelope=Square(duration=1.0, amplitude=0.2),
        )


def test_coherent_and_direct_charge_drives_are_equivalent_when_calibrated() -> None:
    """Calibrated charge and port drives produce the same Hamiltonian term."""
    rate = 0.04
    beta = 0.3
    transmon = DuffingTransmon(
        freq=5.0,
        anharmonicity=-0.25,
        levels=3,
        label="q",
    )
    charge = ChargeDrive(target=transmon, label="charge")
    equipment = ControlEquipment([charge])
    network = PortNetwork(label="line")
    network.port("feedline", target=transmon, rate=rate)
    chip = Chip(
        [transmon],
        control_equipment=equipment,
        port_network=network,
        frame="lab",
    )

    direct = QuantumSequence(chip)
    direct.schedule(
        charge,
        envelope=Square(
            duration=1.0,
            amplitude=beta * np.sqrt(rate) / (2.0 * np.pi),
        ),
    )
    field = QuantumSequence(chip)
    field.schedule(
        CoherentInput("feedline"),
        envelope=Square(duration=1.0, amplitude=beta),
    )

    direct_engine = direct.build_problem(tlist=np.linspace(0.0, 1.0, 11)).engine_result
    field_engine = field.build_problem(tlist=np.linspace(0.0, 1.0, 11)).engine_result
    np.testing.assert_allclose(
        _dynamic_matrix(field_engine, 0.5),
        _dynamic_matrix(direct_engine, 0.5),
    )


def test_coherent_amplitude_remains_batchable() -> None:
    """Coherent amplitude participates in ordinary sequence batching."""
    chip, _ = _one_port_chip()
    sequence = QuantumSequence(chip)
    handle = sequence.schedule(
        CoherentInput("coupler"),
        envelope=Square(duration=1.0, amplitude=0.1),
    )

    batch = sequence.build_batch(
        handle.vary("amplitude", [0.1, 0.25]),
        tlist=np.linspace(0.0, 1.0, 11),
    )

    assert [
        evaluate_signal_program(problem.engine_result.coherent_inputs[0].beta, 0.5)
        for problem in batch
    ] == pytest.approx([0.1, 0.25])


def test_coherent_amplitude_is_jax_differentiable_through_assembly() -> None:
    """Coherent amplitude remains differentiable through solve assembly."""
    chip, _ = _one_port_chip()

    def norm(amplitude: object) -> object:
        sequence = QuantumSequence(chip)
        sequence.schedule(
            CoherentInput("coupler"),
            envelope=Square(duration=1.0, amplitude=amplitude),
        )
        engine = sequence.build_problem(tlist=jnp.linspace(0.0, 1.0, 11)).engine_result
        matrix = sum(
            (
                term.operator.to_dense()
                * evaluate_signal_program(term.time_dependence.signal, 0.5, xp=jnp)
                for term in engine.dynamic_terms
            ),
            start=jnp.zeros((2, 2), dtype=complex),
        )
        return jnp.real(jnp.vdot(matrix, matrix))

    gradient = jax.grad(norm)(jnp.asarray(0.2))
    assert np.isfinite(float(gradient))
    assert float(gradient) > 0.0


def test_qutip_and_dynamiqs_solve_the_same_coherent_input_problem() -> None:
    """QuTiP and Dynamiqs agree on the same coherent-input evolution."""
    pytest.importorskip("dynamiqs")
    final_populations: list[float] = []
    for backend in ("qutip", "dynamiqs"):
        resonator = Resonator(freq=5.0, levels=5, label="r")
        network = PortNetwork(label="line")
        network.port("feedline", target=resonator, rate=0.04)
        chip = Chip(
            [resonator],
            port_network=network,
            frame="rotating",
            backend=backend,
        )
        sequence = QuantumSequence(chip)
        sequence.schedule(
            CoherentInput("feedline"),
            envelope=Square(duration=4.0, amplitude=0.5),
            freq=5.0,
        )
        result = sequence.simulate(
            tlist=np.linspace(0.0, 4.0, 41),
            e_ops={resonator: resonator.number_operator()},
            partition=False,
        )
        final_populations.append(float(np.real(result.expect_values(resonator)[-1])))

    assert final_populations[0] > 0.1
    assert final_populations[1] == pytest.approx(final_populations[0], rel=1e-5)
