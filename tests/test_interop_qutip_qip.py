"""qutip-qip processor and circuit conversion for gate sets."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("qutip_qip", minversion="0.5.0.dev0")

import qutip  # noqa: E402
from qutip_qip.circuit import QubitCircuit  # noqa: E402
from qutip_qip.operations.gates import CX, H, IDLE, ISWAP, PHASE, RZ, SQRTX, SQRTXdag, X  # noqa: E402
from qutip_qip.operations.measurement import Mz  # noqa: E402

from quchip import Capacitive, ChargeDrive, Chip, DuffingTransmon, Resonator  # noqa: E402
from quchip.gateset import GateSet, drag_sx  # noqa: E402
from quchip.interop.qutip_qip import from_qutip_qip, to_qutip_qip  # noqa: E402


def two_qubit_gates(**noise: float) -> GateSet:
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0", **noise)
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q1", **noise)
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003, label="J")])
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    return GateSet.standard(chip, ["q0", "q1"], pairs=[("q0", "q1")], sx_duration=20.0, cr_duration=200.0)


@pytest.mark.unit
def test_from_qutip_qip_keeps_gate_set_names_and_resolves_the_rest():
    """Known names keep their angles, PHASE becomes a virtual Z, and ISWAP resolves through qutip-qip."""
    gates = two_qubit_gates()
    qc = QubitCircuit(2)
    qc.add_gate(H, targets=0)
    qc.add_gate(CX, controls=0, targets=1)
    qc.add_gate(RZ(0.3), targets=1)
    qc.add_gate(ISWAP, targets=[0, 1])
    qc.add_gate(PHASE(0.7), targets=0)
    with warnings.catch_warnings():
        # The conversion itself uses no deprecated qutip-qip API.
        warnings.filterwarnings("error", category=DeprecationWarning, module="quchip")
        instructions = from_qutip_qip(qc, gates)
    assert instructions[:3] == [("H", ("q0",), ()), ("CX", ("q0", "q1"), ()), ("RZ", ("q1",), (0.3,))]
    assert {name for name, *_ in instructions[3:]} <= {"CX", "RX", "RY", "RZ", "GLOBALPHASE"}
    assert np.abs(gates.ideal(instructions) - qc.compute_unitary().full()).max() < 1e-12
    unresolved = QubitCircuit(2)
    unresolved.add_gate(SQRTXdag, targets=0)
    with pytest.raises(ValueError, match="cannot resolve"):
        from_qutip_qip(unresolved, gates)
    measured = QubitCircuit(2, num_cbits=1)
    measured.add_measurement(Mz, targets=[0], classical_store=[0])
    with pytest.raises(ValueError, match="Measurements are not supported"):
        from_qutip_qip(measured, gates)


@pytest.mark.e2e
def test_processor_runs_circuits_like_the_gate_set():
    """run_state on the chip Hamiltonian reproduces GateSet.simulate in the dressed computational frame."""
    gates = two_qubit_gates()
    with (pytest.warns(UserWarning, match="closed-form seeds of SX on q0"),
          pytest.warns(UserWarning, match="poor_fit")):
        processor = to_qutip_qip(gates)
    bell = QubitCircuit(2)
    bell.add_gate(H, targets=0)
    bell.add_gate(CX, controls=0, targets=1)
    result = processor.run_state(processor.generate_init_processor_state(), qc=bell)
    final = processor.get_final_circuit_state(result.states[-1])
    reference = gates.simulate(from_qutip_qip(bell, gates), "00")
    assert final.isket and final.dims[0] == [2, 2]
    assert np.abs(final.full().ravel() - reference.state).max() < 1e-7
    assert {"xy0.i", "xy0.q", "xy0@q1.i", "xy0@q1.q", "xy1.i"} <= set(processor.get_control_labels())
    virtual = QubitCircuit(2)
    virtual.add_gate(RZ(0.4), targets=0)
    plus = qutip.Qobj(np.full(4, 0.5), dims=[[2, 2], [1, 1]])
    result = processor.run_state(processor.generate_init_processor_state(plus), qc=virtual)
    final = processor.get_final_circuit_state(result.states[-1])
    assert np.abs(final.full() - (virtual.compute_unitary() * plus).full()).max() < 1e-12


@pytest.mark.e2e
def test_processor_takes_noise_from_the_chip():
    """noisy=True adds the chip's T1 and T2 channels; qutip-qip noise objects are refused."""
    from qutip_qip.noise import RelaxationNoise

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0", T1=2000.0, T2=1500.0)
    chip = Chip([q0])
    chip.wire(ChargeDrive(q0, label="xy0"))
    gates = GateSet.standard(chip, ["q0"], sx_duration=20.0)
    gates.calibrate(max_nfev=0)
    processor = to_qutip_qip(gates)
    qc = QubitCircuit(1)
    qc.add_gate(X, targets=0)
    qc.add_gate(IDLE(200.0), targets=0)
    result = processor.run_state(processor.generate_init_processor_state(), qc=qc)
    final = processor.get_final_circuit_state(result.states[-1])
    reference = gates.simulate(from_qutip_qip(qc, gates), "0")
    assert np.abs(final.full() - reference.state).max() < 1e-7
    assert 0.05 < final.full()[0, 0].real < 0.15
    closed = processor.run_state(processor.generate_init_processor_state(), noisy=False)
    assert closed.states[-1].isket
    with pytest.raises(NotImplementedError, match="Noise comes from the chip"):
        processor.add_noise(RelaxationNoise(t1=1000.0))
    with pytest.raises(NotImplementedError):
        processor.run_state(processor.generate_init_processor_state(), analytical=True)


@pytest.mark.e2e
def test_processor_runs_gates_at_once_in_the_space_of_simulate():
    """Two short SX at once need one more excitation than either alone; run_state embeds the state like simulate."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.2, levels=3, label="q1")
    chip = Chip([q0, q1, Resonator(freq=7.0, levels=8, label="r")])
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    gates = GateSet(chip, ["q0", "q1"])
    gates.add(drag_sx(chip, "q0", duration=4.0))
    gates.add(drag_sx(chip, "q1", duration=4.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate(max_nfev=0)
    processor = to_qutip_qip(gates)
    qc = QubitCircuit(2)
    qc.add_gate(SQRTX, targets=0)
    qc.add_gate(SQRTX, targets=1)
    ones = qutip.Qobj(np.eye(4)[:, 3], dims=[[2, 2], [1, 1]])
    with pytest.warns(UserWarning, match=r"2 gates at once solve with ExcitationCutoff\(max_total=4\)"):
        result = processor.run_state(processor.generate_init_processor_state(ones), qc=qc)
    final = processor.get_final_circuit_state(result.states[-1])
    reference = gates.simulate(from_qutip_qip(qc, gates), "11", dissipation=False)
    assert result.states[-1].shape[0] > processor.generate_init_processor_state().shape[0]
    assert np.abs(final.full().ravel() - reference.state).max() < 1e-7
