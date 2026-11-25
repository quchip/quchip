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

from quchip import AutoChoiceWarning, Capacitive, ChargeDrive, Chip, DuffingTransmon, Qubit, Resonator  # noqa: E402
from quchip.gateset import GateSet, drag_sx  # noqa: E402
from quchip.interop.qutip_qip import from_qutip_qip, to_qutip_qip  # noqa: E402
from quchip.interop.qutip_qip.processor import ChipCompiler  # noqa: E402


def two_qubit_gateset(**noise: float) -> GateSet:
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0", **noise)
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q1", **noise)
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003, label="J")])
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    return GateSet.standard(chip, ["q0", "q1"], pairs=[("q0", "q1")], sx_duration=20.0, cr_duration=200.0)


@pytest.mark.unit
def test_from_qutip_qip_keeps_gate_set_names_and_resolves_the_rest():
    """Known names keep their angles, PHASE becomes a virtual Z, and ISWAP resolves through qutip-qip."""
    gateset = two_qubit_gateset()
    qc = QubitCircuit(2)
    qc.add_gate(H, targets=0)
    qc.add_gate(CX, controls=0, targets=1)
    qc.add_gate(RZ(0.3), targets=1)
    qc.add_gate(ISWAP, targets=[0, 1])
    qc.add_gate(PHASE(0.7), targets=0)
    with warnings.catch_warnings():
        # The conversion itself uses no deprecated qutip-qip API.
        warnings.filterwarnings("error", category=DeprecationWarning, module="quchip")
        instructions = from_qutip_qip(qc, gateset)
    assert instructions[:3] == [("H", ("q0",), ()), ("CX", ("q0", "q1"), ()), ("RZ", ("q1",), (0.3,))]
    assert {name for name, *_ in instructions[3:]} <= {"CX", "RX", "RY", "RZ", "GLOBALPHASE"}
    assert np.abs(gateset.ideal(instructions) - qc.compute_unitary().full()).max() < 1e-12
    subclass = type("Subclass", (QubitCircuit,), {})(2)
    subclass.add_gate(H, targets=0)
    subclass.add_gate(CX, controls=0, targets=1)
    assert np.abs(gateset.ideal(subclass) - subclass.compute_unitary().full()).max() < 1e-12
    unresolved = QubitCircuit(2)
    unresolved.add_gate(SQRTXdag, targets=0)
    with pytest.raises(ValueError, match="cannot resolve"):
        from_qutip_qip(unresolved, gateset)
    measured = QubitCircuit(2, num_cbits=1)
    measured.add_measurement(Mz, targets=[0], classical_store=[0])
    with pytest.raises(ValueError, match="Measurements are not supported"):
        from_qutip_qip(measured, gateset)


@pytest.mark.e2e
def test_processor_runs_circuits_like_the_gate_set():
    """run_state on the chip Hamiltonian reproduces GateSet.run in the dressed computational frame."""
    gateset = two_qubit_gateset()
    with warnings.catch_warnings():
        # Building the processor calibrates nothing.
        warnings.simplefilter("error")
        processor = to_qutip_qip(gateset)
    bell = QubitCircuit(2)
    bell.add_gate(H, targets=0)
    bell.add_gate(CX, controls=0, targets=1)
    # The run calibrates the natives that it uses at their seeds, as run() does.
    with pytest.warns(AutoChoiceWarning, match="Not calibrated"), pytest.warns(UserWarning, match="poor_fit"):
        result = processor.run_state(qc=bell)
    final = processor.get_final_circuit_state(result.states[-1])
    reference = gateset.run(bell, "00")
    assert final.isket and final.dims[0] == [2, 2]
    assert np.abs(final.full().ravel() - reference.state).max() < 1e-7
    assert gateset.unitary(bell).fidelity() > 0.9
    plus_bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert abs(reference.fidelity() - abs(np.vdot(plus_bell, reference.state)) ** 2) < 1e-12
    assert abs(reference.fidelity("00") - abs(reference.state[0]) ** 2) < 1e-12
    assert np.abs(gateset.run(bell, qutip.basis([2, 2], [0, 0])).state - reference.state).max() < 1e-12
    # After a calibration, run_state() without a circuit compiles the loaded one again.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gateset.calibrate(("SX", ("q0",)), max_nfev=5)
    fresh = processor.get_final_circuit_state(processor.run_state().states[-1]).full().ravel()
    assert np.abs(fresh - gateset.run(bell, "00").state).max() < 1e-7
    assert np.abs(fresh - final.full().ravel()).max() > 1e-6
    with pytest.raises(KeyError, match="pairs="):
        GateSet.standard(gateset.chip, ["q0", "q1"]).schedule([("CX", ("q0", "q1"))])
    assert {"xy0.i", "xy0.q", "xy0@q1.i", "xy0@q1.q", "xy1.i"} <= set(processor.get_control_labels())
    virtual = QubitCircuit(2)
    virtual.add_gate(RZ(0.4), targets=0)
    plus = qutip.Qobj(np.full(4, 0.5), dims=[[2, 2], [1, 1]])
    for start, register in ((plus, plus), ("01", qutip.basis([2, 2], [0, 1]))):
        final = processor.get_final_circuit_state(processor.run_state(start, qc=virtual).states[-1])
        assert np.abs(final.full() - (virtual.compute_unitary() * register).full()).max() < 1e-12
    with pytest.raises(ValueError, match="bit string"):
        processor.run_state("012", qc=virtual)
    with pytest.raises(ValueError, match="dimensions"):
        processor.run_state(qutip.basis(5, 0), qc=virtual)


@pytest.mark.e2e
def test_processor_takes_noise_from_the_chip():
    """noisy=True adds the chip's T1 and T2 channels; qutip-qip noise objects are refused."""
    from qutip_qip.noise import RelaxationNoise

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0", T1=2000.0, T2=1500.0)
    chip = Chip([q0])
    chip.wire(ChargeDrive(q0, label="xy0"))
    gateset = GateSet.standard(chip, ["q0"], sx_duration=20.0)
    gateset.calibrate(max_nfev=0)
    processor = to_qutip_qip(gateset)
    qc = QubitCircuit(1)
    qc.add_gate(X, targets=0)
    qc.add_gate(IDLE(200.0), targets=0)
    result = processor.run_state(processor.generate_init_processor_state(), qc=qc)
    final = processor.get_final_circuit_state(result.states[-1])
    reference = gateset.run(from_qutip_qip(qc, gateset), "0")
    assert np.abs(final.full() - reference.state).max() < 1e-7
    assert 0.05 < final.full()[0, 0].real < 0.15
    assert abs(reference.fidelity() - reference.state[1, 1].real) < 1e-12
    closed = processor.run_state(processor.generate_init_processor_state(), noisy=False)
    assert closed.states[-1].isket
    with pytest.raises(NotImplementedError, match="Noise comes from the chip"):
        processor.add_noise(RelaxationNoise(t1=1000.0))
    with pytest.raises(NotImplementedError):
        processor.run_state(processor.generate_init_processor_state(), analytical=True)
    processor.pulses[0].coeff[:] = 0.0
    with pytest.raises(ValueError, match="pulses changed"):
        processor.run_state()


@pytest.mark.e2e
def test_processor_runs_gates_at_once_in_the_space_of_run():
    """Two short SX at once need one more excitation than either alone; run_state embeds the state like run()."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.2, levels=3, label="q1")
    chip = Chip([q0, q1, Resonator(freq=7.0, levels=8, label="r")])
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    gateset = GateSet(chip, ["q0", "q1"])
    gateset.add(drag_sx(chip, "q0", duration=4.0))
    gateset.add(drag_sx(chip, "q1", duration=4.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gateset.calibrate(max_nfev=0)
    processor = to_qutip_qip(gateset)
    qc = QubitCircuit(2)
    qc.add_gate(SQRTX, targets=0)
    qc.add_gate(SQRTX, targets=1)
    with pytest.warns(UserWarning, match=r"2 gates at once solve with ExcitationCutoff\(max_total=4\)"):
        result = processor.run_state("11", qc=qc)
    final = processor.get_final_circuit_state(result.states[-1])
    reference = gateset.run(from_qutip_qip(qc, gateset), "11", dissipation=False)
    assert result.states[-1].shape[0] > processor.generate_init_processor_state().shape[0]
    assert np.abs(final.full().ravel() - reference.state).max() < 1e-7


@pytest.mark.e2e
def test_processor_keeps_states_prepared_before_the_first_calibration():
    """A chip state made before seeding raises the excitation cutoff runs in the larger space, as run() does."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.21, levels=4, label="q")
    chip = Chip([q, Resonator(freq=6.5, levels=17, label="r")])
    chip.wire(ChargeDrive(q, label="xy"))
    gateset = GateSet.standard(chip, ["q"], sx_duration=6.0)
    processor = to_qutip_qip(gateset)
    start = processor.generate_init_processor_state()
    qc = QubitCircuit(1)
    qc.add_gate(X, targets=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = processor.run_state(start, qc=qc)
        reference = gateset.run(qc, "0", dissipation=False)
    assert result.states[-1].shape[0] > start.shape[0]
    final = processor.get_final_circuit_state(result.states[-1])
    assert np.abs(final.full().ravel() - reference.state).max() < 1e-7


@pytest.mark.e2e
def test_processor_reads_register_states_on_two_level_chips():
    """On a chip of two-level gate-set qubits only, a state of register dimensions is a register state."""
    q0, q1 = Qubit(freq=5.0, label="q0"), Qubit(freq=5.1, label="q1")
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003)])
    chip.wire(ChargeDrive(q0), ChargeDrive(q1))
    gateset = GateSet.standard(chip, ["q1", "q0"])
    processor = to_qutip_qip(gateset)
    empty = QubitCircuit(2)
    register = qutip.basis([2, 2], [0, 1])
    for start in ("01", register, processor.generate_init_processor_state(register)):
        final = processor.get_final_circuit_state(processor.run_state(start, qc=empty).states[-1])
        assert abs(np.vdot(register.full().ravel(), final.full().ravel())) ** 2 > 1 - 1e-9
    with pytest.raises(TypeError, match="pass the circuit as qc="):
        processor.run_state(empty)


@pytest.mark.e2e
def test_processor_compiles_again_with_its_compiler_after_a_chip_change():
    """A chip edit after load_circuit, here T1, compiles the circuit again with its compiler before the next run."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q")
    chip = Chip([q])
    chip.wire(ChargeDrive(q))
    processor = to_qutip_qip(GateSet(chip, ["q"]))

    class Counting(ChipCompiler):
        calls = 0

        def compile(self, *args, **kwargs):
            Counting.calls += 1
            return super().compile(*args, **kwargs)

    qc = QubitCircuit(1)
    qc.add_gate(IDLE(200.0), targets=0)
    processor.load_circuit(qc, compiler=Counting(processor.num_qubits, processor.params))
    q.T1 = 100.0
    final = processor.get_final_circuit_state(processor.run_state("1").states[-1])
    assert Counting.calls == 2
    assert final.full()[1, 1].real == pytest.approx(np.exp(-2.0), rel=1e-4)
