"""Gate sets: ideals and rules, knobs, the calibration residual, the cache, and circuits."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    CrossKerr,
    DuffingTransmon,
    Exact,
    FluxDrive,
    Gaussian,
    GaussianDRAG,
    GaussianEdge,
    QuantumSequence,
    Qubit,
    AutoChoiceWarning,
    Resonator,
    Square,
    analyze_cr_susceptibility,
)
from quchip.gateset import GateSet, Knob, NativeGate, Seed, cross_resonance, drag_sx
from quchip.gateset.cache import covariance, split_pieces
from quchip.gateset.calibration import _Residual, average_fidelity, error_generator
from quchip.gateset.ideals import FALLBACK_RULES, IDEALS, PAULIS, RULES, embed, ideal_unitary
from quchip.gateset.native import ResolvedKnob, bind_pulses, resolve_knob
from quchip.gateset.blocks import project_computational


def two_transmons(df: float = 0.2, approximation=None, g: float = 0.003, **noise):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0", **noise)
    q1 = DuffingTransmon(freq=5.0 + df, anharmonicity=-0.33, levels=3, label="q1", **noise)
    extra = {} if approximation is None else {"approximation": approximation}
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=g, label="J")], **extra)
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    return chip


def three_transmons(bus: bool = False):
    """Return transmons at 5.0, 5.1, and 5.2 GHz in a chain with g = 3 MHz, or each on a 6.2 GHz bus with g = 60 MHz."""
    qubits = [DuffingTransmon(freq=5.0 + 0.1 * i, anharmonicity=-0.33, levels=3, label=f"q{i}") for i in range(3)]
    if bus:
        resonator = Resonator(freq=6.2, levels=3, label="bus")
        chip = Chip([*qubits, resonator],
                    couplings=[Capacitive(q, resonator, g=0.06, label=f"g{i}") for i, q in enumerate(qubits)])
    else:
        chip = Chip(qubits, couplings=[Capacitive(qubits[0], qubits[1], g=0.003, label="J01"),
                                       Capacitive(qubits[1], qubits[2], g=0.003, label="J12")])
    chip.wire(*(ChargeDrive(q, label=f"xy{i}") for i, q in enumerate(qubits)))
    return chip


def cr_turns(chip, area):
    """Return the IX and ZX turns per half of q0 -> q1 tone amplitudes, and the target line's elements t0 and t1."""
    response = analyze_cr_susceptibility(chip, "q0", "q1", drive="xy0")
    line = next(item for item in chip.control_equipment.lines if item.label == "xy1")
    q0, q1 = chip.device_map["q0"], chip.device_map["q1"]
    t0 = complex(chip.drive_matrix_elements(({}, {q1: 1}), drives=[line])[line])
    t1 = complex(chip.drive_matrix_elements(({q0: 1}, {q0: 1, q1: 1}), drives=[line])[line])

    def turns(values):
        cr = values["cr_amplitude"] * np.exp(1j * values["cr_phase"])
        tone = values["cancel_amplitude"] * np.exp(1j * values["cancel_phase"])
        return ((response.IX_per_amplitude * cr + (t0 + t1) / 2 * tone) * area,
                (response.ZX_per_amplitude * cr + (t0 - t1) / 2 * tone) * area)

    return turns, t0, t1


def one_qubit_set():
    """Return a two-level qubit and a gate set with a square SX that starts 20% short and an X of two SX."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q")
    chip = Chip([qubit])
    chip.wire(ChargeDrive(qubit, label="xy"))
    element = complex(chip.drive_matrix_elements("q")["xy"])
    seq = QuantumSequence(chip)
    pulse = seq.schedule("xy", envelope=Square(duration=2.0, amplitude=0.1 / abs(element)), freq=chip.freq("q"),
                         phase=-float(np.angle(element)))
    gates = GateSet(chip, ["q"])
    gates.add(NativeGate("SX", ("q",), seq, knobs={"amplitude": Knob(pulse, "amplitude")}, free_z=("q",)))

    def build(gates):
        twice = QuantumSequence(chip)
        twice.extend(gates.schedule([("SX", "q"), ("SX", "q")]))
        return twice, {}

    gates.add(NativeGate.composite("X", ("q",), build, free_z=("q",)))
    return qubit, gates


def same_up_to_phase(left, right):
    overlap = np.trace(np.asarray(right).conj().T @ np.asarray(left))
    return np.abs(left - np.exp(1j * np.angle(overlap)) * right).max()


@pytest.mark.unit
@pytest.mark.parametrize("name, rule", [*sorted(RULES.items()), *sorted(FALLBACK_RULES.items())])
def test_built_in_rules_reproduce_their_ideals(name, rule):
    """Each built-in expansion and fallback multiplies out to its ideal up to a global phase."""
    count, expansion = rule
    params = (0.7,) if name in ("RX", "RY") else ()
    natives = {"SX": IDEALS["SX"], "ZX90": IDEALS["ZX90"]}

    def unitary(body, n):
        result = np.eye(2 ** n, dtype=complex)
        for item in body:
            gate, slots, *rest = item
            slots = slots if isinstance(slots, tuple) else (slots,)
            if gate == "RZ":
                matrix = ideal_unitary("RZ", rest)
            elif gate in natives:
                matrix = natives[gate]
            else:
                sub_count, sub = RULES[gate]
                matrix = unitary(sub(*rest), sub_count)
            result = embed(matrix, list(slots), n) @ result
        return result

    expected = ideal_unitary(name, params)
    assert same_up_to_phase(unitary(expansion(*params), count), expected) < 1e-12


@pytest.mark.unit
def test_embedding_follows_the_first_qubit_most_significant_order():
    """A gate on qubits (2, 0) of three acts like the matching permuted Kronecker product."""
    cx = IDEALS["CX"]
    x = IDEALS["X"]
    projector = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    expected = np.kron(np.kron(np.eye(2), np.eye(2)), projector[0]) + np.kron(np.kron(x, np.eye(2)), projector[1])
    assert np.allclose(embed(cx, [2, 0], 3), expected)
    with pytest.raises(ValueError, match="not unitary"):
        ideal_unitary(np.array([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(KeyError, match="No registered ideal"):
        ideal_unitary("FOO")


@pytest.mark.unit
def test_residual_norm_is_the_pedersen_infidelity_plus_a_phase_term():
    """|r|² = 1 - F_avg + (d - |Tr M|)² / (d (d + 1)) at γ = arg Tr M, with leakage."""
    rng = np.random.default_rng(7)
    ideal = IDEALS["CX"]
    generator = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    generator = 1e-2 * (generator + generator.conj().T)
    full = expm(-1j * generator)
    embedded = np.zeros((6, 4), dtype=complex)
    embedded[:4, :4] = ideal
    image = full @ embedded
    block, leaked = image[:4], np.vstack([np.zeros((4, 4)), image[4:]])
    knob = ResolvedKnob("x", (("pulse.0.amplitude", 1.0),), (0.0,), 0.0, 1.0, -np.inf, np.inf)
    residual = _Residual(lambda x: (block, leaked, 0.0), ideal, np.zeros((4, 0)), [knob])
    product = ideal.conj().T @ block
    gamma = float(np.angle(np.trace(product)))
    r = residual(np.array([0.0, gamma]))
    expected = 1 - average_fidelity(block, ideal) + (4 - abs(np.trace(product))) ** 2 / 20
    assert abs(float(r @ r) - expected) < 1e-14
    assert np.sum(np.abs(leaked) ** 2) > 1e-5


@pytest.mark.unit
def test_error_generator_and_distance_of_a_known_over_rotation():
    """An SX over-rotated by ε about X has generator coefficient ε/2 on X and distance sin(ε/2)."""
    epsilon = 3e-3
    over = expm(-0.5j * epsilon * IDEALS["X"])
    generator, distance = error_generator(over @ IDEALS["SX"], IDEALS["SX"])
    assert abs(generator["X"] - epsilon / 2) < 1e-12
    assert abs(generator["Y"]) < 1e-12 and abs(generator["Z"]) < 1e-12
    assert abs(distance - np.sin(epsilon / 2)) < 1e-12
    assert abs(1 - average_fidelity(over @ IDEALS["SX"], IDEALS["SX"]) - (2 / 3) * np.sin(epsilon / 2) ** 2) < 1e-15


@pytest.mark.unit
def test_error_generator_puts_the_error_after_the_ideal_gate():
    """A Z error after SX reads as Z, and as Y before the gate; the two readings differ by the ideal."""
    epsilon = 3e-3
    block = expm(-0.5j * epsilon * IDEALS["Z"]) @ IDEALS["SX"]
    after, _ = error_generator(block, IDEALS["SX"])
    before, _ = error_generator(IDEALS["SX"].conj().T @ block, np.eye(2))
    assert after["Z"] == pytest.approx(epsilon / 2, abs=1e-12) and abs(after["Y"]) < 1e-12
    assert before["Y"] == pytest.approx(epsilon / 2, abs=1e-12) and abs(before["Z"]) < 1e-12
    hermitian = np.random.default_rng(3).normal(size=(4, 4, 2)) @ np.array([1.0, 1.0j])
    block = expm(-1e-2j * (hermitian + hermitian.conj().T)) @ IDEALS["CX"]
    after, distance = error_generator(block, IDEALS["CX"])
    before, same = error_generator(IDEALS["CX"].conj().T @ block, np.eye(4))

    def matrix(components):
        return sum(value * np.kron(PAULIS[name[0]], PAULIS[name[1]]) for name, value in components.items())

    assert np.abs(matrix(after) - IDEALS["CX"] @ matrix(before) @ IDEALS["CX"].conj().T).max() < 1e-12
    assert distance == pytest.approx(same, abs=1e-14)


@pytest.mark.unit
def test_zero_idle_adds_nothing():
    """IDLE(0), which qutip-qip allows, adds no delay; a negative IDLE still raises."""
    gates = GateSet(two_transmons(), ["q0", "q1"])
    assert gates.schedule([("IDLE", "q0", 0.0), ("RZ", "q0", 0.3)]).total_duration == 0.0
    with pytest.raises(ValueError, match="must be > 0"):
        gates.schedule([("IDLE", "q0", -1.0)])


@pytest.mark.unit
def test_knobs_resolve_handles_paths_and_ties():
    """Handles after a leading vz resolve to their pulse number; ties move every value by its factor."""
    chip = two_transmons()
    seq = QuantumSequence(chip)
    seq.vz("q0", 0.1)
    first = seq.schedule("xy0", envelope=GaussianDRAG(duration=20.0, sigmas=4.0, amplitude=0.04, beta=0.2),
                         freq=5.0)
    second = seq.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=5.0, phase=np.pi)
    amplitude = resolve_knob("amplitude", Knob(first, "amplitude"), seq)
    assert amplitude.targets == (("pulse.0.amplitude", 1.0),)
    assert amplitude.scale == pytest.approx(0.04) and amplitude.lower == -np.inf
    detuning = resolve_knob("detuning", Knob(first, "detuning"), seq)
    assert detuning.initial == 0.0 and detuning.scale == 1e-3
    duration = resolve_knob("duration", Knob("pulse.0.duration"), seq)
    assert duration.lower == 0.0 and duration.scale == 20.0
    tie = resolve_knob("phase", Knob([((first, "phase"), 1.0), ((second, "phase"), -1.0)]), seq)
    bound = bind_pulses(seq, tie.bindings(0.3))
    assert bound._chip is chip
    assert bound.parameters["pulse.0.phase"] == pytest.approx(0.3)
    assert bound.parameters["pulse.1.phase"] == pytest.approx(np.pi - 0.3)
    with pytest.raises(ValueError, match="pulse fields"):
        resolve_knob("g", Knob("J.g"), seq)
    with pytest.raises(KeyError, match="Pulse paths"):
        resolve_knob("x", Knob("pulse.2.amplitude"), seq)
    other = QuantumSequence(chip)
    with pytest.raises(ValueError, match="another sequence"):
        resolve_knob("x", Knob(first, "amplitude"), other)
    seq.schedule("xy0", envelope=Gaussian(duration=10.0, amplitude=0.01), freq=5.0)
    seq._entries.insert(0, seq._entries.pop())
    with pytest.raises(RuntimeError, match="no longer matches"):
        resolve_knob("x", Knob(first, "amplitude"), seq)


@pytest.mark.unit
def test_gate_set_validates_natives_and_instructions():
    """Templates stay on the chip, in the window, and in gate-qubit frames; unknown gates raise."""
    chip = two_transmons()
    gates = GateSet(chip, ["q0", "q1"])
    gates.add(drag_sx(chip, "q0"))
    assert set(drag_sx(chip, "q1", detuning=True).knobs) == {"amplitude", "beta", "detuning"}
    with pytest.raises(ValueError, match="already has"):
        gates.add(drag_sx(chip, "q0"))
    foreign = QuantumSequence(chip)
    foreign.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=5.2, frame="q1")
    with pytest.raises(ValueError, match="frame"):
        gates.add(NativeGate("Y90", ("q0",), foreign, ideal=IDEALS["SX"]))
    late = QuantumSequence(chip)
    late.schedule("xy1", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=5.2)
    with pytest.raises(ValueError, match="outside"):
        gates.add(NativeGate("SX", ("q1",), late, duration=10.0))
    with pytest.raises(ValueError, match="acts on 2 qubits"):
        NativeGate("CX", ("q0",), late)
    with pytest.raises(ValueError, match="free_z"):
        NativeGate("SX", ("q1",), late, free_z=("q0",))
    with pytest.raises(ValueError, match="cr_duration"):
        GateSet.standard(chip, ["q0", "q1"], pairs=[("q0", "q1")])
    with pytest.raises(KeyError, match="No native or rule"):
        gates._expand([("ZX90", ("q0", "q1"))])
    # Without a ZX90 in either direction, the CX fallback does not loop back to itself.
    with pytest.raises(KeyError, match="No native or rule gives CX"):
        gates._expand([("CX", ("q1", "q0"))])
    cr = QuantumSequence(chip)
    cr.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=chip.freq("q1"), frame="q1")
    gates.add(NativeGate("ZX90", ("q0", "q1"), cr, ideal="ZX90"))
    gates.add(drag_sx(chip, "q1"))
    assert ("native", ("ZX90", ("q0", "q1"))) in gates._expand([("CX", ("q1", "q0"))])
    expanded = gates._expand([("H", "q0"), ("RZ", "q0", 0.2), ("IDLE", "q0", 5.0)])
    assert [item[0] for item in expanded] == ["RZ", "native", "RZ", "RZ", "IDLE"]
    assert np.allclose(gates.ideal([("RZ", "q1", 0.4)]), np.kron(np.eye(2), ideal_unitary("RZ", (0.4,))))


@pytest.mark.unit
def test_covariance_check_is_structural():
    """Placement reuse needs one carrier per piece, terms that turn as their charge requires, and RWA statics."""
    from quchip.control.signal import AnalyticSignal, Delay, Gain, SignalTransform
    from quchip.engine.problem import build_solve_problem
    from quchip.engine import ir
    from quchip.gateset.cache import (Delivered, DeliveredTerm, _harmonics, _Phase, _reached, lower_pulse,
                                      model_check, signal_check)

    chip = two_transmons()
    seq = QuantumSequence(chip)
    seq.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=chip.freq("q0"), detuning=0.002)
    seq.schedule("xy1", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=chip.freq("q1"))
    ops = list(seq.scheduled_ops)
    delivered = [lower_pulse(chip, op) for op in ops]
    assert [item.support for item in delivered] == [(0.0, 20.0), (0.0, 20.0)]
    assert split_pieces([item.support for item in delivered]) == [(0.0, 20.0, [0, 1])]
    assert covariance(ops, delivered, [0, 1])[0] is False
    assert covariance(ops, delivered, [0]) == (True, "", chip.freq("q0"), "q0")
    assert sorted(next(iter(term.charges)) for term in delivered[0].terms) == [-1, 1]
    assert all(term.follows_phase for term in delivered[0].terms)

    # At zero delivered frequency both sidebands of a flux pulse sit at 0 GHz, and the phase sets the field.
    qubit = DuffingTransmon(freq=0.1, anharmonicity=-0.3, levels=2, label="q")
    flux = Chip([qubit])
    flux.wire(FluxDrive(qubit, label="z"))

    def flux_pulse(chip, phase):
        pulse = QuantumSequence(chip)
        pulse.schedule("z", envelope=Square(duration=2.0, amplitude=0.125), freq=0.1, detuning=-0.1, phase=phase,
                       frame="q")
        return pulse

    fields = [np.asarray(flux_pulse(flux, phase).propagator(frame="lab", dissipation=False).matrix)
              for phase in (0.0, np.pi / 2)]
    assert np.abs(fields[0] - fields[1]).max() > 1.0
    # Its two harmonics fail at every initial phase.
    for phase in (0.0, -0.5, np.pi / 2):
        zero = list(flux_pulse(flux, phase).scheduled_ops)
        covariant, reason, _, _ = covariance(zero, [lower_pulse(flux, zero[0])], [0])
        assert not covariant and "cannot show that a charge-0 term of the pulse on 'z' stays the same" in reason

    # The same field with the phase folded into a scale factor, E(t) cos φ, also fails, including at φ = -π/4099,
    # where cos φ is equal before and after a turn by 2π/4099.
    class FoldedFlux(FluxDrive):
        def signal(self, pulse, target):
            local = ir.Window(ir.EnvelopeRef(pulse.envelope), 0.0, pulse.envelope.duration)
            return AnalyticSignal(program=ir.Scale(ir.Shift(local, pulse.start_time), np.cos(pulse.phase_offset)))

    folded = Chip([qubit])
    folded.wire(FoldedFlux(qubit, label="z"))
    for phase in (0.0, -np.pi / 4099, np.pi / 2):
        assert np.allclose(flux_pulse(folded, phase).propagator(frame="lab", dissipation=False).matrix,
                           flux_pulse(flux, phase).propagator(frame="lab", dissipation=False).matrix, atol=1e-8)
        zero = list(flux_pulse(folded, phase).scheduled_ops)
        result = lower_pulse(folded, zero[0])
        assert not any(term.follows_phase for term in result.terms)
        covariant, reason, _, _ = covariance(zero, [result], [0])
        assert not covariant and "FoldedFlux.signal()" in reason

    # A transform that folds each polar scale into a complex scale factor leaves the field unchanged, but the gate
    # set can no longer follow the phase, so a charge pulse that passes without it fails with it.
    def fold(node):
        node = node.rebuild_children(fold)
        if isinstance(node, ir.PolarScale):
            return ir.Scale(node.child, node.amplitude * np.exp(1j * node.theta))
        return node

    class FoldPhase(SignalTransform):
        linear_time_invariant = True

        def apply(self, signals):
            return {key: type(s)(program=fold(s.program), carrier=s.carrier, phase_reference=s.phase_reference)
                    for key, s in signals.items()}

    for chain, passes in (([], True), ([FoldPhase()], False)):
        single = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q")
        wired = Chip([single])
        wired.wire(ChargeDrive(single, label="xy"), signal_chain=chain)
        pulse = QuantumSequence(wired)
        pulse.schedule("xy", envelope=Gaussian(duration=20.0, amplitude=0.04), freq=5.0, phase=-np.pi / 4099)
        charge = list(pulse.scheduled_ops)
        covariant, reason, _, _ = covariance(charge, [lower_pulse(wired, charge[0])], [0])
        assert covariant is passes and (passes or "cannot show that a charge-" in reason)

    # A transform that doubles the field when the pulse phase is 0 makes the field depend on the phase value, not
    # only on a turn. The check fails already at phase 0.3, where nothing doubles.
    class PhaseBranch(SignalTransform):
        linear_time_invariant = True

        def apply(self, signals):
            return {key: type(s)(program=ir.Scale(s.program, 2.0 if s.program.children[0].theta == 0 else 1.0),
                                 carrier=s.carrier, phase_reference=s.phase_reference)
                    for key, s in signals.items()}

    single = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q")
    branching = Chip([single])
    branching.wire(ChargeDrive(single, label="xy"), signal_chain=[PhaseBranch()])

    def branch_pulse(phase):
        pulse = QuantumSequence(branching)
        pulse.schedule("xy", envelope=Gaussian(duration=20.0, amplitude=0.02), freq=5.0, phase=phase)
        return pulse

    # Conjugation by exp(iθN) keeps every |U_ij|, so no placement maps the gate at 0.3 to the gate at 0.
    magnitudes = [np.abs(np.asarray(branch_pulse(phase).propagator(frame="lab", dissipation=False).matrix))
                  for phase in (0.3, 0.0)]
    assert np.abs(magnitudes[0] - magnitudes[1]).max() > 0.1
    turned = list(branch_pulse(0.3).scheduled_ops)
    covariant, reason, _, _ = covariance(turned, [lower_pulse(branching, turned[0])], [0])
    assert not covariant and "cannot show that a charge-" in reason

    # Harmonics come from structure, not from values: |s|^2 + (|s|^2 - 1) Re s has harmonics -1, 0, 1 at every
    # amplitude, although at unit amplitude it does not change with the phase; |s|^2 alone has only 0.
    def nonlinear(theta, amplitude):
        s = ir.PolarScale(ir.EnvelopeRef(Square(duration=2.0, amplitude=amplitude)), 1.0, theta)
        power = ir.Multiply((s, ir.Conjugate(s)))
        return power, ir.Add((power, ir.Multiply((ir.Add((power, ir.Constant(-1.0 + 0j))), ir.RealPart(s)))))

    stand_in = _Phase()
    for amplitude in (1.0, 0.125):
        (power, mixed), (power_probed, mixed_probed) = nonlinear(0.3, amplitude), nonlinear(stand_in, amplitude)
        assert _harmonics(power, power_probed, 0.3) == {0}
        assert _harmonics(mixed, mixed_probed, 0.3) == {-1, 0, 1}
    square = ir.SignalPower(ir.PolarScale(ir.Constant(1.0 + 0j), 1.0, 0.3), 2)
    assert _harmonics(square, ir.SignalPower(ir.PolarScale(ir.Constant(1.0 + 0j), 1.0, stand_in), 2), 0.3) == {2}
    # The stand-in must sit where the phase sits.
    assert _harmonics(square, ir.SignalPower(ir.PolarScale(ir.Constant(1.0 + 0j), 1.0, stand_in), 2), 0.4) is None

    pulses = QuantumSequence(chip)
    pulses.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.0), freq=5.0)
    pulses.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.0), freq=5.0, detuning=5.0)
    two = list(pulses.scheduled_ops)
    double = Delivered((0.0, 20.0), (DeliveredTerm(frozenset({2}), ""),))
    stray = Delivered((0.0, 20.0), (DeliveredTerm(frozenset({1}), "", follows_phase=False),))
    quiet = Delivered((0.0, 20.0), ())
    assert covariance(two, [double, quiet], [0, 1])[0] is True
    covariant, reason, _, _ = covariance(two, [quiet, stray], [0, 1])
    assert not covariant and "cannot show that a charge-1 term of the pulse on 'xy0' turns by exp(1iθ)" in reason
    absolute = Delivered((0.0, 20.0), (DeliveredTerm(frozenset({0}), "CoefficientRef"),))
    covariant, reason, _, _ = covariance(two, [absolute, quiet], [0, 1])
    assert not covariant and "CoefficientRef" in reason

    # Placement turns every charge-q term at the piece's carrier: a transverse baseband term next to a carrier
    # pulse fails, and a longitudinal one passes.
    mixed = QuantumSequence(chip)
    mixed.schedule("xy0", envelope=Gaussian(duration=20.0, amplitude=0.0), freq=5.0)
    mixed.schedule("xy1", envelope=Gaussian(duration=20.0, amplitude=0.0), freq=None)
    pair = list(mixed.scheduled_ops)
    transverse = Delivered((0.0, 20.0), (DeliveredTerm(frozenset({1}), ""),))
    longitudinal = Delivered((0.0, 20.0), (DeliveredTerm(frozenset({0}), ""),))
    covariant, reason, _, _ = covariance(pair, [quiet, transverse], [0, 1])
    assert not covariant and "baseband term of the pulse on 'xy1' has charge 1" in reason
    assert covariance(pair, [quiet, longitudinal], [0, 1])[0] is True

    # A term reaches a device unless it acts there as the identity: X on q0 times Z on q2 reaches q2.
    levels = np.array([[a, b] for a in range(2) for b in range(2)])
    x, z = np.array([[0.0, 1.0], [1.0, 0.0]]), np.diag([1.0, -1.0])
    assert _reached(np.kron(x, z), levels).tolist() == [True, True]
    assert _reached(np.kron(x, np.eye(2)), levels).tolist() == [True, False]
    assert _reached(np.kron(np.eye(2), z), levels).tolist() == [False, True]

    # Only transforms that declare themselves linear and time-invariant keep reuse.
    class Clip(SignalTransform):
        def apply(self, signals):
            return signals

    for chain, expected in (([Delay("xy", delta_t=1.0), Gain("xy", factor=0.9)], ""), ([Clip()], "Clip")):
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q")
        wired = Chip([qubit])
        wired.wire(ChargeDrive(qubit, label="xy"), signal_chain=chain)
        assert (expected in signal_check(wired)) and (bool(signal_check(wired)) == bool(expected))

    exact = two_transmons(approximation=Exact())
    problem = build_solve_problem(exact, [], np.array([0.0, 1.0]), frame="lab", states="final",
                                  dissipation=False, _product_space=True)
    assert "excitation number" in model_check(problem)
    covariant, reason, _, _ = covariance(ops, delivered, [0], model_check(problem))
    assert not covariant and "excitation number" in reason


@pytest.mark.unit
def test_rotary_seeds_turn_the_target_where_its_zz_with_a_spectator_averages_out():
    """Each rotary seed keeps ZX at π/4 per half and turns the target by N turns, with ∫cos(θ - θ_T/2) dt = 0."""
    from scipy.integrate import cumulative_trapezoid, trapezoid

    from quchip.gateset.builtins import _rotary_turns

    # On a square tone the rotary angle grows linearly, so the integral vanishes at whole turns, also at the limit.
    square = Square(duration=200.0, amplitude=1.0)
    for most, zeros in ((4.5, [1, 2, 3, 4]), (1.02, [1]), (2.01, [1, 2]), (0.9, [])):
        np.testing.assert_allclose(_rotary_turns(square, most), zeros, atol=1e-6)
    chip = three_transmons()
    native = cross_resonance(chip, "q0", "q1", duration=400.0, edge=50.0, rotary=True)
    assert len(native.seeds) >= 6 and all(seed.screen for seed in native.seeds)
    times = np.linspace(0.0, 200.0, 40001)
    shape = np.real(GaussianEdge(duration=200.0, edge_duration=50.0, sigmas=2.5, amplitude=1.0).sample(times))
    running = cumulative_trapezoid(shape, times, initial=0.0)
    area = running[-1]
    turns_of, t0, t1 = cr_turns(chip, area)
    rate = abs(chip.dressed_anharmonicity("q1")) / 8
    for seed in native.seeds:
        turns, zx = turns_of(seed.values)
        assert abs(zx - 1 / 8) < 1e-9
        assert abs(turns.imag) < 1e-9 and (turns.real > 0) == ("along" in seed.provenance)
        n = abs(turns.real)
        assert abs(n - round(n)) > 0.05
        assert abs(trapezoid(np.cos(2 * np.pi * n * (running / area - 0.5)), times)) < 1e-6 * 200.0
        assert seed.values["cancel_amplitude"] * max(abs(t0), abs(t1)) < rate


@pytest.mark.unit
def test_standard_adds_rotary_seeds_when_the_target_has_a_spectator():
    """On a chain, q0 -> q1 gets rotary seeds and q1 -> q2 does not; a bus is a coupling path."""
    chip = three_transmons()
    qubits = ["q0", "q1", "q2"]
    gates = GateSet.standard(chip, qubits, pairs=[("q0", "q1"), ("q1", "q2")], cr_duration=400.0)
    assert gates.native("ZX90", ("q0", "q1")).seeds and not gates.native("ZX90", ("q1", "q2")).seeds
    forced = GateSet.standard(chip, qubits, pairs=[("q1", "q2")], cr_duration=400.0, rotary=True)
    assert forced.native("ZX90", ("q1", "q2")).seeds
    for options in ({"spectators": "none"}, {"echo": False}):
        assert not GateSet.standard(chip, qubits, pairs=[("q0", "q1")], cr_duration=400.0,
                                    **options).native("ZX90", ("q0", "q1")).seeds
    on_bus = GateSet.standard(three_transmons(bus=True), qubits, pairs=[("q0", "q1")], cr_duration=400.0)
    assert on_bus.native("ZX90", ("q0", "q1")).seeds
    with pytest.raises(ValueError, match="echo"):
        cross_resonance(chip, "q0", "q1", duration=400.0, echo=False, rotary=True)


@pytest.mark.unit
def test_screened_seeds_cost_one_solve_each_and_the_two_best_are_fitted():
    """A screen solves each seed once. Without a knob search the best start wins; with one, the two best are fitted."""
    from quchip.gateset.calibration import fit_block

    x = np.array([[0, 1], [1, 0]], dtype=complex)

    def rotation(values):
        # X turned by an error b(k), with a local minimum b = 0.2 at k = 1 and b = 0 near k = 3.
        k = values[0]
        error = 0.2 + 0.5 * (k - 1) ** 2 * (k - 3) ** 2 - 0.2 * np.exp(-((k - 3) ** 2) / 0.1)
        angle = np.pi + error
        return np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * x, np.zeros((1, 2)), 0.0

    knob = ResolvedKnob("k", (("pulse.0.amplitude", 1.0),), (0.0,), 0.0, 1.0, -np.inf, np.inf)
    # Screened errors b: 0.7, 0.27, 0.47 (on the wall of the deeper minimum), and 0.22.
    screened = [(np.array([k]), f"k = {k}") for k in (2.0, 1.2, 2.5, 0.9)]
    common = dict(starts=[(np.array([0.6]), "template")], explore=False, phase_names=(), screened=screened)
    seeded = fit_block(rotation, x, np.zeros((2, 0)), [knob], max_nfev=0, **common)
    # The template, four screened seeds, and the two Jacobians of the report.
    assert seeded.seed == "k = 0.9" and seeded.solves == 7
    two = fit_block(rotation, x, np.zeros((2, 0)), [knob], max_nfev=100, **common)
    three = fit_block(rotation, x, np.zeros((2, 0)), [knob], max_nfev=100, keep=3, **common)
    assert abs(two.values[0] - 1.0) < 1e-4 and two.cost > 1e-3
    assert three.seed == "k = 2.5" and three.cost < 1e-12


@pytest.mark.unit
def test_coupled_spectators_include_qubits_that_line_crosstalk_reaches():
    """q2 has no coupling to q0, but crosstalk from q0's line drives it, so it joins the SX block."""
    from quchip.control.signal import Crosstalk

    devices = [DuffingTransmon(freq=f, anharmonicity=-0.33, levels=3, label=f"q{i}")
               for i, f in enumerate((5.0, 5.2, 5.4))]
    for chain, expected in (([], ("q1",)), ([Crosstalk("xy0", "xy2", beta=0.03)], ("q1", "q2"))):
        chip = Chip(devices, couplings=[Capacitive(devices[0], devices[1], g=0.003, label="J")])
        chip.wire(*(ChargeDrive(d, label=f"xy{i}") for i, d in enumerate(devices)), signal_chain=chain)
        gates = GateSet(chip, ["q0", "q1", "q2"])
        assert gates.spectators_of(gates.add(drag_sx(chip, "q0"))) == expected


def calibrated_sx(df=0.2, duration=30.0, spectators="coupled", **noise):
    chip = two_transmons(df, **noise)
    gates = GateSet(chip, ["q0", "q1"], spectators=spectators)
    gates.add(drag_sx(chip, "q0", duration=duration, sigmas=4.0))
    gates.add(drag_sx(chip, "q1", duration=duration, sigmas=4.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate()
    return chip, gates


def test_drag_sx_calibrates_to_the_leakage_floor():
    """With q1 in |0>, a 30 ns ±4σ DRAG SX reaches 1 - F_avg < 1e-8, with β near 1/(4π|α|)."""
    _, gates = calibrated_sx(spectators="none")
    calibration = gates.calibration("SX", "q0")
    assert calibration.block_qubits == ("q0",)
    assert calibration.infidelity < 1e-8
    assert calibration.status in ("ok", "solver_floor")
    assert calibration.knobs["beta"].value == pytest.approx(1 / (4 * np.pi * 0.33), rel=0.05)
    assert set(calibration.knobs) == {"amplitude", "beta"}
    assert calibration.solves < 120
    low, high = calibration.placement_infidelity
    assert set(calibration.spectators) == {"q1"} and high - low < 1e-12
    assert calibration.repetitions[1] == pytest.approx(calibration.infidelity, rel=0.2, abs=1e-9)
    text = gates.describe()
    assert "SX" in text and "q0" in text


def test_spectator_block_carries_the_static_zz_of_the_gate():
    """The joint-block SX error is the static ZZ phase, between the ZZ angle with and without the half-turn echo."""
    chip, gates = calibrated_sx()
    calibration = gates.calibration("SX", "q0")
    assert calibration.block_qubits == ("q0", "q1") and set(calibration.free_z) == {"q0", "q1"}
    assert calibration.ideal.shape == (4, 4)
    zeta = abs(float(chip.kerr_matrix().values[0, 1]))
    angle = np.pi * zeta * 30.0 / 2
    assert 0.2 * 0.8 * angle**2 < calibration.infidelity < 0.8 * angle**2
    assert calibration.repetitions[1] == pytest.approx(calibration.infidelity, rel=0.05)
    # The knobs stay at their physical values; only the error budget changes.
    assert calibration.status == "ok"
    assert calibration.knobs["beta"].value == pytest.approx(1 / (4 * np.pi * 0.33), rel=0.6)
    # The ground-spectator fit, scored on the same joint block, is worse.
    _, restricted = calibrated_sx(spectators="none")
    block, _ = gates._block(restricted.calibration("SX", "q0").sequence, ("q0", "q1"), 30.0)
    assert 1 - average_fidelity(np.asarray(block.matrix), calibration.ideal) > calibration.infidelity


def test_cached_circuit_matches_the_direct_schedule_solve():
    """unitary() composes placed, frame-shifted gates and idles to the direct propagator of schedule()."""
    _, gates = calibrated_sx()
    circuit = [("SX", "q0"), ("RZ", "q1", 0.3), ("IDLE", "q0", 7.3), ("H", "q1"), ("RX", "q0", 0.5),
               ("SX", "q1"), ("RZ", "q0", -1.1), ("SX", "q0")]
    composed = gates.unitary(circuit, policy="serial")
    direct = gates.schedule(circuit, policy="serial").propagator(frame="rotating", options=gates.options,
                                                                 dissipation=False)
    assert composed.solves == 0
    assert np.abs(composed.full - np.asarray(direct.matrix)).max() < 2e-5
    # Static ZZ acts for the whole 250 ns circuit, so the circuit is not ideal.
    assert composed.fidelity() > 0.99
    overlapping = gates.unitary(circuit)
    assert overlapping.solves > 0
    direct = gates.schedule(circuit).propagator(frame="rotating", options=gates.options, dissipation=False)
    assert np.abs(overlapping.full - np.asarray(direct.matrix)).max() < 2e-5


def test_cached_cross_resonance_matches_direct_solves_at_later_starts():
    """A CR tone drives q0 at q1's frequency; the cache turns it with that carrier, not with q0's frame."""
    chip = two_transmons(df=0.1)
    gates = GateSet(chip, ["q0", "q1"])
    gates.add(cross_resonance(chip, "q0", "q1", duration=120.0, echo=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate(max_nfev=0)
    idles = [("IDLE", "q0", 7.1), ("IDLE", "q1", 7.1)]
    circuit = [("RZ", "q0", 0.4), ("RZ", "q1", -1.1), *idles, ("ZX90", ("q0", "q1")), *idles[:1],
               ("ZX90", ("q0", "q1"))]
    composed = gates.unitary(circuit, policy="serial")
    direct = gates.schedule(circuit, policy="serial").propagator(frame="rotating", options=gates.options,
                                                                 dissipation=False)
    assert composed.solves == 0
    assert np.abs(composed.full - np.asarray(direct.matrix)).max() < 2e-5


def test_rotary_target_tone_turns_a_two_level_target_by_its_dressed_element():
    """Alone for one half, a rotary seed's target tone turns the target by 2π t_0 c ∫E dt with the control in 0."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=2, label="q1")
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003, label="J")])
    line = ChargeDrive(q1, label="xy1")
    chip.wire(ChargeDrive(q0, label="xy0"), line)
    t0 = complex(chip.drive_matrix_elements(({}, {q1: 1}), drives=[line])[line])
    x, y = np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]])
    seeds = cross_resonance(chip, "q0", "q1", duration=400.0, edge=50.0, rotary=True).seeds[:2]
    tones = [(seed.values["cancel_amplitude"], seed.values["cancel_phase"]) for seed in seeds]
    # The seeds turn the target about ±X. A third tone turns it about the axis at π/4, so the sign of Y counts.
    tones.append((tones[0][0], np.pi / 4 - np.angle(t0)))
    for amplitude, phase in tones:
        shape = GaussianEdge(duration=200.0, edge_duration=50.0, sigmas=2.5, amplitude=amplitude)
        sequence = QuantumSequence(chip)
        sequence.schedule("xy1", envelope=shape, freq=chip.freq("q1"), frame="q1", phase=phase)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            block = np.asarray(sequence.subspace_propagator(["q0", "q1"], tlist=[0.0, 200.0],
                                                            options={"atol": 1e-12, "rtol": 1e-10}).matrix)
        rotation = block[:2, :2] / np.sqrt(np.linalg.det(block[:2, :2]))
        # The control stays in 0 and the target is resonant there, so the rotation is exact up to the dressing.
        angle = 2 * np.pi * abs(t0) * float(np.real(shape.area()))
        axis = np.angle(t0 * np.exp(1j * phase))
        expected = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * (np.cos(axis) * x + np.sin(axis) * y)
        assert min(np.abs(rotation - expected).max(), np.abs(rotation + expected).max()) < 1e-4


def test_rotary_cross_resonance_starts_at_a_rotary_seed_and_caches_like_a_direct_solve():
    """The screen picks a rotary start, and its strong target tone composes like a direct solve at other starts."""
    chip = two_transmons(df=0.1, g=0.006)
    gates = GateSet.standard(chip, ["q0", "q1"], pairs=[("q0", "q1")], cr_duration=200.0, rotary=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibration = gates.calibrate(("ZX90", ("q0", "q1")), max_nfev=0)[("ZX90", ("q0", "q1"))]
    assert calibration.seed.startswith("rotary")
    # Idles without a common period with the carriers place the gates at unrelated start times.
    circuit = [("IDLE", "q0", 7.1), ("IDLE", "q1", 7.1), ("ZX90", ("q0", "q1")), ("IDLE", "q0", 13.3),
               ("IDLE", "q1", 13.3), ("ZX90", ("q0", "q1"))]
    composed = gates.unitary(circuit, policy="serial")
    direct = gates.schedule(circuit, policy="serial").propagator(frame="rotating", options=gates.options,
                                                                 dissipation=False)
    assert composed.solves == 0
    assert np.abs(composed.full - np.asarray(direct.matrix)).max() < 2e-5


def test_simulate_maps_the_final_state_into_the_computational_frame():
    """The simulated state equals the composed unitary on |00>, and T1 and T2 lower the fidelity."""
    _, gates = calibrated_sx(T1=5000.0, T2=4000.0)
    circuit = [("H", "q0"), ("RZ", "q0", 0.4), ("SX", "q1")]
    run = gates.simulate(circuit, "00", dissipation=False)
    composed = gates.unitary(circuit)
    assert np.abs(run.state - composed.computational[:, 0]).max() < 1e-6
    assert sum(run.probabilities.values()) == pytest.approx(1.0 - run.leakage, abs=1e-12)
    pure = np.outer(np.eye(4)[0], np.eye(4)[0])
    closed = gates.simulate(circuit, pure, dissipation=False)
    assert np.abs(closed.state - np.outer(run.state, run.state.conj())).max() < 1e-6
    ideal = gates.ideal(circuit)[:, 0]
    lossy = gates.simulate(circuit, "00")
    loss = abs(ideal.conj() @ run.state) ** 2 - np.real(ideal.conj() @ lossy.state @ ideal)
    duration = gates.schedule(circuit).total_duration
    assert 0.2 * duration / 5000.0 < loss < duration / 4000.0
    virtual = [("RZ", "q0", 0.4), ("RZ", "q1", -1.1)]
    plus = np.full(4, 0.5)
    assert np.abs(gates.simulate(virtual, plus).state - gates.ideal(virtual) @ plus).max() < 1e-12


def test_auto_cutoff_keeps_the_gate_and_warns():
    """With two readout resonators (225 states), the gate set picks ExcitationCutoff(3) and warns."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.33, levels=3, label="q1")
    r0 = Resonator(freq=7.0, levels=5, label="r0")
    r1 = Resonator(freq=7.2, levels=5, label="r1")
    chip = Chip([q0, q1, r0, r1], couplings=[Capacitive(q0, q1, g=0.003, label="J"),
                                             Capacitive(q0, r0, g=0.08, label="g0"),
                                             Capacitive(q1, r1, g=0.08, label="g1")])
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    gates = GateSet(chip, ["q0", "q1"], spectators="none")
    gates.add(drag_sx(chip, "q0", duration=30.0))
    with pytest.warns(UserWarning, match=r"ExcitationCutoff\(max_total=3\)"):
        gates.calibrate()
    assert gates.cutoff == 3
    calibration = gates.calibration("SX", "q0")
    assert calibration.infidelity < 1e-6
    assert getattr(calibration, "cutoff_difference") < 1e-6
    reference, _ = GateSet(chip, ["q0", "q1"], space=None)._block(calibration.sequence, ("q0",), 30.0)
    cut, _ = gates._block(calibration.sequence, ("q0",), 30.0)
    assert np.abs(np.asarray(reference.matrix) - np.asarray(cut.matrix)).max() < 1e-6


def test_exact_chip_places_gates_by_direct_solves():
    """Without RWA, the cache flags its pieces and solves each placement, which keeps circuits exact."""
    chip = two_transmons(approximation=Exact())
    gates = GateSet(chip, ["q0", "q1"], frame="lab", options={"atol": 1e-10, "rtol": 1e-8, "nsteps": 10**7})
    seq = QuantumSequence(chip)
    m = chip.drive_matrix_elements("q0")["xy0"]
    pulse = seq.schedule("xy0", envelope=GaussianDRAG(duration=20.0, sigmas=4.0, amplitude=0.04, beta=0.24),
                         freq=chip.freq("q0"), phase=-np.angle(m))
    gates.add(NativeGate("SX", ("q0",), seq, knobs={"amplitude": Knob(pulse, "amplitude")}, free_z=("q0",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate(max_nfev=3)
    entry = gates.entry("SX", "q0")
    assert not entry.covariant and "excitation number" in entry.pieces[0].reason
    circuit = [("SX", "q0"), ("RZ", "q0", 0.3), ("SX", "q0")]
    composed = gates.unitary(circuit)
    assert composed.solves in (1, 2)  # the gate at t = 0 may already be placed by the calibration report
    direct = gates.schedule(circuit).propagator(frame="lab", options=gates.options, dissipation=False)
    columns, energies = gates._states(gates.qubits)
    block = project_computational(np.asarray(direct.matrix) @ columns, columns, energies, 0.0, composed.duration,
                                  gates.qubits)
    correction = gates._frame_correction(gates.qubits, composed.frame_offsets)
    # Lab-frame solves at rtol 1e-8 set the agreement.
    assert np.abs(composed.computational - correction[:, None] * np.asarray(block.matrix)).max() < 2e-5


def test_cached_circuits_follow_delivered_fields():
    """With a 5 ns line delay, cached and jointly solved q0 gates match a direct solve."""
    from quchip.control.signal import Delay

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.33, levels=3, label="q1")
    chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003, label="J")])
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"), signal_chain=[Delay("xy0", delta_t=5.0)])
    gates = GateSet(chip, ["q0", "q1"])
    gates.add(drag_sx(chip, "q0", duration=20.0))
    gates.add(drag_sx(chip, "q1", duration=20.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate(max_nfev=30)
    assert gates.entry("SX", "q0").duration == pytest.approx(25.0)
    assert gates.entry("SX", "q1").duration == pytest.approx(20.0)
    # The idles exceed the delay, so no delivered field crosses into the next gate.
    separated = [("SX", "q0"), ("RZ", "q0", 0.4), ("IDLE", "q0", 10.0), ("SX", "q0"), ("IDLE", "q0", 10.0),
                 ("RZ", "q1", -0.7), ("SX", "q1")]
    back_to_back = [("SX", "q0"), ("RZ", "q0", 0.4), ("SX", "q0")]
    for circuit, overlapping in ((separated, False), (back_to_back, True)):
        composed = gates.unitary(circuit, policy="serial")
        assert (composed.solves > 0) is overlapping
        direct = gates.schedule(circuit, policy="serial").propagator(
            tlist=[0.0, composed.duration], frame="rotating", options=gates.options, dissipation=False)
        assert np.abs(composed.full - np.asarray(direct.matrix)).max() < 2e-6
    assert gates.unitary(back_to_back, policy="serial").duration == pytest.approx(45.0)


def test_calibration_leaves_a_zero_amplitude_start():
    """A fit that starts at zero amplitude reaches the exact SX of a 2 ns square pulse on a two-level qubit."""
    from quchip import Square

    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="q")
    chip = Chip([qubit])
    chip.wire(ChargeDrive(qubit, label="xy"))
    element = complex(chip.drive_matrix_elements("q")["xy"])
    seq = QuantumSequence(chip)
    pulse = seq.schedule("xy", envelope=Square(duration=2.0, amplitude=0.0), freq=chip.freq("q"),
                         phase=-float(np.angle(element)))
    gates = GateSet(chip, ["q"])
    gates.add(NativeGate("SX", ("q",), seq, knobs={"amplitude": Knob(pulse, "amplitude", scale=0.1)}, ideal="SX",
                         free_z=("q",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibration = gates.calibrate()[("SX", ("q",))]
    assert calibration.infidelity < 1e-8
    assert calibration.knobs["amplitude"].value == pytest.approx(0.125 / abs(element), rel=1e-5)
    assert calibration.identifiability["jacobian_step_change"] < 1e-3
    # Bounds closer than the step on both sides shorten it, so the difference stays a valid pulse.
    narrow = QuantumSequence(chip)
    short = narrow.schedule("xy", envelope=Square(duration=1.5e-6, amplitude=0.1), freq=chip.freq("q"))
    tight = GateSet(chip, ["q"])
    tight.add(NativeGate("SX", ("q",), narrow, knobs={"duration": Knob(short, "duration", bounds=(1e-6, 2e-6),
                                                                       scale=0.1)}, ideal="SX", free_z=("q",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = tight.calibrate(max_nfev=5)[("SX", ("q",))].knobs["duration"].value
    assert 1e-6 <= value <= 2e-6


def test_calibration_status_flags():
    """Seeds, cut-off fits, inner refits, other start times, and states of the other qubits each set a status."""
    qubit, gates = one_qubit_set()
    chip = gates.chip
    element = complex(chip.drive_matrix_elements("q")["xy"])
    with pytest.warns(UserWarning, match="poor_fit"):
        seeded = gates.calibrate("SX", max_nfev=0)[("SX", ("q",))]
    assert seeded.knobs["amplitude"].value == seeded.knobs["amplitude"].initial
    with pytest.warns(UserWarning, match="unconverged"):
        assert gates.calibrate("SX", max_nfev=1)[("SX", ("q",))].status == "unconverged"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate()
        assert gates.calibration("X", "q").status != "stale"
        gates.calibrate("SX")
    assert gates.calibration("X", "q").status == "stale"
    assert "stale" in gates.describe()
    with pytest.warns(UserWarning, match="calibrated again"):
        gates.unitary([("X", "q")])
    # A second pulse 10 MHz off the qubit turns its axis with the start time: good at 0, poor elsewhere.
    seq = QuantumSequence(chip)
    pulse = seq.schedule("xy", envelope=Square(duration=2.0, amplitude=0.125 / abs(element)), freq=chip.freq("q"),
                         phase=-float(np.angle(element)))
    seq.schedule("xy", envelope=Square(duration=2.0, amplitude=0.01 / abs(element)), freq=chip.freq("q") + 0.01,
                 phase=-float(np.angle(element)))
    gates = GateSet(chip, ["q"])
    gates.add(NativeGate("SX", ("q",), seq, knobs={"amplitude": Knob(pulse, "amplitude")}, free_z=("q",)))
    with pytest.warns(UserWarning, match="poor_fit"):
        calibration = gates.calibrate()[("SX", ("q",))]
    assert calibration.infidelity < 1e-3 < calibration.placement_infidelity[1]
    # |101> meets |020> (f0 + f2 = 2 f1 + α): the middle SX is good unless both neighbors are excited.
    qubits = [DuffingTransmon(freq=f, anharmonicity=-0.33, levels=3, label=f"q{i}")
              for i, f in enumerate((4.95, 5.10, 4.92))]
    chip = Chip(qubits, couplings=[Capacitive(qubits[0], qubits[1], g=0.003, label="J01"),
                                   Capacitive(qubits[1], qubits[2], g=0.003, label="J12")])
    chip.wire(*(ChargeDrive(q, label=f"xy{i}") for i, q in enumerate(qubits)))
    gates = GateSet(chip, ["q0", "q1", "q2"], spectators="none")
    gates.add(drag_sx(chip, "q1"))
    with pytest.warns(UserWarning, match="poor_fit"):
        calibration = gates.calibrate()[("SX", ("q1",))]
    assert calibration.infidelity < 1e-3 and max(1 - f for f in calibration.spectators.values()) < 1e-3
    state, fidelity = calibration.worst_spectators
    assert state == {"q0": 1, "q2": 1} and 1 - fidelity > 0.05


@pytest.mark.unit
def test_uncalibrated_gates_run_at_their_seeds_with_a_notice():
    """A circuit calibrates its uncalibrated gates, and the natives inside them, at their seeds, and says so."""
    _, gates = one_qubit_set()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        implicit = gates.unitary([("X", "q")])
    notices = [w for w in caught if issubclass(w.category, AutoChoiceWarning)]
    assert len(notices) == 1 and "X on q" in str(notices[0].message) and notices[0].filename == __file__
    amplitude = gates.calibration("SX", "q").knobs["amplitude"]
    assert amplitude.value == amplitude.initial
    _, explicit = one_qubit_set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explicit.calibrate(max_nfev=0)
    assert abs(implicit.fidelity() - explicit.unitary([("X", "q")]).fidelity()) < 1e-12


@pytest.mark.unit
def test_an_edited_chip_makes_its_gates_stale_until_they_are_calibrated_again():
    """An in-place change of the chip marks every calibration stale. Calibrating again keeps a fit from the old
    knobs only while its error at most doubles; after a seed-only calibration, every start competes."""
    qubit, gates = one_qubit_set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate(max_nfev=0)
        first = gates.calibrate()[("SX", ("q",))]
        again = gates.calibrate()[("SX", ("q",))]
    # The seed-only knobs are the template's, which comes first among equal fits.
    assert first.seed == "template" and again.seed == "previous calibration"
    # A start of five quarter turns: with a detuned carrier, its stronger drive leaves less error.
    gates.native("SX", "q").seeds = (Seed({"amplitude": 5 * again.knobs["amplitude"].value}, "five quarter turns"),)
    qubit.freq = 5.001
    with pytest.warns(UserWarning, match="chip changed"):
        gates.unitary([("SX", "q")])
    assert {gates.calibration(name, "q").status for name in ("SX", "X")} == {"stale"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = gates.calibrate()
    # The SX keeps its carrier, so a fit from the old knobs gives 1e-5, far above the previous error. Every start
    # competes, and the longer one wins at 4e-7.
    assert calibrated[("SX", ("q",))].seed == "five quarter turns"
    assert all(calibration.status != "stale" for calibration in calibrated.values())


def test_an_edited_chip_declares_the_built_in_natives_again():
    """After a frequency edit, the built-in SX takes its carrier from the chip and calibrates as a new gate set does."""

    def one_transmon(freq):
        qubit = DuffingTransmon(freq=freq, anharmonicity=-0.33, levels=3, label="q")
        chip = Chip([qubit])
        chip.wire(ChargeDrive(qubit, label="xy"))
        gates = GateSet(chip, ["q"])
        gates.add(drag_sx(chip, "q"))
        return qubit, gates

    qubit, gates = one_transmon(5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate()
        qubit.freq = 5.0003
        edited = gates.calibrate()[("SX", ("q",))]
        fresh = one_transmon(5.0003)[1].calibrate()[("SX", ("q",))]
    low, high = edited.placement_infidelity
    assert edited.status == "ok" and edited.seed == "previous calibration"
    assert high - low < 1e-9
    assert edited.knobs["beta"].value == pytest.approx(fresh.knobs["beta"].value, rel=1e-3)


@pytest.mark.unit
def test_report_channel_and_eigenvalue_infidelity():
    """The report's channel maps rho to B rho B^dag, and its eigenvalue infidelity leaves out a fixed basis tilt."""
    _, gates = one_qubit_set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibration = gates.calibrate("SX")[("SX", ("q",))]
    block, rho = calibration.block, np.array([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]])
    mapped = calibration.channel.superoperator @ rho.reshape(-1, order="F")
    assert np.allclose(mapped.reshape(2, 2, order="F"), block @ rho @ block.conj().T, atol=1e-12)
    tilt = expm(-0.05j * PAULIS["Z"])
    tilted = replace(calibration, block=tilt @ calibration.ideal @ tilt.conj().T)
    assert 1 - average_fidelity(tilted.block, tilted.ideal) > 1e-4 and tilted.eigenvalue_infidelity < 1e-12


def test_automatic_cutoff_recalibrates_until_the_gate_converges():
    """When a gate changes at the next excitation cutoff, the gate set raises the cutoff and calibrates again."""
    from quchip import Square
    from quchip.gateset import Seed

    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.01, levels=4, label="q")
    cavity = Resonator(freq=7.0, levels=17, label="r")
    chip = Chip([qubit, cavity])
    chip.wire(ChargeDrive(qubit, label="xy"))
    element = complex(chip.drive_matrix_elements("q")["xy"])
    seq = QuantumSequence(chip)
    pulse = seq.schedule("xy", envelope=Square(duration=2.0, amplitude=0.0), freq=chip.freq("q"),
                         phase=-float(np.angle(element)))
    gates = GateSet(chip, ["q"])
    gates.add(NativeGate("SX", ("q",), seq, knobs={"amplitude": Knob(pulse, "amplitude", scale=0.1)}, ideal="SX",
                         free_z=("q",), seeds=(Seed({"amplitude": 0.1}, "test"),)))
    with pytest.warns(UserWarning, match="recalibrating"):
        calibration = gates.calibrate(max_nfev=40)[("SX", ("q",))]
    assert gates.cutoff is None or gates.cutoff >= 3
    assert calibration.cutoff_difference is None or calibration.cutoff_difference <= gates.cutoff_tolerance


@pytest.mark.parametrize("spectators", ["all", "none"])
def test_cutoff_checks_include_excited_spectators(spectators):
    """The cutoff rises until the gate with the other qubit excited matches the full space, in or out of the block."""
    from quchip import Square
    from quchip.gateset import Seed

    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.01, levels=4, label="q")
    spectator = DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=2, label="s")
    chip = Chip([qubit, spectator, Resonator(freq=7.0, levels=9, label="r")])
    chip.wire(ChargeDrive(qubit, label="xy"))
    element = complex(chip.drive_matrix_elements("q")["xy"])
    seq = QuantumSequence(chip)
    pulse = seq.schedule("xy", envelope=Square(duration=2.0, amplitude=0.0), freq=chip.freq("q"),
                         phase=-float(np.angle(element)))
    gates = GateSet(chip, ["q", "s"], spectators=spectators)
    gates.add(NativeGate("SX", ("q",), seq, knobs={"amplitude": Knob(pulse, "amplitude", scale=0.1)}, ideal="SX",
                         free_z=("q",), seeds=(Seed({"amplitude": 0.1}, "test"),)))
    with pytest.warns(UserWarning, match="recalibrating"):
        calibration = gates.calibrate(max_nfev=40)[("SX", ("q",))]
    assert gates.cutoff is None or gates.cutoff >= 4
    cut, _ = gates._block(calibration.sequence, ("q", "s"), 2.0)
    full, _ = GateSet(chip, ["q", "s"], space=None)._block(calibration.sequence, ("q", "s"), 2.0)
    assert np.abs(np.asarray(cut.matrix) - np.asarray(full.matrix)).max() < 1e-6


def test_gates_at_once_solve_with_one_more_excitation_each():
    """Two short SX at once from |11> reach |22>, above the cutoff that holds each gate; the layer solves above it."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.2, levels=3, label="q1")
    chip = Chip([q0, q1, Resonator(freq=7.0, levels=8, label="r")])
    chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))
    gates = GateSet(chip, ["q0", "q1"])
    gates.add(drag_sx(chip, "q0", duration=4.0))
    gates.add(drag_sx(chip, "q1", duration=4.0))
    with pytest.warns(UserWarning, match=r"ExcitationCutoff\(max_total=3\)"):
        gates.calibrate(max_nfev=0)
    assert gates.cutoff == 3
    # The qubits do not interact, so the layer is the product of the two gates alone.
    alone = gates.unitary([("SX", "q0")]).computational @ gates.unitary([("SX", "q1")]).computational
    layer = [("SX", "q0"), ("SX", "q1")]
    with pytest.warns(UserWarning, match=r"2 gates at once solve with ExcitationCutoff\(max_total=4\)"):
        together = gates.unitary(layer)
    assert np.abs(together.computational - alone).max() < 1e-6
    run = gates.simulate(layer, "11", dissipation=False)
    assert np.abs(run.state - alone[:, 3]).max() < 1e-6


@pytest.mark.validation
def test_echoed_cross_resonance_calibrates_a_cx():
    """The echoed ZX90 gives a CX with 1 - F < 1e-3; its error changes with start time, and the SX error does not."""
    chip = two_transmons(df=0.1)
    gates = GateSet.standard(chip, ["q0", "q1"], pairs=[("q0", "q1")], sx_duration=40.0, cr_duration=400.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate()
    calibration = gates.calibration("ZX90", ("q0", "q1"))
    assert calibration.infidelity < 1e-3
    assert calibration.dependencies and gates.entry("ZX90", ("q0", "q1")).covariant
    assert gates.unitary([("CX", ("q0", "q1"))]).fidelity() > 1 - 2e-3
    low, high = calibration.placement_infidelity
    assert low <= calibration.infidelity + 1e-9 and high > 1.5 * calibration.infidelity
    # The cached range matches a gate placed after an idle and composed like any circuit.
    idle = gates.unitary([("IDLE", "q0", 37.1), ("IDLE", "q1", 37.1)]).computational
    placed = gates.unitary([("IDLE", "q0", 37.1), ("IDLE", "q1", 37.1), ("ZX90", ("q0", "q1"))]).computational
    assert low <= 1 - average_fidelity(placed @ idle.conj().T, calibration.ideal) <= high
    sx = gates.calibration("SX", "q0").placement_infidelity
    assert sx[1] - sx[0] < 1e-12 and sx[0] == pytest.approx(gates.calibration("SX", "q0").infidelity, abs=1e-9)


@pytest.mark.validation
def test_default_tolerances_follow_the_backend():
    """On dynamiqs the default tolerances become a Dopri8 method, and the gate block matches the QuTiP solve."""
    dq = pytest.importorskip("dynamiqs")
    chip = two_transmons()
    sequence = drag_sx(chip, "q0").sequence
    on_qutip = GateSet(chip, ["q0", "q1"])
    on_dynamiqs = GateSet(chip, ["q0", "q1"], backend="dynamiqs")
    method = on_dynamiqs.options["method"]
    assert isinstance(method, dq.method.Dopri8) and (method.atol, method.rtol) == (1e-12, 1e-10)
    assert on_dynamiqs.options is on_dynamiqs.options
    left, _ = on_qutip._block(sequence, ("q0", "q1"), 40.0)
    right, _ = on_dynamiqs._block(sequence, ("q0", "q1"), 40.0)
    assert np.abs(np.asarray(left.matrix) - np.asarray(right.matrix)).max() < 1e-7


def rotary_model(shape, ix, zx, zeta, window=0.0):
    """Return the propagator on (control, target, spectator) of an IX/ZX/ZZ model of the echoed CR, solved exactly.

    In each half the target turns by ``ix`` turns about X and by ``zx`` with the sign of Z_c, and the second half
    reverses both. The control flips at once after each half, and ``ζ n_t n_s`` acts throughout, alone for
    ``window`` ns after each half. For the control in c, the target thus turns by ``ix + s zx`` and then by
    ``-ix + s zx`` turns, with ``s = (-1)^c``.
    """
    from scipy.integrate import solve_ivp

    area = float(np.real(shape.area()))
    number = np.diag([0.0, 1.0])
    static = 2 * np.pi * zeta * np.kron(number, number)
    idle = expm(-1j * static * window)
    blocks = []
    for sign in (1, -1):
        u = np.eye(4, dtype=complex)
        for turns in (ix + sign * zx, -ix + sign * zx):
            def rhs(t, y, turns=turns):
                drive = np.pi * turns / area * float(np.real(shape.sample(np.array([t]))[0]))
                return (-1j * (drive * np.kron(PAULIS["X"], np.eye(2)) + static) @ y.reshape(4, 4)).ravel()

            solved = solve_ivp(rhs, (0.0, shape.duration), u.ravel(), method="DOP853", rtol=1e-12, atol=1e-13)
            u = idle @ solved.y[:, -1].reshape(4, 4)
        blocks.append(u)
    return np.block([[blocks[0], np.zeros((4, 4))], [np.zeros((4, 4)), blocks[1]]])


THREE_QUBIT_PAULIS = ["".join(p) for p in itertools.product("IXYZ", repeat=3)][1:]
# The terms that act on both the target and the spectator.
TARGET_SPECTATOR = [i for i, label in enumerate(THREE_QUBIT_PAULIS) if label[1] != "I" and label[2] != "I"]


@pytest.mark.validation
def test_rotary_seeds_leave_six_percent_of_the_first_order_zz_in_an_ix_zx_zz_model():
    """quchip solves the IX/ZX/ZZ model of the echoed CR at the seeds' exact turns to 1e-6 rad in each signed
    error term. The error beyond first order in ζ falls as ζ², and at the 2.59-turn seeds the first-order
    target-spectator part is 6% of the template's.

    With instant echoes the control enters only as Z_c, so each control state is one run of a two-level target
    with a cross-Kerr ``ζ n_t n_s`` to a two-level spectator.
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.linalg import logm

    chip = three_transmons()
    shape = GaussianEdge(duration=200.0, edge_duration=50.0, sigmas=2.5, amplitude=1.0)
    area = float(np.real(shape.area()))
    turns_of, _, _ = cr_turns(chip, area)
    # The template, and the seeds at 2.59 turns along ZX and against it.
    seeds = cross_resonance(chip, "q0", "q1", duration=400.0, edge=50.0, rotary=True).seeds[4:6]
    starts = [(0.0, 0.125), *((float(np.real(ix)), float(np.real(zx))) for ix, zx in map(turns_of, (
        seed.values for seed in seeds)))]
    assert abs(abs(starts[1][0]) - 2.59) < 0.01 and starts[1][0] * starts[2][0] < 0
    ideal = np.kron(ideal_unitary("ZX90"), np.eye(2))
    zeta = float(chip.dispersive_shift("q1", "q2"))

    def coefficients(k):
        return np.array([np.real(np.trace(np.kron(np.kron(PAULIS[a], PAULIS[b]), PAULIS[c]) @ k)) / 8
                         for a, b, c in THREE_QUBIT_PAULIS])

    def exact(ix, zx, zeta):
        k = 1j * logm(rotary_model(shape, ix, zx, zeta) @ ideal.conj().T)
        return coefficients((k + k.conj().T) / 2)

    # The first-order generator after the gate, 2π ζ ∫ U0(T, t) n_t n_s U0(T, t)† dt, with U0 the target's turn.
    times = np.linspace(0.0, 200.0, 20001)
    weights = np.full(times.size, times[1])
    weights[[0, -1]] /= 2
    running = cumulative_trapezoid(np.real(shape.sample(times)), times, initial=0.0) / area
    number = np.diag([0.0, 1.0])

    def first_order(ix, zx, zeta):
        k = np.zeros((8, 8), dtype=complex)
        for c, sign in enumerate((1, -1)):
            first, second = ix + sign * zx, -ix + sign * zx
            left = 2 * np.pi * np.concatenate([first * (1 - running) + second, second * (1 - running)])
            turn = np.cos(left / 2)[:, None, None] * np.eye(2) - 1j * np.sin(left / 2)[:, None, None] * PAULIS["X"]
            inner = np.einsum("t,tij,jk,tlk->il", np.concatenate([weights, weights]), turn, number, turn.conj())
            k[4 * c:4 * c + 4, 4 * c:4 * c + 4] = 2 * np.pi * zeta * np.kron(inner, number)
        return coefficients(k)

    for ix, zx in starts:
        blocks = []
        for sign in (1, -1):
            target, spectator = Qubit(freq=5.1, label="t"), Qubit(freq=5.2, label="s")
            model = Chip([target, spectator], couplings=[CrossKerr(target, spectator, chi=zeta, label="zz")],
                         frame="rotating")
            line = ChargeDrive(target, label="xy")
            model.wire(line)
            element = complex(model.drive_matrix_elements(({}, {target: 1}), drives=[line])[line])
            sequence = QuantumSequence(model)
            for turns in (ix + sign * zx, -ix + sign * zx):
                sequence.schedule("xy", envelope=shape.with_params({"amplitude": abs(turns) / (abs(element) * area)}),
                                  freq=model.freq("t"), frame="t", phase=float(np.angle(np.sign(turns) / element)))
            blocks.append(np.asarray(sequence.subspace_propagator(["t", "s"], tlist=[0.0, 400.0],
                                                                  options={"atol": 1e-11, "rtol": 1e-9}).matrix))
        generator, _ = error_generator(np.block([[blocks[0], np.zeros((4, 4))], [np.zeros((4, 4)), blocks[1]]]),
                                       ideal)
        reference = exact(ix, zx, zeta)
        assert np.abs(np.array([generator.get(label, 0.0) for label in THREE_QUBIT_PAULIS]) - reference).max() < 1e-6
        beyond = np.linalg.norm(reference - first_order(ix, zx, zeta))
        assert beyond / np.linalg.norm(exact(ix, zx, zeta / 2) - first_order(ix, zx, zeta / 2)) == pytest.approx(
            4.0, abs=0.1)
    template, *seeded = (np.linalg.norm(first_order(ix, zx, zeta)[TARGET_SPECTATOR]) for ix, zx in starts)
    assert all(0.05 < norm / template < 0.07 for norm in seeded)


@pytest.mark.validation
def test_rotary_seeds_cut_the_zz_error_on_the_chain_as_the_ix_zx_zz_model_predicts():
    """On the chain, the ZX90's target-spectator error at the template and at the 2.59-turn seeds is within 5% of
    the IX/ZX/ZZ model with the 40 ns echoes as windows of ZZ alone, and the seeds cut it to about 11%."""
    chip = three_transmons()
    # The norms are compared at 5%, so the solves can use looser tolerances than the gate set's defaults.
    gates = GateSet.standard(chip, ["q0", "q1", "q2"], pairs=[("q0", "q1")], cr_duration=400.0, rotary=False,
                             options={"atol": 1e-10, "rtol": 1e-8})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gates.calibrate(("ZX90", ("q0", "q1")), max_nfev=0)
    native = gates.native("ZX90", ("q0", "q1"))
    knobs = native.resolved_knobs()
    seeds = cross_resonance(chip, "q0", "q1", duration=400.0, rotary=True).seeds[4:6]
    assert all("2.59" in seed.provenance for seed in seeds)
    # The native's envelope, with the default edge from the dressed detuning.
    edge = 2 * 2.5 * 6.0 / (2 * np.pi * abs(chip.freq("q0") - chip.freq("q1")))
    shape = GaussianEdge(duration=200.0, edge_duration=edge, sigmas=2.5, amplitude=1.0)
    turns_of, _, _ = cr_turns(chip, float(np.real(shape.area())))
    ideal = np.kron(native.ideal_matrix(), np.eye(2))
    z_bits = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
    zeta = float(chip.dispersive_shift("q1", "q2"))
    window = gates.native("X", "q0").duration
    full, model = [], []
    for values in ({knob.name: knob.initial for knob in knobs}, *(seed.values for seed in seeds)):
        bindings = {}
        for knob in knobs:
            bindings.update(knob.bindings(float(values[knob.name])))
        block, _ = gates._block(bind_pulses(native.sequence, bindings), ("q0", "q1", "q2"), native.duration)
        matrix, leaked = np.asarray(block.matrix), np.asarray(block.leaked)
        phases = _Residual(None, ideal, z_bits, knobs).phase_start(matrix, leaked)[:-1]
        generator, _ = error_generator(np.exp(1j * (z_bits @ phases))[:, None] * matrix, ideal)
        full.append(np.linalg.norm([generator.get(THREE_QUBIT_PAULIS[i], 0.0) for i in TARGET_SPECTATOR]))
        ix, zx = turns_of(values)
        generator, _ = error_generator(rotary_model(shape, float(np.real(ix)), float(np.real(zx)), zeta, window), ideal)
        model.append(np.linalg.norm([generator.get(THREE_QUBIT_PAULIS[i], 0.0) for i in TARGET_SPECTATOR]))
    assert np.allclose(full, model, rtol=0.05, atol=0.0)
    assert np.allclose(np.array(full[1:]) / full[0], np.array(model[1:]) / model[0], rtol=0.1, atol=0.0)
