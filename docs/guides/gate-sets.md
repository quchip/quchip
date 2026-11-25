# Gate sets and circuits

A gate set turns a chip into calibrated gates. You declare native gates as pulse
templates with knobs, calibrate each native against its ideal unitary, and then
compose circuits from the calibrated gates. quchip solves each native once and
reuses its propagator at every place in a circuit where the physics allows it.

## Calibrate a standard gate set

`GateSet.standard()` declares a DRAG SX and X on each qubit and an echoed
cross-resonance ZX90 on each directed pair. Without `qubits`, it takes the
computational devices with one charge line. Without `cr_duration`, it picks
the shortest CR duration, in steps of 10 ns, at which the seed CR drive stays
weak on the control. The seed's Rabi rate on the control's 0-1 and 1-2
transitions, and its perturbative two-photon rate on the 0-2 transition, stay
at most 0.4 of each transition's detuning from the tone. The fit can raise the
drive past this bound. By default, the nearest of these transitions also sets
the Gaussian edges of the CR pulses.

```python
from quchip import Capacitive, ChargeDrive, Chip, DuffingTransmon
from quchip.gateset import GateSet

q0 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q0")
q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q1")
chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003, label="J")])
chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))

gateset = GateSet.standard(chip, pairs=[("q0", "q1")])
gateset.calibrate()
print(gateset.describe())
```

One notice gives the choices as arguments to pass:
`qubits=['q0', 'q1']` and `cr_duration=290.0`. The control's 0-1 transition,
100 MHz from the tone, sets the duration. The control sits 100 MHz above the target.
With the control 100 MHz below the target, the same rule gives 420 ns. The
rule refuses a tone less than 5 MHz from one of these transitions. It warns
when the tone is less than 60 MHz from the 1-2 transition, near the avoided
crossing of `|11>` and `|20>`, and when the qubits are less than 60 MHz apart.

`calibrate()` prints a line for each native as its fit ends, and
`progress=False` turns this off. Each calibration reports the average gate
infidelity, the leakage, the fitted knobs, the generator of the error after the
ideal gate, and a status. A status other than `"ok"` comes with a warning:

- `"at_bound"`: a knob ends at a bound, so the optimum is outside the range.
- `"weak_direction"`: the fit does not determine some knob combination.
- `"unconverged"`: the fit used all `max_nfev` evaluations before it converged.
- `"solver_floor"`: the error is below ten times the integrator's unitarity
  defect. This is a hint, not a bound: the defect does not measure coherent
  phase errors of the solver.
- `"poor_fit"`: the infidelity is above the target (default `1e-3`) at start
  time 0, at the worst of the start times that the placement check samples,
  or in the worst computational state of the other qubits. Each state of the
  other qubits is scored on its own, so this conditional-state score does not
  see phases that these qubits gather among themselves during the gate, such
  as their own static ZZ. The joint-block score sees them only between
  spectators in the block. Circuits see them all through `unitary()` and
  `run()`.
- `"stale"`: the chip changed after the calibration, for example through
  `q0.freq = ...`, or a native inside this composite was calibrated again
  after it. The gate keeps its earlier pulses, and a circuit that uses it
  warns. Calibrate it again. The built-in natives then take their carriers
  and seeds from the changed chip. A CR keeps its duration, and warns when the
  automatic choice for the changed chip differs. Its edges get longer as the
  qubits come closer in frequency, and a change that makes them too long for
  the kept duration raises an error. For a sweep that brings the qubits close,
  pass a `cr_duration` that is long enough at the closest point. A native that
  you declare keeps its template, so declare it again when its carrier comes
  from the chip.

`gateset.calibration("ZX90", ("q0", "q1")).summary()` prints the full report.
For gates on up to five qubits it also gives the eigenvalue infidelity, a
spectral score that pyGSTi also reports. It compares eigenvalues only, so it
does not see errors in the eigenvectors, such as a small tilt of the rotation
axis, and it can be much smaller than the average gate infidelity. Some of
these errors only tilt the gate against state preparation and measurement, but
not all: an SY in place of an SX scores 0. `.channel` gives the gate as a
`SubspaceChannel`.

Calibrating a native again starts from its previous knobs. When that fit
converges within the target and at most doubles the previous error, the gate
set keeps it and tries no other start, so a gate that still fits costs a few
solves. Otherwise, and after `calibrate(max_nfev=0)`, every start competes,
the previous knobs among them.

`gateset.calibrate(max_nfev=0)` skips the knob search. Each native keeps its
best start, which for the built-in natives is a closed-form seed from the
chip, and the fit sets only the virtual-Z phases. On the chip above, the seeds
give SX and X errors within a factor of three of the calibrated ones. The ZX90
seed gives `7e-3`, against `4e-5` after calibration, so calibration matters
most for the two-qubit gate. A circuit that uses an uncalibrated gate
calibrates it this way, with a notice, so `gateset.unitary(...)` also works
before `calibrate()`.

## What the fit targets

The calibration block holds the gate's qubits and the gate-set qubits that
couple to them or that the gate's pulses reach through line crosstalk. The fit
targets the ideal gate times the identity on these spectators, with a free
virtual Z on every qubit of the block. Static ZZ and drive crosstalk during
the gate are part of the error. The report also gives the error with all
spectators in `|0>`, with each other qubit in `|1>`, and in the state of the
other qubits where it is largest. Pass
`GateSet(..., spectators="none")` to fit with the other qubits in `|0>`.

Static ZZ between a CR target and another qubit adds an error during the CR
tones that the echo does not remove. When the target couples to another
gate-set qubit, directly or through a bus, `GateSet.standard()` gives the
echoed ZX90 rotary starts. In each, the target tone also turns the target
about the ZX axis by N turns per half, along the axis or against it, and the
echo turns it back. N takes the values at which this turn alone would average
the ZZ with the other qubit out to first order over a half. With the Gaussian
edges, these lie between whole turns. The CR also turns the target, by an
eighth of a turn per half in a direction set by the control, so the ZZ error
is reduced, not removed. The calibration solves each start once, fits from the
template and from the two best starts, and keeps the best fit.
`calibration.seed` names the start. On a chain of three transmons 100 MHz
apart with `g = 3 MHz`, the ZX90 on the first pair goes from `5.5e-3` to
`1.6e-4`, with about three times the solves. Pass `rotary=True` or
`rotary=False` to `GateSet.standard()` or `cross_resonance()` to choose.

The fit places the gate at time 0. A gate with one carrier, such as the
built-in SX and X, has the same error at every start time. The echoed CR uses
the target's carrier for its CR pulses and the control's for its echo, so its
error changes with the start time. The report gives the smallest and largest
error over start times in [0, 100) ns.

RZ is virtual: a circuit `RZ(λ)` becomes `vz(q, -λ)` and costs no time. The
free Z phases of each fit become trailing `vz` entries of the calibrated
sequence.

## Compose and run circuits

```python
chip.gateset = gateset
circuit = [("H", "q0"), ("CX", ("q0", "q1"))]
run = chip.run(circuit)
print(run.probabilities, 1 - run.fidelity())
print(1 - gateset.unitary(circuit).fidelity())
```

`chip.run()` runs a circuit with the gate set in `chip.gateset`. It solves
the circuit's schedule from `|00>`, or from a bit string or register state
that you give. The solve includes the chip's collapse channels unless you pass
`dissipation=False`. `run.fidelity()` compares the final state with the
circuit's ideal output and gives `1 - F = 1.5e-4` here.
`gateset.run(circuit)` gives the same result, and
`chip.run(circuit, gateset=other)` runs a second gate set on this chip, for
example to compare two calibrations. Copies of the chip, such as those from
`with_params()`, have no gate set.

`unitary()` composes cached gate propagators, so a long circuit costs a few
matrix products. Its `fidelity()` averages over all inputs and gives
`1 - F_avg = 2.1e-4` here. Gates that overlap in time, gates on a chip
without the RWA, gates whose pulses depend on absolute time, and gates on a
drive that overrides `signal()` are solved directly, once per placement.
Under an automatic excitation cutoff, a circuit with gates that overlap in
time is solved in one piece (see Large chips).

The decomposition rules turn H, X, Y, Z, S, T, RX, RY, CX, CZ, and SWAP into
the natives. A CX against the direction of the gate set's ZX90 gets an H on
both qubits before and after it. Add your own with `gateset.rule(name, qubits, expansion)`, where
`qubits` is the gate's qubit count and `expansion` returns the instructions
that replace it.

## Gradients and drift

On the dynamiqs backend, `unitary()` and `run()` take JAX-traced RZ angles,
so `jax.grad` and `jax.jit` reach a circuit's fidelity and probabilities.
Inside `jax.jit`, the gate set solves each gate once, outside the trace, and
the compiled function keeps only the algebra of the angles.

`gateset.with_params(bindings)` evaluates the calibrated gates at changed or
traced chip and pulse values, without calibrating again. `gateset.parameters`
lists the paths: chip values such as `"q0.freq"`, knobs such as
`"SX[q0].beta"`, and pulse values such as `"SX[q0].pulse.0.amplitude"`. The
pulses keep their calibrated carriers and timing, and the frame clocks keep
the calibrated qubit frequencies, so a drifted qubit shows its error as a Z
rotation. The dressed states and the idle evolution follow the changed chip.

```python
import jax
from quchip.gateset import drag_sx

d0 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q0")
d1 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q1")
pair = Chip([d0, d1], couplings=[Capacitive(d0, d1, g=0.003, label="J")], backend="dynamiqs")
pair.wire(ChargeDrive(d0, label="xy0"), ChargeDrive(d1, label="xy1"))
sx_set = GateSet(pair, ["q0", "q1"], backend="dynamiqs")
sx_set.add(drag_sx(pair, "q0"))
sx_set.calibrate()

def sx_error(frequency):
    drifted = sx_set.with_params({"q0.freq": frequency})
    return 1 - drifted.unitary([("SX", "q0")]).fidelity()

print(jax.grad(sx_error)(5.101))
```

A binding of a pulse's start time or duration, or of a delay in the signal
chain, raises, since the timing stays as calibrated. The copy evaluates the
gate set as it is when `with_params()` runs, so a later change of the gate
set does not reach it. A composite native, such as the echoed CR, holds its own copy
of the pulses of its parts, so a binding of a part does not reach it. A
binding of a pulse carrier, and any traced chip value, makes the gate set
solve each gate at each placement.

## Declare your own native

A native is a `QuantumSequence` that starts at time 0, its knobs, and its
ideal. Knobs name pulse fields, with optional bounds and scales. This SX on
`q0` goes into a new gate set, because the standard set already has one.

```python
import numpy as np
from quchip import GaussianDRAG, QuantumSequence
from quchip.gateset import Knob, NativeGate

element = chip.drive_matrix_elements("q0")["xy0"]
sequence = QuantumSequence(chip)
pulse = sequence.schedule("xy0", envelope=GaussianDRAG(duration=40.0, sigmas=3.0, amplitude=0.012, beta=0.24),
                          freq=chip.freq("q0"), phase=-np.angle(element))
native = NativeGate("SX", ("q0",), sequence,
                    knobs={"amplitude": Knob(pulse, "amplitude"), "beta": Knob(pulse, "beta", scale=0.1)},
                    ideal="SX", free_z=("q0",))
custom = GateSet(chip, ["q0", "q1"])
custom.add(native)
custom.calibrate()
```

A composite native, such as the echoed CR, builds its sequence from other
calibrated natives, so it follows their calibration.

## Large chips

When the chip's product space has more than 64 states and the chip sets no
space policy, gate-set solves use an excitation cutoff. The gate set starts at
one excitation above the number of gate-set qubits and raises the cutoff until
the gates stop changing, with the other qubits in `|0>` and in `|1>`. Each
drive adds a transient excitation, so `unitary()` and `run()` solve a
circuit with gates that run at the same time in one piece, with one more
excitation for each gate after the first. A warning states each cutoff. These
rules are heuristics, not bounds. Pass
`space=ExcitationCutoff(max_total=K)` to use one cutoff for every solve, or
`space=None` for the full product space.

## Solver settings

Gate-set solves use QuTiP's `vern9` integrator at `rtol=1e-8` and `atol=1e-10`,
or a dynamiqs `Dopri8` method at `rtol=1e-10` and `atol=1e-12`. In checks on
chips of two and three transmons, the calibrated errors changed by less than
2e-5 of their value against solves with `adams` at `rtol=1e-10` and
`atol=1e-12`, and the calibrations ran six to seven times faster. Pass
`options=` to `GateSet` or `GateSet.standard()` to replace these settings, in
the form of the backend that solves.

## Run qutip-qip circuits

`run()`, `unitary()`, `ideal()`, and `schedule()` also take a qutip-qip
`QubitCircuit`. They need qutip-qip 0.5 or later, which needs Python 3.12.
Install the optional dependency with `pip install quchip[qutip-qip]`. Until
qutip-qip 0.5 is on PyPI, install it from GitHub first with
`pip install git+https://github.com/qutip/qutip-qip.git`.

```python
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations.gates import CX, H

qc = QubitCircuit(2)
qc.add_gate(H, targets=0)
qc.add_gate(CX, controls=0, targets=1)
print(1 - chip.run(qc).fidelity(), 1 - gateset.unitary(qc).fidelity())
```

Circuit qubit `i` is `gateset.qubits[i]`, so this is the circuit above, with
the same errors. A gate that the gate set has as a native or a rule keeps its
name and angles. qutip-qip first resolves other gates, such as ISWAP, into CX,
RX, RY, and RZ. `from_qutip_qip(qc, gateset)` gives the gate-set
instructions.

`to_qutip_qip()` turns the gate set into a
[qutip-qip](https://qutip-qip.readthedocs.io) processor, for tools that work
on qutip-qip processors, such as `plot_pulses()`.

```python
from quchip import to_qutip_qip

processor = to_qutip_qip(gateset)
result = processor.run_state(qc=qc)
state = processor.get_final_circuit_state(result.states[-1])
processor.plot_pulses()
```

The processor compiles each gate with the natives and rules of the gate set,
and `run_state` integrates the chip's Hamiltonian. Static ZZ, crosstalk, and
leakage are in the result, which matches `chip.run(qc)`. `run_state` starts
in the dressed `|0...0⟩`. Give it a bit string such as `"01"`, or a register
state, to start elsewhere. Register states use the dressed computational basis,
as in `run()`, and leakage shows as missing norm or trace.
`run_state(noisy=True)`, the default, includes the chip's collapse channels:
set T1 and T2 on the devices, not with `add_noise()`. A circuit that uses a
native without a calibration runs its closed-form seed with fitted virtual-Z
phases, with a warning.
