# Gate sets and circuits

A gate set turns a chip into calibrated gates. You declare native gates as pulse
templates with knobs, calibrate each native against its ideal unitary, and then
compose circuits from the calibrated gates. quchip solves each native once and
reuses its propagator at every place in a circuit where the physics allows it.

## Calibrate a standard gate set

`GateSet.standard()` declares a DRAG SX and X on each qubit and an echoed
cross-resonance ZX90 on each directed pair. The CR duration depends on the
coupling and the detuning, so you give it.

```python
from quchip import Capacitive, ChargeDrive, Chip, DuffingTransmon
from quchip.gateset import GateSet

q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.33, levels=3, label="q0")
q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.33, levels=3, label="q1")
chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.003, label="J")])
chip.wire(ChargeDrive(q0, label="xy0"), ChargeDrive(q1, label="xy1"))

gates = GateSet.standard(chip, ["q0", "q1"], pairs=[("q0", "q1")], cr_duration=400.0)
gates.calibrate()
print(gates.describe())
```

Each calibration reports the average gate infidelity, the leakage, the fitted
knobs, the error generator, and a status. A status other than `"ok"` comes with
a warning:

- `"at_bound"`: a knob ends at a bound, so the optimum is outside the range.
- `"weak_direction"`: the fit does not determine some knob combination.
- `"unconverged"`: the fit used all `max_nfev` evaluations before it converged.
- `"solver_floor"`: the error is close to the integrator's unitarity defect.
- `"poor_fit"`: the infidelity is above the target (default `1e-3`) at start
  time 0, at the worst start time of the placement check below, or in the
  worst state of the other qubits.
- `"stale"`: a native inside this composite was calibrated again after it.
  The composite keeps the earlier pulses, and a circuit that uses it warns.
  Calibrate the composite again to use the new ones.

`gates.calibration("ZX90", ("q0", "q1")).summary()` prints the full report.

`gates.calibrate(max_nfev=0)` skips the knob search. Each native keeps its
template values, which for the built-in natives are closed-form seeds from the
chip, and the fit sets only the virtual-Z phases. On the chip above, the seeds
give SX and X errors within a factor of three of the calibrated ones and a ZX90
error near `4e-3`, so calibration matters most for the two-qubit gate.

## What the fit targets

The calibration block holds the gate's qubits and the gate-set qubits that
couple to them or that the gate's pulses reach through line crosstalk. The fit
targets the ideal gate times the identity on these spectators, with a free
virtual Z on every qubit of the block. Static ZZ and drive crosstalk during
the gate are part of the error. The report also gives the error with all
spectators in `|0>`, with each other qubit in `|1>`, and in the state of the
other qubits where it is largest. Pass
`GateSet(..., spectators="none")` to fit with the other qubits in `|0>`.

The fit places the gate at time 0. A gate with one carrier, such as the
built-in SX and X, has the same error at every start time. The echoed CR uses
the target's carrier for its CR pulses and the control's for its echo, so its
error changes with the start time. The report gives the smallest and largest
error over start times in [0, 100) ns.

RZ is virtual: a circuit `RZ(λ)` becomes `vz(q, -λ)` and costs no time. The
free Z phases of each fit become trailing `vz` entries of the calibrated
sequence.

## Compose and simulate circuits

```python
circuit = [("H", "q0"), ("CX", ("q0", "q1"))]
result = gates.unitary(circuit)
print(result.fidelity(), result.leakage)

run = gates.simulate(circuit, "00")
print(run.probabilities)
```

`unitary()` composes cached gate propagators, so a long circuit costs a few
matrix products. Gates that overlap in time, gates on a chip without the RWA,
gates whose pulses depend on absolute time, and gates on a drive that
overrides `signal()` are solved directly, once per placement. `simulate()` solves the circuit's schedule, with the chip's
collapse channels unless you pass `dissipation=False`.

The decomposition rules turn H, X, Y, Z, S, T, RX, RY, CX, CZ, and SWAP into
the natives. A CX against the direction of the gate set's ZX90 gets an H on
both qubits before and after it. Add your own with `gates.rule(name, qubits, expansion)`, where
`qubits` is the gate's qubit count and `expansion` returns the instructions
that replace it.

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
drive adds a transient excitation, so `unitary()` and `simulate()` solve gates
that run at the same time with one more excitation for each gate after the
first. A warning states each cutoff. Pass `space=ExcitationCutoff(max_total=K)`
to use one cutoff for every solve, or `space=None` for the full product space.
