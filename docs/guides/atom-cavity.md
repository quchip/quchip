# An atom coupled to a cavity

Calculate the cavity-induced shift of a two-level atom, drive it, and inspect
its reflected field. The Jaynes–Cummings model uses

```{math}
H/h = -f_a\sigma_z/2 + f_c a^\dagger a
      + g(\sigma_- a^\dagger + \sigma_+ a).
```

This model neglects counter-rotating interactions and higher atomic levels.
See [Jaynes and Cummings (1963)](https://doi.org/10.1109/PROC.1963.1664).
Frequencies are in GHz, times in ns, and decay rates in inverse ns. The
calculation below requires the `dynamiqs` extra for its JAX derivative.

## Declare the atom and interaction

`SpinHalf` supplies a two-level Hamiltonian and spin operators through a
`CustomSpace`. `Resonator` supplies a harmonic cavity. A coupling declares the
interaction using each endpoint's own operator names.

```python
import jax
import numpy as np

from quchip import (
    Chip, CouplingModel, DeviceDrive, PortNetwork, QuantumSequence,
    Resonator, Scalar, Square, VNA, as_operator_expr, eliminate, fit_a_dress, parameter,
)
from quchip.extensions import SpinHalf


class AtomCavity(CouplingModel):
    g: Scalar = parameter(unit="GHz", symbol="g")

    def interaction(self, atom, cavity, p):
        return p.g * (atom.sigma_minus * cavity.adag + atom.sigma_plus * cavity.a)


atom = SpinHalf(freq=5.0, T1=1000.0, label="atom")
cavity = Resonator(freq=5.3, levels=3, T1=500.0, label="cavity")
interaction = AtomCavity(atom, cavity, g=0.02, label="exchange")
chip = Chip([atom, cavity], [interaction], backend="dynamiqs", frame="rotating")

shift = float(chip.freq(atom) - atom.freq)
delta = atom.freq - cavity.freq
analytic_shift = (abs(delta) - np.sqrt(delta**2 + 4 * interaction.g**2)) / 2
np.testing.assert_allclose(shift, analytic_shift, atol=1e-12)
print(f"Atomic shift: {1000 * shift:.6f} MHz")
```

Output:

```text
Atomic shift: -1.327460 MHz
```

## Drive the atomic transition

A device drive maps the delivered classical signal to an operator. Here the
in-phase signal couples to the declared spin-x matrix. Use `chip.wire(a, b)`
to attach several lines together; calling `wire` again replaces the wiring.

```python
class AtomicDrive(DeviceDrive):
    def hamiltonian(self, target, signal):
        dipole = as_operator_expr(target.local_operator("sigma_x"),
                                  labels=(target.label,), dims=(2,), name="dipole")
        return signal.i * dipole


line = AtomicDrive(atom, label="laser")
chip.wire(line)
sequence = QuantumSequence(chip)
sequence.schedule(line, envelope=Square(duration=25.0, amplitude=0.02),
                  freq=chip.freq(atom))
pulse = sequence.simulate(tlist=np.linspace(0, 25, 101))
print(f"Excited population: {float(pulse.population(atom, 1)[-1]):.6f}")
```

Output:

```text
Excited population: 0.986225
```

For a scheduled interaction, subclass `CouplingDrive` and return
`signal.i * target.interaction_hamiltonian()` from `hamiltonian` to modulate
its full declared interaction by a dimensionless signal. Use the existing
`parametric_interaction` hook when the pump amplitude should instead carry
an independent coupling strength in GHz. This supports sideband or exchange
pulses without putting waveform logic in a device Hamiltonian.

## Observe the cavity and eliminate unobserved loss

Attach an accessible cavity channel with rate `0.003 / ns`. Intrinsic T1 loss
remains a separate, unobserved channel. A spin port can similarly select
`operator="sigma_minus"`.

```python
network = PortNetwork()
port = network.port("cavity_port", target=cavity, rate=0.003)
probe = network.expose("probe", at=port)
readout = Chip([atom, cavity], [interaction], port_network=network)
frequencies = np.linspace(5.28, 5.32, 21)
reflection = VNA(readout).sweep(frequencies).s(probe, probe)
assert np.isfinite(reflection).all()
print(f"Minimum reflection magnitude: {np.abs(reflection).min():.6f}")

reduced = eliminate(Chip([atom, cavity], [interaction]), cavity)
print(f"Inherited decay: {float(reduced.effective_params[atom]['purcell_rate']):.9f} / ns")
```

Output:

```text
Minimum reflection magnitude: 0.867345
Inherited decay: 0.000008876 / ns
```

This reduction removes the cavity from the model with intrinsic loss only.
Its decay is retained as a transformed channel, separate from the atom's T1.
The SW Hamiltonian is accurate through second order in coupling over
detuning; the result reports `g_over_delta` and the retained state map.
Eliminating a network boundary has additional restrictions described in the
{doc}`model-reduction guide <chip-transformations>`.

## Differentiate and fit a specified observable

The atom-like branch has derivative
$\partial f_a^{\rm dressed}/\partial g=-2g/\sqrt{\Delta^2+4g^2}$ at this
negative detuning. Compare the JAX result with that expression.

```python
def atomic_frequency(g):
    return chip.with_params({"exchange.g": g}).freq("atom")


derivative = float(jax.grad(atomic_frequency)(interaction.g))
expected = -2 * interaction.g / np.sqrt(delta**2 + 4 * interaction.g**2)
np.testing.assert_allclose(derivative, expected, rtol=1e-10, atol=1e-12)
print(f"Frequency derivative: {derivative:.6f}")

fit = fit_a_dress(
    chip,
    constraints={atom: {"freq": 5.0}, cavity: {"freq": None},
                 interaction: {"coupling_strength": None}},
    vary={atom: ["freq"]},
)
np.testing.assert_allclose(fit.chip.freq(atom), 5.0, atol=1e-8)
print(f"Required bare atomic frequency: {float(fit.chip[atom].freq):.6f} GHz")
```

Output:

```text
Frequency derivative: -0.132164
Required bare atomic frequency: 5.001333 GHz
```

The constraints remove the cavity and coupling defaults and fit only the
atomic transition. A custom coupling defaults to targeting its bare
`coupling_strength`; declare `default_fit_observable` or pass explicit
constraints when the desired quantity is exchange or a conditional shift.
The fitter currently supports frequency, anharmonicity, cross-Kerr, exchange,
and bare coupling targets. Other observables can be differentiated through
quchip and passed to an external optimizer.
