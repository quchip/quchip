# From the SQA 2026 talk

Define the chip once, then ask static, dynamic, structural, or derivative
questions through public APIs. The snippets below are small enough to run as
written. Each of the five entry points has one link to its full guide.

quchip uses GHz for frequencies and couplings, ns for time, and mK for
temperature.

## Define and inspect a chip

```python
from quchip import Capacitive, Chip, DuffingTransmon, Resonator

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
chip = Chip([q, r], [Capacitive(q, r, g=0.05, label="qr")])

print("devices:", [device.label for device in chip.devices])
print("authored H:", chip.unresolved_hamiltonian().latex())
```

Output:

```text
devices: ['q', 'r']
authored H: \omega_{q}\,\hat n_{q} + 0.5\,\alpha_{q}\,\hat n_{q}\,(\hat n_{q} - \hat I_{q}) + \omega_{r}\,\hat n_{r} + g_{qr}\,(\hat a_{q} + \hat a^\dagger_{q})\,(\hat a_{r} + \hat a^\dagger_{r})
```

[Continue with chip definition and inspection](https://docs.quchip.org/guides/defining-and-inspecting-a-chip).

## Read and sweep statics

```python
import numpy as np

from quchip import Capacitive, Chip, DuffingTransmon, Resonator

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
chip = Chip([q, r], [Capacitive(q, r, g=0.05, label="qr")])

frequencies = np.linspace(4.9, 5.1, 21)
dressed_f01 = np.array(
    [chip.with_params({"q.freq": value}).freq("q") for value in frequencies]
)

print("first and last dressed f01 (GHz):", dressed_f01[[0, -1]])
```

Output:

```text
first and last dressed f01 (GHz): [4.89859099 5.09846968]
```

[Continue with statics and parameter studies](https://docs.quchip.org/guides/statics-and-parameter-studies).

## Simulate one pulse

```python
import numpy as np

from quchip import ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
chip = Chip([q], frame="rotating")
line = ChargeDrive(q, label="xy")
chip.wire(line)

sequence = QuantumSequence(chip)
sequence.schedule(
    line,
    envelope=Gaussian(duration=20.0, sigmas=3.0, amplitude=0.04),
    freq=chip.freq(q),
)
result = sequence.simulate(tlist=np.linspace(0.0, 30.0, 121))

print("final excited-state population:", result.population("q", 1)[-1])
```

Output:

```text
final excited-state population: 0.7450958150852982
```

[Continue with dynamics, pulses, observables, and readout](https://docs.quchip.org/guides/dynamics-pulses-and-readout).

## Transform a chip

```python
from quchip import Capacitive, Chip, DuffingTransmon, Resonator, eliminate

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
chip = Chip([q, r], [Capacitive(q, r, g=0.05, label="qr")])

fold = eliminate(chip, "r")
validity = fold.validity["qr"]

print("before:", [device.label for device in chip.devices])
print("after:", [device.label for device in fold.chip.devices])
print("g / detuning:", validity["g_over_delta"])
print("valid:", validity["is_valid"])
```

Output:

```text
before: ['q', 'r']
after: ['q']
g / detuning: 0.025
valid: True
```

[Continue with chip transformations](https://docs.quchip.org/guides/chip-transformations).

## Differentiate a static loss

The second residual uses the common sigma-z convention
$\chi_{\sigma_z}=\left(E_{11}-E_{10}-E_{01}+E_{00}\right)/2$.
`dispersive_shift()` itself returns the full pull in the numerator.

```python
import jax
import jax.numpy as jnp

from quchip import Capacitive, Chip, DuffingTransmon, Resonator
from quchip.backend.dynamiqs import DynamiqsBackend

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=4, label="r")
chip = Chip(
    [q, r],
    [Capacitive(q, r, g=0.05, label="qr")],
    backend=DynamiqsBackend(),
)

names = ("q.freq", "q.anharmonicity", "qr.g")
theta = jnp.array([5.0, -0.25, 0.05])
target = jnp.array([5.05, -0.0010])  # [f01, chi_sigma_z] in GHz


def residual(values):
    rebound = chip.with_params(dict(zip(names, values)))
    chi_sigma_z = rebound.dispersive_shift("q", "r") / 2
    observables = jnp.stack([rebound.freq("q"), chi_sigma_z])
    return observables - target


loss = lambda values: jnp.sum(residual(values) ** 2)

print("gradient:", jax.grad(loss)(theta))
print("Jacobian:\n", jax.jacrev(residual)(theta))
```

Output:

```text
gradient: [-1.02871002e-01 -2.85237283e-06  6.02561564e-03]
Jacobian:
 [[ 9.99395008e-01  3.62188602e-05 -5.86342171e-02]
 [-1.29839799e-04  5.10942405e-04 -5.70796457e-03]]
```

[Continue with experimental static fitting, dynamic losses, and multi-sequence analysis](https://docs.quchip.org/guides/differentiability).
