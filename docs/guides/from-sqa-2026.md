# From the SQA 2026 talk

The talk used four views of the same idea: declare a chip once, then ask static,
dynamic, structural, or derivative questions through public APIs. The snippets
below are small enough to run as written. Each topic has one link to the guide
that develops it.

quchip uses GHz for frequencies and couplings, ns for time, and mK for
temperature.

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

[Continue with dynamics, pulses, observables, and readout](https://docs.quchip.org/guides/dynamics-pulses-and-readout).

## Transform a chip

```python
from quchip import Capacitive, Chip, DuffingTransmon, Resonator, eliminate

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
chip = Chip([q, r], [Capacitive(q, r, g=0.05, label="qr")])

fold = eliminate(chip, "r")

print("before:", [device.label for device in chip.devices])
print("after:", [device.label for device in fold.chip.devices])
print("validity:", fold.validity)
```

[Continue with chip transformations](https://docs.quchip.org/guides/chip-transformations).

## Differentiate a static loss

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
target = jnp.array([5.05, -0.0010])


def residual(values):
    rebound = chip.with_params(dict(zip(names, values)))
    observables = jnp.stack(
        [rebound.freq("q"), rebound.dispersive_shift("q", "r") / 2]
    )
    return observables - target


loss = lambda values: jnp.sum(residual(values) ** 2)

print("gradient shape:", jax.grad(loss)(theta).shape)
print("Jacobian shape:", jax.jacrev(residual)(theta).shape)
```

[Continue with static, dynamic, and multi-sequence losses](https://docs.quchip.org/guides/differentiability).
