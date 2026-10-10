# Your first simulation

How much population does a Gaussian pulse leave in a transmon's first excited
state? Declare a four-level transmon, wire a charge drive, and apply a 20 ns pulse.
Frequencies are in GHz and times are in ns.

## Declare the chip and drive

```python
import numpy as np
from quchip import RWA, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
chip = Chip([q], frame="rotating", approximation=RWA())
line = ChargeDrive(q, label="xy")
chip.wire(line)
```

The model uses a rotating frame and the rotating-wave approximation. It
declares no loss channels, so this calculation describes coherent
evolution.

## Schedule and simulate

```python
sequence = QuantumSequence(chip)
sequence.schedule(
    line,
    envelope=Gaussian(duration=20.0, sigmas=3.0, amplitude=0.04),
    freq=chip.freq(q),
)
result = sequence.simulate(
    tlist=np.linspace(0.0, 30.0, 121),
    initial_state=chip.state({q: 0}),
)
print(f"Final excited-state population: {result.population(q, 1)[-1]:.3f}")
```

```text
Final excited-state population: 0.745
```

The pulse transfers about 75% of the population into |1⟩, because it is not
calibrated as a π pulse and the higher levels let population leak into |2⟩.
Read the |2⟩ population with `result.population(q, 2)`.

For this pulse, five levels instead of four change the final |1⟩ population by
less than one part in a million.

Continue with {doc}`pulses, leakage, and readout <../guides/dynamics-pulses-and-readout>`
to compare short and long pulses on a coupled transmon–resonator model.
