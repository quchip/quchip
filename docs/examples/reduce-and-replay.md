# Reduce and replay a chip

Build the schedule first. `active_patch()` keeps the scheduled device and its
neighbour, folds away the spectator, and rebinds the same schedule to the
smaller chip.

```python
import numpy as np
from quchip import (
    RWA, Capacitive, ChargeDrive, Chip, DuffingTransmon,
    Gaussian, QuantumSequence,
)

qs = [
    DuffingTransmon(freq=f, anharmonicity=-0.25, levels=3, label=f"q{i}")
    for i, f in enumerate([5.0, 5.35, 5.70])
]
chip = Chip(
    qs,
    [Capacitive(qs[i], qs[i + 1], g=0.004) for i in range(2)],
    frame="rotating",
    approximation=RWA(),
)

line = ChargeDrive(qs[0], label="xy")
chip.wire(line)
sequence = QuantumSequence(chip)
sequence.schedule(
    line,
    envelope=Gaussian(duration=20.0, sigmas=3.0, amplitude=0.04),
    freq=chip.freq(qs[0]),
)

patch = sequence.active_patch(hops=1, method="sw")
print("kept:", patch.active_labels)
print("eliminated:", patch.eliminated_labels)
print("validity:", patch.validity)
```

Replay both models on the same time grid and compare one observable.

```python
times = np.linspace(0.0, 40.0, 161)
full = sequence.simulate(tlist=times)
small = patch.simulate(tlist=times)

p_full = np.asarray(full.population("q0", level=1)).real
p_small = np.asarray(small.population("q0", level=1)).real
print("maximum population difference:", float(np.max(np.abs(p_full - p_small))))
```

The full notebook expands the chain to four transmons and plots both traces and
their residual. Read the returned validity record before trusting a reduced
model.

```{figure} ../images/reduce_and_replay.png
:width: 720px
:alt: Full and active-patch driven-qubit populations with their absolute residual below

One schedule drives the 81-state chip and its 9-state active patch.
```

{download}`Download the executed notebook <../../examples/02_reduce_and_replay.ipynb>`
or read its {download}`Jupytext source <../../examples/02_reduce_and_replay.md>`.
