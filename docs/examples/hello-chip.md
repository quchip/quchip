# Drive and read out one chip

This is the smallest pulse example in the guide: one qubit, one resonator, one
Gaussian, and one solve.

```python
import numpy as np
from quchip import (
    RWA, Capacitive, ChargeDrive, Chip, DuffingTransmon,
    Gaussian, QuantumSequence, Resonator,
)

q = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=3, label="q")
r = Resonator(freq=6.8, levels=4, label="r")
chip = Chip(
    [q, r],
    [Capacitive(q, r, g=0.06, label="qr")],
    frame="rotating",
    approximation=RWA(),
)

line = ChargeDrive(q, label="xy")
chip.wire(line)
sequence = QuantumSequence(chip)
sequence.schedule(
    line,
    envelope=Gaussian(duration=30.0, sigmas=3.0, amplitude=0.025),
    freq=chip.freq(q),
)

result = sequence.simulate(
    tlist=np.linspace(0.0, 40.0, 161),
    initial_state=chip.state({q: 0, r: 0}),
)
result.plot_populations(trace_out=r)
```

From here, vary the pulse handle and call `simulate_batch()` to compare pulse
widths. Wire a second `ChargeDrive` to `r`, use
`chip.freq(r, when={q: level})` for the two conditional carriers, and batch the
two prepared qubit states.

The full notebook performs both expansions with the same chip.

```{figure} ../images/hello_qubit_drive_leakage.png
:width: 760px
:alt: Short and long Gaussian pulses with multilevel qubit populations

Two nominal-pi Gaussians with different bandwidths expose multilevel leakage.
```

```{figure} ../images/hello_dispersive_readout_iq.png
:width: 560px
:alt: Conditional resonator IQ paths with emphasized final points

The resonator follows different IQ paths for prepared dressed qubit states
$|0\rangle$ and $|1\rangle$.
```

{download}`Download the executed notebook <../../examples/00_hello_chip.ipynb>`
or read its {download}`Jupytext source <../../examples/00_hello_chip.md>`.
