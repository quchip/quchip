# Resolve and sweep a chip

Start by declaring the devices and reading their dressed transitions.

```python
from quchip import Capacitive, Chip, DuffingTransmon, Resonator

q1 = DuffingTransmon(freq=5.30, anharmonicity=-0.65, levels=4, label="q1")
q2 = DuffingTransmon(freq=5.58, anharmonicity=-0.65, levels=4, label="q2")
bus = Resonator(freq=6.55, levels=4, label="bus")

chip = Chip(
    [q1, q2, bus],
    [Capacitive(q1, bus, g=0.05), Capacitive(q2, bus, g=0.05)],
    frame="rotating",
)

print("dressed f01:", {d.label: float(chip.freq(d)) for d in chip.devices})
print("static ZZ:", float(chip.static_zz(q1, q2)))
```

Now vary one declared parameter. `SpectrumSweep` creates a chip for each point;
it does not mutate `chip`.

```python
import numpy as np
from quchip import SpectrumSweep, Sweep

q2_freqs = np.linspace(5.255, 5.345, 181)
sweep = SpectrumSweep(
    chip,
    [Sweep(q2_freqs, name="q2.freq")],
    evals_count=4,
    overlap_threshold=0.0,
).run(progress=False)

transitions = sweep.eigenvalues[:, 1:3] - sweep.eigenvalues[:, :1]
splitting = transitions[:, 1] - transitions[:, 0]
print("minimum splitting (MHz):", 1e3 * float(splitting.min()))
```

The full notebook adds the talk parameters, the RWA ledger, the avoided-crossing
plot, and the comparison with the second-order exchange scale.

```{figure} ../images/resolve_and_sweep.png
:width: 720px
:alt: Two dressed transmon transitions forming an avoided crossing as one bare frequency is swept

The bare declarations cross while the dressed transitions remain separated by
the 4.4 MHz splitting in Fig. 1.
```

{download}`Download the executed notebook <../../examples/01_resolve_and_sweep.ipynb>`
or read its {download}`Jupytext source <../../examples/01_resolve_and_sweep.md>`.
