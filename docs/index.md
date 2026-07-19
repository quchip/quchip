```{image} _static/quchip-wordmark-light.png
:class: only-light
:width: 380px
:align: center
:alt: quchip
```

```{image} _static/quchip-wordmark-dark.png
:class: only-dark
:width: 380px
:align: center
:alt: quchip
```

# Documentation

`quchip` is an open-source Python toolkit for modeling superconducting quantum chips.

A predictive chip model needs more than a Hamiltonian: device physics, control-line transformations, frames and approximations, dissipation, and measured observables all belong to it. quchip represents each part explicitly. Declare the chip once; the same declaration drives dressed-state analysis, model reduction, control sequencing, open-system simulation, parameter sweeps, and exact JAX gradients.

## Install

quchip requires Python 3.11 or newer.

```bash
pip install quchip
```

Optional extras: `quchip[dynamiqs]` for the JAX-native backend, `quchip[viz]` for graph visualization, `quchip[scqubits]` for scqubits interoperability.

## A minimal chip

```python
import numpy as np
from quchip import (
    Capacitive, ChargeDrive, Chip, DuffingTransmon,
    Gaussian, QuantumSequence, Resonator,
)

qubit = DuffingTransmon(freq=5.24, anharmonicity=-0.26, levels=3)
readout = Resonator(freq=6.65, levels=4)
chip = Chip(
    [qubit, readout],
    couplings=[Capacitive(qubit, readout, g=0.060)],
    frame="rotating",
)
drive = ChargeDrive(qubit)
chip.wire(drive)
sequence = QuantumSequence(chip)
sequence.schedule(
    drive,
    envelope=Gaussian(duration=40.0, amplitude=0.030),
    freq=chip.freq(qubit),
)
result = sequence.simulate(
    tlist=np.linspace(0.0, 40.0, 81),
    initial_state=chip.state({qubit: 0, readout: 0}),
    e_ops={qubit: qubit.projector(1, 1)},
)
print(float(result.expect_final(qubit).real))

fig = result.plot_populations(trace_out=readout)
fig.savefig("populations.png", dpi=200)
```

The pulse carrier comes from the dressed chip frequency; the printed value is the excited-state population after a nominal π pulse. The last two lines plot the qubit populations with the readout resonator traced out — the figure below is the saved output of the snippet.

```{figure} images/populations.png
:width: 560px
:alt: Qubit populations during the pi pulse
```

`quchip` uses GHz for ordinary frequencies, ns for time, and mK for temperature. The implemented conventions and approximations are recorded in the {doc}`physics reference <physics>`.

Worked examples and guides are being added incrementally.

```{toctree}
:maxdepth: 1
:hidden:

physics
api
contributing
conduct
```

## Project

- [GitHub](https://github.com/quchip/quchip)
- [PyPI](https://pypi.org/project/quchip/)
- License: Apache-2.0
