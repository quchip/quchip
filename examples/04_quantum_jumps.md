---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- reader-content -->

# Quantum jumps from a decaying excitation

A single excitation decays at rate 1/T1. Individual runs jump to the ground
state; the ensemble occupation approaches exp(-t/T1). Both backends assemble
the same resonator model. Times are in ns and rates in 1/ns.

```python
import numpy as np
import jax
import dynamiqs as dq
from quchip import Chip, QuantumSequence, Resonator

for backend, solver in (("qutip", "mcsolve"), ("dynamiqs", "jssesolve")):
    r = Resonator(freq=1.0, levels=2, T1=1.0, label="r")
    chip = Chip([r], backend=backend, frame="rotating")
    sequence = QuantumSequence(chip)
    run = ({"ntraj": 256, "seeds": 7} if backend == "qutip" else
           {"keys": jax.random.split(jax.random.key(7), 256),
            "method": dq.method.Event(dtmax=0.01)})
    options = {"keep_runs_results": True, "progress_bar": ""} if backend == "qutip" else {}
    result = sequence.simulate(
        np.linspace(0.0, 2.0, 21), solver=solver, run_args=run, options=options,
        initial_state={"r": 1}, e_ops=chip.e_ops(r="n"), states="all",
    )
    occupation = np.real(result.expect("r"))
    # Five standard errors using the Bernoulli variance bound 1/4.
    assert np.max(np.abs(occupation-np.exp(-result.times))) < 2.5/np.sqrt(256)
    print(f"{backend}: final occupation {occupation[-1]:.3f}; exponential {np.exp(-2):.3f}")
```

<!-- executed-output:start -->

Output:

```text
qutip: final occupation 0.133; exponential 0.135
```

```text
dynamiqs: final occupation 0.156; exponential 0.135
```

<!-- executed-output:end -->

`result.run(0)` provides the usual quchip analysis of one retained trajectory.
`result.native` preserves QuTiP collapse times and channel indices or Dynamiqs
padded click-time buffers. `result.channel_labels` maps their operator order to
the resolved physical channel keys. Refine Event's dtmax to test click-time
precision. More runs reduce sampling error; they do not reduce time-step error.

[QuTiP Monte Carlo solver](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-monte.html)
describes the trajectory and weighted-sampling conventions.
