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

# Conditional diffusive monitoring

Monitor half of a resonator's decay channel. The unobserved half remains a loss,
so each conditional state is a density matrix. Native records describe the
selected coupling in the integration frame; downstream receiver noise and a
coherent incident-field offset require separate analysis.

```python
import numpy as np
import jax
import dynamiqs as dq
from quchip import Chip, QuantumSequence, Resonator, with_monitoring

for backend, solver in (("qutip", "smesolve"), ("dynamiqs", "dsmesolve")):
    r = Resonator(freq=1.0, levels=3, T1=2.0, label="r")
    chip = Chip([r], backend=backend, frame="rotating")
    run = ({"ntraj": 16, "seeds": 11} if backend == "qutip" else
           {"keys": jax.random.split(jax.random.key(11), 16),
            "method": dq.method.EulerMaruyama(dt=0.001)})
    options = ({"dt": 0.001, "store_measurement": "end", "keep_runs_results": True,
                "progress_bar": ""} if backend == "qutip" else {})
    problem = QuantumSequence(chip).build_problem(
        np.linspace(0.0, 0.2, 11), solver=solver, run_args=run, options=options,
        initial_state={"r": 1}, e_ops=chip.e_ops(r="n"), states="all",
    )
    channel = problem.engine_result.slh.channels[0]
    result = chip.solve(with_monitoring(problem, {channel.key: 0.5}))
    conditional = result.run(0)
    print(f"{backend}: conditional final occupation {conditional.expect('r')[-1].real:.3f}")
    print(f"  record channel: {result.monitor_labels[0]}")
```

<!-- executed-output:start -->

Output:

```text
qutip: conditional final occupation 0.950
  record channel: hidden.r.thermal_emission
```

```text
dynamiqs: conditional final occupation 0.825
  record channel: hidden.r.thermal_emission
```

<!-- executed-output:end -->

QuTiP exposes `result.native.measurement`; Dynamiqs exposes
`result.native.measurements`. Keep each library's interval and normalization
conventions. Compare timestep refinements before using a record quantitatively.
Dynamiqs 0.3.4 SME runs eagerly; its public validation prevents full outer JIT.

For efficiency eta, monitored and unobserved couplings are sqrt(eta)L and
sqrt(1-eta)L. Their summed dissipator equals D[L]. See
[Wiseman and Milburn, chapter 4](https://doi.org/10.1017/CBO9780511813948).
