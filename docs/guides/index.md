# Guides

Each guide teaches one quchip workflow on a small chip. Read them in order, or
start with the one that matches your question. New to quchip? Start with
{doc}`your first chip <../get-started/first-chip>`.

1. {doc}`Spectra and parameter sweeps <spectra-and-sweeps>`: compare dressed
   observables, track states through an avoided crossing, and change
   parameters.
2. {doc}`Pulses, leakage, and readout <pulses-and-readout>`: compare pulse
   bandwidth and leakage, read the pulses as gates, and follow the conditional
   resonator response.
3. {doc}`Readout and fridge wiring <steady-state-and-vna>`: calculate VNA
   traces, their resonances, and qubit readout with receiver noise.
4. {doc}`Model reduction <model-reduction>`: compare a reduced model with the
   full pulse simulation, remove a readout from its feedline, solve in an exact
   excitation space, and truncate the excitation number.
5. {doc}`Gradients and parameter fitting <gradients-and-fitting>`:
   differentiate dressed observables and pulse responses, and fit parameters.
6. {doc}`Gate sets and circuits <gate-sets>`: calibrate native gates and
   compose circuits from their cached propagators.
7. {doc}`Resonance fluorescence and photon correlations <resonance-fluorescence>`:
   read the output spectrum, `g1` and `g2` of a driven emitter.
8. {doc}`Import circuits from scqubits <scqubits>`: turn scqubits devices and
   coupled systems into quchip models and back.
9. {doc}`Extending quchip <../extensions>`: define your own operators,
   models, and controls.

Guides 1, 2, 4, and 5 are also Jupyter notebooks, and each of these pages
links its notebook. The {doc}`studies <../studies/index>` apply the workflows
to specific physical questions. The {doc}`cookbook <../cookbook>` collects API
choices and common pitfalls. The {doc}`SQA 2026 page <from-sqa-2026>` links
each calculation of the talk to its guide.

```{toctree}
:hidden:
:maxdepth: 1

Spectra and parameter sweeps <spectra-and-sweeps>
Pulses, leakage, and readout <pulses-and-readout>
Readout and fridge wiring <steady-state-and-vna>
Model reduction <model-reduction>
Gradients and parameter fitting <gradients-and-fitting>
Gate sets and circuits <gate-sets>
Resonance fluorescence and photon correlations <resonance-fluorescence>
Import circuits from scqubits <scqubits>
Extending quchip <../extensions>
```
