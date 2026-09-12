# Guides

Learn the main quchip workflows through coupled models, pulse simulations,
microwave measurements, reductions and fitting.

New to quchip? Start with {doc}`your first chip <defining-and-inspecting-a-chip>`.

## Spectra and dynamics

- {doc}`Spectra and parameter sweeps <statics-and-parameter-studies>`:
  compare dressed observables and track states through avoided crossings.
- {doc}`Pulses, leakage, and readout <dynamics-pulses-and-readout>`:
  compare pulse bandwidth, leakage, and conditional resonator response.
- {doc}`What happens during a qubit measurement? <continuous-measurement>`:
  follow single trajectories, map their density, and condition on an endpoint.

## Microwave networks

- {doc}`Readout and fridge wiring <steady-state-and-vna>`:
  calculate VNA traces and qubit readout with receiver noise.

## Model reduction and fitting

- {doc}`Model reduction <chip-transformations>`:
  compare a reduced model with the full pulse simulation.
- {doc}`Gradients and parameter fitting <differentiability>`:
  differentiate observables and fit shared model parameters.

## From the talk

The {doc}`SQA 2026 examples <from-sqa-2026>` cover five short calculations
from the talk.

For studies of specific physical questions, see {doc}`../studies/index`.
For API choices and common pitfalls, see the {doc}`cookbook <../cookbook>`.

```{toctree}
:hidden:
:maxdepth: 1

Spectra and parameter sweeps <statics-and-parameter-studies>
Pulses, leakage, and readout <dynamics-pulses-and-readout>
Readout and fridge wiring <steady-state-and-vna>
Model reduction <chip-transformations>
Gradients and parameter fitting <differentiability>
Cookbook <../cookbook>
SQA 2026 examples <from-sqa-2026>
What happens during a qubit measurement? <continuous-measurement>
```
