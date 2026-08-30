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

`quchip` is an open-source Python toolkit for modelling superconducting quantum chips.

A predictive chip model needs more than a Hamiltonian: device physics, control-line transformations, frames and approximations, dissipation, and measured observables all belong to it. quchip represents each part explicitly. Declare the chip once; the same declaration drives dressed-state analysis, model reduction, control sequencing, open-system simulation, parameter sweeps, and exact JAX gradients.

The declared and resolved physics remain inspectable. `chip.unresolved_hamiltonian()` shows the authored static model, while `chip.hamiltonian()` applies the same basis, frame, and approximation strategy used by simulation. A sequence's Hamiltonian also includes its scheduled drives.

## Install

quchip requires Python 3.11 or newer.

```bash
pip install quchip
```

Optional extras: `quchip[dynamiqs]` for the JAX-native backend, `quchip[viz]` for graph visualization, `quchip[scqubits]` for scqubits interoperability.

The {doc}`backend guide <guides/choosing-a-backend>` shows how to select QuTiP
or dynamiqs, choose an integration method, and set tolerances, step controls,
batching, and gradients.

## Start with a physical question

The guides begin with a small runnable calculation and add one idea at a time:

- {doc}`Define and inspect a chip <guides/defining-and-inspecting-a-chip>`
- {doc}`Backend and solver options <guides/choosing-a-backend>`
- {doc}`Statics and parameter studies <guides/statics-and-parameter-studies>`
- {doc}`Dynamics, pulses, observables, and readout <guides/dynamics-pulses-and-readout>`
- {doc}`Chip transformations <guides/chip-transformations>`
- {doc}`Differentiability <guides/differentiability>`

The {doc}`cookbook` defines the conventions used by executable quchip examples.

```{figure} images/hello_qubit_drive_leakage.png
:width: 760px
:alt: Short and long Gaussian pulses with multilevel qubit populations
```

```{figure} images/hello_dispersive_readout_iq.png
:width: 560px
:alt: Conditional resonator IQ paths with emphasized final points
```

`quchip` uses GHz for ordinary frequencies, ns for time, and mK for temperature. The implemented conventions and approximations are recorded in the {doc}`physics reference <physics>`.

## Start from the SQA 2026 talk

The {doc}`post-talk page <guides/from-sqa-2026>` contains a short runnable
snippet and one documentation link for each topic. The guides stand on their
own; no knowledge of the presentation is required.

The accompanying paper is [quchip: A Differentiable Toolkit for Modeling Quantum Devices](https://arxiv.org/abs/2607.17081) (arXiv:2607.17081); citation metadata is in the repository's [CITATION.cff](https://github.com/quchip/quchip/blob/main/CITATION.cff).

```{toctree}
:maxdepth: 1
:hidden:

guides/from-sqa-2026
guides/defining-and-inspecting-a-chip
guides/choosing-a-backend
guides/statics-and-parameter-studies
guides/dynamics-pulses-and-readout
guides/chip-transformations
guides/differentiability
cookbook
extensions
physics
api
contributing
conduct
```

## Project

- [GitHub](https://github.com/quchip/quchip)
- [PyPI](https://pypi.org/project/quchip/)
- License: Apache-2.0
