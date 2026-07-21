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

Start with {doc}`Hello, drive and readout <examples/hello-chip>` to declare an explicitly labeled Duffing transmon and lossy resonator, compare broadband and selective nominal-pi qubit drives, then follow the conditional readout responses for a duration derived from the resonator pull and linewidth. The {doc}`cookbook` collects the conventions behind executable quchip recipes.

```{figure} images/hello_qubit_drive_leakage.png
:width: 760px
:alt: Short and long Gaussian pulses with multilevel qubit populations
```

```{figure} images/hello_dispersive_readout_iq.png
:width: 560px
:alt: Conditional resonator IQ paths with emphasized final points
```

`quchip` uses GHz for ordinary frequencies, ns for time, and mK for temperature. The implemented conventions and approximations are recorded in the {doc}`physics reference <physics>`.

The accompanying paper is [quchip: A Differentiable Toolkit for Modeling Quantum Devices](https://arxiv.org/abs/2607.17081) (arXiv:2607.17081); citation metadata is in the repository's [CITATION.cff](https://github.com/quchip/quchip/blob/main/CITATION.cff).

Worked examples and guides are being added incrementally.

```{toctree}
:maxdepth: 1
:hidden:

examples/hello-chip
cookbook
physics
api
contributing
conduct
```

## Project

- [GitHub](https://github.com/quchip/quchip)
- [PyPI](https://pypi.org/project/quchip/)
- License: Apache-2.0
