> [!WARNING]
> quchip is an alpha-stage 0.x project. Minor releases may change public APIs. Pin an exact version when reproducibility matters.

<p align="center">
  <a href="https://quchip.org">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/quchip/quchip/main/docs/images/quchip-wordmark-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/quchip/quchip/main/docs/images/quchip-wordmark-light.png">
      <img src="https://raw.githubusercontent.com/quchip/quchip/main/docs/images/quchip-wordmark-light.png" alt="quchip" width="400">
    </picture>
  </a>
</p>

<p align="center">
  <a href="https://quchip.org">Website</a> ·
  <a href="https://docs.quchip.org">Documentation</a> ·
  <a href="https://arxiv.org/abs/2607.17081">Paper</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/quchip/"><img src="https://img.shields.io/pypi/v/quchip" alt="PyPI version"></a>
  <a href="#project-status-and-contributing"><img src="https://img.shields.io/badge/status-alpha-orange" alt="Project status: alpha"></a>
  <a href="https://pypi.org/project/quchip/"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11 or newer"></a>
  <a href="https://github.com/quchip/quchip/actions/workflows/ci.yml"><img src="https://github.com/quchip/quchip/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status"></a>
  <a href="https://docs.quchip.org"><img src="https://github.com/quchip/quchip/actions/workflows/docs.yml/badge.svg?branch=main" alt="Documentation build status"></a>
</p>

`quchip` is an open-source Python toolkit for modelling superconducting quantum chips.

A predictive chip model needs more than a Hamiltonian. Device physics, control-line transformations, frames, approximations, dissipation, and measured observables all need explicit places in the model. Gain, delay, and crosstalk remain properties of the control chain instead of being folded into Hamiltonian coefficients by hand.

Declare the chip once, then use the same model for dressed-state analysis, model reduction, control sequences, open-system simulation, parameter sweeps, and JAX gradients.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/quchip/quchip/main/docs/images/quchip_pipeline_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/quchip/quchip/main/docs/images/quchip_pipeline_light.png">
  <img src="https://raw.githubusercontent.com/quchip/quchip/main/docs/images/quchip_pipeline_light.png" alt="quchip pipeline from declared devices and control parameters through model resolution, simulation, observables, and gradients" width="1084">
</picture>

QuTiP is the default simulation backend. The optional dynamiqs backend is JAX-native and keeps declared device and control parameters differentiable through a solve. The [backend guide](https://docs.quchip.org/guides/choosing-a-backend.html) compares their numerical and workflow tradeoffs. The scqubits integration imports and exports selected device and composite models.

`quchip` uses GHz for ordinary frequencies, ns for time, and mK for temperature. The implemented conventions and approximations are documented in the [physics guide](https://docs.quchip.org/physics).

## Install

`quchip` requires Python 3.11 or newer.

```bash
python -m pip install quchip
```

Install optional support for dynamiqs, graph visualization, or scqubits as needed:

```bash
python -m pip install 'quchip[dynamiqs]'
python -m pip install 'quchip[viz]'
python -m pip install 'quchip[scqubits]'
```

Extras can be combined. To install the current source instead:

```bash
git clone https://github.com/quchip/quchip.git
cd quchip
python -m pip install .
```

## Define and inspect a chip

```python
from quchip import RWA, Capacitive, Chip, DuffingTransmon, Resonator

qubit = DuffingTransmon(
    freq=5.0,
    anharmonicity=-0.30,
    levels=6,
    label="q",
)
readout = Resonator(
    freq=6.8,
    levels=10,
    internal_quality_factor=6800,
    label="r",
)
coupling = Capacitive(qubit, readout, g=0.060, label="qr")

chip = Chip(
    [qubit, readout],
    couplings=[coupling],
    frame="rotating",
    approximation=RWA(),
)

authored_hamiltonian = chip.unresolved_hamiltonian()
resolved_hamiltonian = chip.hamiltonian()

f01 = chip.freq(qubit)
f12 = chip.transition_frequency(qubit, 1, 2)
fr0 = chip.freq(readout, when={qubit: 0})
fr1 = chip.freq(readout, when={qubit: 1})
chi = (fr1 - fr0) / 2
```

`unresolved_hamiltonian()` preserves the local device and coupling expressions you declared. `hamiltonian()` applies the chip's basis, frame, and approximation. Both return inspectable symbolic expressions; call `.matrix(t=...)` when a resolved expression is time-dependent and you need its numerical array.

The remaining calls read dressed transition frequencies and the resonator frequency conditioned on the qubit state. Their half-difference gives the dispersive shift $\chi$.

The [defining and inspecting a chip guide](https://docs.quchip.org/guides/defining-and-inspecting-a-chip) continues from this example with LaTeX output, term inspection, frame transformations, projections, and graph views.

## Add dynamics and readout

The [dynamics guide](https://docs.quchip.org/guides/dynamics-pulses-and-readout) adds control lines and pulse sequences to the chip above. It compares short and selective Gaussian qubit drives in the full multilevel model, then simulates conditional resonator readout.

![Short and long Gaussian pulses with multilevel qubit populations](https://raw.githubusercontent.com/quchip/quchip/main/docs/images/hello_qubit_drive_leakage.png)

![Conditional resonator IQ paths with emphasized final points](https://raw.githubusercontent.com/quchip/quchip/main/docs/images/hello_dispersive_readout_iq.png)

## Guides

- [Define and inspect a chip](https://docs.quchip.org/guides/defining-and-inspecting-a-chip): build a model, inspect its Hamiltonian, and see how frames and approximations change it.
- [Statics and parameter studies](https://docs.quchip.org/guides/statics-and-parameter-studies): read dressed observables, sweep parameters, and follow states through an avoided crossing.
- [Dynamics, pulses, observables, and readout](https://docs.quchip.org/guides/dynamics-pulses-and-readout): build pulse schedules, batch experiments, inspect states, and simulate resonator readout.
- [Chip transformations](https://docs.quchip.org/guides/chip-transformations): rebind, serialize, partition, eliminate, fit, and replay reduced models.
- [Differentiability](https://docs.quchip.org/guides/differentiability): differentiate static and dynamic losses, fit a published fluxonium spectrum, and combine experiments that share parameters.
- [Extension guide](https://docs.quchip.org/extensions): define new devices, couplings, drives, envelopes, dissipation, local spaces, and interoperability mappings.
- [Cookbook](https://docs.quchip.org/cookbook): the conventions used throughout quchip's examples.
- [From the SQA 2026 talk](https://docs.quchip.org/guides/from-sqa-2026): short, runnable entry points into the main topics.

## Project status and contributing

Report bugs and model requests through [GitHub Issues](https://github.com/quchip/quchip/issues). Use [Discussions](https://github.com/quchip/quchip/discussions) for questions and open-ended proposals. Read the [contributing guide](https://github.com/quchip/quchip/blob/main/CONTRIBUTING.md) before making code or physics changes.

## Paper and citation

The accompanying paper is [quchip: A Differentiable Toolkit for Modeling Quantum Devices](https://arxiv.org/abs/2607.17081) (arXiv:2607.17081).

The [interactive walkthrough](https://quchip.org) follows one five-device model through declaration, crosstalk identification and correction, adiabatic reduction from 576 to 16 dimensions, and gradient-based recovery of four directed crosstalk parameters.

If you use quchip in your work, please cite:

```bibtex
@misc{alyousef2026quchip,
      title={quchip: A Differentiable Toolkit for Modeling Quantum Devices},
      author={Ibraheem AlYousef},
      year={2026},
      eprint={2607.17081},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      doi={10.48550/arXiv.2607.17081},
      url={https://arxiv.org/abs/2607.17081},
}
```

Citation metadata for the software is also available in [CITATION.cff](https://github.com/quchip/quchip/blob/main/CITATION.cff).

## License

`quchip` is distributed under the Apache License 2.0. See [LICENSE](https://github.com/quchip/quchip/blob/main/LICENSE).
