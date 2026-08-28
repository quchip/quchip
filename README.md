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

**Development status:** quchip is currently an alpha-stage 0.x release. Minor releases may refine public APIs; pin an exact version for reproducible work.

A predictive chip model needs more than a Hamiltonian: device physics, control-line transformations, frames and approximations, dissipation, and measured observables all belong to it. quchip represents each part explicitly. Line properties such as gain, delay, and crosstalk belong to the control chain, not to Hamiltonian terms written by hand.

Declare the chip once. The same declaration drives dressed-state analysis, model reduction, control sequencing, open-system simulation, parameter sweeps, and exact JAX gradients. The engine resolves each device's frame, applies the requested approximations, and records the bands it drops.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/quchip/quchip/main/docs/images/quchip_pipeline_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/quchip/quchip/main/docs/images/quchip_pipeline_light.png">
  <img src="https://raw.githubusercontent.com/quchip/quchip/main/docs/images/quchip_pipeline_light.png" alt="quchip pipeline from declared devices and control parameters through basis and frame resolution, physics assembly, observable preparation, backend solving, and one reverse-mode gradient" width="1084">
</picture>

`Chip + QuantumSequence` → `ResolvedFrame` → `EngineResult` → `SolveProblem` → QuTiP or dynamiqs → `SimulationResult`

QuTiP is the default backend. The dynamiqs backend is JAX-native and keeps declared device and control parameters differentiable through the solve. The optional scqubits integration imports and exports selected device and composite models.

`quchip` uses GHz for ordinary frequencies, ns for time, and mK for temperature. The implemented conventions and approximations are documented in the [physics guide](https://docs.quchip.org/physics).

## Install

`quchip` requires Python 3.11 or newer.

```bash
python -m pip install quchip
```

Optional extras are available for the dynamiqs backend, graph visualization, and scqubits interoperability:

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

## Declare and inspect a chip

```python
from quchip import RWA, Capacitive, ChargeDrive, Chip, DuffingTransmon, Resonator

qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=6, label="q")
readout = Resonator(freq=6.8, levels=10, quality_factor=6800, label="r")
coupling = Capacitive(qubit, readout, g=0.060, label="qr")
chip = Chip(
    [qubit, readout],
    couplings=[coupling],
    frame="rotating",
    approximation=RWA(),
)
qubit_line = ChargeDrive(qubit, label="qubit-charge")
readout_line = ChargeDrive(readout, label="readout-charge")
chip.wire(qubit_line, readout_line)

authored_hamiltonian = chip.unresolved_hamiltonian()
resolved_hamiltonian = chip.hamiltonian()

f01 = chip.freq(qubit)
f12 = chip.transition_frequency(qubit, 1, 2)
fr0 = chip.freq(readout, when={qubit: 0})
fr1 = chip.freq(readout, when={qubit: 1})
```

The authored Hamiltonian preserves the device and coupling expressions in their
declared local spaces. The resolved view applies the chip's basis, frame, and
approximation strategy through the same engine path used by simulation. Both remain
inspectable symbolic expressions; call `.matrix()` when a numerical array is
needed.

The complete example derives short and selective nominal-pi Gaussian drives from $|f_{12}-f_{01}|$, then derives a Gaussian-edge readout duration from the conditional pull and resonator linewidth. Both parts run the real multilevel, lossy chip with compact reproducibility receipts.

![Short and long Gaussian pulses with multilevel qubit populations](https://raw.githubusercontent.com/quchip/quchip/main/docs/images/hello_qubit_drive_leakage.png)

![Conditional resonator IQ paths with emphasized final points](https://raw.githubusercontent.com/quchip/quchip/main/docs/images/hello_dispersive_readout_iq.png)

The complete walkthrough is available in the [dynamics guide](https://docs.quchip.org/guides/dynamics-pulses-and-readout).

## Examples

- [From the SQA 2026 talk](https://docs.quchip.org/guides/from-sqa-2026): five runnable entry points, from defining a chip through differentiating it.
- [Statics and parameter studies](https://docs.quchip.org/guides/statics-and-parameter-studies): read dressed observables, sweep parameters, and track assignments through an avoided crossing.
- [Dynamics, pulses, observables, and readout](https://docs.quchip.org/guides/dynamics-pulses-and-readout): build pulse schedules, batch experiments, inspect states, and simulate a resonator response.
- [Chip transformations](https://docs.quchip.org/guides/chip-transformations): rebind, serialize, partition, eliminate, fit, and replay reduced models.
- [Differentiability](https://docs.quchip.org/guides/differentiability): differentiate static and dynamic losses, fit a published fluxonium spectrum, and combine shared-parameter experiments.
- [Cookbook](https://docs.quchip.org/cookbook): conventions for writing and using quchip examples.
- [Extension guide](https://docs.quchip.org/extensions): author devices, couplings, time-dependent terms, drives, envelopes, dissipation, local spaces, and interop mappings.

## Project status and contributing

Report bugs and model requests through [GitHub Issues](https://github.com/quchip/quchip/issues). Use [Discussions](https://github.com/quchip/quchip/discussions) for questions and open-ended proposals. See the [contributing guide](https://github.com/quchip/quchip/blob/main/CONTRIBUTING.md) before making code or physics changes.

## Paper and citation

The accompanying paper is [quchip: A Differentiable Toolkit for Modeling Quantum Devices](https://arxiv.org/abs/2607.17081) (arXiv:2607.17081).

The [interactive walkthrough](https://quchip.org) follows one five-device model through declaration, crosstalk identification and correction, adiabatic reduction from 576 to 16 dimensions, and gradient-based recovery of four directed crosstalk parameters.

If you use quchip in your work, please cite it:

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

Citation metadata for the software itself is in [CITATION.cff](https://github.com/quchip/quchip/blob/main/CITATION.cff).

## License

`quchip` is distributed under the Apache License 2.0. See [LICENSE](https://github.com/quchip/quchip/blob/main/LICENSE).
