<p align="center">
  <a href="https://quchip.org">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://docs.quchip.org/_static/quchip-wordmark-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="https://docs.quchip.org/_static/quchip-wordmark-light.png">
      <img src="https://docs.quchip.org/_static/quchip-wordmark-light.png" alt="quchip" width="400">
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

`quchip` is an open-source Python toolkit to model quantum devices. Its
built-in models cover circuit QED, and custom Hamiltonians, operators, and
loss channels describe other systems. Define the chip once and use the same
model for dressed-state analysis, model reduction, control sequences,
open-system simulation, parameter sweeps, and JAX gradients.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://docs.quchip.org/_static/readme/guide_overview-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://docs.quchip.org/_static/readme/guide_overview.png">
  <img src="https://docs.quchip.org/_static/readme/guide_overview.png" alt="Four quchip calculations: an avoided crossing of two transmons, the populations of a transmon during a short Gaussian pulse that leaks into the second excited state, a VNA trace sampled through fridge wiring, and fluxonium energies fitted to measured spectroscopy" width="1084">
</picture>

Each panel uses the model of one of the [guides](#guides). The
fluxonium measurements are from [Stefanski *et al.*](https://arxiv.org/abs/2411.13437).

> [!WARNING]
> quchip is an alpha-stage 0.x project. Minor releases can change public APIs, so pin an exact version when reproducibility matters.

## Install

`quchip` requires Python 3.11 or newer.

```bash
python -m pip install quchip
```

Optionally, install support for dynamiqs, graph visualization, or scqubits:

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
chi_z = (fr0 - fr1) / 2
print(f"f01 = {f01:.4f} GHz, f12 = {f12:.4f} GHz, chi_Z = {1e3 * chi_z:.3f} MHz")
```

```text
f01 = 4.9980 GHz, f12 = 4.6986 GHz, chi_Z = 0.285 MHz
```

`unresolved_hamiltonian()` keeps the local device and coupling expressions as you defined them, and `hamiltonian()` applies the chip's basis, frame, and approximation. Both return inspectable symbolic expressions. If a resolved expression is time-dependent and you need its numerical array, call `.matrix(t=...)`.

The remaining calls read dressed transition frequencies and the resonator frequency conditioned on the qubit state. With quchip's $Z=|0\rangle\langle0|-|1\rangle\langle1|$, half the ground-minus-excited difference is the coefficient $\chi_Z$ of $Z\,n_r$.

[Your first chip](https://docs.quchip.org/get-started/first-chip) inspects a similar model's Hamiltonian and dropped terms, changes a parameter, and fits bare parameters to dressed targets.

## Add dynamics and readout

The [pulses guide](https://docs.quchip.org/guides/pulses-and-readout) adds control lines and pulse sequences to the chip above. It compares short and selective Gaussian qubit drives in the full multilevel model, reads them as gates, then simulates conditional resonator readout.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://docs.quchip.org/_static/readme/hello_dispersive_readout_iq-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://docs.quchip.org/_static/readme/hello_dispersive_readout_iq.png">
  <img src="https://docs.quchip.org/_static/readme/hello_dispersive_readout_iq.png" alt="Conditional resonator IQ paths with emphasized final points" width="560">
</picture>

During the readout pulse, the resonator field follows a different path for
each prepared qubit state.

## Guides

- [Your first chip](https://docs.quchip.org/get-started/first-chip): build a coupled model, inspect its Hamiltonian, and fit dressed targets.
- [Spectra and parameter sweeps](https://docs.quchip.org/guides/spectra-and-sweeps): follow an avoided crossing, check state labels, and change parameters.
- [Pulses, leakage, and readout](https://docs.quchip.org/guides/pulses-and-readout): compare pulse selectivity, read pulses as gates, and simulate conditional resonator response.
- [Readout and fridge wiring](https://docs.quchip.org/guides/steady-state-and-vna): calculate VNA traces, their resonances, and qubit readout with receiver noise.
- [Model reduction](https://docs.quchip.org/guides/model-reduction): compare a reduced model with the full pulse simulation, remove a readout from its feedline, solve in an exact excitation space, and truncate the excitation number.
- [Gradients and parameter fitting](https://docs.quchip.org/guides/gradients-and-fitting): differentiate spectra and pulse responses, and fit model parameters.
- [Gate sets and circuits](https://docs.quchip.org/guides/gate-sets): calibrate native gates and compose circuits from them.
- [Resonance fluorescence and photon correlations](https://docs.quchip.org/guides/resonance-fluorescence): read the output spectrum, `g1` and `g2` of a driven emitter.
- [Import circuits from scqubits](https://docs.quchip.org/guides/scqubits): turn scqubits devices and coupled systems into quchip models and back.
- [Extending quchip](https://docs.quchip.org/extensions): define custom models, controls, dissipation and interoperability mappings.
- [Cookbook](https://docs.quchip.org/cookbook): practical API choices, tips and common pitfalls.

The spectra, pulses, model reduction and gradients guides are also Jupyter notebooks in [`examples/`](https://github.com/quchip/quchip/tree/main/examples).

## Studies

- [Purcell filtering and T1](https://docs.quchip.org/studies/purcell-filter): how much can a Purcell filter suppress qubit decay while keeping the readout bandwidth?
- [Fluxonium spectroscopy](https://docs.quchip.org/studies/fluxonium-spectroscopy): does one model reproduce published fluxonium and readout data, and does a gradient fit recover the circuit energies?
- [Resonator depletion with CLEAR](https://docs.quchip.org/studies/resonator-depletion): how much faster does a CLEAR pulse empty a filtered readout resonator than waiting?
- [What happens during a qubit measurement?](https://docs.quchip.org/studies/continuous-measurement): follow single measurement trajectories, map their density, and condition on an endpoint.

## How quchip works

A predictive chip model needs more than a Hamiltonian: device physics, control-line transformations, frames, approximations, dissipation, and measured observables each need an explicit place in it. Gain, delay, and crosstalk stay properties of the control chain instead of being folded into Hamiltonian coefficients by hand.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://docs.quchip.org/_static/readme/quchip_pipeline_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://docs.quchip.org/_static/readme/quchip_pipeline_light.png">
  <img src="https://docs.quchip.org/_static/readme/quchip_pipeline_light.png" alt="quchip pipeline from defined devices and control parameters through model resolution, simulation, observables, and gradients" width="1084">
</picture>

QuTiP is the default simulation backend. The optional dynamiqs backend is JAX-native and keeps the defined device and control parameters differentiable through a solve. The [backend guide](https://docs.quchip.org/reference/backends-and-solvers.html) compares their numerical and workflow tradeoffs. The scqubits integration imports and exports selected device and composite models.

`quchip` uses GHz for ordinary frequencies, ns for time, and mK for temperature. The [physics guide](https://docs.quchip.org/physics) documents the implemented conventions and approximations.

## Project status and contributing

Report bugs and model requests through [GitHub Issues](https://github.com/quchip/quchip/issues). Use [Discussions](https://github.com/quchip/quchip/discussions) for questions and open-ended proposals. Before making code or physics changes, read the [contributing guide](https://github.com/quchip/quchip/blob/main/CONTRIBUTING.md).

## Paper and citation

The accompanying paper is [quchip: A Differentiable Toolkit for Modeling Quantum Devices](https://arxiv.org/abs/2607.17081) (arXiv:2607.17081).

If you use quchip, cite:

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

Software citation metadata is also in [CITATION.cff](https://github.com/quchip/quchip/blob/main/CITATION.cff).

## License

`quchip` is distributed under the Apache License 2.0. See [LICENSE](https://github.com/quchip/quchip/blob/main/LICENSE).
