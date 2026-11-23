```{raw} html
<p class="docs-eyebrow">quchip documentation</p>
```

# quchip documentation

```{container} docs-intro
Model quantum devices in Python. Declare devices, couplings,
controls, and losses, then calculate spectra, simulate pulses, or fit parameters.
```

## Install

Python 3.11 or newer is required.

```bash
python -m pip install quchip
```

See {doc}`installation <get-started/installation>` for optional backends and integrations.

## Your first calculation

Couple a transmon to a resonator and read their dressed frequencies. Frequencies and
couplings are in GHz. This model uses the rotating-wave approximation.

```python
from quchip import RWA, Capacitive, Chip, DuffingTransmon, Resonator

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
chip = Chip(
    [q, r], [Capacitive(q, r, g=0.05, label="qr")], approximation=RWA()
)

print(f"Qubit: {chip.freq(q):.6f} GHz")
print(f"Resonator: {chip.freq(r):.6f} GHz")
```

Output:

```text
Qubit: 4.998751 GHz
Resonator: 7.001249 GHz
```

The coupling shifts each frequency from its bare value. Continue with
{doc}`your first chip <get-started/first-chip>` to inspect the
Hamiltonian and change a parameter.

## What would you like to calculate?

```{figure} images/guide_overview.svg
:alt: Four quchip calculations: an avoided crossing of two transmons, the populations of a transmon during a short Gaussian pulse that leaks into the second excited state, a VNA trace sampled through fridge wiring, and fluxonium energies fitted to measured spectroscopy

The panels use the models of the spectra, pulses, and readout guides and of
the fluxonium study. The fluxonium measurements are from
[Stefanski *et al.*](https://arxiv.org/abs/2411.13437).
```

```{raw} html
<div class="docs-paths">
  <a href="guides/spectra-and-sweeps.html"><strong>Spectra and parameter sweeps</strong><span>Sweep parameters and follow dressed states.</span><b aria-hidden="true">→</b></a>
  <a href="guides/pulses-and-readout.html"><strong>Pulses, leakage, and readout</strong><span>Schedule drives, compare leakage, and read observables.</span><b aria-hidden="true">→</b></a>
  <a href="guides/steady-state-and-vna.html"><strong>Readout and fridge wiring</strong><span>Calculate microwave response and receiver noise.</span><b aria-hidden="true">→</b></a>
  <a href="guides/model-reduction.html"><strong>Model reduction</strong><span>Remove spectators and check the reduced dynamics.</span><b aria-hidden="true">→</b></a>
  <a href="guides/gradients-and-fitting.html"><strong>Gradients and parameter fitting</strong><span>Differentiate observables and infer model parameters.</span><b aria-hidden="true">→</b></a>
  <a href="guides/gate-sets.html"><strong>Gate sets and circuits</strong><span>Calibrate native gates and compose circuits.</span><b aria-hidden="true">→</b></a>
</div>
```

Browse all {doc}`guides <guides/index>`, the {doc}`studies <studies/index>`
of specific physical questions, or the {doc}`reference <reference/index>` for
the API, the physics conventions, and the cookbook.

```{toctree}
:maxdepth: 1
:hidden:

get-started/index
guides/index
studies/index
reference/index
contribute/index
```
