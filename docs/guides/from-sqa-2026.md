# From the SQA 2026 talk

The talk declares one chip, then reuses the same physical description for
static analysis, pulse-level dynamics, model transformations, and derivatives.
These executed examples follow that order using quchip's public API.

Install the base package for the first three stages. The differentiability
example uses the optional dynamiqs backend.

```bash
python -m pip install quchip
python -m pip install 'quchip[dynamiqs]'
```

| Talk stage | Start here | What it reproduces |
| --- | --- | --- |
| Define and resolve | {doc}`Resolve and sweep a chip <../examples/resolve-and-sweep>` | The bus-mediated avoided crossing and four-band RWA ledger from Fig. 1 |
| Drive and batch | {doc}`Hello, drive and readout <../examples/hello-chip>` | Pulse scheduling, open-system dynamics, batched initial states, and measured readout paths |
| Transform | {doc}`Reduce and replay a chip <../examples/reduce-and-replay>` | A compact active-patch reduction with validity records and a forward check |
| Differentiate | {doc}`Differentiate a driven chip <../examples/differentiate-a-driven-chip>` | The amplitude, Gaussian-shape, and detuning derivatives from Fig. 6 |

Chips can be written directly, fitted from dressed observables with
`fit_a_dress()`, or imported from scqubits with `from_scqubits()`. The examples
below use direct declarations so every physical assumption remains visible.

## Resolve and sweep a chip

{doc}`Resolve and sweep a chip <../examples/resolve-and-sweep>` declares two
multilevel transmons coupled through a bus, reads dressed static quantities,
audits the chip's RWA, and reproduces the talk's 4.4 MHz avoided crossing
without mutating the original model.

## Drive and read out a chip

{doc}`Hello, drive and readout <../examples/hello-chip>` uses one chip for
dressed drive frequencies, pulse-level open-system simulation, batched initial
states, populations, and conditional resonator IQ paths.

To request the device-frame coordinates shown in the talk, build the
observables through the chip and pass them to the same solve:

```python
result = sequence.simulate(e_ops=chip.e_ops(q=["X", "Y", "Z"]))
```

The returned observable traces contain the processed device-frame values;
their `.raw` arrays retain the pre-processing values for frame debugging.

## Reduce and replay a chip

{doc}`Reduce and replay a chip <../examples/reduce-and-replay>` keeps the
scheduled neighbourhood, reports the validity of each eliminated spectator,
and replays the same pulse on the full and reduced models before plotting their
residual.

The deck shows separate 324-to-9 and 768-to-48 reductions. This notebook uses
the same public `active_patch()` workflow on an 81-state model and reduces it
to 9, so readers can rerun the comparison quickly.

## Differentiate a driven chip

{doc}`Differentiate a driven chip <../examples/differentiate-a-driven-chip>`
differentiates the final population with respect to pulse amplitude, the
Gaussian shape parameter $N_\sigma$, and detuning, then checks all three
derivatives against converged central differences. The dynamiqs backend keeps
these parameters differentiable while the device graph, Hilbert-space
dimensions, and RWA band selection remain fixed. Eigenvector derivatives near
degeneracy require separate care.
