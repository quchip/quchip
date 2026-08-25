# quchip 0.2.1

quchip 0.2.1 keeps JAX-traced crosstalk and sequence parameters intact and adds an executable path through the examples presented at SQA 2026.

## Fixes

- `ControlEquipment.set_crosstalk_matrix()` accepts nested Python lists. When those lists contain JAX tracers, quchip stacks them without converting them to NumPy.
- `QuantumSequence.with_params()` preserves traced array leaves while rebuilding engine records, so dynamiqs simulations remain differentiable with respect to rebound pulse and device parameters.

## From the SQA 2026 talk

The new [post-talk guide](https://docs.quchip.org/guides/from-sqa-2026.html) links four public-API workflows in the order used by the presentation:

- [Resolve and sweep a chip](https://docs.quchip.org/examples/resolve-and-sweep.html) reproduces the bus-mediated 4.4 MHz avoided crossing and records the four counter-rotating bands removed by `RWA()`.
- [Hello, drive and readout](https://docs.quchip.org/examples/hello-chip.html) covers pulse scheduling, open-system simulation, batched initial states, and conditional readout paths.
- [Reduce and replay a chip](https://docs.quchip.org/examples/reduce-and-replay.html) reduces an 81-state scheduled model to a 9-state active patch, reports the elimination validity record, and compares the full and reduced dynamics.
- [Differentiate a driven chip](https://docs.quchip.org/examples/differentiate-a-driven-chip.html) differentiates the final population with respect to pulse amplitude, Gaussian shape, and detuning, then checks all three derivatives against central finite differences.

Each new example includes its Markdown source, an executed notebook, a rendered figure, and a machine-readable result receipt. The examples use public quchip APIs and state where the compact documentation model differs from the larger model shown in the talk.

## Compatibility

This patch release adds no migration steps beyond 0.2.0.

**Full changes:** [v0.2.0...v0.2.1](https://github.com/quchip/quchip/compare/v0.2.0...v0.2.1)
