# quchip 0.2.1

quchip 0.2.1 keeps JAX-traced crosstalk and sequence parameters intact and adds an executable path through the examples presented at SQA 2026.

## Fixes

- `ControlEquipment.set_crosstalk_matrix()` accepts nested Python lists. When those lists contain JAX tracers, quchip stacks them without converting them to NumPy.
- `QuantumSequence.with_params()` preserves traced array leaves while rebuilding engine records, so dynamiqs simulations remain differentiable with respect to rebound pulse and device parameters.

## Dressed-chip fitting

`fit_a_dress(desired)` now reads device spectra and coupling values as numerical
dressed targets. Add observables with `constraints=`, select bare parameters
with `vary=`, and override optimizer starts with `start=`. The fit result records
target sources, parameter seeds and bounds, coupling-sign choices, and final
Jacobian identifiability diagnostics; `fit.summary()` prints the same receipt.

## Dressed Kerr matrix

`chip.kerr_matrix()` returns one immutable, labeled view of the coupled chip's
dressed nonlinearities in ordinary GHz. Its diagonal contains
`dressed_anharmonicity()` for each device; off-diagonal entries contain the
full-pull `dispersive_shift()` for each pair. Axes follow `chip.devices`, and
entries can be read with device objects or labels, for example
`kerr[q0, "bus"]`.

The matrix is symmetric and JAX-compatible. A device with fewer than three
resolved levels has `NaN` on its diagonal while its pairwise entries remain
available. These are derived dressed observables, not the authored values of
individual `CrossKerr` edges. For `KerrCavity(freq=..., kerr=K)`, the diagonal
is `-2*K` under the existing `H = omega*n - K*n*(n-1)` convention.

## Flux-tunable transmons

`FluxTunableTransmon.flux_bias` is now a public chip parameter, so static
studies can sweep reduced flux directly with
`Sweep(..., name="coupler.flux_bias")`. A flux-only rebind preserves the
inferred SQUID calibration and updates the local frequency and Hamiltonian.
Supplying `freq` and `flux_bias` together defines a new calibration anchor,
independent of mapping order. The operation remains immutable through
`Chip.with_params()` and differentiable with JAX.

## From the SQA 2026 talk

The new [post-talk guide](https://docs.quchip.org/guides/from-sqa-2026) links five public-API workflows:

- [Define and inspect a chip](https://docs.quchip.org/guides/defining-and-inspecting-a-chip) starts with a small declaration, then covers dressed fitting, symbolic Hamiltonians, frames, projections, and scqubits conversion.
- [Statics and parameter studies](https://docs.quchip.org/guides/statics-and-parameter-studies) reproduces the bus-mediated 4.4 MHz avoided crossing and records the counter-rotating bands removed by `RWA()`.
- [Dynamics, pulses, and readout](https://docs.quchip.org/guides/dynamics-pulses-and-readout) covers pulse scheduling, open-system simulation, batched initial states, observables, and conditional readout.
- [Chip transformations](https://docs.quchip.org/guides/chip-transformations) reduces an 81-state scheduled model to a 9-state active patch, reports the elimination validity record, and compares the full and reduced dynamics.
- [Differentiability](https://docs.quchip.org/guides/differentiability) covers losses through statics, one driven sequence, and multi-sequence analysis, with central-finite-difference checks.

The four paired examples include Markdown source, an executed notebook, a rendered figure, and a machine-readable result receipt. The defining guide executes its shown blocks directly. All five use public quchip APIs and state where the compact documentation model differs from the larger model shown in the talk.

## Compatibility

The seed-chip keywords `coupling_targets=`, `observable_targets=`, and
`fit_parameters=` are deprecated in 0.2.1 and will be removed in 0.3.0. They
retain their existing behavior during 0.2.x and emit one `DeprecationWarning`
per call. Replace them with the desired-chip API:

- put standard dressed frequencies, anharmonicities, and edge cross-Kerr values
  on the desired `Chip`;
- replace `observable_targets=` and `coupling_targets=` with `constraints=` for
  additional or overridden observables;
- replace `fit_parameters=` with `vary=`; and
- use `start=` only when the automatic starting point or coupling-sign branch
  is not the one you want.

**Full changes:** [v0.2.0...v0.2.1](https://github.com/quchip/quchip/compare/v0.2.0...v0.2.1)
