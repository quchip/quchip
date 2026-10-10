# Changelog

This file records notable user-visible changes to quchip.

## Unreleased

### Changes since 0.4.0

#### Changes and migration

- `eliminate()` of a port-coupled mode keeps the mode's reflection on the
  port's plane as a `PortNetwork.mode_reflection(...)` reference section.
  Before this change, the reduced boundary kept only the transformed port. So
  VNA on the reduced chip missed the mode's reflection by approximately κ/Δ
  (1e-2 in the reported example). Now it matches the full chip up to a
  correction of order (g/Δ)²κ/Δ (8e-5). A mode with several ports, or a port
  whose plane also carries other fields, now raises, so keep such a mode in the
  model. ([#76](https://github.com/quchip/quchip/issues/76))

#### New features

- You can eliminate port-coupled modes on chips with more than two devices. The
  transformed port acts jointly on every survivor. Stationary-tone frame
  planning keeps each port band's sign, so VNA accepts the joint operator. A
  later elimination transforms that port again, so you can eliminate both
  readout modes of a chip in either order.
  ([#77](https://github.com/quchip/quchip/issues/77))
- `PortNetwork.mode_reflection(...)` adds a serializable two-sided reference
  section that reflects like a damped linear mode.
  ([#76](https://github.com/quchip/quchip/issues/76))
- `QuantumSequence.schedule()`, `charge()` and `phase()` accept `detuning=`, a
  carrier offset in GHz referenced to the pulse start. The pulse is then the
  same in its frame at every start time, so a detuned calibrated gate keeps its
  rotation axis when it moves. `pulse.<i>.detuning` rebinds, sweeps and
  differentiates like `freq`. A carrier at `freq = f + δ` stays a fixed
  oscillator. ([#93](https://github.com/quchip/quchip/issues/93))

#### Fixes

- `eliminate(..., method="exact")` of a chip whose approximation conserves total
  excitation number, such as `RWA()`, now diagonalizes each excitation sector
  separately. Retained terms no longer carry ~1e-12 entries between sectors,
  which made the rotating-frame collapse-operator check fail at some parameter
  values. ([#78](https://github.com/quchip/quchip/issues/78))
- Reduced chips resolve the same under `jax.jit` and `jax.grad` as eagerly.
  `EffectiveTerms.excitation_changes` declares each retained channel's total excitation change.
  Projected surviving operators keep their authored operator's change, so band decomposition no
  longer depends on traced values. Ports follow the same rule, both on surviving modes and those an
  elimination transformed. ([#78](https://github.com/quchip/quchip/issues/78),
  [#79](https://github.com/quchip/quchip/issues/79))
- A band whose level changes cancel among devices that share a frame frequency
  is now exactly static. Its frame frequency was summed one device at a time,
  so level changes such as (3, −1, −2) left carriers of approximately 1e-14
  GHz. An exactly reduced two-qubit readout chip in a 5.2 GHz frame resolved
  44 static couplings as time-dependent terms. It also split drive bands with
  equal carriers into separate terms.
  ([#78](https://github.com/quchip/quchip/issues/78))
- QuTiP `mesolve`, `smesolve` and stationary solves raise `MemoryError` before they
  assemble a superoperator whose estimated peak exceeds the available memory.
  Previously, the operating system killed them. The dynamiqs stationary Liouvillian
  uses the same check. ([#80](https://github.com/quchip/quchip/issues/80))
- Carrier pulses accept `frame=` to select the device whose virtual-Z phase
  they follow. For cross-resonance control, set the target qubit as the frame
  of both the cross-resonance and cancellation tones. The default remains the
  driven device. ([#105](https://github.com/quchip/quchip/pull/105))

#### Performance

- `VNA.sweep()` solves weak-probe scattering of pump-free chips that conserve
  total excitation number in their one-excitation block, if every input is
  vacuum. These chips include Duffing transmons, pure dephasing, and reduced
  chips with retained terms. Previously, any such term fell back to a dense
  D²×D² stationary solve. On one machine, the reported transmon-resonator
  sweep at Hilbert dimension 64 took 540 s. Now it takes 10 s, mostly for the
  frame's reference-frequency analysis. Dimension 400 runs in 10 s and no
  longer uses approximately 410 GB. Diagnostics name the route
  `"vacuum_response"`. ([#81](https://github.com/quchip/quchip/issues/81))
- The QuTiP backend stores dense operators with at most one quarter nonzero
  entries as CSR, so superoperator terms built from reduced-model bands and
  jumps stay sparse. A lossy `mesolve` of a reduced model at Hilbert dimension
  144 now peaks near 1.3 GB instead of an estimated 1.2 TB.
  ([#80](https://github.com/quchip/quchip/issues/80))
- dynamiqs solves sum the time-dependent Hamiltonian into one array per step
  from operators assembled on the host. A traced chip reuses its resolution and
  dressed analysis within one trace. On one machine, a jitted reverse-mode
  gradient through an exact elimination and a dynamiqs pulse of a 576-state
  chip took 830 s per call and now takes 3.7 s. A batch of eight pulse phases
  takes 4.1 s instead of 20 s.
- QuTiP `mesolve` of a Hermitian density matrix integrates its packed upper
  triangle with one sparse product per step. Drive carriers that act through
  the same operator share one coefficient. A lossy pulse on the same chip
  takes 200 s instead of 400 s. A coherent 16-pulse train takes 7.3 s
  instead of 49 s.
- Large packed QuTiP master-equation products split their rows over up to four
  threads, which brings that lossy pulse to 145 s. `QUCHIP_NUM_THREADS`, or
  else `OMP_NUM_THREADS`, sets the count. Worker processes use one thread
  unless `QUCHIP_NUM_THREADS` is set. Each row sums in the same order, so
  results do not depend on the thread count.
- Static analysis of concrete chips sums, diagonalizes, reduces and labels
  operators with NumPy instead of compiling a JAX program for each array shape.
  `fit_a_dress()` on the 576-state chip takes 3.1 s per fit instead of 15 s.
- On the general stationary route, `VNA.sweep()` solves an operating point that
  stays stationary across probe frequencies once, not at every frequency. The
  QuTiP backend runs its default direct steady-state solve without QuTiP's
  per-call option scope, which rebuilt every data-layer dispatcher.
  `steadystate_batch()` over 40 points runs 5.4 times faster, and
  `VNA.finite_power()` over 44 points runs 3.2 times faster.

#### Compatibility

- `Backend.steadystate()` takes an optional `guess`: a native stationary
  state of a related problem, which a backend can return if it is also
  stationary here. A backend that overrides `steadystate()` must accept the
  keyword but can ignore it.

#### Documentation and examples

- The NV-centre guide is removed.
- The example notebooks and the guides are re-executed against the current
  code. The transmon avoided crossing in example 01 now reports the RWA
  splitting of 3.99 MHz, which agrees with second-order theory to 0.3%. The fluxonium
  readout comparison uses `Exact()`, because counter-rotating terms contribute
  strongly to its dispersive shift.

## [0.4.0] - 2026-09-29 <a id="quchip-0-4-0"></a>

### 0.4 series highlights

The 0.4 series adds continuous-measurement trajectories and extends custom-device
and effective-Hamiltonian workflows:

- Quantum-jump and diffusive trajectories on QuTiP and Dynamiqs, with native
  results, parameter batches, and explicit monitored channels.
- Custom operator names across noise, ports, observables, and elimination,
  plus an ideal two-level `Qubit` model.
- Elimination of retained `EffectiveTerms`, with dressed parameter reports
  and inspectable approximation notes. Dressed analysis follows the chip's
  selected approximation.

### Changes since 0.3.2

[Full comparison: v0.3.2 → v0.4.0](https://github.com/quchip/quchip/compare/v0.3.2...v0.4.0).

#### Changes and migration

- Dressed analysis now follows `chip.approximation`, including frequencies,
  states, Kerr shifts and drive matrix elements. Previously these queries
  silently retained all coupling bands even on an `RWA()` chip. Construct the
  chip with `approximation=Exact()` to recover the previous full-Hamiltonian
  analysis. Dressed queries remain independent of the integration frame.
  Elimination and its χ report follow the same choice; `method="exact"`
  diagonalizes the selected Hamiltonian without restoring discarded bands. ([#71](https://github.com/quchip/quchip/pull/71))
- Solves with `initial_state=None` now use the all-ground-labeled eigenstate of
  the undriven static lab-frame Hamiltonian retained by the solve's
  approximation. The default RWA with ordinary couplings remains a bare
  product. Before the frame transform, a solve using the chip's approximation
  selects the same physical state as `chip.state()` for the all-ground label.
  Retained bands, effective terms, or network Hamiltonian terms that couple
  the vacuum can dress an RWA start. Rotating-frame solves apply `U(tlist[0])†`. Pass `chip.bare_state()`
  explicitly to keep the previous bare start. ([#64](https://github.com/quchip/quchip/pull/64))
- Deterministic Dynamiqs solves default to `Dopri8(rtol=1e-9, atol=1e-11)` instead of the
  native `Tsit5()` (`rtol = atol = 1e-6`), at which pulse-parameter gradients
  could be off by a factor of two. Pass `method` to restore the old integrator. ([#57](https://github.com/quchip/quchip/pull/57))
- VNA S-parameters, finite-power fields, field correlations, and IQ statistics now follow the
  engineering `e^{+jωt}` convention, where `j = −i`, including network phases
  and delays. Remove manual conjugation of VNA outputs; conjugate old complex
  probe and pump amplitudes to reproduce the same physical drive. Internal
  mode observables and authored network parameters keep their physics convention. ([#46](https://github.com/quchip/quchip/pull/46))
- Wiring-derived IQ readouts share the VNA convention for supplied boundary
  means, detector fields, and receiver filtering. Conjugate raw simulation
  fields before supplying them as readout templates. Receiver-filter validation
  now also accepts JAX arrays. ([#46](https://github.com/quchip/quchip/pull/46))
- Truncation diagnostics are now explicit. Remove `check_truncation` and
  `truncation_threshold` from solve calls; use `result.check_truncation()` or
  prepare `with_truncation(problem)` when saving only diagnostics. ([#45](https://github.com/quchip/quchip/pull/45))
- Omitted stochastic storage uses native defaults. Put native call keywords
  in `run_args` and integrator options in `options`. For stochastic Dynamiqs
  calls, `options` holds native solver keywords; `method` and `gradient` go
  in `run_args`. See the
  [backend options](https://docs.quchip.org/guides/choosing-a-backend.html#native-stochastic-trajectories). ([#45](https://github.com/quchip/quchip/pull/45))

#### New features

- Added native quantum-jump trajectories (`mcsolve` on QuTiP, `jssesolve` on
  Dynamiqs), diffusive stochastic Schrödinger equations (`ssesolve` /
  `dssesolve`), and diffusive stochastic master equations (`smesolve` /
  `dsmesolve`). Supports parameter batches, native results, and explicit
  monitored-channel selection. See the
  [solver and storage guide](https://docs.quchip.org/guides/choosing-a-backend.html#native-stochastic-trajectories). ([#45](https://github.com/quchip/quchip/pull/45))
- Added the ideal two-level `Qubit` model. ([#45](https://github.com/quchip/quchip/pull/45))
- Custom devices can use declared operator names for ports and observables,
  and existing lowering, raising and number hooks for T1/T2 channels.
  Elimination supports non-capacitive mediated exchange, derives decay
  summaries from declared channels, and accepts custom harmonic Fock boundaries. ([#58](https://github.com/quchip/quchip/pull/58))
- `eliminate()` accepts the label of an `EffectiveTerms` contribution. With
  `method="exact"` it diagonalizes those terms together with the local
  Hamiltonians of their devices, keeps every device and edge authored, and
  retains the level shifts as a correction, so dressed queries on the reduced
  chip return the source spectrum. `effective_params` reports each device's
  dressed transition, Lamb shift, anharmonicity and cross-Kerr shifts.
  `Chip` rejects effective-term labels that collide with device or coupling labels. ([#69](https://github.com/quchip/quchip/pull/69))
- `EffectiveTerms` carry producer `notes`. `Chip.physics_notes()` lists each
  contribution under `effective:<label>` with its support, assembly rule,
  channels and notes, and `describe()` lists them. Device and coupling
  eliminations record their approximations on the retained terms, including
  the notes of earlier contributions they absorb. Serialized effective terms
  include `notes` when present; payloads without them still load.
  `Resonator.physics_notes()` states its harmonic approximation once. ([#69](https://github.com/quchip/quchip/pull/69))
- `fit_a_dress` summaries mark targets whose error exceeds 1% of the target. ([#57](https://github.com/quchip/quchip/pull/57))

#### Fixes

- `from_scqubits` and `to_scqubits` now convert energies between scqubits'
  global unit, `scqubits.get_units()`, and GHz. Previously they assumed GHz, so
  objects built after `scqubits.set_units("MHz")`, such as Quantum Metal's LOM
  composites, imported 1000 times too large. ([#72](https://github.com/quchip/quchip/pull/72))
- Per-call backend overrides no longer re-project QuTiP `Qobj` or dynamiqs
  `QArray` initial states returned by `chip.state()`, `chip.bare_state()`, or
  `chip.superposition()` in `simulate()` or `simulate_batch()`. Hand-built
  foreign-backend states now also use resolved solver coordinates. NumPy/JAX
  arrays and symbolic or callable states remain authored-space kets. ([#63](https://github.com/quchip/quchip/pull/63))
- `effective_hamiltonian` and `effective_hamiltonian_between_states` index the
  resolved product basis, so eigen-projected charge-basis devices no longer
  raise a singular-Gram-matrix error. Traced dressed states and `describe()`
  report the same retained dimensions. ([#50](https://github.com/quchip/quchip/pull/50))
- Collective and other multi-device collapse channels are now checked for a
  single removable frame phase. Unequal device frames raise before solving
  instead of silently retaining a static jump operator. ([#50](https://github.com/quchip/quchip/pull/50))
- Retained effective terms enter the dressed-analysis cache key; a chip
  resolved before a coupling elimination no longer reports bare energies. ([#57](https://github.com/quchip/quchip/pull/57))
- Preserve shifted square-pulse endpoints when floating-point subtraction
  rounds the local time past the pulse duration. ([#46](https://github.com/quchip/quchip/pull/46))
- The hybridization warning has fixed text and is shown once per call site. ([#57](https://github.com/quchip/quchip/pull/57))

#### Compatibility

- The dynamiqs backend targets dynamiqs 0.3.6, whose qarray rewrite broke the
  previous import. The `dynamiqs` extra requires dynamiqs 0.3.6 or newer and
  caps jax below 0.11.1 until a dynamiqs release includes dynamiqs/dynamiqs#1145.
  Stochastic Dynamiqs `options` are passed as native solver keywords:
  `save_states`, `cartesian_batching` and `save_extra`, plus `t0` and
  `nmaxclick` for `jssesolve`. ([#59](https://github.com/quchip/quchip/pull/59))

#### Documentation and examples

- Added an NV-centre guide modelling a spin-1 defect and its ¹⁴N nucleus through
  field-dependent resonances, pulsed ODMR, and Ramsey fringes.
  ([#66](https://github.com/quchip/quchip/pull/66))
- Added a continuous-measurement guide for stochastic trajectories.
  ([#45](https://github.com/quchip/quchip/pull/45))
- Clarified that `Capacitive` authors `g n_a n_b` on charge and phase-grid endpoints;
  `PortNetwork.filter` leaves Purcell decay unchanged. ([#57](https://github.com/quchip/quchip/pull/57))

#### Development

- The contributor workflow starts PRs as drafts. Marking a PR ready for
  review runs validation; later pushes rerun it. Documentation, workflow changes and version-only
  release metadata use lightweight checks. ([#73](https://github.com/quchip/quchip/pull/73))
- Historical release notes are consolidated in this changelog. Tagged
  releases extract their notes from the matching dated section. ([#68](https://github.com/quchip/quchip/pull/68))

## [0.3.2] - 2026-09-10 <a id="quchip-0-3-2"></a>

Changes since [v0.3.1](https://github.com/quchip/quchip/compare/v0.3.1...v0.3.2).

### Network noise API

- Passive network components now use `thermal_occupation`; replace previous
  `occupation` and `loss_occupation` arguments and saved parameter keys.
- Amplifiers require explicit `added_noise`. Temperature, noise-figure, and
  `noise_frequency` constructor arguments have been removed; convert these
  inputs to noise quanta before declaring components. See the
  [network-noise migration guidance](docs/cookbook.md#declare-network-noise).

### API documentation

- Added parameter options, defaults, units, `None` behavior, result shapes,
  and physics references across public constructors and methods.
- Reuse inherited backend contracts and suppress empty type-only parameter
  tables. A coverage check detects missing descriptions and stale names.
- Serve README images from the documentation site for consistent rendering.

### Development and releases

- PR fast and pre-merge test selections cover the suite without repeating the
  fast tests on Python 3.11. Merges require checks against current `main`.
- Remove post-merge test reruns, make benchmarks manual, and fail change
  classification when Git cannot read the compared revisions.
- Tagged releases verify package metadata and release notes, publish to PyPI,
  and create the corresponding GitHub Release after publication.

## [0.3.1] - 2026-09-08 <a id="quchip-0-3-1"></a>

Changes since [v0.3.0](https://github.com/quchip/quchip/compare/v0.3.0...v0.3.1).

### Measurements and fridge noise

- `VNA.measure()` captures the driven steady-state response, internal mode amplitudes and photon numbers, and output noise spectra. Receiver bandwidth, integration time, calibration and repeated IQ sampling reuse those results without another physical solve.
- Fridge noise propagates through declared attenuators, isolators, circulators, filters and amplifiers. Measurements report output noise density and contributions by source, including cross-output IQ covariance.
- `result.measure()` samples saved quantum states after either Schrödinger or master-equation evolution. Joint measurements preserve correlations and support assignment errors and calibrated conditional IQ distributions.
- `result.iq_readout()` uses captured downstream wiring; `IQReadout.from_wiring()` uses a supplied wired model. Both transform supplied conditional coherent fields and accumulate receiver noise without requiring a readout pulse. These detector models do not infer IQ signals from qubit populations or add measurement backaction to the simulated evolution.

### Documentation

- Added section navigation and aligned page titles across headings, sidebars and the README.
- Extended the fridge guide with steady-state versus sampled resonator responses and Rabi counts and IQ using the same wiring.
- Moved the Purcell calculation into Focused studies, retaining its existing URL.
- Shortened the cookbook to practical API choices and common pitfalls; example-authoring guidance now lives under Contribute.

### Public names and compatibility

Public names now distinguish bath occupation, jump rates and network ports:

| Previous name | Preferred name |
|---|---|
| `thermal_population` | `thermal_occupation` |
| `result.collapse_flux(...)` | `result.jump_rate(...)` |
| `component.side(...)`, `block.side(...)` | `component.port(...)`, `block.port(...)` |
| `network.exposure(...)`, `network.exposures` | `network.external_port(...)`, `network.external_ports` |
| `FieldExposure`, `FieldSide` | `NetworkPort`, `ComponentPort` |

The previous names remain compatibility aliases through 0.4 and are scheduled
for removal in 0.5. Old thermal constructor arguments, parameter bindings,
noise configurations and saved device dictionaries are accepted. Parameter
discovery and new serialized dictionaries use `thermal_occupation` only;
supplying both spellings in one update raises an error. The value is the bath's
mean occupation, not an initial qubit population. Jump rates include absorption
and dephasing channels and are not generally emitted photon fluxes.

The new measurement results are named `StateMeasurement` and `StateSamples`
for saved quantum states, and `VNAMeasurement`, `VNAMeasurementStatistics` and
`VNAMeasurementSamples` for VNA calculations. `result.measure(...)` and
`VNA.measure(...)` keep their existing call syntax. `fit_a_dress` is unchanged.

## [0.3.0] - 2026-09-06 <a id="quchip-0-3-0"></a>

### Fixes

- Fixed solver selection so density-matrix initial states use `mesolve` even without collapse terms. QuTiP and dynamiqs now reject a density matrix passed explicitly to `sesolve`.
- QuTiP no longer selects `method="diag"` automatically when the resolved SLH Hamiltonian contains network-generated static terms, avoiding diagonal-propagator failures for cascaded degenerate modes. Solver failure messages now include the underlying exception details.
- Replaced the local eigensolver's custom VJP with a custom JVP. `jax.jacfwd` and `jax.hessian` now work through traced device parameters, while `jax.grad` is unchanged; exact degeneracies mask the eigenvector connection to zero, and second derivatives through a degeneracy remain undefined.
- Automatic QuTiP `method="diag"` now covers open systems up to Hilbert dimension 32 (Liouvillian dimension 1024), up from 12. It made a static 20-dimensional transmon-resonator chip 100× faster than adaptive stepping.

### Network and state diagnostics

- `chip.state()` now warns when the requested label's assignment overlap is below `0.9` and points to `chip.bare_state()` for the product state. Degenerate cascaded modes can dress into superpositions and weaken the product-state assignment.
- `PHYSICS.md` now states the multi-port Lamb-shift convention: `phase_shift(phase=2π f τ)` gives `+γ sin φ`, matching Kockum et al. It also states that the SLH core is Markovian: `delay()` shifts reference planes and is not retardation.
- `Port` now documents that `rate` and `external_quality_factor` remain constant within each solve. Shaped emission uses an explicit buffer or coupler device with a static `Port` and a modulated Hamiltonian coupling.
- Added `SimulationResult.collapse_channels`, `collapse_flux()`, and `collapse_integral()` for resolved per-channel jump rates and cumulative expected jump counts. Batch results provide the same methods with `reduce=`; an exposed plane's `raw_photon_flux` matches its collapse flux only for vacuum input.

### Frames

- Added opt-in `frame="auto"`, which chooses per-device frame frequencies from retained couplings and delivered scheduled signals. Drive tones include control gain, attenuation, delay, and crosstalk; coherent-input tones include network scattering and the `2π` conversion to ordinary GHz. Frequencies, pins, clusters, and residual oscillations are available through `resolved_frame.plan`, `chip.describe()`, and `sequence.describe()`.
- `QuantumSequence.build_problem()`, `build_solve_problem()`, and `prepare_solve_problem_context()` now accept `frame=`. Entry-axis batches resolve `"auto"` at each point and use per-point problems when the selected frames differ. Stationary VNA and steady-state analyses retain their existing errors for incompatible tones or dynamic Hamiltonians; chips still default to `"lab"`, and `"rotating"` is unchanged.

### Input-output architecture

- Added an immutable, input-free scalar-S SLH normal form to every resolved engine snapshot. With no ports, ordinary closed/open-system quchip workflows retain their existing behavior.
- Added public `series_product`, `concatenate`, and `feedback_reduce` helpers in `quchip.engine` for textbook composition of resolved SLH triples.
- Added `PortNetwork` for symbolic series composition, named exposures, convenient scalar scattering, and unitary vacuum dilation of attenuation. Two-sided reference sections use `network.delay(...)`; their sweepable, differentiable duration lives at `network.component.<label>.duration`, outside the Markovian `S`, `L`, and `H`. The engine applies their reference-plane factors, so backends no longer apply reference-plane phases. `network.filter(...)` adds passive two-sided reference sections with sweepable, differentiable transfer parameters; continuous-wave response uses `H(f)` exactly, while transients use its narrowband carrier value. `network.amplifier(...)` adds phase-preserving output-line gain with sweepable, differentiable input-referred added noise.
- `VNA(chip)` selects every external plane, while `VNA(chip, ports=...)` selects a subset. Each sweep returns the complete small-signal matrix as `result.matrix`; `result.s(output, input)` selects one entry. Backends solve all input columns from one factorization per frequency. Ordinary chip-parameter `Sweep` axes are accepted beside pump axes and rebind the chip at each point.
- `VNA.sweep()` now also reports the phase-conjugating small-signal matrix `T` as `result.conjugate_matrix`, with the same `[..., output, input]` layout as `result.matrix`; `result.t(output, input)` selects one entry. Around a phase-sensitive operating point, `delta <b_out> = S delta beta + T conj(delta beta)`. The stationary route obtains both matrices from one shifted-Liouvillian factorization, while the passive-linear route returns zero for `T`.
- Added `vna.finite_power(...)`, which solves the stationary Liouvillian for a finite coherent probe and returns a `MeanFieldResponseResult` containing `<b_out>`, `<b_out>/beta`, and the broadcast incident field at every selected plane.
- `PortNetwork.cascade(*items)` accepts variadic chains of ports, single-channel components, and explicit field terminals; `PortNetwork.expose(...)` accepts ports or components as shorthand for their sole or signal terminals. VNA pump tones own their frequency and amplitude axes through `pump.vary(...)`, with `name=` setting the result axis name; `vna.zip(...)` pairs axes point by point.
- Added physical connector sides through `port.side` and `component.side(k)`, bidirectional cabling through `PortNetwork.link(...)`, side exposures through `PortNetwork.expose(..., at=...)`, and ideal `PortNetwork.circulator(...)` and `PortNetwork.isolator(...)` components. `PortNetwork.attenuator(...)` is two-sided and reciprocal, with two hidden vacuum channels.
- `PortNetwork` now compiles instantaneous feedback loops inside the Markov core. It reduces each connection cycle with the scalar Gough–James feedback rule, including loop gain in `S`, `L`, structural reachability, and the generated series/feedback Hamiltonian; closing the same connections in turn with `feedback_reduce` gives the same resolved triple. Reference sections cannot lie inside a loop. Concrete singular loops raise, while traced JAX scattering produces non-finite values at the singular point.
- Added external-plane input scheduling through `network.expose(...).input`; coherent amplitudes are in `sqrt(photons/ns)` and are not stored on `ResolvedSLH`.
- Added complete transient field traces through `result.output(plane)`, with complex amplitude, arbitrary post-solve quadratures, normally ordered photon flux, and the Markov-boundary values before outbound reference sections derived from the same `b_out = S b_in + L` model. `VNA.sweep()` remains small signal, while `VNA.finite_power()` reports the stationary mean field.
- Renamed the `OutputSpectrumResult` fields to distinguish `total_fluctuation_spectrum` from `signal_fluctuation_spectrum` and `added_noise_spectrum`, and to mark `signal_photon_flux`, `signal_coherent_flux`, and `signal_incoherent_flux` as signal-only fluxes. Added amplifier noise remains a spectral density because converting it to flux requires a detection bandwidth; `total_flux` was removed.
- `beam_splitter` now uses the directional convention `[[sqrt(eta), sqrt(1-eta)], [-sqrt(1-eta), sqrt(eta)]]`; directional splitters and 90-degree hybrids have no physical sides, so `component.side(...)` points callers to terminals or `cascade()`.
- `PortNetwork.to_dict()` now records every built-in component by factory `kind` and `parameters`, and `PortNetwork.from_dict()` rebuilds it through that factory. Generic `component(...)` entries retain their terminals and scattering matrix; unknown kinds raise `TypeError`.
- Added `PortNetwork.restrict(ports)`, which copies the independent field-graph components reached from selected ports, including their connections, exposures, tracked parameters, filter callables, and boundary scattering. `Chip.partition()` now carries separable field lines into their device-group sub-chips; if a field graph spans groups, contains components unreachable from any one group's ports, or has boundary scattering that mixes groups, it keeps one joint solve and records why in `partition.notes`. This includes a passive swap between otherwise independent ports.
- Added reusable network blocks through `host.include(template, prefix=...)`. A portless template without boundary scattering can be copied repeatedly under distinct label prefixes; its exposures become interfaces reached through `block.side()`, `block.input()`, or `block.output()`, and copied parameters use paths such as `network.component.<prefix>/<label>.<name>`. Exposure membership checks are now direction-aware, so one side's input and output may belong to different asymmetric planes.

### Visualization

- Added `plot_port_network(...)` for field-network schematics and `plot_sparameters(...)` for magnitude, dB-and-phase, and complex-plane views of small-signal scattering results.

### Resolved analysis and transformations

- Added `EngineResult.dress(at_time=...)`. Static snapshots may omit the time; dynamic snapshots require it and return an instantaneous eigensystem rather than a Floquet spectrum. `Chip.dress()` keeps its exact intrinsic lab-static meaning.
- Partition connectivity now records resolved multi-device support, including Hamiltonian terms generated by SLH cascade composition. Passive scattering alone does not connect independent devices, and field-reference-plane requests safely use the joint solve.
- `eliminate()` can transform default ports on a linear resonator into an effective lowering channel on one unprojected Fock-space survivor while preserving the attached network and exposure. Custom or collective ports, active cascades, projected bases, and multi-survivor field reductions raise rather than dropping or double-counting field physics.

### Current scope

- Scattering is scalar and instantaneous in 0.3; operator-valued scattering, time-dependent collapse channels, Floquet dressing, and thermal input fields remain outside this release. Static composition of several quantum-port couplings requires a shared rotating-frame frequency.

### Breaking changes and migration

- Labels are immutable; create a replacement component to rename one. Replace `device.dressed_freq` and chip-bound `device.drive_freq` with `chip.freq(device)`.
- Local state indices, populations, and Pauli operators use isolated energy levels; excited-state Z is −1.
- Replace `population_array()` / `overlap_array()` with `population()` / `overlap()` (NumPy on QuTiP, JAX on dynamiqs).
- Set `states="all"`, `"final"`, or `"none"` instead of native storage options. For the old nearest-time behavior, pass `method="nearest"` to `state_at()` / `dm_at()`; the default is now `"exact"`.
- `chip.parameters` includes unset optional fields as `None`. Skip them before numerical conversion; unchanged rebinding still works.
- Replace deprecated fitting arguments `coupling_targets`, `observable_targets`, and `fit_parameters` with `constraints`, `vary`, and `start`. Fitting defaults to `evaluator="full"`; choose `"local"` explicitly.
- Replace reduction metadata key `"folded_into"` with `"coupling"`. Keep the returned chip's effective terms; parameter summaries no longer reconstruct the reduction.
- Custom signal transforms use `parameter()` / `setting()` instead of `_parameter_names`; custom reductions implement `retained_hamiltonian(ctx)` and `embedding(ctx)`. See [extensions](docs/extensions.md).
- Saved models require `format_version: 1`; recreate older models from Python declarations.

## [0.2.1] - 2026-08-28

### Fixed

- `ControlEquipment.set_crosstalk_matrix()` now accepts nested Python lists and preserves JAX tracers when stacking their rows.
- Sequence parameter rebinding preserves traced array leaves when rebuilding engine records, keeping `QuantumSequence.with_params()` differentiable through dynamiqs solves.

### Documentation

- Added a post-talk guide that follows the SQA 2026 presentation from dressed statics and pulse-level dynamics through model reduction and differentiation using public APIs.
- Added three executed notebooks covering a bus-mediated avoided crossing with an RWA audit, active-patch reduction with a forward comparison, and pulse-gradient checks against central finite differences.

### Inverse design

- `fit_a_dress(desired)` treats component values as numerical dressed targets and accepts additional `constraints=`, an explicit `vary=` allowlist, and `start=` overrides.
- Fit results now report target sources, bare-parameter seed and sign choices, bounds, final Jacobian rank, condition number, and weak parameter directions through structured fields and `fit.summary()`.

### Analysis

- Added `chip.kerr_matrix()`, returning a frozen, labeled, differentiable matrix of dressed self-Kerr and full-pull cross-Kerr coefficients in chip device order.
- Devices with fewer than three resolved levels report `NaN` only on the self-Kerr diagonal; their defined pairwise cross-Kerr entries remain available.
- `FluxTunableTransmon.flux_bias` is now a bindable, sweepable chip parameter. A flux-only rebind preserves the SQUID calibration and retunes the local Hamiltonian; supplying `freq` and `flux_bias` together defines a new anchor.

### Deprecated

- `fit_a_dress()` keyword arguments `coupling_targets=`, `observable_targets=`, and `fit_parameters=` retain their 0.2.x seed-chip behavior but now emit `DeprecationWarning`. Use the desired-chip API with `constraints=`, `vary=`, and `start=`; the compatibility keywords will be removed in 0.3.0.

## [0.2.0] - 2026-08-14

### Highlights

- Inspect authored and solver-resolved physics as backend-neutral symbolic expressions without first evaluating a numerical matrix.
- Choose explicit Fock, charge, phase-grid, or custom local spaces, with native or energy-eigenstate solver bases.
- Extend every part of the model, from custom devices to classical signal transforms, through tested public contracts.

### Physics and modelling

- Added `PhysicsExpr` for symbolic parameters, matrices, time-dependent scalars, labels, and opaque JAX callables. Expressions support semantic display, immutable parameter rebinding, and explicit numerical evaluation through `.matrix()`. [#4]
- Added `LocalSpace`, `FockSpace`, `ChargeSpace`, `PhaseGridSpace`, and `CustomSpace`. A chip or individual device can use its authored native basis or project into a retained local energy basis with `projection_levels`. [#6]
- Basis resolution now transforms Hamiltonians, couplings, drives, pumps, states, observables, collapse operators, frames, and RWA bands through one engine-owned boundary. Native solving remains the default. [#6]
- Added distinct inspection paths: `unresolved_hamiltonian()` preserves authored static physics, while `hamiltonian()` reports the canonical result after basis, frame, and RWA resolution. Sequence Hamiltonians also include scheduled drives. [#6]
- Added declarative surfaces for custom devices, couplings, component-owned time dependence, drives, envelopes, scalar time coefficients, dissipation, local spaces, classical signal transforms, and scqubits mappings. Installed references exercise each supported path. [#10]
- Drives now map complete delivered analytic signals to quantum Hamiltonians through `hamiltonian(target, signal)`. Envelopes define local pulse shapes, while scheduling and `ControlEquipment` own carrier, phase, gain, delay, filtering, and crosstalk. [#10]
- Added isolated and dressed transition queries through `device.transition_frequency(...)` and `chip.transition_frequency(...)`; `chip.freq(target)` remains the concise dressed `0 -> 1` query. [#10]

### Engine and performance

- Replaced the previous Hamiltonian containers with the frozen `EngineResult`, `SolveProblem`, and `SolveBatch` contracts shared by QuTiP and dynamiqs. [#4]
- Preserved sparse canonical operators through assembly, compiled native batches once, avoided unnecessary result densification, and selected compact dynamiqs storage. [#6]
- Reused resolved engine assembly and chip snapshots when their physics inputs are unchanged, including the common state-preparation and sequence-build path. [#8]
- Added reproducible QuTiP and dynamiqs benchmark CI with separate cold-build, repeated-build, first-solve, and warm-solve measurements, physics-parity checks, environment receipts, and raw samples. Timing changes remain informational. [#7]

### Developer tooling and CI

- Added per-Python CI constraint files and a weekly unconstrained dependency canary. Published package metadata remains unpinned. [#3]
- Added a format-aware prose audit for Python docstrings, Markdown, and rendered HTML. It reports recognizable patterns and coverage without guessing authorship. [#9]
- Added a pull-request template covering verification, documentation, paired notebooks, `PHYSICS.md`, and AI-assistance disclosure. [#9]
- Declarative constructors now expose required fields, defaults, and positional or keyword-only arguments to third-party type checkers through PEP 681 metadata, without generated stubs. [#10]

### Documentation

- Updated the README, physics reference, cookbook, API docstrings, documentation home, and executed hello-chip example for symbolic inspection, local spaces, basis projection, and authored versus resolved Hamiltonians. [#9]
- Kept the hello-chip plots unchanged after clean execution, receipt checks, strict Jupytext pairing, and image comparison. [#9]

### Breaking changes and migration

0.2.0 intentionally breaks compatibility with 0.1.x. The removed APIs below have no compatibility aliases, and the 0.2.0 loader rejects serialized chip payloads produced by 0.1.x. Rebuild those models in code and serialize them again with 0.2.0.

- Replace `HamiltonianDescription` with `EngineResult`, `build_hamiltonian_description()` with `build_engine_result()`, and `SolveProblem.hamiltonian` with `SolveProblem.engine_result`.
- Replace `ProblemBatch` and `BatchedHamiltonianDescription` with `SolveBatch` and the `QuantumSequence` batch APIs.
- `CircuitDevice` is no longer public. Custom devices should declare an explicit `LocalSpace` through `BaseDevice` or `FockDevice`, as appropriate; built-in charge-basis and phase-grid devices use the same boundary.
- Declarative methods receive the symbolic parameter namespace `p`. Custom models use `local_hamiltonian(op, p)`, `interaction(a, b, p)`, and `time_terms(...)` returning `TimeDependentTerm` values.
- Replace Boolean and per-component `rwa=` arguments with the chip-level `approximation=RWA()` or `approximation=Exact()` strategy. `RWA(keep_bands=...)` supports an explicit structural band selection.
- Replace `DriveChannel`, `DriveModulation`, and `DriveSignalSpec` with drive methods: implement `hamiltonian(target, signal)` and override `signal(pulse, target)` only when needed. The delivered signal exposes physical `signal.i` and `signal.q` quadratures after the classical signal chain.
- Replace `EnvelopeShape` with `Envelope`. Custom envelopes implement `value(local_time)`; pulse timing, global phase, and carrier frequency belong to scheduling. In particular, move `Square.phase` to `QuantumSequence.schedule(..., phase=...)`.
- Replace `Modulation` with component `time_terms(...)` returning `TimeDependentTerm` values and a `TimeCoefficient`, such as `CosineCoefficient`.
- Replace `NoiseChannel` with `dissipation(...)` returning `CollapseChannel` values containing unscaled operators and rates in inverse nanoseconds.
- `BaseDrive` and `SignalTransform` are no longer top-level exports. Import them from `quchip.control`; extension authors should normally start from `DeviceDrive` or `CouplingDrive`.
- Use `chip.unresolved_hamiltonian()` when authored lab-frame physics is required. `chip.hamiltonian()` now returns the resolved basis/frame/RWA view used by the engine.
- Required CI currently constrains `qutip<5.3.1` because scqubits 4.3.1 and earlier cannot consume the SciPy sparse arrays returned by qutip 5.3.1. This is a CI compatibility constraint, not a package dependency pin. [#3]

## [0.1.1] - 2026-07-21

### Added

- Added the executed [Hello, drive and readout] example, its reader-facing walkthrough, and a cookbook for executable quchip studies. [#1]
- Added `CITATION.cff` with the accompanying paper as the preferred citation.
- Published the API and physics documentation at [docs.quchip.org].

### Fixed

- Heterogeneous QuTiP problem lists now use the loky process pool while preserving input order. [#2]
- Aligned the physics reference with the implementation and widened one physics-sentinel symmetry bound to a platform-independent solver-accuracy floor.

### Packaging and infrastructure

- Added PyPI trusted publishing for version tags, project links, Python 3.11/3.12 pull-request checks, and scheduled full-suite CI.
- Served README figures from quchip.org so they render consistently on GitHub and PyPI.

## [0.1.0] - 2026-07-19

- Initial public release of the open-source Python toolkit for modelling superconducting quantum chips.
- Included device, coupling, control, frame, RWA, dissipation, transformation, sweep, visualization, and inverse-design APIs; QuTiP and dynamiqs backends; and JAX-compatible differentiation paths.
- Published the README, contribution guide, code of conduct, physics reference, and test suite.

[0.3.2]: https://github.com/quchip/quchip/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/quchip/quchip/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/quchip/quchip/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/quchip/quchip/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/quchip/quchip/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/quchip/quchip/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/quchip/quchip/tree/v0.1.0
[#1]: https://github.com/quchip/quchip/pull/1
[#2]: https://github.com/quchip/quchip/pull/2
[#3]: https://github.com/quchip/quchip/pull/3
[#4]: https://github.com/quchip/quchip/pull/4
[#6]: https://github.com/quchip/quchip/pull/6
[#7]: https://github.com/quchip/quchip/pull/7
[#8]: https://github.com/quchip/quchip/pull/8
[#9]: https://github.com/quchip/quchip/pull/9
[#10]: https://github.com/quchip/quchip/pull/10
[Hello, drive and readout]: https://docs.quchip.org/examples/hello-chip.html
[docs.quchip.org]: https://docs.quchip.org
