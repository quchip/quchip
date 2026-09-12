# Changelog

This file records notable user-visible changes to quchip.

## Unreleased

- Add native quantum-jump and diffusive SSE/SME solvers on QuTiP and Dynamiqs,
  with parameter batches, native results, and explicit monitored-channel selection.
- Truncation diagnostics are now explicit. Remove `check_truncation` and
  `truncation_threshold` from solve calls; use `result.check_truncation()` or
  prepare `with_truncation(problem)` when saving only diagnostics.
- Omitted stochastic storage uses native defaults. Native run keywords go in
  `run_args`; integrator options remain in `options`.

## [0.3.0] - 2026-09-06

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

- Removed `SimulationResult.population_array()` and `overlap_array()` without aliases. Use `population()` and `overlap()`; they return NumPy arrays with QuTiP and JAX arrays with dynamiqs.
- Reduction entries now name their resulting edge with `effective_params[<edge>]["coupling"]` instead of `"folded_into"`. Update code that reads this metadata.
- `chip.parameters` now includes unset optional device fields such as `T1` and `T2` as `None`. Filter these entries before numerical conversion; passing them unchanged to `with_params()` remains supported.

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
