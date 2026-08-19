# Changelog

This file records notable user-visible changes to quchip.

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
