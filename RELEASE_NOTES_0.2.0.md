# quchip 0.2.0

quchip 0.2.0 makes the physics assembled by the engine inspectable and gives custom models a smaller, tested extension surface.

## Highlights

- Inspect the model before and after engine resolution. `unresolved_hamiltonian()` preserves authored lab-frame physics, while `hamiltonian()` reports the basis-, frame-, and approximation-resolved expression used for simulation.
- Author devices in Fock, charge, phase-grid, or custom local spaces. Use the native basis or project each device into a retained local energy basis without changing the attached couplings, drives, states, observables, or dissipation channels by hand.
- Extend devices, couplings, component-owned time dependence, drives, envelopes, dissipation, local spaces, classical signal transforms, and scqubits mappings through documented contracts backed by installed reference implementations.
- Query isolated or dressed transitions with `device.transition_frequency(...)` and `chip.transition_frequency(...)`, including conditional dressed transitions. `chip.freq(target)` remains the short form for the dressed `0 -> 1` transition.

## Control and approximation model

Envelopes now describe only the local pulse shape. Scheduling adds timing, phase, and an optional carrier; `ControlEquipment` applies gain, delay, filtering, distortion, and crosstalk to complete analytic signals; the drive then maps the delivered `signal.i` and `signal.q` quadratures into the quantum Hamiltonian.

Approximation selection is explicit at the chip boundary:

```python
chip = Chip(..., approximation=Exact())  # retain every authored term
chip = Chip(..., approximation=RWA())    # structural first-order RWA
```

`RWA(keep_bands=...)` replaces the default total-excitation selection with an explicit set of structural operator bands. Removed bands remain available as `DroppedTerm` records.

## Engine and performance

- `EngineResult` is the frozen backend-neutral record of the resolved Hamiltonian and noise terms. `SolveProblem` and `SolveBatch` carry the complete QuTiP- or dynamiqs-ready request.
- Sparse canonical operators remain sparse through assembly where possible.
- Native batches compile once, resolved chip contracts are reused when their physics is unchanged, and result inspection avoids unnecessary densification.
- Benchmark CI records cold build, repeated build, first solve, and warm solve separately for QuTiP and dynamiqs, with physics-parity checks and environment receipts.

## Upgrade notes

- Replace `HamiltonianDescription` with `EngineResult`, `build_hamiltonian_description()` with `build_engine_result()`, and `SolveProblem.hamiltonian` with `SolveProblem.engine_result`.
- Replace Boolean `rwa=` arguments with `approximation=RWA()` or `approximation=Exact()`.
- Custom declarative methods receive a symbolic parameter namespace: `local_hamiltonian(op, p)`, `interaction(a, b, p)`, and `time_terms(...)`.
- Custom drives implement `hamiltonian(target, signal)`; most inherit the default scheduled-signal construction. Custom envelopes implement `value(local_time)`.
- Dissipation hooks return `CollapseChannel` values containing an unscaled operator and a rate in inverse nanoseconds.
- `CircuitDevice` is no longer public. Custom devices declare an explicit `LocalSpace` through `BaseDevice` or `FockDevice`.

See the [changelog](https://github.com/quchip/quchip/blob/v0.2.0/CHANGELOG.md) for the complete list and the [extension guide](https://docs.quchip.org/extensions.html) for working implementations.

**Full changes:** [v0.1.1...v0.2.0](https://github.com/quchip/quchip/compare/v0.1.1...v0.2.0)

Previous releases: [0.1.1](https://github.com/quchip/quchip/releases/tag/v0.1.1), [0.1.0](https://github.com/quchip/quchip/tree/v0.1.0).
