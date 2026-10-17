# Changelog

This file records notable user-visible changes to quchip.

## Unreleased

### Changes since 0.4.0

[Full comparison: v0.4.0 → main](https://github.com/quchip/quchip/compare/v0.4.0...main).

#### Changes and migration

- `frame="auto"` ranks coupling bands whose integer weights sum to a nonzero
  value after tones, exchange bands and reference pins. Previously, under
  `Exact()`, a counter-rotating band could set a negative-frequency frame, or
  the lab frame as the solve window grew. Without tones, `"auto"` now follows
  exchange constraints and device references. Pass `frame="lab"` to keep the
  previous no-tone frame. The chip default stays `"lab"`, and frame planning
  removes no Hamiltonian terms. Stored integration-frame phases and numerical
  integration errors can change.
  ([#135](https://github.com/quchip/quchip/issues/135),
  [#141](https://github.com/quchip/quchip/pull/141))
- Devices whose Hamiltonian is diagonal by construction, such as `Resonator`,
  `DuffingTransmon` or `KerrCavity`, keep their levels in authored order when
  `jax.jit` or `jax.grad` traces their parameters. Previously, a traced
  eigensolve sorted the levels by energy, as plain evaluation does. A call now
  raises an error when traced values put the levels out of energy order, or put
  a dropped level below a kept one. For example, a `DuffingTransmon` whose
  spectrum turns over within its levels raises. Use fewer levels or an
  eigenbasis truncation below the turnover, or evaluate without JAX
  transformations. ([#138](https://github.com/quchip/quchip/pull/138))
- Static-coupling `DroppedTerm` records describe the entry with the largest
  `|V|²/ΔE`, where `ΔE = |E_col - E_row|` uses the isolated energies.
  `amplitude` is that entry's `|V|`, and `frequency` is its `ΔE` in any
  frame. Previously, `amplitude` was the band's largest entry and `frequency`
  was the frame carrier, which is 0 GHz in the lab frame. The carrier moves to
  the new `frame_frequency` field. Time-dependent coupling and drive records
  leave `amplitude` and `frequency` as `None`. For records from a coupling's
  `dropped_terms()`, assembly fills a missing `frame_frequency` instead of
  `frequency`. Labels print the unchanged `column - row` weights without `Δ`,
  and the reason reads "discarded by RWA band selection".
  ([#132](https://github.com/quchip/quchip/issues/132),
  [#134](https://github.com/quchip/quchip/issues/134),
  [#137](https://github.com/quchip/quchip/pull/137))
- `RWA()` warns when a discarded static band has a nonzero resonant entry or
  a larger `|V|²/ΔE` than every retained off-diagonal band. The band mask is
  unchanged. For uneven spectra such as `Fluxonium`, compare with `Exact()`
  or list every band to keep in `RWA(keep_bands=...)`, which replaces the
  default selection. Each resolved snapshot reports these warnings and the
  "vanishes entirely" warning for its own approximation. Chip analysis
  reports them once, at the first query that uses its cached model. Implicit
  reference-frequency lookups do not report them.
  ([#133](https://github.com/quchip/quchip/issues/133),
  [#137](https://github.com/quchip/quchip/pull/137))
- An explicit `thermal_occupation=0` keeps a zero-rate absorption channel.
  Previously, a concrete zero dropped it. Dynamics and gradients do not
  change, but each emission channel of that device gains a zero-rate partner,
  and dissipative pulse solves of the chip use the full product space. Omit
  `thermal_occupation` for a vacuum-only model.
  ([#125](https://github.com/quchip/quchip/pull/125))
- An eligible `SolveProblem` stores its initial state, operators and
  `engine_result.dims` in compact excitation coordinates, and
  `EngineResult.dress()` returns compact eigenvectors. Results keep physical
  `dims`. Rebuild a problem with `build_problem(..., initial_state=...)`
  instead of replacing `initial_state` with a product-space state.
  ([#125](https://github.com/quchip/quchip/pull/125))
- Custom `RWA(keep_bands=...)` selections must contain every adjoint band.
  Add each tuple's negation to preserve Hermiticity. Asymmetric selections
  now raise instead of producing non-Hermitian Hamiltonians.
  ([#122](https://github.com/quchip/quchip/pull/122))
- CR susceptibility now includes real-carrier halving. `IX_per_amplitude` and
  `ZX_per_amplitude` are half their previous values, while raw matrix elements
  stay unchanged. Recalculate gate-time estimates from the corrected rates.
  ([#122](https://github.com/quchip/quchip/pull/122))
- Under `RWA()`, carrier drives keep their weight-zero (longitudinal) bands
  with the complete waveform, including at an explicit `freq=0.0`.
  Previously, quchip dropped these bands and listed them as dropped terms, so
  a transmon `FluxDrive` pulse scheduled with `freq=` had no effect. Pulse
  results change for drive operators with diagonal energy-basis elements.
  ([#122](https://github.com/quchip/quchip/pull/122))
- `eliminate()` of a port-coupled mode keeps the mode's scattering as a
  reference section at its dressed transition. A port alone on its plane gets
  `PortNetwork.mode_reflection(...)`. Previously, the reduced chip kept only
  the transformed port, so its VNA response missed the mode's reflection by
  about κ/Δ. The residual is of order (g/Δ)²κ/(2π|Δ|) from Purcell
  dispersion, plus omitted survivor damping and, for SW, higher-order energy
  shifts. A port that shares its plane with fields outside its series path
  now raises `NotImplementedError`. Keep such a mode in the model.
  ([#76](https://github.com/quchip/quchip/issues/76),
  [#85](https://github.com/quchip/quchip/pull/85),
  [#115](https://github.com/quchip/quchip/pull/115))
- Capacitive couplings use each device's declared charge operator, as charge
  drives do. For ordinary Fock devices, this changes `a + a†` to `i(a − a†)`.
  Exchange terms keep their sign, and counter-rotating terms change sign.
  Driven `Exact()` results and mixed-basis exchange phases can change.
  `EigenbasisDevice` now requires `charge_operator` for capacitive coupling.
  ([#108](https://github.com/quchip/quchip/pull/108))
- `FluxTunableTransmon` scales charge by `s = (E_J(Φ)/E_J,max)^(1/4)` and phase
  by `1/s`. Coupling strengths and charge-drive amplitudes now refer to the
  sweet spot. To retain a calibration at another static bias, divide `g` by
  each flux-tunable endpoint's `s`, and charge-drive amplitude by its target's
  `s`. Phase-drive amplitudes require multiplication by `s`. Named charge and
  phase observables and port operators use these scales. A `FluxDrive` pulse
  still changes only the frequency and does not vary the charge scale in time.
  ([#108](https://github.com/quchip/quchip/pull/108))
- `pulse.<i>.detuning` now names the carrier detuning. A custom envelope
  parameter named `detuning` moves to `pulse.<i>.envelope.detuning`, like one
  named `phase`. Code that rebinds or sweeps `pulse.<i>.detuning` for such an
  envelope now adds a carrier offset and keeps the envelope's value. Use the
  new path with `with_params()` and `QuantumSequence.vary()`.
  ([#106](https://github.com/quchip/quchip/pull/106))

#### New features

- `eliminate()` removes a readout with one port in each direction of a two-way
  hanger feedline. The reduced network keeps the readout's dressed pole at the
  original ports, so vacuum `VNA.sweep()` keeps the dip and the reflection into
  the other direction. Both transformed ports keep each qubit's directional
  Purcell decay. In conserving models, a one-port readout that shares a line
  with such a mode also keeps a pole instead of a `mode_transmission(...)`
  section. Only vacuum sweeps use the pole. Pumped, finite-power, noisy and
  excited-reference field calculations raise, and so do time-domain coherent
  inputs at the pole's ports. Elimination raises `NotImplementedError` for two
  ports of one mode on one directional line. It also raises for a multiport
  mode in a model that does not conserve excitation number.
  ([#131](https://github.com/quchip/quchip/issues/131),
  [#140](https://github.com/quchip/quchip/pull/140))
- Deterministic pulse simulations on QuTiP and dynamiqs evolve in an invariant
  excitation space when quchip can prove one. The proof covers the initial state
  and every retained Hamiltonian, control, collapse operator and no-jump
  product. A batch proves one space for the initial states of all its points. A
  closed one-excitation evolution of a ring of 24 three-level devices uses 24
  coordinates, or 25 when decay adds the vacuum, instead of 3²⁴. Transverse
  drives and thermal raising can require the full space, and stochastic solvers
  keep it. When the dense operators of a proven space do not fit in memory, the
  solve uses sparse operators in the full space. Local observables, measurements
  and saved-state replay stay compact. `Chip.compact_state()` prepares bare or
  conserving dressed states without allocating the product space. Full-space
  operator assembly and state exports check available memory first. Under JAX
  tracing, the proof holds for traced coupling strengths and for traced
  parameters of diagonal device Hamiltonians, such as a `DuffingTransmon`
  frequency. quchip applies no excitation cutoff.
  ([#125](https://github.com/quchip/quchip/pull/125),
  [#138](https://github.com/quchip/quchip/pull/138))
- `VNA.sweep(..., when={q: 1})` computes weak-probe scattering from a frozen
  dressed reference with zero or one excitation. The response includes all
  declared loss channels and coherence recycling. Diagnostics report reference
  escape and stationarity rates. This is an approximate CW response when the
  reference decays, and does not simulate a finite acquisition. QuTiP and
  dynamiqs both compute it, with JAX derivatives on fixed structure. On
  dynamiqs, eager sweeps of every length reuse one compiled solve for one
  frequency, and a traced sweep compiles one loop over its frequencies.
  Iterative blocks stop at convergence and check the residual of every
  response and derivative solve.
  ([#124](https://github.com/quchip/quchip/pull/124),
  [#154](https://github.com/quchip/quchip/pull/154),
  [#176](https://github.com/quchip/quchip/pull/176))
- Labeled dressed queries, `drive_matrix_elements()`, weak-probe VNA and
  `fit_a_dress()` diagonalize only the excitation sectors they need when the
  chip conserves total excitation number. The engine proves conservation from
  the retained operators, so conserving `Exact()` models qualify as well as
  `RWA()` models. Other chips keep the full-space route. `fit_a_dress()`
  applies `max_hilbert_dim` to the largest matrix it diagonalizes and checks
  it again after rebinding or tracing. A 24-transmon `Exact()` exchange chain
  now fits with `max_hilbert_dim=24`. Full spectra, dressed states and
  `dress()` still diagonalize the complete chip.
  ([#98](https://github.com/quchip/quchip/issues/98),
  [#114](https://github.com/quchip/quchip/pull/114),
  [#123](https://github.com/quchip/quchip/pull/123))
- Dispersive readout accepts `kappa_detected` and `efficiency`. Total linewidth
  controls pointer states and dephasing. Observed loss controls detected SNR.
  Defaults preserve ideal collection. Invalid concrete input domains raise.
  ([#122](https://github.com/quchip/quchip/pull/122))
- `eliminate()` removes a readout whose port shares a feedline with other ports.
  `PortNetwork.mode_transmission(...)` keeps its dressed transmission as a
  one-pass reference section. Reference sections include internal loss that
  earlier eliminations carried onto several devices. The result reports a
  dispersive field-error scale for kept ports beyond the section as
  `kappa_over_delta`, not a strict bound.
  ([#116](https://github.com/quchip/quchip/pull/116))
- `eliminate()` removes port-coupled modes on chips with more than two devices.
  The transformed port acts jointly on every survivor, and VNA accepts it. A
  later elimination transforms that port again, so you can eliminate both
  readout modes of a chip in either order.
  ([#77](https://github.com/quchip/quchip/issues/77),
  [#85](https://github.com/quchip/quchip/pull/85))
- `PortNetwork.mode_reflection(...)` adds a serializable two-sided reference
  section that reflects like a damped linear mode.
  ([#76](https://github.com/quchip/quchip/issues/76),
  [#85](https://github.com/quchip/quchip/pull/85))
- `eliminate(chip, device, local=True)` computes the SW reduction from a patch
  around the device. The retained Hamiltonian, channels and map match the
  full reduction up to round-off omissions, which the result's notes report
  in GHz. Each correction part and channel keeps its own support, so repeated
  local eliminations around a ring stay local. Noisy channels can still
  spread along the captured maps. Ports on a shared feedline enter each patch
  as independent copies, and the reduced chip keeps the complete feedline.
  On one machine, a lossy 24-mode ring with every readout on one feedline
  reduces to its eight readouts in 8 s with a 1.3 GB peak, as with
  independent ports. The removed mode's reference section includes spectator
  dressing from interactions outside the patch, as with `local=False`. `chi`
  describes only the patch. `Chip.partition()` keeps the devices of each
  captured map in one part. Both `active_patch()` entry points forward
  `local`. `method="exact"` and chips with baths raise `NotImplementedError`.
  ([#112](https://github.com/quchip/quchip/pull/112),
  [#113](https://github.com/quchip/quchip/pull/113),
  [#130](https://github.com/quchip/quchip/issues/130),
  [#139](https://github.com/quchip/quchip/pull/139))
- `fit_a_dress(evaluator="local")` accepts chips with a `PortNetwork`. Each
  neighborhood keeps the ports whose targets it holds, with their field
  subgraphs. A cut through a shared feedline, a joint port or unsupported
  boundary scattering raises `NotImplementedError`. Previously, any port
  network raised. ([#100](https://github.com/quchip/quchip/issues/100),
  [#109](https://github.com/quchip/quchip/pull/109))
- `QuantumSequence.schedule()`, `charge()` and `phase()` accept `detuning=`, a
  carrier offset in GHz referenced to the pulse start. The pulse is then the
  same in its frame at every start time, so a detuned calibrated gate keeps its
  rotation axis when it moves. `pulse.<i>.detuning` rebinds, sweeps and
  differentiates like `freq`. A carrier at `freq = f + δ` stays a fixed
  oscillator. ([#106](https://github.com/quchip/quchip/pull/106))
- `QuantumSequence.schedule()`, `charge()` and `phase()` accept `frame=`, the
  device whose virtual-Z phase a carrier pulse follows on any line. For
  cross-resonance control, set the target qubit as the frame of both the
  cross-resonance and cancellation tones. The default remains the driven
  device. ([#92](https://github.com/quchip/quchip/issues/92),
  [#105](https://github.com/quchip/quchip/pull/105))

#### Fixes

- Pump-free `VNA.sweep()` computes models that are static in the lab frame but
  dynamic in the probe frame, e.g. capacitive couplings under `Exact()`.
  Previously, it raised. It solves one lab-frame stationary state per chip point
  and a shifted Liouvillian at each probe frequency, and `T` is zero away from
  DC. Other pump-free stationary sweeps also use this route, so their residual
  and condition-number diagnostics describe the lab-frame generator.
  ([#148](https://github.com/quchip/quchip/issues/148),
  [#155](https://github.com/quchip/quchip/pull/155))
- On a chip reduced by `eliminate()` or `active_patch()`,
  `drive_matrix_elements()` passes each drive operator through the retained
  coordinate maps, as pulses do. Previously, it used the authored operator and
  missed the dressing by removed devices, e.g. 0.125% at g/Δ = 0.05.
  ([#146](https://github.com/quchip/quchip/issues/146),
  [#150](https://github.com/quchip/quchip/pull/150))
- QuTiP calculations that receive traced JAX values, e.g. under `jax.grad`,
  raise a `ValueError` that names `backend='dynamiqs'`. Previously, they
  failed inside operator arithmetic with `TypeError` or
  `ConcretizationTypeError`. The compact weak-probe VNA route applies the
  same check to port rates and phases.
  ([#128](https://github.com/quchip/quchip/issues/128),
  [#142](https://github.com/quchip/quchip/pull/142))
- `jax.grad` and `jax.jit` with a traced device frequency pass the
  collapse-channel frame check after `eliminate()` and for a collective `Bath`
  in a common frame, also on a reduced chip. Previously, they raised a
  `ValueError`. `jax.jit` of the rate of a collective `Bath` on
  `EigenbasisDevice` targets also passes the check. After `eliminate()`, a
  traced coupling also passes the check on the port of a surviving mode in
  per-device frames. Traced `VNA.sweep()` calls keep the mode-space route of
  plain evaluation, also on reduced chips, and `when=` no longer raises.
  ([#126](https://github.com/quchip/quchip/issues/126),
  [#127](https://github.com/quchip/quchip/issues/127),
  [#144](https://github.com/quchip/quchip/issues/144),
  [#138](https://github.com/quchip/quchip/pull/138))
- Local eigenvector derivatives retain resolved small-gap responses. Their
  cutoff now uses matrix dimension and floating-point precision instead of
  `1e-9` of the spectral span. Degenerate eigenvector derivatives remain
  undefined.
  ([#122](https://github.com/quchip/quchip/pull/122))
- `eliminate(..., method="exact")` diagonalizes each total-excitation sector
  separately when the static model conserves excitation number, as under
  `RWA()`. Retained terms no longer carry ~1e-12 entries between sectors.
  These entries made the rotating-frame collapse-operator check fail at some
  parameter values. ([#78](https://github.com/quchip/quchip/issues/78),
  [#82](https://github.com/quchip/quchip/pull/82))
- `resolve()` of a reduced chip, including kept and transformed ports, gives
  the same result under `jax.jit` and `jax.grad` as eagerly. Previously, an
  exactly reduced chip could fail the rotating-frame check under `jax.jit`,
  even when no traced value entered the reduction.
  `EffectiveTerms(..., excitation_changes=...)` and
  `PhysicsExpr.from_matrix(..., excitation_changes=...)` declare the structure
  that keeps this check independent of traced values.
  ([#78](https://github.com/quchip/quchip/issues/78),
  [#79](https://github.com/quchip/quchip/issues/79),
  [#82](https://github.com/quchip/quchip/pull/82))
- A band whose level changes cancel among devices that share a frame frequency
  is now exactly static. Its frame frequency was summed one device at a time,
  so level changes such as (3, −1, −2) left carriers of approximately 1e-14
  GHz. An exactly reduced two-qubit readout chip in a 5.2 GHz frame resolved
  44 static couplings as time-dependent terms. It also split drive bands with
  equal carriers into separate terms.
  ([#78](https://github.com/quchip/quchip/issues/78),
  [#82](https://github.com/quchip/quchip/pull/82))
- QuTiP `mesolve`, `smesolve` and stationary solves raise `MemoryError` before
  they assemble a superoperator whose estimated peak exceeds the available
  memory. Previously, the operating system killed them. The dynamiqs
  stationary Liouvillian uses the same check.
  ([#80](https://github.com/quchip/quchip/issues/80),
  [#82](https://github.com/quchip/quchip/pull/82))
- `eliminate()` reports `j_eff` and `dJ_domega_c` in the emitted edge's units,
  accounting for complex or non-unit charge matrix elements. Previously,
  taking the real exchange element could produce a zero or mis-scaled edge.
  Retained terms still carry the full reduced Hamiltonian. `g_over_delta`
  now uses resolved exchange rather than authored strength. Parallel device
  couplings report their combined exchange.
  ([#111](https://github.com/quchip/quchip/pull/111))
- QuTiP preserves envelope steps listed in `sampling_times()` after pulse
  starts and line delays, within floating-point precision. Previously, these
  steps could become interpolation ramps. Unlisted steps still require explicit
  feature times. Pulse edges at zero no longer cause interpolation overflow
  warnings when the solve starts before zero.
  ([#104](https://github.com/quchip/quchip/pull/104))

#### Performance

- Multi-device operators project factor by factor into the retained local
  eigenbases instead of through the dense native product space. Two coupled
  default `Fluxonium` devices previously needed an estimated 820 GB. Their
  `chip.freq()` now peaks below 256 MB. Opaque operators keep the dense route.
  The engine estimates both routes, takes the smaller, and raises
  `MemoryError` before an allocation that exceeds available memory.
  ([#96](https://github.com/quchip/quchip/issues/96),
  [#107](https://github.com/quchip/quchip/pull/107))
- `VNA.sweep()` solves weak-probe scattering in the one-excitation block when a
  pump-free chip conserves total excitation number and every input is vacuum.
  This covers Duffing transmons, pure dephasing, reduced chips with retained
  terms, and shared feedlines whose ports each lower excitation by one.
  Previously, these chips fell back to a dense D²×D² stationary solve. On one
  machine, the reported transmon-resonator sweep at Hilbert dimension 64 took
  540 s. Now it takes 10 s, mostly for the frame's reference-frequency
  analysis. Dimension 400 runs in 10 s instead of needing about 410 GB.
  Diagnostics name the route `"vacuum_response"`.
  ([#81](https://github.com/quchip/quchip/issues/81),
  [#85](https://github.com/quchip/quchip/pull/85),
  [#110](https://github.com/quchip/quchip/pull/110))
- The QuTiP backend stores dense operators that are at most one quarter
  nonzero as CSR. Superoperator terms built from reduced-model bands and jumps
  then stay sparse. A lossy `mesolve` of a reduced model at Hilbert dimension
  144 now peaks near 1.3 GB instead of an estimated 1.2 TB.
  ([#80](https://github.com/quchip/quchip/issues/80),
  [#82](https://github.com/quchip/quchip/pull/82))
- dynamiqs solves sum the time-dependent Hamiltonian into one array per step
  from operators assembled on the host. A traced chip reuses its resolution and
  dressed analysis within one trace. On one machine, a 576-state chip's jitted
  reverse-mode gradient through an exact elimination and a dynamiqs pulse took
  830 s per call. It now takes 3.7 s. A batch of eight pulse phases takes
  4.1 s instead of 20 s. ([#85](https://github.com/quchip/quchip/pull/85))
- QuTiP `mesolve` of a Hermitian density matrix integrates its packed upper
  triangle with one sparse product per step. Drive carriers that act through
  the same operator share one coefficient. A lossy pulse on the same chip
  takes 200 s instead of 400 s. A coherent 16-pulse train takes 7.3 s
  instead of 49 s. ([#85](https://github.com/quchip/quchip/pull/85))
- Large packed QuTiP master-equation products split their rows over up to four
  threads, which brings that lossy pulse to 145 s. `QUCHIP_NUM_THREADS`, or
  else `OMP_NUM_THREADS`, sets the count. Worker processes use one thread
  unless `QUCHIP_NUM_THREADS` is set. Each row sums in the same order, so
  results do not depend on the thread count.
  ([#85](https://github.com/quchip/quchip/pull/85))
- Static analysis of concrete chips sums, diagonalizes, reduces and labels
  operators with NumPy instead of compiling a JAX program for each array shape.
  `fit_a_dress()` on the 576-state chip takes 3.1 s per fit instead of 15 s.
  ([#85](https://github.com/quchip/quchip/pull/85))
- On the general stationary route, `VNA.sweep()` solves an operating point that
  stays stationary across probe frequencies once, not at every frequency. The
  QuTiP backend runs its default direct steady-state solve without QuTiP's
  per-call option scope, which rebuilt every data-layer dispatcher.
  `steadystate_batch()` over 40 points runs 5.4 times faster, and
  `VNA.finite_power()` over 44 points runs 3.2 times faster.
  ([#85](https://github.com/quchip/quchip/pull/85))

#### Compatibility

- The dynamiqs backend supports dynamiqs 0.3.7, where `QArray` is abstract,
  and keeps support for 0.3.6. Sparse operators use the public constructor
  and keep JAX gradients. Batched integration uses the public time-array
  conversion and native save callbacks, and still reports numerical failures.
  ([#136](https://github.com/quchip/quchip/issues/136),
  [#149](https://github.com/quchip/quchip/pull/149))
- `Backend.steadystate()` takes an optional `guess`: a native stationary
  state of a related problem, which a backend can return if it is also
  stationary here. A backend that overrides `steadystate()` must accept the
  keyword but can ignore it. ([#85](https://github.com/quchip/quchip/pull/85))
- `Backend.coherence_response()` solves conditional VNA responses. Its default
  raises `NotImplementedError`, so a custom backend must implement it to
  support `VNA.sweep(..., when=...)`. The QuTiP and dynamiqs backends
  implement it. ([#124](https://github.com/quchip/quchip/pull/124))

#### Documentation and examples

- `PHYSICS.md` §10.4 identifies the Löwdin map of
  `eliminate(..., method="exact")` with the canonical Schrieffer–Wolff
  rotation to all orders, provided the kept/dressed overlap block is
  nonsingular. Off-diagonal reads differ from `method="sw"` because SW
  truncates at second order.
  ([#129](https://github.com/quchip/quchip/issues/129),
  [#143](https://github.com/quchip/quchip/pull/143))
- Correct dispersive signs, readout rate definitions, VNA phase notation, and
  reduction limits. Document device and bath models, convergence requirements,
  and parametric control. Add primary references for the physical conventions.
  Align guide calculations, fitting targets, and plots with ground-positive Z.
  ([#122](https://github.com/quchip/quchip/pull/122))
- Rewrite docstrings, guides, examples, the README and `PHYSICS.md` in
  shorter, plain sentences. Code, error messages and notebook code cells are
  unchanged. ([#87](https://github.com/quchip/quchip/pull/87))
- Remove the NV-centre guide, its figures and its links.
  ([#85](https://github.com/quchip/quchip/pull/85))
- Re-execute the example notebooks and guides against the current code. The
  transmon avoided crossing in example 01 now reports the RWA splitting of
  3.99 MHz, which agrees with second-order theory to 0.3%. The fluxonium
  readout comparison uses `Exact()`, because counter-rotating terms contribute
  strongly to its dispersive shift.
  ([#88](https://github.com/quchip/quchip/pull/88))

#### Development

- `CONTRIBUTING.md` documents the release-note format, and the pull request
  template uses checkboxes for AI assistance.
  ([#75](https://github.com/quchip/quchip/pull/75),
  [#89](https://github.com/quchip/quchip/pull/89))
- `tools/theme_figures.py` renders a PDF or dark PNG only when its light SVG
  changed. `--force` renders every figure.
  ([#88](https://github.com/quchip/quchip/pull/88))

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
