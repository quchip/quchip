# Changelog

This file records notable user-visible changes to quchip.

## Unreleased

### Changes since 0.4.0

[Full comparison: v0.4.0 → main](https://github.com/quchip/quchip/compare/v0.4.0...main).

#### Changes and migration

- quchip 0.5 removes the names that 0.4 deprecated. Use `thermal_occupation`
  for `thermal_population` (in constructors, `with_params()`, `set_noise()`,
  declared parameters, symbolic noise and saved device dictionaries),
  `jump_rate()` for `collapse_flux()`, `port()` for `side()`,
  `external_port()` and `external_ports` for `exposure()` and `exposures`,
  and `NetworkPort` and `ComponentPort` for `FieldExposure` and `FieldSide`.
  The `check_truncation()` result methods stay until 0.6.
- Warnings are shorter, name the one action that sets or silences them, and
  point at the caller's line. The default-frame and automatic-carrier
  warnings have fixed text, so Python shows each once per line of code
  instead of once per chip in a loop. A dressed label that keeps less than
  90% of its bare product state warns once per chip, not at every query, and
  `chip.freq()` and frame references still raise for an ambiguous label
  (overlap below one half or a vanishing margin). Partitioned results no
  longer warn when they build the joint state; the memory guard stays.
- The dynamiqs `steadystate()` counts the Liouvillian nullity only up to
  `diagnostic_max_dimension` (default 16), as the QuTiP backend does. Above
  it, the singular value decomposition, which cost about ten times the solve,
  does not run: `nullity` is `None`, and the solve gives no uniqueness
  guarantee: a singular system can return NaN or Inf, and a nearly singular
  one an unreliable state. Pass a larger
  `diagnostic_max_dimension` to keep the check. A
  `D = 24` steady state takes about 0.05–0.1 s instead of 0.5–0.8 s.
- A QuTiP batch falls back to one process only when a worker process stops.
  Other pool errors, such as a pickling failure, now raise instead of
  rerunning the batch.
- A pulse scheduled without `freq` on a charge, phase, or two-photon line takes
  a carrier at the line's device and warns. Charge and
  phase lines use `chip.freq(device)`, the dressed `0 -> 1` frequency, and a
  two-photon line uses twice it. Previously, such a pulse had no carrier, so a
  charge drive stayed about 5 GHz below a transmon and left it in `|0>`: a
  20 ns Gaussian π pulse from `chip.rotation_amplitude()` gave `P1 = 5e-11`
  without a warning, and now gives `0.99`. The new default is `freq="auto"`,
  and its warning comes once per line in each sequence. Flux and coupling
  drives, custom drives, and coherent inputs keep a baseband pulse. A custom
  drive can set the class attribute `carrier_harmonic` to take an automatic
  carrier. To keep a previous pulse without a carrier, pass `freq=None`.
  `freq=0.0` stays a carrier at zero frequency. `QuantumSequence.charge()`
  takes the same default and warning, and `charge(freq=None)` now gives a
  pulse without a carrier, as `schedule(freq=None)` does. Previously,
  `charge(freq=None)` used `chip.freq(target)`. If you passed `freq=None` to
  `charge()` for that carrier, omit `freq`, or pass `freq=chip.freq(target)`
  to keep it without the warning.
  ([#213](https://github.com/quchip/quchip/pull/213))
- A time-domain solve on a chip without a frame uses the rotating frame and
  warns with that choice. `Chip(frame=None)` is the new default, and static
  calculations keep the lab frame: `chip.resolve()`, `chip.hamiltonian()`,
  dressed analysis and steady states. Previously, `Chip()` set
  `frame="lab"`, so a solve resolved each device's phase at its transition
  frequency. A transmon with `T1` and a resonator, 15 states, driven for
  20 ns and solved for 100 ns with automatic sampling, took 2.8 s and
  1.76 GB. It now takes 0.43 s and 0.33 GB, and its `P1` error at QuTiP's
  default tolerances falls from 1.0e-4 to 3.5e-6. Expectation values do not
  depend on the frame. Stored states and trajectory measurement records are
  now in the rotating frame, and `result.state(..., frame="lab")` converts a
  state. A ket `initial_state` is an integration-frame vector at `tlist[0]`,
  so when `tlist[0]` is not zero it is now a vector of the frame that the
  warning names. When a collapse or port operator does not fit the rotating
  frame, the solve uses the frame that `frame="auto"` plans and warns. Examples are a reduced chip whose eliminated bus carries
  each transmon's jump onto the other, and two readouts on one feedline.
  When the planned frame does not fit either, the solve uses the lab frame.
  A batch checks every point and solves all of them in the first of these
  frames that fits each point. In the planned frame, each point gets its own
  plan, as with `frame="auto"`, and each point starts in the same physical
  state as its own solve. When a device has no reliably assigned dressed
  `0 -> 1` frequency, as in a resonant pair of identical devices, the solve
  raises, names the devices, and asks for a frame such as
  `Chip(frame="lab")`. `chip.frame` is `None` on a chip without a frame.
  `chip.frame_info()` returns the frequencies of the frame that a solve
  without pulses would use, without a warning, or raises as the solve does. To keep the previous behavior, pass
  `Chip(frame="lab")`. A saved chip keeps its saved frame.
  ([#214](https://github.com/quchip/quchip/pull/214))
- `eliminate(..., method="sw")` keys the exchange `pathways` by occupation
  tuples in the source chip's device order, as `analyze_static_zz()` does.
  Previously, the keys were flat product indices, such as `1` and `17` for
  `(0, 0, 1)` and `(1, 1, 1)` on a chip with dims `(3, 3, 4)`. The pathways of
  both calls now list every nonzero contribution, so they sum to the
  second-order part of their element. Previously, contributions with
  `|V_ik V_kj| <= 1e-12` GHz² were left out, so a weak coupling across a small
  gap could list no pathway for a shift of `-1e-9` GHz.
  ([#199](https://github.com/quchip/quchip/issues/199),
  [#208](https://github.com/quchip/quchip/pull/208))
- A level label such as `{"q0": 1}` or `"eg"` names its dressed state
  everywhere. An `initial_state` label starts a solve in the dressed state of
  the label, under the solve's approximation and excitation cutoff, placed in
  the solve frame at `tlist[0]` as an omitted `initial_state` is. So
  `initial_state={}` equals an omitted state. Previously, a label
  `initial_state` was the bare product, so a coupled chip exchanged
  excitation from the start. Two transmons 100 MHz apart with `g = 30 MHz`
  exchanged 26% of the population. `chip.superposition()` and
  `chip.compact_state()` also give dressed states, and
  `compact_state(..., dressed=False)` keeps the bare form. Uncoupled chips do
  not change. To keep a bare start, pass `initial_state=chip.bare_state(...)`,
  which also takes `(amplitude, label)` pairs for bare superpositions. Inside
  `jax.jit`, `chip.bare_state()` now stays a constant when it depends on no
  traced value, so this start keeps the exact excitation space of its levels.
  Previously, the traced ket made the solve use the full product space. A
  mapping with a local ket raises as an `initial_state`. Pass
  `chip.bare_state(...)` instead, or `chip.compact_state(..., dressed=False)`,
  which now also takes local kets, on a chip too large for its product space.
  `population()` stays the local projector.
  ([#206](https://github.com/quchip/quchip/issues/206),
  [#205](https://github.com/quchip/quchip/pull/205))
- `active_patch()` eliminates spectators in the default order of
  `eliminate_spectators()`. Previously, it took the spectator farthest from the
  active devices first, then label order. On rings of transmons and coupling
  resonators, that order spread a drive over 768 states and dropped the
  near-resonant exchange shifts of kept transmons. The reduced chip, its notes
  and `eliminated_labels` can change. To keep a previous order, pass its
  `eliminated_labels` as `order=`.
  ([#151](https://github.com/quchip/quchip/issues/151),
  [#183](https://github.com/quchip/quchip/pull/183))
- `chip.state()`, `DressedResult.eigenstates` and
  `DressedResult.eigenvector_matrix` give each dressed vector a real,
  nonnegative amplitude on its assigned bare state. Previously, their signs
  came from the eigensolver, so a state could differ in sign between QuTiP
  and dynamiqs. Dressed-basis operators and drive matrix elements already
  used this phase. When the assigned amplitude is zero, the vector's largest
  amplitude now sets the phase of all these outputs. Energies, populations
  and overlap magnitudes do not change. Code that transforms with the
  eigenvector matrix directly can see off-diagonal entries change phase.
  ([#163](https://github.com/quchip/quchip/issues/163),
  [#170](https://github.com/quchip/quchip/pull/170))
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
- Without an excitation cutoff, `VNA.finite_power()` and `VNA.measure()` with
  `options=None` solve a chip that conserves the total excitation and has more
  than 64 product states on the labels with `N <= K`, and they warn which `K`
  they kept. When `K + 1` fits in memory, they compare the results with it at
  a tolerance of `1e-6`, keep `K + 1`, and the warning says whether this
  check ran. The
  check does not bound the error against the full product space. Pass
  `options={}` to keep the full product-space solve.
  ([#156](https://github.com/quchip/quchip/issues/156),
  [#185](https://github.com/quchip/quchip/pull/185))
- The truncation check is now the function `quchip.check_truncation(result)`,
  next to `with_truncation`. It accepts the same results and returns the same
  reports as the `check_truncation()` methods of `SimulationResult`,
  `TrajectoryResult` and `PartitionedSimulationResult`. These methods still
  work but warn with `DeprecationWarning`, and quchip 0.6 removes them. Under
  an `ExcitationCutoff`, a request now records only the cutoff and its boundary
  totals for the check, and the check finds the compact rows when it runs.
  ([#188](https://github.com/quchip/quchip/pull/188))

- On a chip that conserves the total excitation `N` and has more than 64
  product states or an `ExcitationCutoff`, `steadystate()` returns the vacuum
  without a Liouvillian solve when the lowering channels prove it unique. The
  result holds the vacuum on the labels with `N <= 1`, which
  `result.excitation_space` names, so `result.state` is no longer a
  product-space matrix. `reduced_state()` and `e_ops` values do not change.
  The call no longer raises in a frame where the resolved Hamiltonian is not
  static, because the vacuum is the same in every frame. Pass `options={}` to
  keep the product-space solve.
  ([#191](https://github.com/quchip/quchip/issues/191),
  [#203](https://github.com/quchip/quchip/issues/203),
  [#205](https://github.com/quchip/quchip/pull/205))
- Without an excitation cutoff, `VNA.output_spectrum()`, `VNA.g1()` and
  `VNA.g2()` with `options=None` solve a chip that conserves `N` and has more
  than 64 product states on the labels with `N <= K`, as `finite_power()`
  does, and warn which `K` they kept. They compare each offset or delay with
  `K + 1` as `finite_power()` does. Pass `options={}` to keep the full
  product-space solve. `finite_power()` and `measure()` also take this route
  when the VNA has pumps at the probe frequency.
  ([#191](https://github.com/quchip/quchip/issues/191),
  [#205](https://github.com/quchip/quchip/pull/205))
- QuTiP batches take their default worker count from loky's `cpu_count()`,
  which follows `LOKY_MAX_CPU_COUNT`, CPU affinity and container limits,
  instead of `os.cpu_count()`. The default pool shrinks to the number of
  workers that the available memory holds, and each result records its count
  in `stats["workers"]`.
  ([#196](https://github.com/quchip/quchip/issues/196),
  [#205](https://github.com/quchip/quchip/pull/205))
- `pulse.<n>` parameter paths name the n-th scheduled pulse. Virtual-Z
  shifts, delays and barriers no longer count, so `vz()` or `delay()` does not
  move the paths of later pulses in `parameters`, `with_params()` and
  `vary()`. Sequences without such entries keep their paths. In a sequence
  with a `vz()`, `delay()` or `barrier()` before a pulse, renumber the paths:
  after `vz()`, `delay()` and one pulse, `pulse.2.amplitude` is now
  `pulse.0.amplitude`. ([#217](https://github.com/quchip/quchip/issues/217),
  [#221](https://github.com/quchip/quchip/pull/221))
- Every built-in device now uses the textbook charge sign `[φ, n] = +i`, as
  Krantz et al. (2019), Koch et al. (2007), Blais et al. (2021) and scqubits
  do. The Fock devices (`DuffingTransmon`, `Resonator`, `KerrCavity`, `Qubit`)
  and `FluxTunableTransmon` use `Q = i(a† − a)` instead of `i(a − a†)`, and
  the charge basis of `ChargeBasisTransmon` has `e^{iφ}` raise `n`, which
  flips its `sin φ` phase operator. `Fluxonium` and the scqubits imports
  already used this sign. From energy eigenstates, populations under a charge
  drive alone or charge couplings alone do not change. Drive matrix elements
  on the Fock devices change sign, for example `m = i` instead of `−i` on the
  `0 -> 1` transition of a Duffing transmon, so a real π pulse gives `−iY`
  instead of `iY`. The cross-resonance coefficients `IX_per_amplitude` and
  `ZX_per_amplitude` between these devices change sign together. To keep a
  calibrated gate, add π to the phase of its `ChargeDrive` pulses on these
  devices. A `PhaseDrive` on a `ChargeBasisTransmon` changes sign in the same
  way, and at `n_g ≠ 0` its populations can also change, because the sign flip
  is then not a symmetry of the device. The relative sign between a
  `CoherentInput` port drive and a `ChargeDrive` flips. `PHYSICS.md` gained a
  cross-walk from quchip's conventions to these references.
  ([#222](https://github.com/quchip/quchip/pull/222))
- `ChargePhaseDrive` now drives a transition at a positive carrier, as
  `ChargeDrive` does, and takes the automatic carrier `chip.freq(target)`. Its
  Hamiltonian stays `I·Q_c − Q·P`, but with the textbook charge sign above it
  is `i(s a† − s* a)` on a Fock device: the resonant sideband at twice the
  rate of `ChargeDrive` with no counter-rotating term, so its π amplitude is
  half that of `ChargeDrive`. It now also accepts devices whose operators are
  matrices, such as `ChargeBasisTransmon` and `Fluxonium`, where the resonant
  sideband is the stronger of two. Previously, a 20 ns Gaussian at
  ChargeDrive's π amplitude and +5 GHz left a 5 GHz transmon in `|0>`, and
  only a negative carrier drove it. To keep the Hamiltonian of a 0.4 schedule
  on a Fock device, negate `freq` and replace the pulse phase φ by π − φ.
  ([#221](https://github.com/quchip/quchip/pull/221),
  [#222](https://github.com/quchip/quchip/pull/222))

#### New features

- `VNA(chip).resonances()` returns the poles of the weak-probe response on
  the routes of `VNA.sweep()`: for each pole, its lab-frame frequency, its
  response linewidth `κ = −2 Re λ`, its composition over the modes, the
  dominant label and the residue of each S entry. `scattering(frequencies)`
  rebuilds the swept response from the poles, and `dark` marks poles whose
  residues are negligible at the selected ports. Ill-conditioned poles that are too
  close to resolve, as near an exceptional point, raise.
  Frequencies and linewidths have JAX derivatives.
  ([#192](https://github.com/quchip/quchip/issues/192),
  [#208](https://github.com/quchip/quchip/pull/208))
- The exact route of `eliminate()` reports the margins of its labeling checks
  under `validity["<mode>_labeling"]`: the largest dressed weight of each
  diagnostic label, the runner-up and the margin to `0.5 + 1e-6`, and the
  smallest eigenvalue and orthonormalization residual of the kept overlap
  matrix. A failed check of either route raises `EliminationError`, a
  `ValueError` that carries the route, the check, and the label with its
  weight, the residual, or the two degenerate levels with their coupling. Its
  message is the previous message followed by these values, also inside
  `jax.jit`.
  ([#198](https://github.com/quchip/quchip/issues/198),
  [#208](https://github.com/quchip/quchip/pull/208))
- `EliminationResult.pathways(final, initial)` attributes any element of an
  SW-retained Hamiltonian to virtual states, including dispersive and Lamb
  shifts. It takes level mappings such as `{"qa": 1}`.
  ([#199](https://github.com/quchip/quchip/issues/199),
  [#208](https://github.com/quchip/quchip/pull/208))
- `sequence.subspace_propagator(devices, levels=(0, 1))` returns the
  propagator of a sequence on the dressed states of the selected levels, as a
  `PropagatorResult`. One batch starts each label in its dressed state, and
  the record converts the label amplitudes to the dressed frame by default, or
  to the lab or integration frame. The dressed frame turns each device at its
  dressed frequency under the solve's approximation and cutoff, so an idle
  pair of qubits gives `diag(1, 1, 1, exp(-2πiζT))` with the static ZZ `ζ`.
  The record gives the leakage of each input, and `fidelity(target)` returns
  the average gate fidelity, which counts that leakage. The solves leave out
  collapse channels, which `omitted_channels` names, and `dissipation=True`
  raises and points to `subspace_channel()`.
  ([#193](https://github.com/quchip/quchip/issues/193),
  [#207](https://github.com/quchip/quchip/pull/207),
  [#209](https://github.com/quchip/quchip/issues/209),
  [#210](https://github.com/quchip/quchip/pull/210))
- `sequence.subspace_channel(devices, levels=(0, 1))` returns the gate with
  the chip's collapse channels as a `ChannelResult`: a superoperator on the
  same dressed labels and frames, from `d²` pure-state master-equation
  solves. It gives the infidelity in quchip's frame, with ideal preparation
  and measurement, and the eigenvalue infidelity, a spectral score that a
  frame tilt does not change, with the average leakage beside it. Labels may
  include leakage levels, and `seepage()` reads them. `error(target)` returns
  the error map after the ideal gate, with its Pauli transfer matrix, Pauli
  fidelities, Pauli rates and twirl. PHYSICS.md §9.5 gives the definitions.
  ([#223](https://github.com/quchip/quchip/pull/223))
- `Envelope.area()` returns the complex area of an envelope: closed forms for
  the built-in shapes, and Gauss-Legendre quadrature between the sampling
  times for other shapes, also under JAX tracing.
  `chip.rotation_amplitude(transition, line, envelope, angle=π)` returns a
  first-guess amplitude from `θ = 2π |m| |A a|`, with the dressed matrix
  element `m` of the line. PHYSICS.md §6.2.1 states the relation, its rotation
  axis and its limits.
  ([#197](https://github.com/quchip/quchip/issues/197),
  [#207](https://github.com/quchip/quchip/pull/207))
- `TrajectoryResult.records()`, `clicks()` and `run_expect(key)` read the
  measurement currents, click events and per-run expectations of QuTiP and
  Dynamiqs in one layout. Records have shape `(runs, channels, intervals)` and
  name their native sampling. Click records keep the sampling weight of each
  run, the no-click weight and a full-buffer flag. `result.native` does not
  change.
  ([#194](https://github.com/quchip/quchip/issues/194),
  [#207](https://github.com/quchip/quchip/pull/207))
- `eliminate_spectators(chip, keep)` reduces a chip to the kept devices, one
  `eliminate()` step at a time, and returns every step. Its default order keeps
  the operators of kept devices small under `local=True`. A spectator waits
  while a spectator neighbour links it to a kept device near it in frequency,
  so the chain keeps their fourth-order exchange shift. It also waits while its
  step would couple two near-degenerate spectators beyond the fold limit
  `g/Δ = 0.1`. On a ring of eight transmons with four kept, the drives stay
  within 256 states. The largest f01 error is 22.8 kHz, against 98.1 kHz when
  each arc goes from one end. `order=` sets the steps, and the default order
  raises under JAX tracing of device parameters. Each `eliminate(...,
  local=True)` step reports its widest operator support in `notes` and warns
  when one dense copy needs 256 MiB or more. `active_patch()` accepts `order=`.
  ([#151](https://github.com/quchip/quchip/issues/151),
  [#183](https://github.com/quchip/quchip/pull/183))
- `eliminate_spectators()` and `active_patch()` estimate the shift that each
  Schrieffer-Wolff step omits from a kept device when a device that stays on
  the chip links a removed spectator to it. An example is the coupler that
  `active_patch(hops=1)` keeps next to a driven transmon. `omitted_shifts`
  holds the estimate in GHz, and a note names each term. For a transmon with a
  kept coupler and a spectator 40 MHz away, the estimate is -39.9 kHz and the
  f01 error is +39.7 kHz. Outside the dispersive regime, and next to a direct
  exchange, the estimate is `None`. `method="exact"` with `local=False` keeps
  the shift.
  ([#184](https://github.com/quchip/quchip/issues/184),
  [#183](https://github.com/quchip/quchip/pull/183))
- `quchip.gateset` calibrates native gates and composes circuits from them.
  `GateSet.standard()` declares DRAG SX and X gates and echoed
  cross-resonance ZX90 gates; `NativeGate` and `Knob` declare your own. Each
  fit targets the ideal gate on a block that includes coupled spectator
  qubits, with free virtual Z phases, and reports the infidelity, leakage,
  error generator, knob bounds, and fit identifiability. `GateSet.unitary()`
  reuses each gate's propagator at new start times and frames when its
  Hamiltonian allows it, and `GateSet.simulate()` solves the circuit's
  schedule with noise. On a chip above 64 states, gate-set solves pick and
  raise an excitation cutoff with a warning.
- `QuantumSequence.propagator()` returns the lab-frame propagator of a
  schedule, and `Propagator.computational()` gives its block on dressed
  computational states. `QuantumSequence.extend()` appends another sequence.
- `ExcitationCutoff(max_total=K)` keeps the product energy labels whose levels
  sum to at most `K`. Set it with `Chip(..., space=...)` or `chip.with_space(...)`.
  It is separate from `RWA()` and `Exact()`, and quchip never sets it. The
  truncated model keeps `P H(t) P`, with every element between retained
  sectors, and the projected jumps `P L P`. Labeled dressed queries,
  `Chip.state()`, `Chip.compact_state()`, `fit_a_dress(evaluator="full")` and
  deterministic pulse solves on QuTiP and dynamiqs use it, also under JAX
  tracing. Under `Exact()`, the frequencies of a ring of 24 three-level devices
  then need blocks of 301 and 2600 states at `K = 3`. Labels and supplied states
  above the cutoff raise. `quchip.check_truncation(result)` reports the sampled
  population in the sectors that connect above the cutoff, as a heuristic.
  `Chip.dress()`, VNA sweeps, output spectra and correlations, steady-state,
  stochastic, elimination, local-fit and partition workflows raise for an
  active cutoff. Compare results at increasing `K`, because convergence need
  not be monotone.
  ([#180](https://github.com/quchip/quchip/pull/180))
- `VNA.finite_power()` and `VNA.measure()` solve a chip that conserves the
  total excitation on the product labels with `N <= K`, without the product
  space. This route applies when every port lowers `N`, other channels lower
  or conserve it, the inputs are vacuum, and the chip has no pump or thermal
  occupation. A finite probe couples every `N` to `N ± 1`, so the route
  truncates. With `chip.with_space(ExcitationCutoff(max_total=K))`, `K` is the
  chip's. Without a cutoff, a chip with more than 64 product states starts at
  `K = 2`, compares every grid point with `K + 1`, raises `K` until the change
  is at most `1e-6` of each value, keeps `K + 1`, and warns which cutoff it
  kept. A truncation
  that leaves more than one stationary state raises. Two
  transmon-readout pairs take 0.06 s instead of 18 s for two grid points, and
  12 pairs keep 325 of 3²⁴ product states at `K = 2`. Under a cutoff, a chip
  outside this route and a JAX-traced call solve the compact model directly.
  ([#156](https://github.com/quchip/quchip/issues/156),
  [#185](https://github.com/quchip/quchip/pull/185))
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
  mode in a model that does not conserve excitation number. The result's
  `validity` reports `self_energy_over_width` for each kept pole. It compares
  the return path through the omitted pole with the damping of each kept mode
  on the other ports, and `describe()` shows it.
  ([#131](https://github.com/quchip/quchip/issues/131),
  [#140](https://github.com/quchip/quchip/pull/140),
  [#174](https://github.com/quchip/quchip/issues/174),
  [#177](https://github.com/quchip/quchip/pull/177))
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
- `FitADressResult.history_params` and `FitADressResult.history_residuals`
  give the parameter vector and the target residuals in GHz at each `history`
  entry. Previously, the path of a fit needed one rerun with `max_nfev` per
  point. With a numerical Jacobian, they include the finite-difference
  probes, as `history` does.
  ([#152](https://github.com/quchip/quchip/issues/152),
  [#168](https://github.com/quchip/quchip/pull/168))

- Every stationary result records its numerical route under
  `diagnostics["route"]`. The `SolveRoute` gives the route, the reason for it,
  the dimension that it solves on, its array library and an estimate of its
  peak memory. `SteadyStateResult`, `OutputSpectrumResult` and
  `OutputCorrelationResult` gain `diagnostics`. One planner chooses the route
  of each stationary workflow, and `PHYSICS.md` §6.6 lists the routes. On
  dynamiqs, the sector route solves with SciPy on the host, and its reason
  says so. ([#203](https://github.com/quchip/quchip/issues/203),
  [#205](https://github.com/quchip/quchip/pull/205))
- `steadystate()`, `VNA.output_spectrum()`, `VNA.g1()` and `VNA.g2()` accept a
  chip with an `ExcitationCutoff`. A chip that conserves `N` solves on the
  excitation sectors, and other chips solve the projected model with the
  backend's stationary Liouvillian. Three transmon-readout pairs have 729
  product states, and at `N <= 2` each of these calls takes less than 0.2 s.
  ([#191](https://github.com/quchip/quchip/issues/191),
  [#205](https://github.com/quchip/quchip/pull/205))
- Every result records its integration frame as `result.frame`. `state()`,
  `state_at()`, `dm_at()` and `reduced_state()` take `frame="lab"`.
  `overlap()`, `amplitude_array()`, `final_overlap_magnitudes()` and
  `final_amplitudes()` take `frame="lab"`, which reads a ket target as a
  lab-frame vector. Their label targets, such as `{"q0": 1}`, are dressed
  eigenstates that turn with the label. An idle dressed state then reads 1
  in every frame. A ket target keeps its meaning, a vector of the
  integration frame. ([#190](https://github.com/quchip/quchip/issues/190),
  [#202](https://github.com/quchip/quchip/issues/202),
  [#205](https://github.com/quchip/quchip/pull/205))
- `chip.excitation_structure()` reports whether the static model conserves the
  total excitation `N`, the change of `N` of each coupling, retained term,
  port, drive and collapse channel, the largest entry of each term that breaks
  conservation, and the largest block that dressed queries diagonalize.
  `describe()` shows the verdict and the block size. When only entries below
  `1e-12` of the largest local level span break conservation, dressed queries
  warn and name the terms. ([#195](https://github.com/quchip/quchip/issues/195),
  [#201](https://github.com/quchip/quchip/issues/201),
  [#205](https://github.com/quchip/quchip/pull/205))
- `simulate_batch()`, `Chip.solve_many()`, `solve_many()` and `solve_batch()`
  take `max_workers`. `max_workers=1` solves in the calling process, and a
  count that does not fit in memory raises `MemoryError`.
  ([#196](https://github.com/quchip/quchip/issues/196),
  [#205](https://github.com/quchip/quchip/pull/205))
- quchip warnings have categories. `QuchipWarning`, a `UserWarning`, holds
  `AutoChoiceWarning` for notices about automatic choices (frame, carrier,
  cutoff, solver fallbacks) and `AccuracyWarning` for advice that a result can
  be inaccurate (truncation, approximation, dressed labels, fits). One filter
  such as `warnings.filterwarnings("ignore", category=AutoChoiceWarning)`
  silences a group. Renamed APIs keep `DeprecationWarning`.
  ([#221](https://github.com/quchip/quchip/pull/221))
- `chip.rotation_amplitude()` works on lines with signal-chain transforms and
  crosstalk. It passes a tone at the dressed transition frequency through the
  chain and sums each reached line's matrix element times its transfer, so a
  gain `g` divides the seed by `|g|` and delayed or crosstalk paths to one
  device add as phasors. Previously, any transform on the line raised. A
  transform must set the new class attribute
  `SignalTransform.linear_time_invariant` to take part; `Gain`, `Delay`,
  `Crosstalk`, `CrosstalkMatrix` and `CableLoss` set it, and a transform
  without it still raises. `ControlEquipment.tone_transfer(line, frequency)`
  returns the transfer to each line.
  ([#218](https://github.com/quchip/quchip/issues/218),
  [#221](https://github.com/quchip/quchip/pull/221))

#### Fixes

- Under `RWA()`, a pulse with a zero carrier (`freq=0.0`, or
  `freq + detuning = 0`) keeps its complete real drive, as under `Exact()`.
  Both partners of a zero-frequency band are resonant, but `RWA()` kept one,
  so the drive had half its amplitude at phase 0 and did not go to zero at
  phase `π/2`. On a 50 MHz mode, a 20 ns Gaussian pulse of amplitude 0.02
  gave `P1 = 0.085` with `freq=0.0` and `0.276` with `freq=None`; both now
  give `0.276`. A zero carrier still follows `phase=` and `vz()`: on a
  charge line its carrier phase `p` scales a real envelope by `cos p`. A
  carrier near zero
  but not zero keeps one partner under `RWA()`; use `Exact()` there.
  ([#224](https://github.com/quchip/quchip/pull/224))
- Every calculation evaluates a generic `Coupling` with the chip's backend.
  Previously, `resolve()`, `eliminate()`, `chip.state()` and `chip.freq()` used
  the process default backend, so on a dynamiqs chip while QuTiP was the
  default, the first three raised `TypeError`. The callable form now receives
  the chip's backend.
  ([#158](https://github.com/quchip/quchip/issues/158),
  [#166](https://github.com/quchip/quchip/pull/166))
- The sequence of `active_patch()` keeps the timing of the source schedule.
  Previously, a delay on the line of a removed device no longer reached a
  later barrier, so later pulses of the patch sequence started earlier, and a
  `vz()` on a removed device made `simulate()` raise `KeyError`.
  ([#162](https://github.com/quchip/quchip/issues/162),
  [#167](https://github.com/quchip/quchip/pull/167))
- The conservation check of the total energy-level index reads each
  cascaded port in the chip's retained coordinates, as assembly does.
  Previously, it read the authored port. A retained map that mixes the
  port's levels could then put the chip on the excitation-sector route,
  which drops the parts of the generated Hamiltonian that change the total
  level.
  ([#171](https://github.com/quchip/quchip/pull/171))
- After `eliminate()` keeps a removed readout's response on its line,
  `simulate()` raises `NotImplementedError` for an `OutputField` whose channel
  carries emission from the response's ports, and for diffusive monitoring of
  such a channel with a nonzero efficiency, including pure-state SSE.
  Monitoring a hidden channel that acted on the removed readout, such as its
  internal loss or a bath on it, also raises.
  `EffectiveTerms.omitted_responses` and the baths record these channels. The
  record survives serialization and later reductions, and stays when the port
  network drops the response. Previously, these outputs omitted the response
  with no error, e.g. a steady output amplitude 160 times too large at the
  response frequency. Keep the boundary mode for these outputs. Other outputs,
  SME records of other channels, and population observables still run.
  ([#159](https://github.com/quchip/quchip/issues/159),
  [#172](https://github.com/quchip/quchip/pull/172))
- After `eliminate()` removes a readout with a one-port reflection or
  transmission section, diffusive monitoring with a nonzero efficiency raises
  `NotImplementedError` for the line output through the section and for a
  hidden channel that acted on the readout, such as its internal loss or a
  bath on it. Native monitoring applies no reference section, so these
  records missed the readout's response, e.g. a mean of the internal-loss
  field almost 800 times too small under a weak resonant input. The
  transformed port, the carried channels and the baths record the readout.
  `OutputField` traces, other records and unconditional solves still run.
  ([#175](https://github.com/quchip/quchip/issues/175),
  [#178](https://github.com/quchip/quchip/pull/178))
- Cascaded ports keep the Hamiltonian that they generate in two more cases.
  Previously, a compact pulse solve with `dissipation=False` dropped it, so a
  photon stayed in the first of two cascaded resonators instead of moving to
  the second. A small port rate, e.g. 1e-9/ns, also dropped it, because an
  absolute tolerance treated the term as zero. Now an element counts as zero
  only below a round-off threshold set by the port products that form it.
  ([#179](https://github.com/quchip/quchip/pull/179))
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

- QuTiP stationary solves estimate the memory of their LU factors before they
  factorize, and raise `MemoryError` when it does not fit. Previously, they
  checked only the generator. On three transmon-readout pairs, whose factors
  need about 1.6 TB, SuperLU ran for more than 15 minutes, and the call now
  raises after 2.3 s. The dynamiqs steady state counts the six `D² × D²` arrays
  of its solve instead of the generator alone. An 81-state chip peaked at
  4.65 GB against a check of 0.69 GB.
  ([#189](https://github.com/quchip/quchip/issues/189),
  [#205](https://github.com/quchip/quchip/pull/205))
- Memory checks run on macOS and Windows. Previously, without `/proc/meminfo`
  or an installed psutil, every check accepted every request.
  ([#200](https://github.com/quchip/quchip/issues/200),
  [#205](https://github.com/quchip/quchip/pull/205))
- A solve with `dissipation=False` no longer checks the collapse channels
  that it leaves out against its frame. Previously, such a channel raised
  when it did not fit the frame: a collective decay of two detuned devices
  refused `frame="rotating"` even without dissipation. On a chip without a
  frame, these solves and `subspace_propagator()` now use the rotating frame
  instead of a fallback. Ports still need to fit the frame, because drives
  through a port and the Hamiltonian of a cascade stay in the solve.
  ([#214](https://github.com/quchip/quchip/pull/214))

- Dressed label queries, such as `dispersive_shift()`, `freq(..., when=...)`,
  `static_zz()` and `kerr_matrix()`, warn when the couplings mix a label with
  states beyond the levels of a truncated device. The check counts coupling
  steps for dispersive, nondegenerate couplings. It warns at second and fourth
  order, and up to the order at which the excitations of the label first
  interact, such as sixth order for ZZ through two couplers in series. Under
  `Exact()`, steps that create excitations also count. Diagonal couplings,
  such as `CrossKerr`, and couplings that the approximation removes do not.
  Silence is not a convergence check. The warning names the device and the
  `levels` that remove it. Values do not change. Previously, a two-level
  `Resonator` gave a dispersive shift of +9.98 MHz with no warning, against
  -0.99 MHz with three levels. A `Qubit` is not a truncation and never warns.
  ([#211](https://github.com/quchip/quchip/pull/211))
- After an earlier `eliminate(..., method="exact")` step,
  `effective_params[...]["purcell_rate"]` and `["kappa"]` count the removed
  mode's own loss channels. Previously, both read 0.0, because the channel acted
  on several devices in the retained coordinates. On every route, they now
  count only the channels that the mode declares, as `PHYSICS.md` states.
  Previously, `local=False` also counted a retained channel that acted on the
  mode alone. The retained channels did not change.
  ([#211](https://github.com/quchip/quchip/pull/211))
- Chips with a port network, sequences that hold them, `SimulationResult`,
  `SParameterResult` and `ResonanceResult` pickle and load. Previously,
  `pickle.dumps` raised on each of them. A loaded chip rebuilds its caches on
  demand.
  ([#211](https://github.com/quchip/quchip/pull/211))
- Error messages name the cause and the fix in five cases. `fit_a_dress()` says
  when it read a coupling's declared strength as a dressed `cross_kerr` target.
  A solve that meets an unwired line names `chip.wire(...)`. `matrix()` without
  `t` names the time-dependent terms and their rotation frequencies. A carried
  collapse channel or a feedline exposure that a frame cannot compose names
  its devices and the frames that work, such as `frame="auto"`.
  ([#211](https://github.com/quchip/quchip/pull/211))
- `Chip.describe()` prints `T1`, `T2` and `thermal_occupation` once for each
  device. Previously, a device that declares them printed them twice.
  ([#211](https://github.com/quchip/quchip/pull/211))
- `vz(device, θ)` turns a later `TwoPhotonDrive` pump on `device` by 2θ, and
  in general a pulse on a line whose drive sets `carrier_harmonic = k` by kθ
  when its frame is the line's target. Previously, every carrier pulse took
  θ. A virtual Z of π/2 between two squeezing pumps then acted like a second
  pump with phase π/2, not π, and left `P(vacuum)` at 0.50 instead of 1.00.
  One-photon drives, coherent inputs, and pulses in another device's frame
  keep θ.
  ([#216](https://github.com/quchip/quchip/pull/216))
- A dynamiqs batch over a pulse carrier that includes `freq=0.0` runs. In the
  lab frame under `RWA()`, a zero carrier and the frame cancel, so that point
  had another signal structure, and
  `simulate_batch(pulse.vary("freq", [0.0, 0.05]))` raised "Heterogeneous
  signal pytree". QuTiP already ran this batch. `freq=0.0` stays a carrier at
  zero frequency.
  ([#216](https://github.com/quchip/quchip/pull/216))
- An explicit `frame="rotating"` warns when a device's dressed `0 -> 1`
  reference has no reliable label assignment, and names `frame="auto"`. The
  frame does not change. Two identical transmons with `g = 10 MHz` took
  references of 5.01 and 4.99 GHz without a warning, and their idle exchange
  then had an error of 9e-7 against 2e-13 in the lab frame. A chip without a
  frame already raises on the same check.
  ([#216](https://github.com/quchip/quchip/pull/216))
- Steady-state results and `VNA.finite_power()` results pickle and load.
  Previously, `pickle.dumps` raised on their read-only mappings and on the
  deferred memory estimate of the stationary route. Every read-only mapping
  in quchip records now pickles as a copy, and a deferred value pickles its
  calculated value.
  ([#216](https://github.com/quchip/quchip/pull/216))
- `SimulationResult.population()` works on a result saved with
  `states="final"` and holds only the final value, so
  `population(q, 1)[-1]` reads it with either storage choice. Previously, it
  raised "Full state history is unavailable" although the final state was
  saved. The new `final_population()` on `SimulationResult` and
  `SimulationBatchResult` returns the final-time population and needs only
  the final state.
  ([#216](https://github.com/quchip/quchip/pull/216))
- On the QuTiP backend, a pulse envelope reaches the accuracy of the solver's
  `rtol`. QuTiP sampled each windowed envelope at 40 points/ns whatever the
  tolerance, so a Gaussian π pulse kept an error in `P1` of 4.1e-6 at 12 ns
  and 8.6e-5 at 4 ns, also at `rtol = 1e-10`. The sample step now follows
  `h = sqrt(8 rtol max|f| / max|f''|)`, never below 40 points/ns. At QuTiP's
  default `rtol = 1e-6`, the errors fall to 1.0e-6 and 3.1e-7 at the same
  cost, and at `rtol = 1e-8` to 9.4e-9 and 1.6e-8. A window that would need
  more than 2,000,000 samples uses the exact envelope as a callable and warns.
  QuTiP outputs can move at the 1e-6 level.
  ([#216](https://github.com/quchip/quchip/pull/216))
- `IQReadout.from_wiring()` no longer warns about the chip's frame. It reads
  only the output wiring, which does not depend on the frame, so it resolves
  the chip in the lab frame. ([#221](https://github.com/quchip/quchip/pull/221))
- Under `RWA()`, a pulse with a negative carrier keeps the signal band that
  co-rotates with its transition. Previously, `RWA()` picked the band by
  position and kept the counter-rotating one, so a `ChargeDrive` π pulse at
  `freq=-5.0` left a 5 GHz transmon in `|0>` (P1 = 0.99 under `Exact()`).
  A carrier at `-f` now drives as at `+f` with the conjugate phase, also when
  the carrier is traced, and the dropped-term record gives the partner at
  `|f| + f_transition`. A zero carrier keeps its conventional half.
  ([#221](https://github.com/quchip/quchip/pull/221))

#### Performance

- `VNA.measure()` with noise and an automatic cutoff settles the means first
  and starts the noise comparison at the first agreeing pair, which keeps the
  same cutoff and skips the noise solves below it: 49 s instead of 55 s for a
  four-level transmon and a 20-level resonator at five frequencies.
- On QuTiP, exact excitation spaces and `ExcitationCutoff` solves store each
  compact operator as CSR instead of a dense matrix of the compact dimension.
  A dense backend, such as dynamiqs, checks each dense compact operator
  against the available memory before it allocates it, instead of budgeting
  eight such matrices for any number of terms. A driven ring of 24
  three-level transmons at `ExcitationCutoff(max_total=3)` has 2,901 states.
  Its `build_problem()` and `simulate()` now peak at 0.25 GB instead of
  7.35 GB, and `check_truncation()` takes 0.6 s instead of 82 s.
  ([#186](https://github.com/quchip/quchip/issues/186),
  [#187](https://github.com/quchip/quchip/pull/187))
- On a chip whose static model conserves the total energy-level index,
  `chip.state()` diagonalizes only the sector of its label, and the default
  initial state of a solve is the bare product with no static model. The
  full eigensystem's memory check now runs before its decomposition, not
  before the static model build. For example, on a reduced 15,552-state
  chain, `chip.state({})` and the default initial state no longer stop with
  `MemoryError` on an 8 GB machine.
  ([#153](https://github.com/quchip/quchip/issues/153),
  [#171](https://github.com/quchip/quchip/pull/171))
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

- On two transmon-readout pairs with 81 product states, `VNA.output_spectrum()`
  takes 0.07 s instead of 86.7 s on QuTiP, and 0.11 s instead of 144.7 s on
  dynamiqs. On dynamiqs, `chip.steadystate()` takes 0.6 s instead of 124 s.
  ([#191](https://github.com/quchip/quchip/issues/191),
  [#203](https://github.com/quchip/quchip/issues/203),
  [#205](https://github.com/quchip/quchip/pull/205))
- `VNA.measure()` with an automatic cutoff starts each cutoff's noise solves
  from the solutions at the previous cutoff, as its stationary solves already
  did. The solves keep their equations and residual limit, so only the
  solver path changes. On a 4-level transmon with a 20-level resonator, 5
  probe frequencies and 5 noise offsets, the call takes 7.4 s instead of
  11.4 s, with the same kept cutoff and noise spectra within 2e-13. The solutions are
  kept only while they fit in memory.
  ([#221](https://github.com/quchip/quchip/pull/221))

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

- psutil is a required dependency.
  ([#200](https://github.com/quchip/quchip/issues/200),
  [#205](https://github.com/quchip/quchip/pull/205))
- The backend protocol gains `sector_generator()`, `sector_solve_bytes()`,
  `sector_unique()`, `stationary_bytes()`, `compact_operator()`,
  `compact_operators_fit()` and `trajectory_view()`. Their defaults run the
  host SciPy sector solver and dense storage, so a custom backend needs no
  change. `Backend.solve_batch()` takes `max_workers`, and a backend that
  overrides it must accept the keyword.
  ([#200](https://github.com/quchip/quchip/issues/200),
  [#203](https://github.com/quchip/quchip/issues/203),
  [#204](https://github.com/quchip/quchip/issues/204),
  [#205](https://github.com/quchip/quchip/pull/205))

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

- `PHYSICS.md` §4.2 describes result frames and dressed label targets, §6.5
  the sector routes of `steadystate()` and the output statistics, §6.6 the
  route record, and §9 the dressed meaning of level labels. ([#205](https://github.com/quchip/quchip/pull/205))
- `PHYSICS.md` opens with a short orientation and a table of its main
  conventions, and uses one symbol for each quantity. It corrects statements
  in §1, §6.4, §8, §11 and §13 that no longer matched the code, among them the
  default frame. On docs.quchip.org, its source links now point to GitHub, and
  table rows render whole. ([#212](https://github.com/quchip/quchip/pull/212))

- The `effective_params` docstring of `EliminationResult` states what
  `lamb_shift` and `freq_after` mean under each method. `method="sw"` reports
  the shift through the removed mode. `method="exact"` reports the labeled
  dressed transition, which includes couplings that stay on the chip.
  ([#211](https://github.com/quchip/quchip/pull/211))
- `PHYSICS.md` lists `VNA.resonances()` among the workflows that raise under a
  cutoff and corrects two stale source references. The README no longer
  describes the removed quchip.org walkthrough.
  ([#211](https://github.com/quchip/quchip/pull/211))

#### Development

- `CONTRIBUTING.md` documents the release-note format, and the pull request
  template uses checkboxes for AI assistance.
  ([#75](https://github.com/quchip/quchip/pull/75),
  [#89](https://github.com/quchip/quchip/pull/89))
- `tools/theme_figures.py` renders a PDF or dark PNG only when its light SVG
  changed. `--force` renders every figure.
  ([#88](https://github.com/quchip/quchip/pull/88))
- The engine owns the resolution kernels, state preparation, excitation
  structure, frame map and route planner. Engine kernels move out of `chip/`,
  state preparation moves out of `results/`, and `results/` no longer imports
  `analysis/` or dynamiqs. `quchip/utils/memory.py` holds one memory cost
  model. ([#200](https://github.com/quchip/quchip/issues/200),
  [#201](https://github.com/quchip/quchip/issues/201),
  [#204](https://github.com/quchip/quchip/issues/204),
  [#205](https://github.com/quchip/quchip/pull/205))

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
