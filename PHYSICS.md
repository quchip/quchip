# Physics reference

This document states the physics contracts implemented by quchip. It distinguishes authored and resolved Hamiltonians, records where local bases, frames, and RWA are applied, and states the engine's assumptions.

## 1. Units and the 2π Convention

User-facing Hamiltonians express `E/h` in ordinary GHz. Solver assembly converts
them to angular frequency. The units are:

| Quantity | Unit |
| --- | --- |
| Frequency | GHz, ordinary frequency |
| Time | ns |
| Temperature | mK |
| Energy / h | GHz |

The domain layer stays in ordinary GHz. The only Hamiltonian-assembly `2π` conversion is in [`quchip/engine/assembly.py`](quchip/engine/assembly.py), right before the solver-facing Hamiltonian is built.

## 2. What `.hamiltonian()` Means

### 2.1 Device Hamiltonians

`BaseDevice.unresolved_hamiltonian()` returns that device's authored static Hamiltonian in its declared local space, in the lab frame, in ordinary GHz.

Examples:

- `DuffingTransmon.unresolved_hamiltonian()` returns `omega * n + (alpha / 2) * n * (n - I)`.
- `Resonator.unresolved_hamiltonian()` returns `omega * n`.

The authored view does not:

- include `2π`
- include any rotating-frame subtraction
- include any drive term
- include any explicit time dependence

`BaseDevice.hamiltonian()` resolves isolated device physics through the engine, using the device's basis policy and the lab frame. Pass `frame=` to `device.resolve()` for another local frame. Attaching the device to a chip does not change these queries. Use `chip.hamiltonian()` or `chip.freq(device)` for the coupled system.

Local level labels refer to the isolated Hamiltonian's energy order. `chip.bare_state(q=1)` prepares that local excited state, and `result.population(q, 1)` measures its occupation, summing over other devices. A result retains the energy vectors used to build its calculation. The frame generator is the same energy-level index operator in either solver basis.

Numerical device Pauli operators act on the lowest two isolated energy states and vanish on higher levels: `Z = |0><0| - |1><1|`, `X = |0><1| + |1><0|`, and `Y = -i|0><1| + i|1><0|`. Leakage is not renormalized away. Each energy vector's largest-magnitude authored-basis component is made real and positive; the first component wins a tie. State preparation, transitions and default Pauli observables share that convention. This fixes phase locally, not the basis within a degenerate eigenspace; derivatives require separated eigenvalues and a stable phase pivot. No globally continuous eigenvector phase is promised.

Authored `LocalOps` operators, including `op.sigma_z` inside a Hamiltonian declaration, retain their local-space definitions. They do not diagonalize the Hamiltonian they define. Named observable lookup respects the component's operators and transforms them into the selected solver basis.

### 2.2 Coupling `.interaction_hamiltonian()`

`BaseCoupling.interaction_hamiltonian()` returns the coupling's full two-body operator in the pair subspace, still in the lab frame, still in ordinary GHz. It takes no RWA argument — a coupling defines exactly one interaction; RWA is resolved and applied structurally by the chip and engine, not chosen here (§6.1).

For `Capacitive`:

```text
full: g * (a + a†)(b + b†)
```

`interaction_hamiltonian()` always returns this full form. The RWA form `g * (a†b + ab†)` is never authored directly — it is what remains once the bands that change total excitation are masked out by the chip or filtered by the engine.

### 2.3 Chip and sequence Hamiltonians

`Chip.unresolved_hamiltonian()` embeds every authored device Hamiltonian and full coupling interaction into the total declared Hilbert space. It is the exact static lab-frame expression before local-basis resolution, retained-level truncation, frame transformation, or RWA.

`Chip.hamiltonian()` is reconstructed from `Chip.resolve()`, the frozen `EngineResult` used by backends. Resolution:

1. resolves each authored local space into the selected solver basis and retained dimension
2. multiplies solver-facing operators by `2π`
3. subtracts the chosen frame generator
4. decomposes non-static pieces into excitation-change bands and applies the chip's approximation strategy
5. attaches explicit time-dependent phases where needed

The returned expression is an inspectable, ordinary-GHz view of those same canonical terms; the solver-facing `EngineResult` retains the internal `2π` scaling. `QuantumSequence.hamiltonian()` follows the same path and adds the sequence's scheduled drive and crosstalk terms.

These inspection methods return `PhysicsExpr`, the backend-neutral scalar and operator algebra used for authored and resolved physics. It retains declared parameters, matrices, time-dependent scalars, labels, and opaque JAX callables without forcing numerical values. Numerical materialization is explicit through `.matrix()` or backend lowering.

The resulting contracts are:

- device `.unresolved_hamiltonian()` means authored local static lab-frame physics
- device `.hamiltonian()` means the resolved one-device engine view
- coupling `.interaction_hamiltonian()` means local static lab-frame interaction physics
- `Chip.unresolved_hamiltonian()` means the embedded authored static lab-frame chip Hamiltonian
- `Chip.hamiltonian()` means the resolved canonical chip Hamiltonian without scheduled drives
- `QuantumSequence.hamiltonian()` means the resolved canonical Hamiltonian with scheduled drives

## 3. Device Models

### 3.1 Duffing transmon

Source: [`quchip/devices/transmon/duffing.py`](quchip/devices/transmon/duffing.py)

```text
H = omega * n + (alpha / 2) * n * (n - I)
```

`omega` is the `0 -> 1` transition frequency and `alpha` is the anharmonicity.

### 3.2 Resonator

Source: [`quchip/devices/resonator.py`](quchip/devices/resonator.py)

```text
H = omega * n
```

If `internal_quality_factor` is set, the resonator contributes unobserved
photon loss with authored operator `a` and rate `2π * omega / Q_internal` in
`1/ns`. Backend lowering forms the Lindblad operator
`sqrt(2π * omega / Q_internal) * a`.

### 3.3 Collapse operators

Source: [`quchip/devices/base.py`](quchip/devices/base.py)

The standard dissipators are:

- `T1`: relaxation through `a`
- `T2`: pure dephasing through `sqrt(2*gamma_phi) * n` with `gamma_phi = 1/T2 - 1/(2*T1)`. The factor `2` makes the 0–1 coherence decay at `1/(2*T1) + gamma_phi = 1/T2`, so the input `T2` is the resulting coherence time (when `thermal_occupation == 0`). The number operator `n` gives the standard `(m-n)^2` dephasing scaling across higher levels.
- thermal up/down channels when `thermal_occupation` is set

Devices, drives, couplings, and baths author `CollapseChannel` records that
keep the local operator separate from its non-negative rate in `1/ns`. The
engine projects and embeds the operator while preserving the rate; backend
lowering applies `sqrt(rate)` exactly once.

### 3.3.1 Accessible ports

Source: [`quchip/chip/ports.py`](quchip/chip/ports.py),
[`quchip/control/field.py`](quchip/control/field.py),
[`quchip/observables.py`](quchip/observables.py),
[`quchip/engine/input_output.py`](quchip/engine/input_output.py),
[`quchip/analysis/vna.py`](quchip/analysis/vna.py)

A `Port` is an accessible Markovian channel. It owns a dimensionless operator
`A_p`, an external rate `kappa_p`, and a reference-plane phase `phi_p`:

```text
L_p = exp(i phi_p) sqrt(kappa_p) A_p
b_out,p = b_in,p + L_p
```

The dissipator uses `L_p`. Scheduling an external plane's `input` consumes the
same envelope, phase, carrier, and start-time grammar as a classical drive,
with

```text
beta_i(t) = A(t) exp(i theta) exp(-i 2π f t),
|beta_i|^2 = incident photon flux in photons/ns.
```

There is no implicit conjugation or factor of one half. For a general resolved
boundary, the field arriving at each coupling channel is

```text
c_j(t) = sum_i S_ji beta_i(t),
H_input(t) = i sum_j (c_j^* L_j - c_j L_j^dagger).
```

This Hamiltonian is the canonical solver form obtained by composing the
coherent source `W_beta = (I, beta, 0)` with the resolved SLH model and then
gauging the displaced collapse operators back to the input-free `L`. The
collapse channels therefore remain unchanged: applying a coherent field never
adds a second damping channel.

`ResolvedSLH` itself stays input-free. Per-solve beta programs are retained on
`EngineResult.coherent_inputs` for later output-field reconstruction. Classical
`ControlEquipment` transforms do not accept field inputs; field attenuation,
phase, crosstalk, and reference-plane delay sections belong to `PortNetwork`.

For the one-port identity case this reduces to the angular Hamiltonian

```text
H_input = i (beta_p^* L_p - beta_p L_p^dagger).
```

Transient output observables are reconstructed from that same solve-bound
model:

```text
<b_out,j> = c_j + <L_j>,
<X_theta,j> = Re[exp(-i theta) <b_out,j>],
<b_out,j^dagger b_out,j>
  = |c_j|^2 + 2 Re[c_j^* <L_j>] + <L_j^dagger L_j>.
```

An external plane's `output` request lowers `L_j` and `L_j^dagger L_j` into
backend expectation operators. `result.output(plane)` returns the reconstructed
complex amplitude and normally ordered photon flux; any quadrature phase is a
post-solve projection of the complex amplitude. The coherent background is
evaluated from the retained beta programs rather than inserted as an identity
operator. Reference sections shift the incident beta on an exposure's inbound
leg and the complete boundary trace on its outbound leg, with zero field before
the simulation's initial boundary data can arrive. The result also retains both
moments at the Markov boundary before the outbound shift.

For a single resonator, `external_quality_factor` gives
`kappa_p = 2π * freq / Q_external`. Internal resonator loss and every port are
separate collapse channels, so `kappa_total` is their sum.

### 3.3.2 Field networks

Source: [`quchip/chip/port_network.py`](quchip/chip/port_network.py),
[`quchip/engine/ir.py`](quchip/engine/ir.py)

A `PortNetwork` composes accessible ports and instantaneous scalar scattering
before the engine lowers the resolved SLH value. An included block is a plain
copy of template components and connections under a label prefix and adds no
physics of its own. For a series connection with `G2` after `G1`, quchip uses

```text
S = S2 S1
L = L2 + S2 L1
H = H1 + H2 + Im(L2^dagger S2 L1).
```

For two equal-rate ports on one device,
`network.cascade(near, network.phase_shift("phi", phase=phi), far)` gives an
effective channel coefficient with magnitude squared `2γ(1 + cos φ)` and adds
`+γ sin φ · a†a` to `H`. Here `γ` is angular, in rad/ns, so the Lamb shift is
`+γ sin φ / 2π` GHz. Entering the propagation phase as
`φ = 2π f τ = ωτ`, with `f` in GHz and `τ` in ns, matches the
`Δ = γ sin φ` convention of Kockum et al., Phys. Rev. A 90, 013837 (2014).

`port.side` and `component.port(k)` each select one physical connector, pairing
that side's input and output terminals. `network.link(...)` cables consecutive
sides in both directions; each cable compiles to two directed terminal
connections. An ideal circulator routes `1 -> 2 -> 3 -> 1`; an isolator is a
circulator whose third side is a hidden vacuum load. Compilation proceeds per
terminal, so permutation components do not create false cycles; core terminals
are grouped into strongly connected components, and each connection cycle is
reduced algebraically with the feedback rule in 3.3.3. This produces
loop-corrected `S`, `L`, structural reachability, and the generated
series/feedback Hamiltonian, while reference sections cannot sit inside a
loop. A concrete singular `I - M` raises; with traced JAX scattering, the
singularity cannot be detected during tracing and the solve returns non-finite
values at that point.

Concrete scattering must be unitary. Loss is represented by an explicit
unitary dilation: a two-sided reciprocal attenuator with power transmission
`eta` from side 1 to side 2 has amplitude transmission `sqrt(eta)` in both
directions and couples each direction to one of two hidden vacuum channels
with amplitude `sqrt(1-eta)`. A beam splitter is instead a directional
two-input/two-output device: `eta` is the input-k to output-k power, and its
scattering matrix is `[[sqrt(eta), sqrt(1-eta)], [-sqrt(1-eta), sqrt(eta)]]`.
Beam splitters and 90-degree hybrids have no physical sides; connect their
input and output terminals directly or with `network.cascade(...)`. Network
exposures define the external channel order. Their
adjacent reference sections move the incident and reported reference planes;
the compiler peels them from each leg before forming the instantaneous `S`,
`L`, and `H`. A section traversed on a reflection line contributes once inbound
and once outbound. A reference section not adjacent to an exposure raises.
`network.delay(...)` is a reference-plane shift, not retardation. It changes
fields at exposure planes but adds no memory to the dynamical SLH core, which
is Markovian throughout; the propagation phase between giant-atom coupling
points is entered with `network.phase_shift(...)` as above. Retarded feedback, linewidth-scale variation of `γ(ω)` or
the phase across a resonance, and the non-Markovian giant-atom regime are out
of scope.

Static composition of several quantum ports requires a common rotating-frame
frequency. Different local carriers would make both the collective `L` and
`Im(L2^dagger S2 L1)` explicitly time dependent; until dynamic collapse
channels exist, quchip rejects that network and directs the model to the lab
or a common frame.

`network.filter(...)` adds a two-sided passive reference section with complex
transfer `H(f)`. Its `transfer(frequency, **parameters)` callable takes scalar
or array frequencies in ordinary GHz and remains outside the Markovian `S`,
`L`, and `H`; each keyword parameter is tracked at
`network.component.<label>.<name>`. Continuous-wave calculations, including
VNA, small-signal response, spectra, and stationary pumps, apply `H(f)` exactly
on each leg. In particular, `output_spectrum` scales the fluctuation spectrum
at offset `nu` by

```text
|H(f_c + nu)|^2.
```

Transient filtering uses a narrowband carrier approximation rather than a
time-domain convolution:

```text
beta_boundary(t) = H(f_carrier) beta_plane(t),
b_out,plane(t) = H(f_c) b_out,boundary(t),
Phi_plane(t) = |H(f_c)|^2 Phi_boundary(t).
```

Here `f_carrier` is the scheduled input carrier and `f_c` is the channel's
rotating-frame carrier. A lab-frame outbound channel has no such carrier, so
requesting its filtered output raises before the solve. A concrete evaluation
with `|H| > 1` also raises: filter sections are passive; use
`network.amplifier(...)` for gain.

`network.amplifier(...)` adds a phase-preserving output-line reference section
with power gain `G` and input-referred symmetrized added noise `n_add` in
quanta. It amplifies side 1 to side 2 by `sqrt(G)` and is transparent in
reverse. The exposure must lie beyond side 2 on the outbound leg. quchip raises
if the forward direction would amplify an incident field into the chip, or if
an outbound plane lies on side 1. `gain` and `added_noise` are sweepable,
differentiable paths at `network.component.<label>.gain` and
`network.component.<label>.added_noise`; amplifier sections serialize
normally.

The quantum floor and output-line noise recursion are

```text
n_add >= (1 - 1/G)/2,
N <- |H(f)|^2 N,
N <- G N + G n_add + (G - 1)/2.
```

The last line applies at each amplifier; its added term equals `G - 1` at the
quantum limit. Two amplifiers in series obey the input-referred Friis relation
`n_total = n1 + n2/G1`.

Continuous-wave means and small-signal entries, including VNA, acquire
`sqrt(G)`, so `|S|` may exceed one. `output_spectrum` returns the amplified
signal density as `signal_fluctuation_spectrum`, the accumulated density `N`
as `added_noise_spectrum`, and their sum as `total_fluctuation_spectrum`. Its
`signal_coherent_flux`, `signal_incoherent_flux`, and `signal_photon_flux`
fields scale the signal by `G` but exclude amplifier noise, which cannot be
converted to flux without a detection bandwidth. Transient
`result.output(plane)` likewise scales amplitude by `sqrt(G)` and photon flux
by `G`, with no added-noise term. Normalized `g1` and `g2` through an amplifier
raise; request them at a plane before the amplifier.

Noise parameters are ordinary tracked attributes: set (or clear with `None`) at construction **or any time after** — collapse operators are rebuilt from current values on every solve, and post-construction writes get the same validation as the constructor. Chip-level shared/collective dissipation lives in `Bath` ([`quchip/chip/baths.py`](quchip/chip/baths.py)), attached at construction or later via `chip.add_bath(...)`; bath rates are Lindblad-ready 1/ns with no assembly `2π` (that boundary is Hamiltonian-only — a component's *intrinsic* `2π`, e.g. a resonator's `κ = 2π·f/Q`, is its own physics).

### 3.3.3 Composition rules

Source: [`quchip/engine/slh.py`](quchip/engine/slh.py)

For resolved triples `G1 = (S1, L1, H1)` and `G2 = (S2, L2, H2)`,
the Gough–James rules are

```text
G1 ▷ G2 = (S2 S1, L2 + S2 L1,
           H1 + H2 + (L2^dagger S2 L1 - h.c.) / 2i),
G1 ⊞ G2 = (diag(S1, S2), (L1, L2), H1 + H2),
g = (1 - S_xy)^-1,
S_tilde = S_x̄ȳ + S_x̄y g S_xȳ,
L_tilde = L_x̄ + S_x̄y g L_x,
H_tilde = H + ((sum_j L_j^dagger S_jy) g L_x - h.c.) / 2i.
```

`quchip.engine` exposes these as `series_product`, `concatenate`, and
`feedback_reduce` on the input-free `ResolvedSLH` returned by
`chip.resolve().slh`. Operands must share one Hilbert space, and joined or
closed legs must have empty reference runs. A concrete loop with
`1 - S_xy = 0` is singular and raises. Concatenation needs `prefixes=` when
channel keys collide; feedback from `x` to a different `y` keys the merged
channel `x->y`. Composition Hamiltonian corrections are network-origin static
terms.

### 3.4 Authored local spaces and solver bases

Source: [`quchip/devices/spaces.py`](quchip/devices/spaces.py), [`quchip/engine/basis.py`](quchip/engine/basis.py)

Each device authors its Hamiltonian and named operators in a `LocalSpace`. The built-in realizations are `FockSpace`, `ChargeSpace`, `PhaseGridSpace`, and `CustomSpace`. This declared coordinate space is independent of the solver-basis policy.

`Chip(..., basis="native")` preserves each authored local coordinate basis and dimension. `basis="eigen"` diagonalizes each device's exact authored local Hamiltonian and projects all Hamiltonians, coupling and drive operators, states, observables, and collapse operators into the retained local energy subspace. A device can override the chip-wide policy with its own `basis` and selects the retained energy dimension with `projection_levels` where required.

Resolution records every fixed authored-to-solver transformation in `EngineResult.bases`. Local energy ordering is distinct from whole-chip dressing: `Chip.dress()` diagonalizes the coupled static chip for analysis, while local-basis resolution defines the tensor factors sent through the engine and backends.

Truncation diagnostics sample the component's declared boundary projector at the
solver output times and report its maximum population. Native boundary projectors
are transformed with the captured authored-to-solver map; energy projection adds
the projector onto the highest retained energy state. Frame-band reconstruction
returns the physical boundary population independently of the readout reference.
The sampled scalar traces are retained separately from user observables and state
histories, including when no states are saved.

Fock ladders check their highest Fock state; charge and phase grids check both
edges. An intrinsically finite model declares no native cutoff through
`truncation_boundary() -> None`. Unknown custom cutoffs are reported unavailable.
Boundary population is a heuristic, not a truncation-error bound: sampling can
miss intermediate excursions, and convergence requires increasing the relevant
cutoff and comparing observables.

### 3.5 Transitions

`device.transition(lower, upper)` returns the Hermitian transition operator
`|lower><upper| + |upper><lower|` in the device's authored coordinates.
`device.transition_frequency(lower, upper)` returns the isolated energy gap in
GHz. The level indices refer to the energy ordering of the device's static
local Hamiltonian.

`chip.transition_frequency(target, lower, upper, when=...)` uses the complete
undriven static chip Hamiltonian. It assigns dressed eigenstates to the two bare
product labels and returns their energy difference before frame subtraction or
drive approximation. Unspecified spectators are in level zero; `when` sets
spectator occupations and cannot include `target`.

`chip.freq(target, when=...)` is the concise 0-to-1 form. For a higher target
transition, use explicit levels:

```python
f01 = chip.freq(qubit)
f12 = chip.transition_frequency(qubit, 1, 2)
fr_when_excited = chip.freq(readout, when={qubit: 1})
```

Local basis projection and dressed transition assignment are separate. Basis
projection selects the tensor factors used by the solver; dressed assignment
labels eigenstates of the coupled static chip.

## 4. Frames

### 4.1 What frame selection means

Source: [`quchip/engine/frames.py`](quchip/engine/frames.py)

The public frame spec is one of:

- `"lab"`
- `"rotating"`
- `"auto"`
- a shared float
- a per-device dict

The engine resolves that into per-device reference frequencies `omega_ref,i`.

### 4.2 What transform the engine is using

The engine assumes the rotating-frame unitary

```text
U(t) = exp(-i 2π t Σ_i omega_ref,i * n_i)
```

So the solver Hamiltonian is

```text
H_rot = U† H_lab U - 2π Σ_i omega_ref,i * n_i
```

That second term is why the assembler subtracts `omega_ref,i * n_i` from `H0`.

### 4.3 What `"rotating"` means in practice

`"rotating"` means:

- each device gets its own reference frequency
- an explicit `device.reference_freq` supplies that frequency
- when the setting is `None`, the calculation uses the chip's dressed `0 -> 1` transition

This defines the frequencies `omega_ref,i` used for frame subtraction.

### 4.4 What `"auto"` chooses

`"auto"` is opt-in. The chip default remains `"lab"`, and `"rotating"` keeps
the meaning above. `Chip.resolve(frame="auto", approximation=...)` uses the
requested approximation. With `Exact()` and no tones, it selects the lab frame
because every retained band is static there.

The frame planner chooses one frequency `omega_d` per device. A retained
coupling band with excitation-change vector `k` is static when

```text
Σ_d k_d omega_d = 0
```

A band of a scheduled drive or port coupling at frequency `f` is static when

```text
Σ_d k_d omega_d = f
```

For example, a two-photon cavity pump at 10.2 GHz imposes
`2 omega_cavity = 10.2 GHz` and pins the cavity frame to 5.1 GHz. Network
cascades add coupling constraints. Dispersive and cross-Kerr terms have a zero
charge vector, so they do not constrain the frame.

Drive constraints come from signals delivered by the control equipment after
gain, attenuation, delay, and crosstalk. Each nonzero carrier band constrains
every retained operator band of its destination drive. A crosstalk destination
gets its own tone, and gain changes its weight. A zero-frequency carrier band
adds no constraint.

For a coherent field `beta` entering a network exposure, each reached channel
has `c = S beta` and drives

```text
i(c* L - c L†)
```

The instantaneous operator-band amplitude is `|c| |L_band| / 2π` in ordinary
GHz. Equivalently, the frame record stores `|S| |L_band| / 2π` and the weight
integrates `|beta|²`.

When all constraints cannot hold at once, the planner keeps the consistent
subset with the largest integrated strength. A scheduled tone band has weight
`|h_band|² ∫|envelope|² dt`. Each signal is integrated over its own nonzero
extent inside the outer weighting window. That window is
`[tlist[0], tlist[-1]]` when a problem is built, or `[0, last pulse end]`
otherwise. A short pulse deep inside a long solve therefore keeps its full
energy, while the solve window clips any portion outside it. A static coupling
band has weight `|h_band|² T`, where `T` is the outer window's length. An
unknown traced window leaves static-coupling weights unknown. If identical
constraints are merged, any unknown contribution keeps the combined weight
unknown, and equal or unknown weights fall back to declaration order. The
other bands remain time-dependent.
`resolved.resolved_frame.plan.residuals` records each one's source, devices,
oscillation frequency, and weight.

The planner fixes free frequencies from `reference_freq` in device declaration
order. An undriven exchange-coupled cluster therefore rotates together at one
member's reference frequency. An isolated mode keeps its own reference
frequency. `chip.describe()` and `sequence.describe()` show the selected
frequencies, accepted tones, pins, and residuals. Traced JAX tone frequencies
pass through the plan without conversion to Python scalars.

Stationary analyses such as VNA sweeps and steady states use a strict order.
Cascade constraints are mandatory first, followed by exchange coupling bands
with nonzero charge vectors summing to zero, then tones. A conflicting tone
raises the port form of the existing distinct-stationary-tone error when an
accepted tone already addresses the same devices, or the exchange-connected
form otherwise.
Only non-exchange coupling bands, such as counter-rotating bands retained under
`Exact()`, may remain time-dependent; they then trigger the existing
dynamic-Hamiltonian-terms error.

### 4.5 `reference_freq` — the readout / LO reference

Source: [`quchip/devices/base.py`](quchip/devices/base.py)

`device.reference_freq` returns the authored override in GHz, or `None`. Each new chip calculation resolves `None` to that chip's dressed transition and captures the value in its frame record. An explicit value fixes the frame/readout reference across model changes. Setting it off the transition leaves a residual detuning `Δ = omega - omega_ref` in `H0`, producing idle Ramsey precession.

It is a *frame / readout* reference only: it does **not** detune drives (the drive carrier is a separate choice, so a real LO error must also set the drive frequency). It is ordinary GHz, tracked (mutating it invalidates engine caches), and JAX-traceable / differentiable / sweepable.

## 5. Frame Tracking in the Engine

Source: [`quchip/engine/assembly.py`](quchip/engine/assembly.py)

The engine does not rotate whole expressions symbolically. It tracks phases band-by-band.

### 5.1 Single-device operators

A local operator is decomposed into bands with weight

```text
w = col - row
```

In the chosen frame, that band gets phase

```text
exp(-i 2π w * omega_ref * t)
```

That is how the engine knows which part of `a + a†`, `i(a - a†)`, or an observable is still rotating.

### 5.2 Two-device couplings

A two-body operator is decomposed into bands labeled by `(delta_a, delta_b)`, where each value is the excitation change on one subsystem.

That band gets phase

```text
exp(-i 2π (delta_a * omega_ref,a + delta_b * omega_ref,b) * t)
```

If that effective frequency is zero, the band stays static in `H0`. If not, it becomes an explicit time-dependent term.

This band decomposition is the frame-tracking mechanism.

### 5.3 Model time dependence and scheduled control

`DeviceModel.time_terms()` and `CouplingModel.time_terms()` return
`TimeDependentTerm` values for physics that exists without a scheduled pulse. Each
term pairs a local operator with a `TimeCoefficient`; the engine projects,
band-decomposes, and frames it through the same path as other Hamiltonian
terms. Scheduled control remains drive-owned: its finite-duration envelope and
carrier produce a drive modulation only after `QuantumSequence.schedule()`.
`Envelope.value(local_time)` defines local complex I/Q shape. Scheduling owns
the pulse start, carrier, and global phase, so the same shape can be placed and
phase-rotated without changing its physics definition.

Local eigenbasis projection uses the static authored Hamiltonian at the solve's
operating point. Component-owned time-dependent terms are projected into that
fixed basis; quchip does not construct an instantaneous moving basis.

## 6. Approximation strategies

Source: [`quchip/approximations.py`](quchip/approximations.py), [`quchip/engine/approximations.py`](quchip/engine/approximations.py)

The chip owns one explicit approximation strategy. `Exact()` retains every term in the authored finite-dimensional Hamiltonian. `RWA()` applies the engine's first-order structural rotating-wave reduction to static interactions and scheduled drives. Devices, couplings, and drives do not carry their own RWA policy.

`Chip.unresolved_hamiltonian()` preserves the authored static interaction. Engine assembly band-decomposes each interaction and applies the selected strategy. `Chip.hamiltonian()` is reconstructed from those same canonical terms, so inspection and simulation use one decision path. Under `RWA()`, rejected bands become advisory `DroppedTerm` records with their excitation weights, largest matrix-element magnitude, and frame frequency. A retained band with zero frame frequency stays in `H0`; every other retained band is carried as an explicit time-dependent term.

### 6.1 Static operator bands

For `Capacitive`:

```text
full: g * (a + a†)(b + b†)
     = g * (a†b + ab†) + g * (ab + a†b†)
```

- `a†b + ab†` has total excitation weight zero and survives `RWA()`
- `ab + a†b†` has total excitation weight two and is removed by `RWA()`

`interaction_hamiltonian()` always returns the complete authored form. `RWA()` reconstructs its retained bands in the engine; a coupling does not supply an alternative RWA operator or retention hook. The mask depends only on integer band offsets, so it remains concrete when operator parameters are JAX tracers.

The static/dynamic decision is made per band, not per coupling: in a *shared* frame (multiple devices detuned to a common reference), the coupling's counter-rotating band can carry a nonzero carrier even when its co-rotating band is frame-static. The per-band fold evaluates each band's own carrier independently, so a shared frame never suppresses the counter-rotating band's true rotation.

With `Exact()`, no band is dropped; every non-static band is carried at its own frame frequency.

### 6.2 Driven operator bands

For a single-tone drive channel, the engine forms the real lab-frame field

```text
Re[s(t) * exp(-i 2π f_drive t)]
```

and combines it with the operator bands.

`Exact()` retains both co-rotating and counter-rotating pieces.

`RWA()` keeps the conventional partner for each excitation band and drops its counter-rotating partner.

Flux drives are different: they couple through `n`, which is diagonal, so there is no raising/lowering split to RWA away. They are treated as direct real-valued modulation channels.

## 7. Counter-Rotating Terms

Counter-rotating terms appear when the operator changes total excitation in the same direction as the classical or frame rotation instead of cancelling it.

Concrete examples:

- In full capacitive coupling, `ab` and `a†b†` are counter-rotating.
- In a single-tone drive, the fast partner of the real field is counter-rotating relative to the chosen transition band.

In the rotating frame of two detuned modes with frequencies `omega_a` and `omega_b`:

- exchange terms rotate at about `|omega_a - omega_b|`
- counter-rotating terms rotate at about `omega_a + omega_b`

That is why they are usually dropped by RWA: they are much faster and usually average out.

Choose `RWA()` to remove structural first-order counter-rotating bands, or `Exact()` to carry every term explicitly.

## 8. Observables and Demodulation

Source: [`quchip/engine/observables.py`](quchip/engine/observables.py)

Dict-form `e_ops` are decomposed into the same excitation bands used by the frame logic. After the solver returns, the engine recombines them with the demodulation frequencies in `ResolvedFrame.demod_freqs = omega_ref - omega_frame` (per device).

This makes `result.expect` a **co-rotating readout**: observables are always reported in each device's `reference_freq` frame, independent of the integration frame the solver used. So transverse observables (`<a>`, `<sigma_x>`) come back as the non-oscillatory demodulated envelope a lab readout produces — slow, and turning at `Δ = omega - omega_ref` when the reference is detuned; diagonal observables (populations) are frame-invariant. In the default `"rotating"` mode the integration frame *is* the reference frame, so the demodulation is a no-op and `result.expect` equals `Tr(O·rho)` on the same states `result.states` returns. The raw, un-demodulated band sum (the observable in the integration frame) remains available on each `ObservableTrace` as `.raw`.

### 8.1 Stationary solves and scattering

`Chip.steadystate()` solves `L(rho_ss) = 0` together with
`Tr(rho_ss) = 1`. It requires a static resolved Hamiltonian and a unique
normalized stationary state. `VNA.sweep()` adds continuous-wave port terms in
their stationary tone frames and returns the complete scattering matrix
between the selected planes. At each frequency, the stationary route solves
one pumped operating point and uses one shifted-Liouvillian factorization for
all input columns; the passive-linear route uses one multi-right-hand-side
mode-space solve. Small-signal scattering differentiates the output mean
around the fixed-tone state,

```text
S_ji(f) = d <b_out,j> / d beta_in,i  at beta_probe -> 0.
```

Around a phase-sensitive operating point, the full response is
`delta <b_out> = S delta beta + T conj(delta beta)`. `result.conjugate_matrix` stores
`T` with the same `[..., output, input]` layout as `result.matrix`, and
`result.t(output, input)` selects one entry. The stationary route obtains `S`
and `T` from the same shifted-Liouvillian factorization; the passive-linear
route reports zero for `T`.

The direct term comes from the resolved scalar `S`; the system term comes from
the stationary response of `L` under the same coherent-input Hamiltonian.
`VNA.finite_power()` instead adds a probe of amplitude `beta` at one selected
input and solves the stationary Liouvillian in the probe frame at each grid
point. It reports the mean field at every selected plane,

```text
<b_out,j> = H_out,j(f) [sum_i S_ji beta_boundary,i + <L_j>].
```

This path uses the same reference-plane and hidden-channel bookkeeping as the
small-signal response, but has no mode-space shortcut. Every selected plane
must resolve at the probe frequency. The ratio `<b_out,j>/beta` tends to the
corresponding `S_ji` as `beta -> 0` when no fixed pump leaves a coherent mean
at that plane and carrier; at finite `beta` it is a stationary
mean-field response, not a small-signal S-parameter. It does not describe
sweep-rate hysteresis or metastable branches.

For a pump-free passive-linear model, the same authored expressions also admit
the mode-space form

```text
d a/dt = A a + B b_in,          b_out = C a + S b_in,
A = -i Omega - C^dagger C / 2,  B = -C^dagger S.
```

The engine accepts this route only when the retained Hamiltonian is static,
quadratic, and number conserving and every collapse operator is linear in the
mode lowering operators. It applies the Hamiltonian `2π` conversion at the
assembly boundary, includes the series-product Hamiltonian generated by the
`PortNetwork`, and computes the per-frequency inbound and outbound transfer
factor `exp(+i 2π f τ)` for each reference leg. It sends the compact matrices
to the selected backend, which evaluates the undecorated Markov response

```text
S_out,in(f) = S_out,in + C_out (-i 2π f I - A)^(-1) B_in.
```

The engine then applies the inbound and outbound factors to that response.

Nonlinear, pumped, active, dynamic, or opaque operator models retain the
stationary-Liouvillian route. This selection is structural and does not depend
on the numerical value of a traced parameter.
Active local terms also retain the general route: a weight-only RWA does not
establish whether a local parametric term is off resonance in its authored frame.

`VNA.sweep()` and `VNA.finite_power()` cover small-signal scattering and
stationary finite-power mean fields, respectively. Ring-up, ring-down, wave
packets, and other time-resolved fields require a scheduled external-plane
input in a `QuantumSequence`. Fixed finite pumps remain valid VNA
operating-point fields.
The engine supplies canonical sources and observables for stationary response,
spectrum, and correlation queries. Each backend constructs and solves its own
native Liouvillian. A stationary state and its subsequent response or regression
query share that preparation. Public results do not retain the native generator.

Uniqueness checks and residuals run with the solve. Optional condition numbers
and positivity checks run when accessed, using captured inputs. Reading VNA
diagnostics such as `solver` or `residual` does not evaluate the other entries;
converting a diagnostic mapping to a dictionary requests all its values. A
requested stationary condition number rebuilds the native generator without
solving for the state again. Concrete diagnostic values are cached; traced
values are not. QuTiP retains its `diagnostic_max_dimension` limit: a skipped
rank or condition diagnostic is `None`, not a successful check.

If frame and approximation resolution leave dynamic terms, the stationary
APIs raise. Periodic/Floquet stationary states are not implemented.

### 8.2 Captured noisy VNA measurements

`VNA.measure(frequencies, amplitudes, input=..., outputs=...)` prepares one
stationary operating point for each probe/sweep coordinate and captures means
and normally ordered IQ cross-spectra. Receiver integration, calibration, and
Gaussian draws act on these captured arrays. They never call a stationary
solver or consult a later mutable chip. `measurement.parameters` records the
numerical model parameters in flattened sweep order; `noise_frequencies`
records the stored offset grid. The finite-power ratio is output mean divided
by the probe amplitude, not the small-signal derivative around a separate pump.

For eligible passive harmonic models, `measure()` uses the same compact
mode-space lowering and backend response solver as `sweep()`. With response
matrix T(f) and input occupations n, the normal output spectrum at the Markov
boundary is `T(f) diag(n) T(f)†`. Subtracting `S diag(n) S†` supplies the
device-generated excess to the shared downstream propagation; that owner adds
the direct sources once. This is the exact stationary Gaussian field solution,
including thermal fluctuations, without a Fock-space truncation. All modes
must decay. Concrete acquisitions check stability; traced paths retain the
usual host-validation limitation. Thermal device collapse declarations,
nonlinear or active terms, fixed pump configurations, branched output graphs,
and explicit solver options retain the operator-space acquisition. Passing
`options={}` requests that general route for cross-checks.

Measurements also capture internal Fock-mode observables. `mode_amplitude(r)`
returns `<a_r>` in the stationary frame reported by `mode_frequency(r)`;
`photon_number(r)` returns `<a_r† a_r>`. Both follow the measurement sweep
axes and retain the full declared input wiring. The compact backend solves
`A N + N A† + B diag(n) B† = 0` for the centered normal covariance
`N_ij = <delta a_j† delta a_i>` and adds its diagonal to the coherent
occupation `|<a_r>|²`. This covariance is independent of the output spectral
grid. The general path evaluates the authored `a` and `n` operators in the
resolved basis against the solved reduced density matrix, retaining nonlinear
and active physics and the declared truncation. These queries use captured
arrays; receiver processing does not change internal occupation.

An attenuator, isolator load, or `network.termination()` can declare
`thermal_occupation=n`: a finite, non-negative mean thermal population in
quanta. Vacuum remains the default. Amplifiers require input-referred
symmetrized `added_noise` in quanta, subject to the phase-preserving quantum
floor. Both noise values are constant across the modeled band, including
frequency sweeps; no temperature or reference frequency is inferred.
Attenuators accept positive `loss_db` instead of `eta`; amplifiers accept
`gain_db` instead of linear power gain. Authored parameters remain the
rebinding and serialization paths.

For a full unitary scattering matrix S, input j couples through
`K_j = (S† L)_j`. In addition to vacuum `sum_i D[L_i]`, its thermal population
adds `n_j D[K_j] + n_j D[K_j†]`. Thermal input is therefore part of the
stationary and transient quantum dynamics. It does not become a second
independent device bath. SLH composition retains the surviving input's state
and rejects an independent thermal declaration on an input that is connected
away. Arbitrary passive nonideal components use `network.component()` with
unitary scattering and explicit dissipative terminals connected to declared
loads; insertion loss and isolation numbers alone do not define this matrix.

Normal output spectra include the direct term `S diag(n) S†` and the
input-system interference in the regression source
`B_i = (L_i-<L_i>) rho + sum_k (S diag(n) S†)_ik [L_k,rho]`.
This interference prevents double counting fluorescence on top of an
incident thermal field at equilibrium. Real IQ sources constructed from B
retain normal, anomalous, and cross-output second moments. `output_spectrum()`
selects the scalar normal spectrum from the same calculation as `measure()`.
The Fourier convention is `integral exp(+i 2π offset τ) <δb†(0) δb(τ)> dτ`;
a mode above the carrier peaks at positive offset. This corrects the mirrored
detuned-fluorescence spectrum in 0.3.0.

Physical source budgets separate directly propagated fields from
`device.correlations`. The latter includes nonlinear response and interference,
so it need not be positive or independently sampleable. A matched absorptive
filter declares `thermal_occupation` and emits
`(1-|H(f)|²)n` in each direction. A scalar H(f) without that declaration does
not imply an absorptive thermal model. Colored emission may propagate to
external outputs, but colored noise that feeds a quantum coupling is rejected:
a colored reservoir requires an explicit dynamical model. Source color follows
propagation order. A vacuum filter before occupied attenuators does not color
their emission. Source backaction follows `S†L`, so fields mixed only after
the device may carry filtered thermal noise without heating it.

`measurement.noise_spectrum(output)` recovers the normally ordered scalar
spectrum from the captured IQ matrix, retaining upper/lower sideband asymmetry.
It excludes the coherent carrier and final receiver vacuum. `unit="W/Hz"`
multiplies by `h f_absolute`; `unit="dBm/Hz"` reports its power ratio to
1 mW/Hz. Power conversions require positive absolute sideband frequencies.
Receiver source budgets remain integrated IQ covariances in photons/ns;
their trace is the complex-field variance, not a spectral power density.

Acyclic output networks can place an amplifier before a splitter or between
passive components. The compiler retains a unitary Markovian boundary and
captures a separate directed field map with source cross-spectra. These
sections cannot feed quantum couplings or instantaneous feedback loops.
Amplifiers retain the output-line convention above; `added_noise` never
implies reverse HEMT emission. The separate inbound and outbound traversals
of an exposed reference section are preserved. Branched reference networks
currently support stationary fields; transient output observables and direct
SLH composition reject them explicitly. Compose their physical PortNetwork
before resolving it. Their VNA response uses the general stationary solver.

`IQReceiver(integration_time=T)` applies a normalized boxcar with frequency
weight `sinc(offset*T)²`. For `b=I+iQ`, ideal heterodyne detection adds one
complex vacuum quantum at the final plane, or `1/2` on each IQ diagonal.
Thus flat normally ordered noise N gives `Var(I)=Var(Q)=(N+1)/(2T)`.
Joint outputs retain their complex cross-spectrum and relative delays;
independent detector vacuum is added once per output. Calibration multiplies
the mean and transforms both covariance axes. Zero probe amplitude leaves
field statistics defined and ratios undefined.

An optional receiver `transfer(offset)` is a digital complex amplitude
response and also scales the mean by its DC value. White noise is integrated
analytically for an ordinary boxcar; colored terms and digital filters use
the captured grid. The receiver compares full and coarsened quadrature and
checks spectral edges. These local checks do not prove that an arbitrary
spectrum has no unsampled feature. Use a wider or finer capture for unsupported
bandwidths or integration times. Concrete validation must be run outside JAX
tracing; deterministic integration and keyed reparameterized draws remain
differentiable on fixed shapes.

Gaussian samples reproduce the captured second moments. They do not supply
higher-order non-Gaussian photon statistics or continuous correlated records.
Normalized `g1` and `g2` for network thermal fields require a detection bandwidth
and are rejected by the unfiltered correlation API.


## 9. Dressing

Sources: [`quchip/chip/chip.py`](quchip/chip/chip.py), [`quchip/chip/analysis.py`](quchip/chip/analysis.py)

`Chip.dress()` diagonalizes the full static lab-frame Hamiltonian, assigns bare product states to dressed eigenstates by overlap, and stores a `DressedResult` containing:

- eigenvalues and lazily materialized eigenstates
- bare-to-dressed state assignments and the assigned eigenvalue for each bare label
- assignment overlaps and labels below the requested overlap threshold
- the dressed eigenvector matrix used by dressed-basis analysis

`Chip.freq()` evaluates dressed `0 -> 1` frequencies through the traceable array-labeling cache; those frequencies are not stored in `DressedResult`.

`Chip.dress()` is always intrinsic static lab-frame analysis. It is not part
of the runtime frame transform. A resolved `EngineResult` also provides
`dress()`, which diagonalizes that snapshot's selected frame and approximation.
If the snapshot has dynamic Hamiltonian terms, `dress(at_time=...)` is required
and evaluates their signal programs at that instant. This is an instantaneous
eigensystem, not Floquet or cycle-averaged analysis.

### 9.1 Dressed drive matrix elements

Sources: [`quchip/chip/analysis.py`](quchip/chip/analysis.py), [`quchip/control/equipment.py`](quchip/control/equipment.py)

For a drive line `j` with local Hamiltonian operator `D_j`, quchip defines the dressed matrix element

```text
m_j^(fi) = <f~|D_j|i~>
```

with the **final** dressed state as the matrix row and the **initial** dressed state as the matrix column. Thus
`Chip.drive_matrix_elements((initial, final))[j]` reads `[final, initial]` from `U† D_j U`. The device shorthand
`chip.drive_matrix_elements(q)` selects the dressed transition from the all-ground state to the state labeled by
one excitation in `q`. Explicit `(initial_mapping, final_mapping)` arguments select arbitrary transitions. Before
the matrix element is evaluated, every dressed eigenvector is phase-fixed so that its overlap with its assigned bare
state is real and nonnegative. This removes backend-dependent eigenvector signs from comparisons between conditioned
transitions, such as the sum and difference used for the weak-drive `IX` and `ZX` coefficients.

`drive_matrix_elements` evaluates the physical drive operators without applying the signal chain. Declared
control-line mixing is represented separately by `ControlEquipment.crosstalk_matrix()`: column `j` is the source
line, row `l` is the victim line, and each entry carries an amplitude, phase, and delay. The returned matrix
elements can then be combined with those declared line phasors in a chosen weak-drive effective-Hamiltonian model.
Keeping the two pieces separate distinguishes dressed quantum response from microwave-path mixing.

This projection follows the effective driven-Hamiltonian treatment of E. Magesan and J. M. Gambetta,
Phys. Rev. A 101, 052308 (2020), DOI [`10.1103/PhysRevA.101.052308`](https://doi.org/10.1103/PhysRevA.101.052308).

### 9.2 Weak-drive cross-resonance susceptibility

For a charge drive on control `c`, projected onto the target transition `t` with the control fixed in `|z>`, define

```text
m_z = <z_c, 1_t~|D_c|z_c, 0_t~>,    z in {0, 1}.
```

In the cross-resonance convention

```text
H_eff = (IX I X + ZX Z X) / 2,
```

the control-conditioned off-diagonal entries are `(IX + ZX)/2` and `(IX - ZX)/2`. Therefore a signal amplitude
`Omega` multiplying `D_c` gives

```text
IX / Omega = m_0 + m_1,
ZX / Omega = m_0 - m_1.
```

`analyze_cr_susceptibility` reports these complex coefficients per unit amplitude without choosing a pulse or
performing time evolution. A drive phase may rotate the common complex quadrature; `abs(ZX)` is the maximum useful
linear-response rate after that phase choice. The projection remains a weak-drive statement and does not include
strong-drive Stark shifts, pulse-bandwidth leakage, or echo/cancellation calibration.

This convention follows the effective-Hamiltonian decompositions of Magesan and Gambetta, Phys. Rev. A 101,
052308 (2020), and Malekakhlagh, Magesan, and McKay, Phys. Rev. A 102, 042605 (2020).

### 9.3 Dressed Kerr matrix

`Chip.kerr_matrix()` evaluates one labeled eigensystem and returns a symmetric
matrix in `chip.devices` order. For distinct devices,

```text
K[i,j] = E(1_i,1_j) - E(1_i) - E(1_j) + E(0),
```

which is the same full-pull convention as `Chip.dispersive_shift(i, j)` and
the static-ZZ coefficient for two qubits. On the diagonal,

```text
K[i,i] = E(2_i) - 2 E(1_i) + E(0),
```

which is `Chip.dressed_anharmonicity(i)`. A device with fewer than three
resolved levels has `NaN` on the diagonal; its defined off-diagonal entries
remain available. Every entry comes from the complete dressed chip rather
than from a matching authored edge. In particular, an isolated
`KerrCavity` with `H = omega*n - K*n*(n-1)` has `K[i,i] = -2*K`.

## 10. Adiabatic Elimination and Dispersive Readout

Sources: [`quchip/chip/transformations/`](quchip/chip/transformations/), [`quchip/analysis/dispersive_readout.py`](quchip/analysis/dispersive_readout.py)

`eliminate(chip, target, method="sw"|"exact")` performs model reduction, dispatched on the target. A device target removes a far-detuned mode. Both routes retain their complete computed Hamiltonian and each transformed channel from the removed mode and couplings. `chip.effective_terms` carries the matrix correction beyond the reported Lamb shifts and mediated exchange, including higher-level corrections. Intrinsic survivor noise stays separate from inherited loss; a common bus channel remains collective. Scalar readout quantities such as `chi`, `kappa`, and the first-transition Purcell rate remain available in `effective_params`. A coupling target keeps both endpoints and removes the selected edge. Its isolated pair determines a coordinate change applied to the entire chip Hamiltonian, including parallel and spectator interactions. The exact route is a full unitary transformation; SW retains terms through second order in interactions. The retained correction includes per-level shifts while authored endpoint parameters stay unchanged. Surviving component channels follow a captured operator projection while their rates remain component-owned. Removed-component channels already use the retained coordinates. Surviving controls and collective or thermal baths follow the captured coordinate changes. Controls targeting removed components and retargeted ports require their explicit conversion rules. Sources for the reduction math: [`quchip/chip/sw.py`](quchip/chip/sw.py).

### 10.1 The χ convention and related quantities

```text
chi ≡ chi_pull ≡ f_r(qubit in |1>) − f_r(qubit in |0>)     [GHz]
```

the *full* resonator pull per qubit excitation. This is **2×** the σ_z-convention χ of `H_disp = (omega_r + chi_sigma_z * sigma_z) * a†a` used in most textbooks. Three related quantities use different conventions:

- `eliminate(...).effective_params[q]["chi"]` — χ_pull as defined above, computed *numerically* from the pre-elimination dressed spectrum (identically `Chip.dispersive_shift(r, q)`: `E(1,1) − E(1,0) − E(0,1) + E(0,0)`, one shared diagonalization), exact and device-agnostic (works for any survivor type, not just Duffing transmons). The entry is evaluated and cached on first access, so the diagonalization occurs only when `chi` is read.
- `fit_a_dress` constraints use signed full `cross_kerr` (including its `chi` alias). Migrating a pre-0.3 `coupling_targets={edge: "chi"}` half-pull target requires `constraints={edge: {"cross_kerr": 2 * old_target}}`. A previous `zz` target already uses full cross-Kerr and keeps its value.
- `Chip.dispersive_shift(a, b)` (alias `static_zz`) — the general two-mode cross-Kerr `E(1,1) − E(1,0) − E(0,1) + E(0,0)`. For a qubit–resonator pair this *is* χ_pull (which is exactly how the `chi` entry is computed); between two qubits the same expression is the static-ZZ ζ — do not read a qubit–qubit `dispersive_shift` as a readout χ.

Analytic cross-checks (2nd-order dispersive): two-level `chi = 2g^2/Delta`; Duffing transmon `chi = 2g^2*alpha/(Delta*(Delta+alpha))` with `Delta = f_q − f_r` (Koch et al., PRA 76, 042319, §IV). Critical photon number `n_crit = Delta^2/(4g^2)`.

`effective_params[q]["kappa"]` is the eliminated mode's total intrinsic downward decay rate, in 1/ns, as returned by `intrinsic_decay_rate()`. For a resonator it includes `2π*f_r/Q_internal` when `internal_quality_factor` is set and the inherited thermal-emission rate when `T1` or `thermal_occupation` is set. The latter is `(nbar + 1)/T1` with `T1`, or `nbar + 1` when only `thermal_occupation` is present. The reported value is `0.0` only when none of these intrinsic lowering channels is configured. An external default port on a linear resonator is transformed separately as described in §10.5 and is not folded into survivor `T1`; this avoids counting its Purcell channel twice. Bridge legs report `chi = 0.0`: bus/coupler modes are not readout modes, and their dressed pull would double-count the mediated exchange.

Gradients through `chi` follow the same rule as `Chip.freq` (§13): the eigensystem must come from a JAX-capable backend.

### 10.2 Pointer states and readout figures of merit

`analyze_dispersive_readout(chi, kappa, tau, ...)` is closed-form steady-state algebra (driven, damped *linear* resonator, `d<a>/dt = −(i*delta + kappa/2)<a> − i*eps`):

```text
delta_r = f_r|0 − f_drive                   drive placement  [GHz]  (Δ_r = ω_r − ω_d)
delta_j = 2π*(delta_r + chi_eff*j)          resonator−drive detuning, qubit in |j>  [rad/ns]
alpha_j = −i*eps / (kappa/2 + i*delta_j)    coherent pointer state
nbar_j  = |alpha_j|^2                        steady-state photons (emergent)
sigma   = 1/sqrt(2*kappa*tau)                integrated vacuum-noise blob width
SNR     = |alpha_1 − alpha_0| * sqrt(2*kappa*tau)
p_err   = (1/2)*erfc(SNR/(2*sqrt(2)))        two equal Gaussians, optimal discriminant
Gamma_m = kappa*|alpha_1 − alpha_0|^2 / 2    measurement-induced dephasing  [1/ns]
```

with `eps = sqrt(nbar_0*((kappa/2)^2 + delta_0^2))` when the drive is given as a target photon number, and the optional strong-drive collapse `chi_eff = chi/(1 + nbar_0/n_crit)`. In the small-χ limit `Gamma_m → 8*chi_sigma_z,ang^2*nbar/kappa` with `chi_sigma_z,ang = π*chi_pull` in rad/ns (Gambetta et al., PRA 74, 042318; Krantz et al., APR 6, 021318, §V).

The internal `2π`s here are local physics conversions at the module's public boundary — the engine's single Hamiltonian-assembly `2π` (§1) is untouched. Declared approximations (carried in the result's `notes`): steady state only (no ring-up transient), linear resonator, 2nd-order dispersive, no measurement-induced qubit T1.

### 10.3 The Schrieffer-Wolff route (`method="sw"`)

The chip's bare Hamiltonian `H = H0 + V` (GHz, before `2π`, under the selected approximation) is partitioned by the eliminated mode's occupation: `P` = the mode in `|0>`, `Q` = everything else. The generator solves the Sylvester condition on the cross blocks,

```text
S_ij = V_ij / (E_i − E_j)        (i, j straddling P/Q; E = diag H)
H_eff = P (H + (1/2)[S, V]) P
```

(Bravyi, DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011), 2nd order). Nested `where` guards handle the division: an exactly degenerate cross pair with no matrix element contributes zero with a *finite gradient*, whereas a single `where` would propagate a `NaN` backward through the unselected branch. A zero gap with nonzero P/Q coupling is singular and raises; numerical outputs are also marked invalid under tracing. The working-precision threshold only selects diagnostic entries and never turns a nonzero coupling into zero. Exact reduction does not construct an unused SW generator. Survivor parameters are obtained by indexing `H_eff`: `freq_after(s) = E(1_s) − E(0)`; the pair exchange is the `<1_a|H_eff|1_b>` element. Authored direct edges stay unchanged. A separate mediated edge carries the real exchange matrix element of `H_eff − P H P`; `effective_params["exchange"]["coupling"]` names that edge. The complete retained correction carries all remaining matrix elements. Sequential shifts and detunings use the incoming Hamiltonian's diagonal, including earlier retained corrections. Alongside `J`, the bridge reduction records its linearization

```text
dJ/domega_c = (g_a*g_b/2)(1/Delta_a^2 + 1/Delta_b^2)
```

— the weight the flux-drive retarget rule uses (§11). Per-element virtual-state attribution (`pathways`) is `(1/2) V_ik V_kj (1/(E_i−E_k) + 1/(E_j−E_k))` summed over intermediate `|k>`, with the same guarded denominator.

### 10.4 The exact route (`method="exact"`)

One full diagonalization; parameters are read off the *labeled* dressed spectrum (`label_eigensystem`, §9), so kept-block energies are exact to all orders, as required for residual ZZ:

```text
zz(a, b) = E(1,1) − E(1,0) − E(0,1) + E(0,0)        (≡ Chip.dispersive_shift)
```

The complete retained Hamiltonian and its pair exchange are read through the symmetrically (Löwdin-)orthonormalized subspace projection `S^(−1/2) (W E W†) S^(−1/2)` with `W` the overlap block and `S = W W†` — the des-Cloizeaux effective Hamiltonian, whose spectrum equals the labeled energies exactly. The energies are exact, but this basis is not the canonical SW rotation, so off-diagonal reads agree with `method="sw"` only through 2nd order. `method="exact"` validates the ground, touching-survivor single excitations and their pair excitations: each diagnostic label must have a distinct majority dressed eigenstate. The full retained overlap Gram matrix must also permit stable orthonormalization. These checks apply in eager, compiled and gradient-only execution. In that regime, near-degenerate dressed states straddle the bare labels and quantities assigned to a single label are not well defined. Use `method="sw"` or shift the operating point.

### 10.5 Collapse transforms and validity metrics

The eliminated mode's own jump operator is carried into the reduced frame by the same rotation as the Hamiltonian:

```text
c_eff = P (c + [S, c]) P            (sw — leading transformed mode jump)
c_eff = B† c B                      (exact — B is the retained Löwdin embedding)
```

For the exact route, `B = V_selected (S^(-1/2) W)†` uses the same selected eigenvectors and overlap matrix as the retained Hamiltonian. It is isometric and independent of arbitrary eigenvector phases.

For intrinsic mode loss, the survivor-lowering amplitude gives the inherited
(Purcell) rate `|amplitude|^2 * kappa`. Both models retain the complete transformed mode jump
matrix and its own rate, including separate thermal emission and absorption
channels. It does not replace a collective jump by independent T1 channels.
`EffectiveTerms` are captured in authored coordinates and use the ordinary
basis and frame compiler without a second band-removal approximation. A static
collective jump must have one removable global phase in the selected frame;
unequal band phases require a compatible common frame or the lab frame.
For an external default port on a
linear resonator, the complete `c_eff` matrix becomes that port's operator on
one unprojected Fock-space survivor; the port's rate, phase, scalar scattering,
and exposure reference plane are retained. Custom or collective boundary
operators, projected or multiple survivors, a nonlinear eliminated boundary
target, and ports participating in a cascade-generated Hamiltonian are
rejected rather than approximated or double-counted.

The result's `notes` record that the projection is exact for the *spectrum*
but approximate for *dissipation*: the discarded `Q`-block dynamics also
dephase and decay. `validity` reports, per eliminated coupling,
`g_over_delta` (2nd-order smallness; `is_valid` gates at `< 0.1`) and
`min_block_gap` — the smallest bare-energy gap the Sylvester generator
crossed. A small gap with a nonzero matrix element is the perturbative
expansion's failure mode even when every `g/Delta` is small.

### 10.6 State and observable maps

Device elimination leaves surviving devices' authored frequencies and local bases unchanged. The retained Hamiltonian correction owns their shifts; `effective_params` reports the reduction's transition diagnostics. Use `chip.freq(...)` for a reduced chip's coupled transitions. To reduce another operating point, rebind the source model and eliminate again.

`result.mapping` captures source and target labels, dimensions, backend and lab-frame solver coordinates. Its `embedding` maps retained coordinates into the source space. `project_operator(operator)` returns `B† O B`; `project_state(state)` returns `B† psi` or `B† rho B`; `lift_state(state)` performs the reverse embedding. These methods accept full numerical matrices and backend-native objects and return native objects on the captured backend. Projection preserves the lost norm or trace, so discarded population remains visible. No state is silently renormalized. Rotating-frame trajectory states must be expressed in lab coordinates before using this map.

Exact reduction uses the same Löwdin embedding as its Hamiltonian and removed-component channels. SW uses `B = exp(-S) P`, with the first-order generator already used for reduction. This map is isometric; the dynamics remain perturbative. The SW Hamiltonian remains truncated at second order. Jump operators use the same exponentiated first-order coordinate map, preserving their common interpretation with states and observables. This does not make the reduced dynamics or dissipation exact. Maps capture numerical coordinates independently of later source edits.

## 11. Parametric Edge Control

Sources: [`quchip/control/drive.py`](quchip/control/drive.py) (`ParametricDrive`), [`quchip/engine/assembly.py`](quchip/engine/assembly.py) (`EDGE_PUMP`), [`quchip/chip/retarget.py`](quchip/chip/retarget.py)

### 11.1 The pump contract

A `ParametricDrive` pumps a modulable coupling (a `TunableCapacitive` edge): the scheduled envelope is the *real* modulation `A(t)` of the coupling strength, in GHz. Two forms:

```text
freq omitted (baseband):  delta_g(t) = Re s(t)
freq = nu_d (tone):       delta_g(t) = Re[s(t) · e^(−i·2π·nu_d·t)]
```

The tone is never RWA-split by the engine: the *coupling's* `parametric_interaction` hook picks the retained operator structure, and each excitation-change band `(Δa, Δb)` carries its rotating-frame carrier `exp(−i(Δa·ω_a + Δb·ω_b)t)` exactly as static couplings do (§5.2). Pumping at the survivors' difference frequency parametrically activates the exchange with effective rate `A/2` (the rotating-wave halving of a real modulation; Didier et al., PRA 97, 022330 (2018)).

### 11.2 Retargeting stranded control (`chip/retarget.py`)

`eliminate()` converts control lines whose target was removed through a registry keyed by `(drive type, target type, result kind)`. The registry follows each type's MRO, so a rule registered for a base type also covers its subclasses. The built-in rule converts a `FluxDrive` on an eliminated exchange-mediating mode into one baseband `ParametricDrive` per emitted edge. The first pair's pump keeps the flux line's label; further edges receive unit-amplitude `Crosstalk` copies of the scheduled signal; and every pump carries its own `Gain(dJ_ab/domega_c)`. This small-signal conversion is exact to first order in `delta_omega_c` and assumes `delta_omega_c ≪ Delta`; second-order Lamb-shift modulation of the survivors is omitted and recorded.

### 11.3 Schedule portability

The retargeted line keeps its label, and `schedule()` resolves drive-line labels in device → coupling → line order. The same schedule call can therefore run on the full and reduced chips. [`tests/physics_sentinel/test_eliminate_portability.py`](tests/physics_sentinel/test_eliminate_portability.py) applies identical schedules to both models and compares them using tolerances derived from the validity metrics (`g/Delta`, `delta_omega/Delta`).

## 12. Engine Assumptions

The engine relies on four physics assumptions:

1. Frame generators are built from per-device number operators `n_i`.
2. Single-device and two-device operators can be decomposed by excitation-change bands.
3. The `"rotating"` frame uses each device's explicit reference or the chip-resolved dressed transition.
4. Each drive builds a complete analytic signal with an optional carrier, then
   implements `hamiltonian(target, signal)`. Control equipment transforms that
   signal before the destination drive maps its physical I/Q quadratures to
   target-local operators. The engine owns frame and band selection.

What the engine does not hardcode:

- transmon-specific formulas
- resonator-specific formulas
- backend-native operator types
- device-specific noise models beyond asking each device for its collapse operators

Devices, couplings, and drives supply the domain-specific physics. The engine handles the frame and RWA bookkeeping shared by them.

## 13. JAX Traceability Boundaries

Band decomposition, coefficient construction, observable recombination, and the backend-free Hamiltonian IR preserve JAX arrays.

The following operations require concrete Python values:

- `Chip.dress()` returns a concrete dict-based view and is not traceable. The bare→dressed assignment itself is discrete and piecewise. Traced callers should use `Chip.energy()`, `Chip.freq(target, when=...)`, `Chip.dispersive_shift()`, or `Chip.kerr_matrix()`, which route through `label_eigensystem` in the pure-JAX kernel in `quchip/chip/dressing.py`; labeled energy lookup stays differentiable away from label discontinuities. `track_path` is a separate continuation utility for following labels through a stacked eigensystem along a parameter sweep.
- Human-facing serialization and diagnostics coerce to Python scalars.

Other engine paths avoid implicit conversion to host arrays.

## 14. Audit Pointers

When you need to audit a physics path, start here:

- units and assembly boundary: [`quchip/engine/assembly.py`](quchip/engine/assembly.py)
- frame resolution: [`quchip/engine/frames.py`](quchip/engine/frames.py)
- observable preparation and demodulation: [`quchip/engine/observables.py`](quchip/engine/observables.py)
- dressing and public Hamiltonian APIs: [`quchip/chip/chip.py`](quchip/chip/chip.py)
- adiabatic elimination and χ/κ reporting: [`quchip/chip/transformations/`](quchip/chip/transformations/)
- Schrieffer-Wolff kernels and the exact reduction route: [`quchip/chip/sw.py`](quchip/chip/sw.py)
- control-line retargeting across reductions: [`quchip/chip/retarget.py`](quchip/chip/retarget.py)
- readout pointer states and figures of merit: [`quchip/analysis/dispersive_readout.py`](quchip/analysis/dispersive_readout.py)

## Measurement of saved states

`SimulationResult.measure(*devices, t=None, basis="energy")` projects a retained
ket or density matrix in the captured local isolated energy bases. `t=None`
selects the final state. A custom basis supplies orthonormal columns in those
energy coordinates in the stored integration frame, without an automatic
phase-frame conversion; `basis="solver"` selects the solver's product basis.
Joint probabilities are obtained before marginalizing unmeasured devices.
Independent partition components may be combined at the probability level.

Shot sampling applies the Born rule without further quantum evolution.
`assignment[recorded, physical]` is column-stochastic. `IQReadout` defines one
conditional complex Gaussian distribution per physical outcome; its total
mixture covariance includes both within-outcome covariance and the covariance
of conditional means. These readout models do not change solver
selection, return collapsed states, or describe continuous quantum trajectories.

`IQReadout.from_wiring(chip, output, ...)` resolves a detector without quantum
evolution. `result.iq_readout()` uses the simulation's captured wiring instead.
Both propagate supplied conditional coherent fields through
captured output reference sections and downstream mixing. Unspecified boundary
channels are vacuum. It includes downstream added noise and ideal heterodyne
vacuum through the same propagation and integration used by VNA. Boundary
thermal noise, device correlations and transient field correlations are outside
this coherent-field readout model; calibrated conditional distributions may include
those effects instead. A calibrated full covariance must not receive the same
apparatus noise a second time. Input baths and port decay remain in the declared
quantum dynamics, irrespective of the readout model.


## Continuous trajectory monitoring

Native jump and diffusive solvers evolve the captured Hamiltonian and declared
channels. `with_monitoring` selects physical SLH couplings with their authored
phases. At efficiency eta, QuTiP receives sqrt(eta) exp(-i phase) L and
sqrt(1-eta) L; their dissipators sum to D[L]. Dynamiqs receives L and eta through
its native SME interface. Pure SSE cannot omit unobserved loss.

Native measurement records describe selected L in the integration frame, without
adding coherent incident beta or downstream receiver noise. Native weighted jump
averages include deterministic no-click paths. Truncation checks are explicit
and cover available samples, with final-only coverage identified separately.
See [the solver guide](docs/guides/choosing-a-backend.md) and
[Wiseman and Milburn, chapter 4](https://doi.org/10.1017/CBO9780511813948).
