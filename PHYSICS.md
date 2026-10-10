# Physics reference

This document states the physics contracts of quchip. It separates authored from resolved Hamiltonians, records where quchip applies local bases, frames, and RWA, and states the engine's assumptions.

## 1. Units and the 2π Convention

User-facing Hamiltonians express `E/h` in ordinary GHz, and solver assembly
converts them to angular frequency. The units are:

| Quantity | Unit |
| --- | --- |
| Frequency | GHz, ordinary frequency |
| Time | ns |
| Temperature | mK |
| Energy / h | GHz |

The domain layer stays in ordinary GHz. The only `2π` conversion for Hamiltonian assembly is in [`quchip/engine/assembly.py`](quchip/engine/assembly.py), immediately before the solver-facing Hamiltonian is built.

## 2. What `.hamiltonian()` Means

### 2.1 Device Hamiltonians

`BaseDevice.unresolved_hamiltonian()` returns the device's authored static Hamiltonian in its declared local space, in the lab frame, in ordinary GHz.

Examples:

- `DuffingTransmon.unresolved_hamiltonian()` returns `omega * n + (alpha / 2) * n * (n - I)`.
- `Resonator.unresolved_hamiltonian()` returns `omega * n`.

The authored view does not:

- include `2π`
- include a rotating-frame subtraction
- include a drive term
- include explicit time dependence

`BaseDevice.hamiltonian()` resolves the isolated device physics through the engine, with the device's basis policy and the lab frame. For another local frame, pass `frame=` to `device.resolve()`. These queries do not change when you attach the device to a chip. For the coupled system, use `chip.hamiltonian()` or `chip.freq(device)`.

Local level labels refer to the energy order of the isolated Hamiltonian. `chip.bare_state(q=1)` prepares that local excited state, and `result.population(q, 1)` measures its occupation, summed over the other devices. A result keeps the energy vectors that quchip used to build its calculation. The frame generator is the same energy-level index operator in each solver basis.

Numerical device Pauli operators act on the lowest two isolated energy states and vanish on higher
levels: `Z = |0><0| - |1><1|`, `X = |0><1| + |1><0|`, and `Y = -i|0><1| + i|1><0|`. quchip does not
renormalize leakage away. For each energy vector, quchip makes the authored-basis component with the
largest magnitude real and positive, and the first component wins a tie. State preparation,
transitions, and default Pauli observables use the same convention. This convention fixes the phase
locally but does not fix the basis in a degenerate eigenspace.

Derivatives require separated eigenvalues and a stable phase pivot. quchip does not promise a
globally continuous eigenvector phase.

Authored `LocalOps` operators, including `op.sigma_z` in a Hamiltonian declaration, keep their local-space definitions. They do not diagonalize the Hamiltonian they define. Looking up a named observable respects the component's operators and transforms them into the selected solver basis.

### 2.2 Coupling `.interaction_hamiltonian()`

`BaseCoupling.interaction_hamiltonian()` returns the coupling's full two-body operator in the pair subspace, still in the lab frame and in ordinary GHz. It takes no RWA argument, and a coupling defines exactly one interaction. The chip and the engine resolve and apply RWA structurally, not the coupling (§6.1).

For `Capacitive`:

```text
full: g * (a + a†)(b + b†)
```

`interaction_hamiltonian()` always returns this full form, and you never author the RWA form `g * (a†b + ab†)` directly. The RWA form is what remains after the chip masks out, or the engine filters, the bands that change the total excitation.

`a + a†` is the charge-like operator of a `FockSpace` endpoint. On a `ChargeSpace` or
`PhaseGridSpace` endpoint (`ChargeBasisTransmon`, `Fluxonium`), the charge-like operator is `n`. So
the same declaration authors `g * n_a n_b`, or `g * (a + a†) n_b` for a mixed pair
(`EndpointOps.charge`, [`quchip/declarative/ops.py`](quchip/declarative/ops.py)). The two operators
have different matrix elements. For a transmon, `<0|a + a†|1> = 1`, but
`<0|n|1> ≈ (E_J/8E_C)^{1/4}/√2`. One numerical `g` is therefore not one physical coupling across
bases. Match models across bases through dressed quantities (exchange, χ, ZZ), not through `g`.

`ChargeDrive` on a Fock device addresses the quadrature `i(a − a†)`, whereas the coupling charge
operator is `a + a†`. Both operators are charge-like and differ by a phase convention (§2.3).

### 2.3 Chip and sequence Hamiltonians

`Chip.unresolved_hamiltonian()` embeds every authored device Hamiltonian and every full coupling interaction into the total declared Hilbert space. It is the exact static lab-frame expression before local-basis resolution, truncation of retained levels, frame transformation, or RWA.

quchip reconstructs `Chip.hamiltonian()` from `Chip.resolve()`, the frozen `EngineResult` that backends use. Resolution:

1. resolves each authored local space into the selected solver basis and retained dimension
2. multiplies solver-facing operators by `2π`
3. subtracts the chosen frame generator
4. decomposes non-static pieces into excitation-change bands and applies the chip's approximation strategy
5. attaches explicit time-dependent phases where necessary

The returned expression is an inspectable view of those same canonical terms in ordinary GHz, while the solver-facing `EngineResult` keeps the internal `2π` scaling. `QuantumSequence.hamiltonian()` follows the same path and adds the sequence's scheduled drive and crosstalk terms.

These inspection methods return `PhysicsExpr`, the backend-neutral scalar and operator algebra for authored and resolved physics. It keeps declared parameters, matrices, time-dependent scalars, labels, and opaque JAX callables without forcing numerical values. Numerical materialization is explicit through `.matrix()` or backend lowering.

The contracts are:

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

If you set `internal_quality_factor`, the resonator contributes unobserved
photon loss with the authored operator `a` and the rate
`2π * omega / Q_internal` in `1/ns`. Backend lowering forms the Lindblad
operator `sqrt(2π * omega / Q_internal) * a`.

### 3.3 Collapse operators

`Qubit(freq=...)` is an intrinsically two-level device with
`H/h = freq * |1><1|`, with its frequency in GHz. A frame at that frequency
removes free precession. It inherits the T1/T2 channels below and has no
numerical truncation boundary. If higher levels matter, use a transmon model.

Source: [`quchip/devices/base.py`](quchip/devices/base.py)

The standard dissipators are:

- `T1`: relaxation through the device's `lowering_operator()` (the declared `a` by default)
- `T2`: pure dephasing through `sqrt(2*gamma_phi) * n` with `gamma_phi = 1/T2 - 1/(2*T1)`. Because of the factor `2`, the 0–1 coherence decays at `1/(2*T1) + gamma_phi = 1/T2`, so the input `T2` is the resulting coherence time (when `thermal_occupation == 0`). The number operator `n` gives the standard `(m-n)^2` dephasing scaling across higher levels.
- thermal up/down channels when `thermal_occupation` is set

Custom devices select these operators through `lowering_operator()`,
`raising_operator()`, and `number_operator()`, or declare channels
directly. The lifetime interpretation above assumes a unit 0–1 lowering
matrix element and dephasing eigenvalues separated by one. Other
normalizations change this interpretation.

Devices, drives, couplings, and baths author `CollapseChannel` records,
which keep the local operator separate from its non-negative rate in
`1/ns`. The engine projects and embeds the operator and keeps the rate
unchanged. Backend lowering applies `sqrt(rate)` exactly once.

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

The dissipator uses `L_p`. When you schedule the `input` of an external plane,
it uses the same grammar for envelope, phase, carrier, and start time as a
classical drive, with

```text
beta_i(t) = A(t) exp(i theta) exp(-i 2π f t),
|beta_i|^2 = incident photon flux in photons/ns.
```

quchip applies no implicit conjugation or factor of one half. For a general
resolved boundary, the field that arrives at each coupling channel is

```text
c_j(t) = sum_i S_ji beta_i(t),
H_input(t) = i sum_j (c_j^* L_j - c_j L_j^dagger).
```

This Hamiltonian is the canonical solver form. quchip obtains it by composing
the coherent source `W_beta = (I, beta, 0)` with the resolved SLH model. It
then gauges the displaced collapse operators back to the input-free `L`. The
collapse channels therefore stay unchanged: a coherent field never adds a
second damping channel.

`ResolvedSLH` itself stays input-free. quchip keeps each solve's beta programs
on `EngineResult.coherent_inputs` for later reconstruction of the output
field. Classical `ControlEquipment` transforms do not accept field inputs.
Sections for field attenuation, phase, crosstalk, and reference-plane delay
belong to `PortNetwork`.

For the one-port identity case, this reduces to the angular Hamiltonian

```text
H_input = i (beta_p^* L_p - beta_p L_p^dagger).
```

quchip reconstructs transient output observables from that same
solve-bound model:

```text
<b_out,j> = c_j + <L_j>,
<X_theta,j> = Re[exp(-i theta) <b_out,j>],
<b_out,j^dagger b_out,j>
  = |c_j|^2 + 2 Re[c_j^* <L_j>] + <L_j^dagger L_j>.
```

An `output` request for an external plane lowers `L_j` and `L_j^dagger L_j`
into backend expectation operators. `result.output(plane)` returns the
reconstructed complex amplitude and the normally ordered photon flux. Any
quadrature phase is a post-solve projection of the complex amplitude. quchip
evaluates the coherent background from the retained beta programs and does not
insert it as an identity operator. Reference sections shift the incident beta
on an exposure's inbound leg and the complete boundary trace on its outbound
leg. The field is zero before the simulation's initial boundary data can
arrive. The result also keeps both moments at the Markov boundary before the
outbound shift.

For a single resonator, `external_quality_factor` gives
`kappa_p = 2π * freq / Q_external`. Internal resonator loss and each port are
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
effective channel coefficient with magnitude squared `2γ(1 + cos φ)`. It also
adds `+γ sin φ · a†a` to `H`. Here `γ` is angular, in rad/ns, so the Lamb
shift is `+γ sin φ / 2π` GHz. If you enter the propagation phase as
`φ = 2π f τ = ωτ`, with `f` in GHz and `τ` in ns, the result matches the
`Δ = γ sin φ` convention of Kockum et al., Phys. Rev. A 90, 013837 (2014).

`port.side` and `component.port(k)` each select one physical connector and pair
the input and output terminals of that side. `network.link(...)` cables
consecutive sides in both directions, and each cable compiles to two directed
terminal connections. An ideal circulator routes `1 -> 2 -> 3 -> 1`. An
isolator is a circulator whose third side is a hidden vacuum load. Compilation
operates per terminal, so permutation components do not create false cycles.

quchip groups core terminals into strongly connected components and reduces
each connection cycle algebraically with the feedback rule in 3.3.3. This gives
loop-corrected `S`, `L`, structural reachability, and the generated
series/feedback Hamiltonian. Reference sections cannot be in a loop. A concrete
singular `I - M` raises. With traced JAX scattering, quchip cannot detect the
singularity during tracing, and the solve returns non-finite values at that
point.

Concrete scattering must be unitary, and an explicit unitary dilation represents loss. A two-sided
reciprocal attenuator with power transmission `eta` from side 1 to side 2 has amplitude transmission
`sqrt(eta)` in both directions. It couples each direction to one of two hidden vacuum channels with
amplitude `sqrt(1-eta)`. A beam splitter, by contrast, is a directional two-input/two-output device:
`eta` is the power from input k to output k, and its scattering matrix is
`[[sqrt(eta), sqrt(1-eta)], [-sqrt(1-eta), sqrt(eta)]]`. Beam splitters and 90-degree hybrids have
no physical sides. Connect their input and output terminals directly or with `network.cascade(...)`.

Network exposures define the order of the external channels, and their adjacent reference sections
move the incident and reported reference planes. The compiler removes these sections from each leg
before it forms the instantaneous `S`, `L`, and `H`. A section on a reflection line contributes once
inbound and once outbound. A reference section that is not adjacent to an exposure raises.

`network.delay(...)` is a reference-plane shift, not retardation. It changes fields at exposure
planes but adds no memory to the dynamical SLH core, which is Markovian throughout. Enter the
propagation phase between giant-atom coupling points with `network.phase_shift(...)` as above.
Retarded feedback, linewidth-scale variation of `γ(ω)` or of the phase across a resonance, and the
non-Markovian giant-atom regime are out of scope.

Static composition of several quantum ports requires a common rotating-frame
frequency. Different local carriers would make both the collective `L` and
`Im(L2^dagger S2 L1)` explicitly time dependent. Until dynamic collapse
channels exist, quchip rejects that network and directs the model to the lab
frame or a common frame.

`network.filter(...)` adds a two-sided passive reference section with complex
transfer `H(f)`. Its `transfer(frequency, **parameters)` callable takes scalar
or array frequencies in ordinary GHz. This callable stays outside the Markovian
`S`, `L`, and `H`. quchip tracks each keyword parameter at
`network.component.<label>.<name>`. Continuous-wave calculations, including
VNA, small-signal response, spectra, and stationary pumps, apply `H(f)` exactly
on each leg. In particular, `output_spectrum` scales the fluctuation spectrum
at offset `nu` by

```text
|H(f_c + nu)|^2.
```

Transient filtering uses a narrowband carrier approximation, not a
time-domain convolution:

```text
beta_boundary(t) = H(f_carrier) beta_plane(t),
b_out,plane(t) = H(f_c) b_out,boundary(t),
Phi_plane(t) = |H(f_c)|^2 Phi_boundary(t).
```

Here `f_carrier` is the scheduled input carrier and `f_c` is the channel's
rotating-frame carrier. A lab-frame outbound channel has no such carrier, so
requesting its filtered output raises before the solve. A concrete evaluation
with `|H| > 1` also raises, because filter sections are passive. For gain,
use `network.amplifier(...)`.

`network.mode_reflection(...)` adds a two-sided passive reference section. The
square of its per-pass transfer `H(f)` is the reflection `S_r(f)` of a damped
linear mode (section 10.5). A reflection plane therefore gets `S_r(f)`, and a
field that crosses one leg, such as emission from the devices behind it, gets
`H(f)`. `H` is the continuous square root with a nonnegative real part at
`reference_freq`. At that frequency, `H` reproduces to first order the emission
phase `1 - i kappa_e / (2 Delta)` of a device detuned by `Delta` from the mode.
Its parameters are tracked like those of `network.filter(...)`, and the section
serializes.

`network.amplifier(...)` adds a phase-preserving reference section on the
output line with power gain `G` and input-referred symmetrized added noise
`n_add` in quanta. It amplifies side 1 to side 2 by `sqrt(G)` and is
transparent in reverse. The exposure must be beyond side 2 on the outbound leg.
quchip raises if the forward direction would amplify an incident field into the
chip. It also raises if an outbound plane is on side 1. `gain` and
`added_noise` are sweepable, differentiable paths at
`network.component.<label>.gain` and `network.component.<label>.added_noise`.
Amplifier sections serialize normally.

The quantum floor and output-line noise recursion are

```text
n_add >= (1 - 1/G)/2,
N <- |H(f)|^2 N,
N <- G N + G n_add + (G - 1)/2.
```

The last line applies at each amplifier, and at the quantum limit its added
term equals `G - 1`. Two amplifiers in series obey the input-referred Friis
relation `n_total = n1 + n2/G1`.

Continuous-wave means and small-signal entries, including VNA, get a factor
`sqrt(G)`, so `|S|` can exceed one. `output_spectrum` returns the amplified
signal density as `signal_fluctuation_spectrum`, the accumulated density `N` as
`added_noise_spectrum`, and their sum as `total_fluctuation_spectrum`. Its
`signal_coherent_flux`, `signal_incoherent_flux`, and `signal_photon_flux`
fields scale the signal by `G` but exclude amplifier noise. Without a detection
bandwidth, quchip cannot convert this noise to flux. Transient
`result.output(plane)` also scales amplitude by `sqrt(G)` and photon flux by
`G`, with no added-noise term. Normalized `g1` and `g2` through an amplifier
raise, so request them at a plane before the amplifier.

Noise parameters are ordinary tracked attributes. You can set them (or clear them with `None`) at construction **or any time after**. Collapse operators are rebuilt from the current values on each solve, and writes after construction get the same validation as the constructor. Shared/collective dissipation at chip level is in `Bath` ([`quchip/chip/baths.py`](quchip/chip/baths.py)). Attach a bath at construction or later with `chip.add_bath(...)`. Bath rates are Lindblad-ready 1/ns with no assembly `2π`, because that boundary applies only to Hamiltonians. A component's *intrinsic* `2π`, e.g. a resonator's `κ = 2π·f/Q`, is its own physics.

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

`quchip.engine` exposes these rules as `series_product`, `concatenate`, and
`feedback_reduce` on the input-free `ResolvedSLH` that `chip.resolve().slh`
returns. Operands must share one Hilbert space, and joined or closed legs
need empty reference runs. A concrete loop with `1 - S_xy = 0` is singular
and raises. If channel keys collide, concatenation requires `prefixes=`.
Feedback from `x` to a different `y` gives the merged channel the key `x->y`.
Hamiltonian corrections from composition are static terms of network origin.

### 3.4 Authored local spaces and solver bases

Source: [`quchip/devices/spaces.py`](quchip/devices/spaces.py), [`quchip/engine/basis.py`](quchip/engine/basis.py)

Each device authors its Hamiltonian and named operators in a `LocalSpace`. The built-in realizations are `FockSpace`, `ChargeSpace`, `PhaseGridSpace`, and `CustomSpace`. This declared coordinate space is independent of the solver-basis policy.

`Chip(..., basis="native")` keeps each device's authored local coordinate basis and dimension. `basis="eigen"` diagonalizes each device's exact authored local Hamiltonian. It then projects all Hamiltonians, coupling and drive operators, states, observables, and collapse operators into the retained local energy subspace. A device can override the chip-wide policy with its own `basis` and, where necessary, select the retained energy dimension with `projection_levels`.

Resolution records each fixed authored-to-solver transformation in `EngineResult.bases`. Local energy ordering differs from whole-chip dressing: `Chip.dress()` diagonalizes the coupled static chip for analysis, whereas local-basis resolution defines the tensor factors that go through the engine and backends.

Truncation diagnostics sample the component's declared boundary projector at the
solver output times and report its maximum population. quchip transforms native
boundary projectors with the captured authored-to-solver map. Energy projection
adds the projector onto the highest retained energy state. Frame-band
reconstruction returns the physical boundary population, independent of the
readout reference. The sampled scalar traces stay separate from user observables
and state histories, even when no states are saved.

Fock ladders check their highest Fock state, and charge and phase grids check
both edges. An intrinsically finite model declares no native cutoff through
`truncation_boundary() -> None`. quchip reports unknown custom cutoffs as
unavailable. Boundary population is a heuristic, not a bound on the truncation
error, and sampling can miss intermediate excursions. For convergence,
increase the relevant cutoff and compare observables.

### 3.5 Transitions

`device.transition(lower, upper)` returns the Hermitian transition operator
`|lower><upper| + |upper><lower|` in the device's authored coordinates.
`device.transition_frequency(lower, upper)` returns the isolated energy gap in
GHz. The level indices follow the energy ordering of the device's static local
Hamiltonian.

`chip.transition_frequency(target, lower, upper, when=...)` uses the complete
undriven static chip Hamiltonian. It assigns dressed eigenstates to the two bare
product labels and returns their energy difference before frame subtraction or
drive approximation. Unspecified spectators are in level zero. `when` sets
spectator occupations and cannot include `target`.

`chip.freq(target, when=...)` is the concise 0-to-1 form. For a higher target
transition, use explicit levels:

```python
f01 = chip.freq(qubit)
f12 = chip.transition_frequency(qubit, 1, 2)
fr_when_excited = chip.freq(readout, when={qubit: 1})
```

Local basis projection and dressed transition assignment are separate. Basis
projection selects the solver's tensor factors, and dressed assignment
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

The engine resolves that spec into a reference frequency `omega_ref,i` for each device.

### 4.2 What transform the engine is using

The engine assumes the rotating-frame unitary

```text
U(t) = exp(-i 2π t Σ_i omega_ref,i * n_i)
```

The solver Hamiltonian is then

```text
H_rot = U† H_lab U - 2π Σ_i omega_ref,i * n_i
```

Because of that second term, the assembler subtracts `omega_ref,i * n_i` from `H0`.

When `initial_state=None`, the solver receives
`psi_solve(t0) = U(t0)† psi_lab` at `t0 = tlist[0]`, where `psi_lab` is the
lab-frame default that §9 describes. A nonzero rotating-frame start
therefore need not equal the lab-frame vector that `chip.state()` returns.

### 4.3 What `"rotating"` means in practice

`"rotating"` means:

- each device gets its own reference frequency
- an explicit `device.reference_freq` gives that frequency
- when the setting is `None`, the calculation uses the chip's dressed `0 -> 1` transition

This rule defines the frame-subtraction frequencies `omega_ref,i`.

### 4.4 What `"auto"` chooses

`"auto"` is opt-in. The chip default stays `"lab"`, and `"rotating"` keeps the
meaning above. `Chip.resolve(frame="auto", approximation=...)` uses the
requested approximation. With `Exact()` and no tones, it selects the lab
frame, because every retained band is static in the lab frame.

The frame planner chooses one frequency `omega_d` for each device. A
retained coupling band with excitation-change vector `k` is static when

```text
Σ_d k_d omega_d = 0
```

A band of a scheduled drive or port coupling at frequency `f` is static when

```text
Σ_d k_d omega_d = f
```

Assembly evaluates `Σ_d k_d omega_d` by first adding the integer changes of
devices that share a frame frequency. A band whose changes cancel among these
devices is therefore exactly static. Drive bands with equal carriers share one
term, so they do not differ by floating-point residues of approximately 1e-14
GHz.

For example, a two-photon cavity pump at 10.2 GHz sets
`2 omega_cavity = 10.2 GHz` and pins the cavity frame to 5.1 GHz. Network
cascades add coupling constraints. Dispersive and cross-Kerr terms have a zero
charge vector, so they do not constrain the frame.

Drive constraints come from the signals the control equipment delivers after
gain, attenuation, delay, and crosstalk. Each nonzero carrier band constrains
every retained operator band of its destination drive. A crosstalk
destination gets its own tone, and gain changes its weight. A zero-frequency
carrier band adds no constraint.

For a coherent field `beta` that enters a network exposure, each channel that
the field reaches has `c = S beta` and drives

```text
i(c* L - c L†)
```

The operator band's instantaneous amplitude is `|c| |L_band| / 2π` in
ordinary GHz. Equivalently, the frame record stores `|S| |L_band| / 2π` and
the weight integrates `|beta|²`.

When the constraints cannot all hold at once, the planner keeps the consistent
subset with the largest integrated strength. A scheduled tone band has weight
`|h_band|² ∫|envelope|² dt`. Each signal is integrated over its own nonzero
extent in the outer weighting window. That window is `[tlist[0], tlist[-1]]`
when you build a problem, or `[0, last pulse end]` otherwise. A short pulse
deep in a long solve keeps its full energy, but the solve window clips any
portion outside it.

A static coupling band has weight `|h_band|² T`, where `T` is the outer
window's length. An unknown traced window makes the static-coupling weights
unknown. When identical constraints are merged, any unknown contribution makes
the combined weight unknown. Equal or unknown weights fall back to declaration
order. The other bands stay time-dependent, and
`resolved.resolved_frame.plan.residuals` records each one's source, devices,
oscillation frequency, and weight.

The planner fixes free frequencies from `reference_freq` in device
declaration order. An undriven cluster with exchange coupling therefore
rotates together at one member's reference frequency, and an isolated mode
keeps its own. `chip.describe()` and `sequence.describe()` show the selected
frequencies, accepted tones, pins, and residuals. Traced JAX tone frequencies
go through the plan without conversion to Python scalars.

Stationary analyses such as VNA sweeps and steady states use a strict order.
Mandatory cascade constraints come first, then exchange coupling bands with
nonzero charge vectors that sum to zero, and tones last. A conflicting tone
raises the port form of the existing distinct-stationary-tone error if an
accepted tone already addresses the same devices. Otherwise, it raises the
exchange-connected form. Only non-exchange coupling bands, such as
counter-rotating bands retained under `Exact()`, can stay time-dependent, and
they then trigger the existing dynamic-Hamiltonian-terms error.

### 4.5 `reference_freq` — the readout / LO reference

Source: [`quchip/devices/base.py`](quchip/devices/base.py)

`device.reference_freq` returns the authored override in GHz, or `None`. Each new chip calculation resolves `None` to that chip's dressed transition and captures the value in its frame record. An explicit value fixes the frame/readout reference across model changes. If you set it off the transition, a residual detuning `Δ = omega - omega_ref` stays in `H0` and causes idle Ramsey precession.

It is a *frame / readout* reference only and does **not** detune drives. The drive carrier is a separate choice, so a real LO error must also set the drive frequency. The value is in ordinary GHz and is tracked (a change to it invalidates engine caches). It is JAX-traceable / differentiable / sweepable.

## 5. Frame Tracking in the Engine

Source: [`quchip/engine/assembly.py`](quchip/engine/assembly.py)

The engine does not rotate whole expressions symbolically. It tracks phases band-by-band.

### 5.1 Single-device operators

The engine decomposes a local operator into bands with weight

```text
w = col - row
```

In the chosen frame, that band gets phase

```text
exp(-i 2π w * omega_ref * t)
```

With this phase, the engine knows which part of `a + a†`, `i(a - a†)`, or an observable still rotates.

### 5.2 Two-device couplings

The engine decomposes a two-body operator into bands labeled `(delta_a, delta_b)`, where each value is the excitation change on one subsystem.

That band gets phase

```text
exp(-i 2π (delta_a * omega_ref,a + delta_b * omega_ref,b) * t)
```

If that effective frequency is zero, the band stays static in `H0`. Otherwise, it becomes an explicit time-dependent term.

This band decomposition is the frame-tracking mechanism.

### 5.3 Model time dependence and scheduled control

`DeviceModel.time_terms()` and `CouplingModel.time_terms()` return
`TimeDependentTerm` values for physics that exists without a scheduled pulse. Each
term pairs a local operator with a `TimeCoefficient`. The engine projects,
band-decomposes, and frames the term through the same path as other Hamiltonian
terms. Scheduled control stays drive-owned. Its finite-duration envelope and
carrier produce a drive modulation only after `QuantumSequence.schedule()`.
`Envelope.value(local_time)` defines the local complex I/Q shape, and scheduling
owns the pulse start, carrier, and global phase. You can therefore place and
phase-rotate the same shape without changing its physics definition.

Local eigenbasis projection uses the static authored Hamiltonian at the
solve's operating point. The engine projects component-owned
time-dependent terms into that fixed basis. quchip does not construct an
instantaneous moving basis.

## 6. Approximation strategies

Source: [`quchip/approximations.py`](quchip/approximations.py), [`quchip/engine/approximations.py`](quchip/engine/approximations.py)

The chip owns one explicit approximation strategy. `Exact()` keeps every term in the authored finite-dimensional Hamiltonian. `RWA()` applies the engine's first-order structural rotating-wave reduction to static interactions and scheduled drives. Devices, couplings, and drives have no RWA policy of their own.

`Chip.unresolved_hamiltonian()` keeps the authored static interaction. Engine assembly band-decomposes each interaction and applies the selected strategy. The engine reconstructs `Chip.hamiltonian()` from those same canonical terms, so inspection and simulation use one decision path. Under `RWA()`, rejected bands become advisory `DroppedTerm` records with the band's excitation weights, largest matrix-element magnitude, and frame frequency. A retained band with zero frame frequency stays in `H0`, and every other retained band is an explicit time-dependent term.

### 6.1 Static operator bands

For `Capacitive`:

```text
full: g * (a + a†)(b + b†)
     = g * (a†b + ab†) + g * (ab + a†b†)
```

- `a†b + ab†` has total excitation weight zero and stays under `RWA()`
- `ab + a†b†` has total excitation weight two and `RWA()` removes it

`interaction_hamiltonian()` always returns the complete authored form, and `RWA()` reconstructs its retained bands in the engine. A coupling does not supply an alternative RWA operator or retention hook. The mask depends only on integer band offsets, so it stays concrete when operator parameters are JAX tracers.

The engine makes the static/dynamic decision for each band, not for each coupling. In a *shared* frame (multiple devices detuned to a common reference), the coupling's counter-rotating band can have a nonzero carrier even when its co-rotating band is frame-static. The per-band fold evaluates each band's carrier independently, so a shared frame never suppresses the counter-rotating band's true rotation.

With `Exact()`, the engine drops no band, and every non-static band stays at its own frame frequency.

### 6.2 Driven operator bands

For a single-tone drive channel, the engine forms the real lab-frame field

```text
Re[s(t) * exp(-i 2π f_drive t)]
```

and combines it with the operator bands.

`Exact()` keeps both co-rotating and counter-rotating pieces.

`RWA()` keeps the conventional partner for each excitation band and drops its counter-rotating partner.

Flux drives couple through the diagonal `n`, so RWA has no raising/lowering split to remove. The engine uses them as direct real-valued modulation channels.

## 7. Counter-Rotating Terms

Counter-rotating terms occur when the operator changes total excitation in the same direction as the classical or frame rotation, and does not cancel it.

Concrete examples:

- In full capacitive coupling, `ab` and `a†b†` are counter-rotating.
- In a single-tone drive, the real field's fast partner is counter-rotating relative to the chosen transition band.

In the rotating frame of two detuned modes with frequencies `omega_a` and `omega_b`:

- exchange terms rotate at approximately `|omega_a - omega_b|`
- counter-rotating terms rotate at approximately `omega_a + omega_b`

RWA usually drops these terms because they are much faster and average out.

Choose `RWA()` to remove structural first-order counter-rotating bands, or `Exact()` to keep every term explicitly.

## 8. Observables and Demodulation

Source: [`quchip/engine/observables.py`](quchip/engine/observables.py)

The engine decomposes dict-form `e_ops` into the same excitation bands as the frame logic. After the solver returns, the engine recombines them with the per-device demodulation frequencies in `ResolvedFrame.demod_freqs = omega_ref - omega_frame`.

`result.expect` is therefore a **co-rotating readout**. Observables are always reported in each
device's `reference_freq` frame, independent of the solver's integration frame. Transverse
observables (`<a>`, `<sigma_x>`) come back as the non-oscillatory demodulated envelope that a lab
readout produces. This envelope is slow, and it turns at `Δ = omega - omega_ref` when the reference
is detuned. Diagonal observables (populations) are frame-invariant.

In the default `"rotating"` mode, the integration frame *is* the reference frame, so demodulation
is a no-op and `result.expect` equals `Tr(O·rho)` on the states that `result.states` returns. The
raw, un-demodulated band sum (the observable in the integration frame) stays available on each
`ObservableTrace` as `.raw`.

### 8.1 Stationary solves and scattering

S-parameters follow the engineering `e^{+jωt}` convention, where `j = −i`.
`VNA.sweep()`, `finite_power()`, and `measure()` conjugate the complete
physics response at the instrument boundary, including network propagation.
VNA probe and pump amplitudes use this convention, and the engine conjugates
them on entry. Internal Hamiltonians, network declarations, mode
observables, and the engine equations below keep `e^{-iωt}`. A one-port
resonator therefore reports `1 - κe/(κ/2 + j 2π(f-f0))`, and a positive
cable delay adds `exp(-j 2π f τ)` for each traversal.

`Chip.steadystate()` solves `L(rho_ss) = 0` together with `Tr(rho_ss) = 1`.
It requires a static resolved Hamiltonian and a unique normalized stationary
state. `VNA.sweep()` adds continuous-wave port terms in their stationary tone
frames and returns the complete scattering matrix between the selected
planes. At each frequency, the stationary route solves one pumped operating
point with one shifted-Liouvillian factorization for all input columns. The
passive-linear route uses one multi-right-hand-side mode-space solve.
Small-signal scattering differentiates the output mean around the fixed-tone
state,

```text
S_ji(f) = d <b_out,j> / d beta_in,i  at beta_probe -> 0.
```

Around a phase-sensitive operating point, the full response is
`delta <b_out> = S delta beta + T conj(delta beta)`. `result.conjugate_matrix` stores
`T` with the same `[..., output, input]` layout as `result.matrix`, and
`result.t(output, input)` selects one entry. The stationary route gets `S` and `T`
from the same shifted-Liouvillian factorization. The passive-linear route reports
zero for `T`.

The direct term comes from the resolved scalar `S`. The system term comes from
the stationary response of `L` under the same coherent-input Hamiltonian.
`VNA.finite_power()` instead adds a probe of amplitude `beta` at one selected
input. It solves the stationary Liouvillian in the probe frame at each grid
point and reports the mean field at every selected plane,

```text
<b_out,j> = H_out,j(f) [sum_i S_ji beta_boundary,i + <L_j>].
```

This path uses the same reference-plane and hidden-channel bookkeeping as the
small-signal response, but it has no mode-space shortcut. Every selected
plane must resolve at the probe frequency. The ratio `<b_out,j>/beta` tends
to the corresponding `S_ji` as `beta -> 0` if no fixed pump leaves a coherent
mean at that plane and carrier. At finite `beta`, the ratio is a stationary
mean-field response, not a small-signal S-parameter, and does not describe
sweep-rate hysteresis or metastable branches.

For a pump-free passive-linear model, the same authored expressions also admit
the mode-space form

```text
d a/dt = A a + B b_in,          b_out = C a + S b_in,
A = -i Omega - C^dagger C / 2,  B = -C^dagger S.
```

The engine accepts this route only under two conditions. The retained
Hamiltonian must be static, quadratic, and number conserving, and every
collapse operator must be linear in the mode lowering operators. The engine
applies the Hamiltonian `2π` conversion at the assembly boundary, including
the series-product Hamiltonian that the `PortNetwork` generates. It computes
the inbound and outbound transfer factor `exp(+i 2π f τ)` at each frequency
for each reference leg. It sends the compact matrices to the selected
backend, which evaluates the undecorated Markov response

```text
S_out,in(f) = S_out,in + C_out (-i 2π f I - A)^(-1) B_in.
```

The engine then applies the inbound and outbound factors to that response.

`VNA.sweep()` extends this form to pump-free models that conserve the total
energy-level index `N` but are not harmonic, e.g. Duffing transmons, pure
dephasing, or reduced chips with retained terms. The resolved static Hamiltonian
must conserve `N`. The approximation must keep only zero-total bands, as `RWA()`
does, and retained terms must declare conservation. Every port must lower `N` by
one. A cascade-generated term `Im(L2^dagger S2 L1)`, as on a shared feedline,
then also conserves `N`. Every other channel must lower `N` by one or conserve it.
Every input must be vacuum. The vacuum is then stationary, and to first order in
the probe the coherences `|1_j><0|` stay in the one-excitation block. The engine
projects the lab-frame model onto the vacuum `|0>` and the states `|1_j>` that
raise one device to its first excited level:

```text
Omega = H_1 - <0|H|0> I,     C_k = <0|L_k|1>,     c_k = <0|L_k|0>,
A = -i Omega - sum_k (L_k^dagger L_k)_1 / 2 + sum_k [c_k^* (L_k)_1 - |c_k|^2 / 2],
```

where `X_1 = <1|X|1>` is the one-excitation block. A lowering channel has
`(L^dagger L)_1 = C^dagger C` and `c = (L)_1 = 0`, which gives the harmonic `A`
again. The response then follows from the same mode-space formula. It is exact for
an infinitesimal probe, independent of anharmonicities, cross-Kerr terms and
cutoffs, and its size is the number of devices. VNA diagnostics name the route
`"vacuum_response"`. Finite-power and noisy measurements keep the harmonic
condition above, because only a harmonic model responds linearly at finite
amplitude.

Nonlinear models outside these conditions, and pumped, active, dynamic, or opaque
operator models, keep the stationary-Liouvillian route. This selection is
structural and independent of the numerical value of a traced parameter. Active
local terms also keep the general route, because a weight-only RWA does not show
if a local parametric term is off resonance in its authored frame.

`VNA.sweep()` and `VNA.finite_power()` cover small-signal scattering and
stationary finite-power mean fields, respectively. Ring-up, ring-down, wave
packets, and other time-resolved fields require a scheduled external-plane
input in a `QuantumSequence`. Fixed finite pumps stay valid VNA operating-point
fields. The engine supplies canonical sources and observables for stationary
response, spectrum, and correlation queries, and each backend constructs and
solves its own native Liouvillian. A stationary state and its next response or
regression query share that preparation. Public results do not keep the native
generator.

Uniqueness checks and residuals run with the solve. Optional condition numbers
and positivity checks run when you access them, and they use captured inputs.
Reading VNA diagnostics such as `solver` or `residual` does not evaluate the
other entries. Converting a diagnostic mapping to a dictionary requests all its
values. A requested stationary condition number rebuilds the native generator
but does not solve for the state again. Concrete diagnostic values are cached,
but traced values are not. QuTiP keeps its `diagnostic_max_dimension` limit. A
skipped rank or condition diagnostic is `None`, not a successful check.

If frame and approximation resolution leave dynamic terms, the stationary
APIs raise. Periodic/Floquet stationary states are not implemented.

### 8.2 Captured noisy VNA measurements

`VNA.measure(frequencies, amplitudes, input=..., outputs=...)` prepares one
stationary operating point for each probe/sweep coordinate and captures means and
normally ordered IQ cross-spectra. Receiver integration, calibration, and
Gaussian draws act on these captured arrays, and never call a stationary solver
or consult a later mutable chip. `measurement.parameters` records the numerical
model parameters in flattened sweep order. `noise_frequencies` records the stored
offset grid. The finite-power ratio is the output mean divided by the probe
amplitude, not the small-signal derivative around a separate pump.

At capture, the instrument conversion sends `I -> I`, `Q -> -Q` on both
covariance axes. Spectral covariances are also complex conjugated, and they keep
the physical sideband labels `f_carrier + offset`. Receiver transfer functions,
calibration, and sampled IQ fields use the engineering convention. `VNA.g1()`
uses the same field convention, including cross-port phases. Intensity
correlations and scalar noise power are invariant under the conversion.

For eligible passive harmonic models, `measure()` uses the same compact
mode-space lowering and backend response solver as `sweep()`. With response
matrix T(f) and input occupations n, the normal output spectrum at the Markov
boundary is `T(f) diag(n) T(f)†`. Subtracting `S diag(n) S†` gives the
device-generated excess to the shared downstream propagation, which adds the
direct sources once. The result is the exact stationary Gaussian field
solution, including thermal fluctuations, without a Fock-space truncation. All
modes must decay.

Concrete acquisitions check stability. Traced paths keep the usual
host-validation limitation. Thermal device collapse declarations, nonlinear or
active terms, fixed pump configurations, branched output graphs, and explicit
solver options keep the operator-space acquisition. Passing `options={}`
requests that general route for cross-checks.

Measurements also capture internal Fock-mode observables. `mode_amplitude(r)`
returns `<a_r>` in the stationary frame that `mode_frequency(r)` reports, and
`photon_number(r)` returns `<a_r† a_r>`. Both follow the measurement sweep
axes and keep the full declared input wiring. The compact backend solves
`A N + N A† + B diag(n) B† = 0` for the centered normal covariance
`N_ij = <delta a_j† delta a_i>`, then adds its diagonal to the coherent
occupation `|<a_r>|²`. This covariance is independent of the output spectral
grid.

The general path evaluates the authored `a` and `n` operators in the resolved
basis against the solved reduced density matrix. It keeps nonlinear and
active physics and the declared truncation. These queries use captured
arrays, and receiver processing does not change internal occupation.

An attenuator, isolator load, or `network.termination()` can declare
`thermal_occupation=n`, a finite, non-negative mean thermal population in
quanta. Vacuum stays the default. Amplifiers require input-referred
symmetrized `added_noise` in quanta, subject to the phase-preserving
quantum floor. Both noise values are constant across the modeled band,
including frequency sweeps. quchip does not infer a temperature or
reference frequency. Attenuators accept positive `loss_db` instead of
`eta`, and amplifiers accept `gain_db` instead of linear power gain.
Authored parameters stay the rebinding and serialization paths.

For a full unitary scattering matrix S, input j couples through
`K_j = (S† L)_j`. In addition to vacuum `sum_i D[L_i]`, its thermal population
adds `n_j D[K_j] + n_j D[K_j†]`. Thermal input is therefore part of the
stationary and transient quantum dynamics, not a second independent device
bath. SLH composition keeps the surviving input's state and rejects an
independent thermal declaration on an input that is connected away.

Arbitrary passive nonideal components use `network.component()` with unitary
scattering and explicit dissipative terminals connected to declared loads.
Insertion loss and isolation numbers alone do not define this matrix.

Normal output spectra include the direct term `S diag(n) S†` and the input-system
interference in the regression source
`B_i = (L_i-<L_i>) rho + sum_k (S diag(n) S†)_ik [L_k,rho]`. This interference
prevents double counting of fluorescence on top of an incident thermal field at
equilibrium. Real IQ sources constructed from B keep normal, anomalous, and
cross-output second moments. `output_spectrum()` selects the scalar normal spectrum
from the same calculation as `measure()`.

The internal Fourier convention is `integral exp(+i 2π offset τ) <δb†(0) δb(τ)> dτ`,
so a mode above the carrier peaks at positive offset. This corrects the mirrored
detuned-fluorescence spectrum in 0.3.0. The equivalent engineering spectrum uses
`exp(-j 2π offset τ)` and the conjugated field correlation, and it keeps the same
physical frequency axis.

Physical source budgets separate directly propagated fields from
`device.correlations`, which includes nonlinear response and interference and
so is not necessarily positive or independently sampleable. A matched
absorptive filter declares `thermal_occupation` and emits `(1-|H(f)|²)n` in
each direction. A scalar H(f) without that declaration does not imply an
absorptive thermal model.

Colored emission can propagate to external outputs, but quchip rejects colored
noise that feeds a quantum coupling, because a colored reservoir needs an
explicit dynamical model. Source color follows propagation order, so a vacuum
filter before occupied attenuators does not color their emission. Source
backaction follows `S†L`, so fields mixed only after the device can carry
filtered thermal noise without heating the device.

`measurement.noise_spectrum(output)` recovers the normally ordered scalar
spectrum from the captured IQ matrix. It keeps the upper/lower sideband
asymmetry and excludes the coherent carrier and the final receiver vacuum.
`unit="W/Hz"` multiplies by `h f_absolute`. `unit="dBm/Hz"` reports its power
ratio to 1 mW/Hz. Power conversions require positive absolute sideband
frequencies. Receiver source budgets stay integrated IQ covariances in
photons/ns, whose trace is the complex-field variance, not a spectral power
density.

Acyclic output networks can put an amplifier before a splitter or between
passive components. The compiler keeps a unitary Markovian boundary and
captures a separate directed field map with source cross-spectra. These
sections cannot feed quantum couplings or instantaneous feedback loops.
Amplifiers keep the output-line convention above. `added_noise` never
implies reverse HEMT emission. An exposed reference section keeps its
separate inbound and outbound traversals.

Branched reference networks currently support stationary fields. Transient
output observables and direct SLH composition reject them explicitly.
Compose their physical PortNetwork before you resolve it. Their VNA response
uses the general stationary solver.

`IQReceiver(integration_time=T)` applies a normalized boxcar with frequency
weight `sinc(offset*T)²`. For `b=I+iQ`, ideal heterodyne detection adds one
complex vacuum quantum at the final plane, or `1/2` on each IQ diagonal.
Flat normally ordered noise N therefore gives `Var(I)=Var(Q)=(N+1)/(2T)`.
Joint outputs keep their complex cross-spectrum and relative delays.
Independent detector vacuum is added once for each output. Calibration
multiplies the mean and transforms both covariance axes. Zero probe
amplitude leaves field statistics defined and ratios undefined.

An optional receiver `transfer(offset)` is a digital complex amplitude
response that also scales the mean by its DC value. For an ordinary boxcar,
white noise is integrated analytically. Colored terms and digital filters use
the captured grid. The receiver compares full and coarsened quadrature and
checks spectral edges. These local checks do not prove that an arbitrary
spectrum has no unsampled feature. For unsupported bandwidths or integration
times, use a wider or finer capture. Concrete validation must run outside JAX
tracing. Deterministic integration and keyed reparameterized draws stay
differentiable on fixed shapes.

Gaussian samples reproduce the captured second moments. They do not supply
higher-order non-Gaussian photon statistics or continuous correlated records.
Normalized `g1` and `g2` for network thermal fields require a detection
bandwidth, and the unfiltered correlation API rejects them.


## 9. Dressing

Sources: [`quchip/chip/chip.py`](quchip/chip/chip.py), [`quchip/chip/analysis.py`](quchip/chip/analysis.py)

`Chip.dress()` diagonalizes the static lab-frame Hamiltonian that `chip.approximation` keeps, assigns bare product states to dressed eigenstates by overlap, and stores a `DressedResult` with:

- eigenvalues and lazily materialized eigenstates
- bare-to-dressed state assignments and the assigned eigenvalue for each bare label
- assignment overlaps and labels below the requested overlap threshold
- the dressed eigenvector matrix used by dressed-basis analysis

`Chip.freq()` evaluates dressed `0 -> 1` frequencies through the traceable array-labeling cache, and `DressedResult` does not store them. Frequencies, states, Kerr shifts and drive matrix elements all use the chip's approximation, so `RWA(keep_bands=...)` applies here too. Choose `Exact()` to include counter-rotating bands and their Bloch–Siegert shifts.

`Chip.dress()` is always intrinsic static lab-frame analysis and is not part of
the runtime frame transform. A resolved `EngineResult` also provides `dress()`,
which diagonalizes that snapshot's selected frame and approximation. If the
snapshot has dynamic Hamiltonian terms, `dress(at_time=...)` is required, and
it evaluates their signal programs at that instant. The result is an
instantaneous eigensystem, not Floquet or cycle-averaged analysis.

When `initial_state=None`, a solve uses the eigenstate assigned to the
all-ground label of the undriven static lab-frame Hamiltonian that its
approximation keeps. The state is phase-fixed to have a real, nonnegative
overlap on the bare product before the runtime frame transform (§4.2). With
ordinary couplings, the default RWA leaves the bare product unchanged. When
the solve uses the chip's approximation, it starts from the same physical
state as `chip.state()` for the all-ground label before the frame transform.
Retained bands, effective terms, or network Hamiltonian terms that couple the
vacuum can dress an RWA default.

### 9.1 Dressed drive matrix elements

Sources: [`quchip/chip/analysis.py`](quchip/chip/analysis.py), [`quchip/control/equipment.py`](quchip/control/equipment.py)

For a drive line `j` with local Hamiltonian operator `D_j`, quchip defines the dressed matrix element

```text
m_j^(fi) = <f~|D_j|i~>
```

with the **final** dressed state as the matrix row and the **initial** dressed state as the matrix
column, so `Chip.drive_matrix_elements((initial, final))[j]` reads `[final, initial]` from
`U† D_j U`. The device shorthand `chip.drive_matrix_elements(q)` selects the dressed transition from
the all-ground state to the state labeled by one excitation in `q`. Explicit
`(initial_mapping, final_mapping)` arguments select arbitrary transitions. Before evaluating the
matrix element, quchip phase-fixes every dressed eigenvector so that its overlap with its assigned
bare state is real and nonnegative. This removes backend-dependent eigenvector signs from
comparisons between conditioned transitions, e.g. the sum and difference for the weak-drive `IX` and
`ZX` coefficients.

`drive_matrix_elements` evaluates the physical drive operators and does not apply the signal chain.
`ControlEquipment.crosstalk_matrix()` represents declared control-line mixing separately. In this
matrix, column `j` is the source line, row `l` is the victim line, and each entry has an amplitude,
phase, and delay. You can combine the returned matrix elements with those declared line phasors in
a selected weak-drive effective-Hamiltonian model. The two pieces stay separate to distinguish the
dressed quantum response from microwave-path mixing.

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

the control-conditioned off-diagonal entries are `(IX + ZX)/2` and `(IX - ZX)/2`. Therefore a signal
amplitude `Omega` that multiplies `D_c` gives

```text
IX / Omega = m_0 + m_1,
ZX / Omega = m_0 - m_1.
```

`analyze_cr_susceptibility` reports these complex coefficients per unit amplitude and does not
select a pulse or run time evolution. A drive phase can rotate the common complex quadrature,
after which `abs(ZX)` is the maximum useful linear-response rate. The projection remains a
weak-drive statement and excludes strong-drive Stark shifts, pulse-bandwidth leakage, and
echo/cancellation calibration.

This convention follows the effective-Hamiltonian decompositions of Magesan and Gambetta, Phys. Rev. A 101,
052308 (2020), and Malekakhlagh, Magesan, and McKay, Phys. Rev. A 102, 042605 (2020).

### 9.3 Dressed Kerr matrix

`Chip.kerr_matrix()` evaluates one labeled eigensystem and returns a symmetric
matrix in `chip.devices` order. For distinct devices,

```text
K[i,j] = E(1_i,1_j) - E(1_i) - E(1_j) + E(0),
```

the same full-pull convention as `Chip.dispersive_shift(i, j)` and the
static-ZZ coefficient for two qubits. On the diagonal,

```text
K[i,i] = E(2_i) - 2 E(1_i) + E(0),
```

which is `Chip.dressed_anharmonicity(i)`. A device with fewer than three
resolved levels has `NaN` on the diagonal, but its defined off-diagonal
entries stay available. Every entry comes from the complete dressed chip,
not from a matching authored edge. For example, an isolated `KerrCavity`
with `H = omega*n - K*n*(n-1)` has `K[i,i] = -2*K`.

## 10. Adiabatic Elimination and Dispersive Readout

Sources: [`quchip/chip/transformations/`](quchip/chip/transformations/), [`quchip/analysis/dispersive_readout.py`](quchip/analysis/dispersive_readout.py)

`eliminate(chip, target, method="sw"|"exact")` performs model reduction dispatched on the target. A
device target removes a far-detuned mode. Both routes keep their complete calculated Hamiltonian and
each transformed channel from the removed mode and couplings. `chip.effective_terms` carries the
matrix correction beyond the reported Lamb shifts and mediated exchange, including higher-level
corrections. Intrinsic survivor noise stays separate from inherited loss, and a common bus channel
stays collective. Scalar readout quantities such as `chi`, `kappa`, and the first-transition Purcell
rate stay available in `effective_params`.

A coupling target keeps both endpoints and removes the selected edge. Its isolated pair sets a
coordinate change that applies to the entire chip Hamiltonian, including parallel and spectator
interactions. The exact route is a full unitary transformation, and SW keeps terms through second
order in interactions. The retained correction includes per-level shifts, and the authored endpoint
parameters stay unchanged. Surviving component channels follow a captured operator projection, but
their rates stay component-owned. Removed-component channels already use the retained coordinates.
Surviving controls and collective or thermal baths follow the captured coordinate changes. Controls
that target removed components, and retargeted ports, require explicit conversion rules.

An effective-terms target keeps every device and edge. It diagonalizes the selected `EffectiveTerms`
exactly, together with the local Hamiltonians of the devices they act on. Only `method="exact"` is
implemented, because a second-order expansion fails for a strong nonlinearity such as a junction
cosine. Dressed states take the bare label of largest overlap. The rotation acts on the entire chip,
so dressed queries on the reduced chip return the source spectrum at the chip's approximation
(§10.6). `effective_params` reports each device's `freq_after`, `lamb_shift`, `anharmonicity` and
full-pull `cross_kerr`. Every retained contribution states its approximation in
`EffectiveTerms.notes`, including the notes of earlier contributions it absorbs.
`chip.physics_notes()` reports them under `effective:<label>`. Sources for the reduction math:
[`quchip/chip/sw.py`](quchip/chip/sw.py).

### 10.1 The χ convention and related quantities

```text
chi ≡ chi_pull ≡ f_r(qubit in |1>) − f_r(qubit in |0>)     [GHz]
```

the *full* resonator pull per qubit excitation. This is **2×** the σ_z-convention χ of `H_disp = (omega_r + chi_sigma_z * sigma_z) * a†a` that most textbooks use. Three related quantities use different conventions:

- `eliminate(...).effective_params[q]["chi"]`: χ_pull as defined above, calculated *numerically* from the pre-elimination dressed spectrum. It is identical to `Chip.dispersive_shift(r, q)`: `E(1,1) − E(1,0) − E(0,1) + E(0,0)`, with one shared diagonalization. It is exact and device-agnostic, so it works for any survivor type, not only Duffing transmons. quchip evaluates and caches the entry on first access, so the diagonalization occurs only when you read `chi`.
- `fit_a_dress` constraints use signed full `cross_kerr` (including its `chi` alias). To migrate a pre-0.3 `coupling_targets={edge: "chi"}` half-pull target, use `constraints={edge: {"cross_kerr": 2 * old_target}}`. A previous `zz` target already uses full cross-Kerr and keeps its value.
- `Chip.dispersive_shift(a, b)` (alias `static_zz`): the general two-mode cross-Kerr `E(1,1) − E(1,0) − E(0,1) + E(0,0)`. For a qubit–resonator pair, this *is* χ_pull (quchip calculates the `chi` entry exactly this way). Between two qubits, the same expression is the static-ZZ ζ. Do not read a qubit–qubit `dispersive_shift` as a readout χ.

Analytic cross-checks (2nd-order dispersive): two-level `chi = 2g^2/Delta`; Duffing transmon `chi = 2g^2*alpha/(Delta*(Delta+alpha))` with `Delta = f_q − f_r` (Koch et al., PRA 76, 042319, §IV). Critical photon number `n_crit = Delta^2/(4g^2)`.

`effective_params[q]["kappa"]` is the eliminated mode's total intrinsic downward decay rate, in 1/ns, as `intrinsic_decay_rate()` returns it. For a resonator, it includes `2π*f_r/Q_internal` when `internal_quality_factor` is set. It also includes the inherited thermal-emission rate when `T1` or `thermal_occupation` is set. This rate is `(nbar + 1)/T1` with `T1`, or `nbar + 1` when only `thermal_occupation` is present. The reported value is `0.0` only when none of these intrinsic lowering channels is configured. An external default port on a linear resonator is transformed separately as §10.5 describes and is not folded into survivor `T1`, which prevents double-counting its Purcell channel. Bridge legs report `chi = 0.0` because bus/coupler modes are not readout modes, and their dressed pull would double-count the mediated exchange.

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

with `eps = sqrt(nbar_0*((kappa/2)^2 + delta_0^2))` when the drive is given as a target photon number, and the optional strong-drive collapse `chi_eff = chi/(1 + nbar_0/n_crit)`. In the small-χ limit, `Gamma_m → 8*chi_sigma_z,ang^2*nbar/kappa` with `chi_sigma_z,ang = π*chi_pull` in rad/ns (Gambetta et al., PRA 74, 042318; Krantz et al., APR 6, 021318, §V).

The internal `2π` factors here are local physics conversions at the module's public boundary and do not change the engine's single Hamiltonian-assembly `2π` (§1). Declared approximations (carried in the result's `notes`): steady state only (no ring-up transient), linear resonator, 2nd-order dispersive, no measurement-induced qubit T1.

### 10.3 The Schrieffer-Wolff route (`method="sw"`)

The eliminated mode's occupation partitions the chip's bare Hamiltonian, `H = H0 + V` (GHz, before `2π`, under the selected approximation). `P` = the mode in `|0>`, and `Q` = everything else. The generator solves the Sylvester condition on the cross blocks,

```text
S_ij = V_ij / (E_i − E_j)        (i, j straddling P/Q; E = diag H)
H_eff = P (H + (1/2)[S, V]) P
```

(Bravyi, DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011), 2nd order). Nested `where` guards handle
the division, so an exactly degenerate cross pair with no matrix element contributes zero with a
*finite gradient*. A single `where` would propagate a `NaN` backward through the unselected branch.
A zero gap with nonzero P/Q coupling is singular and raises an error, and under tracing quchip also
marks numerical outputs as invalid. The working-precision threshold only selects diagnostic entries
and never changes a nonzero coupling into zero. Exact reduction does not construct an unused SW
generator. Survivor parameters come from indexing `H_eff`: `freq_after(s) = E(1_s) − E(0)`, and the
pair exchange is the `<1_a|H_eff|1_b>` element. Authored direct edges stay unchanged. A separate
mediated edge, named by `effective_params["exchange"]["coupling"]`, carries the real exchange matrix
element of `H_eff − P H P`.

The complete retained correction carries all remaining matrix elements. Sequential shifts and
detunings use the incoming Hamiltonian's diagonal, including earlier retained corrections. With `J`,
the bridge reduction records its linearization

```text
dJ/domega_c = (g_a*g_b/2)(1/Delta_a^2 + 1/Delta_b^2)
```

(the weight that the flux-drive retarget rule uses, §11). Per-element virtual-state attribution (`pathways`) is `(1/2) V_ik V_kj (1/(E_i−E_k) + 1/(E_j−E_k))` summed over intermediate `|k>`, with the same guarded denominator.

### 10.4 The exact route (`method="exact"`)

The exact route fully diagonalizes, once, the Hamiltonian that `chip.approximation` keeps. `method="exact"` selects the reduction algorithm and does not restore discarded bands. Parameters are read from the *labeled* dressed spectrum (`label_eigensystem`, §9), so kept-block energies are exact to all orders in that Hamiltonian, as residual ZZ requires:

```text
zz(a, b) = E(1,1) − E(1,0) − E(0,1) + E(0,0)        (≡ Chip.dispersive_shift)
```

The complete retained Hamiltonian and its pair exchange are read through the symmetrically
(Löwdin-)orthonormalized subspace projection `S^(−1/2) (W E W†) S^(−1/2)`, where `W` is the
overlap block and `S = W W†`. This des-Cloizeaux effective Hamiltonian has a spectrum equal to the
labeled energies exactly. Its basis is not the canonical SW rotation, so off-diagonal reads agree
with `method="sw"` only through 2nd order.

When `chip.approximation` keeps only bands of zero total weight, as `RWA()` does, the static model
conserves the total energy-level index. The exact route then diagonalizes each total-excitation
sector separately, so its retained Hamiltonian, jump operators and coordinate map have exact zeros
between sectors, with no round-off from near-degeneracies across sectors. This requires earlier
retained terms to declare the same conservation. Each port pair that generates a cascade
Hamiltonian, as on a shared feedline, must lower the index by one at both ports. Otherwise, the
route diagonalizes the full matrix.

`method="exact"` validates the ground, the single excitations of touching survivors, and their pair
excitations. Each diagnostic label needs a distinct majority dressed eigenstate, and the full
retained overlap Gram matrix must permit stable orthonormalization. These checks apply in eager,
compiled and gradient-only execution. In that regime, near-degenerate dressed states straddle the
bare labels, and quantities assigned to a single label are not well defined. Use `method="sw"` or
shift the operating point.

### 10.5 Collapse transforms and validity metrics

The Hamiltonian's rotation carries the eliminated mode's jump operator into the reduced frame:

```text
c_eff = P (c + [S, c]) P            (sw — leading transformed mode jump)
c_eff = B† c B                      (exact — B is the retained Löwdin embedding)
```

For the exact route, `B = V_selected (S^(-1/2) W)†` uses the same selected eigenvectors and overlap matrix as the retained Hamiltonian. It is isometric and independent of arbitrary eigenvector phases.

For intrinsic mode loss, the inherited (Purcell) diagnostic sums
`rate * |<0|L_eff|1_survivor>|^2` over the eliminated device's declared channels.
`kappa` similarly sums its isolated 1-to-0 rates, which for a unit lowering channel
gives `kappa * |amplitude|^2`. Both models keep the complete transformed mode jump
matrix and its own rate, including separate thermal emission and absorption channels.
It does not replace a collective jump by independent T1 channels. `EffectiveTerms` are
captured in authored coordinates and use the ordinary basis and frame compiler without
a second band-removal approximation. A static collective jump requires one removable
global phase in the selected frame. Unequal band phases require a compatible common
frame or the lab frame.

Reducing an excitation-conserving model records each retained channel's total level
change in `EffectiveTerms.excitation_changes`. A projected surviving operator keeps
its authored operator's change. Ports follow the same rule: a port on a surviving mode
keeps the change of its target's operator, built at compile time. A transformed port
operator declares the change of the port it replaces. Band decomposition treats every
other total change as a structural zero, so this check gives the same answer for
concrete values and under `jax.grad` or `jax.jit`.

Consider an external default port on a declared harmonic Fock mode. The complete
`c_eff` matrix becomes that port's joint operator on every survivor, in their
authored coordinates. The port's rate, phase, scalar scattering and existing
reference sections are kept. The eliminated mode's own reflection,

```text
S_r(f) = ((kappa_i - kappa_e)/2 - i Omega) / ((kappa_e + kappa_i)/2 - i Omega),
Omega = 2π (f - f_mode),
```

is not part of `c_eff`. It is kept as a `network.mode_reflection(...)` section
at the core end of the port's plane, so a reflection sweep obeys
`S_full(f) ≈ S_r(f) S_reduced(f)`. The remaining difference is the frequency
dependence of the Purcell coupling across the sweep, of order
`(g/Delta)^2 kappa_e/Delta`. `kappa_e` is the port rate. `kappa_i` is the
eliminated mode's internal damping of `<a>`, read from its own channels.
Lowering channels add their rate, raising channels subtract it, and pure
dephasing adds it. The section's internal-loss bath is vacuum.

A port that an earlier elimination transformed already acts on every survivor
and reaches a later eliminated mode only through that dressing. The later
reduction transforms it like an inherited channel and gives it no section, so
you can eliminate both readout modes of a chip in either order. Like an
inherited channel, it drops its direct scattering through the later mode, of
order `rate |<0|L|1>|^2 / Delta`. Rather than approximate or double-count them,
quchip rejects custom or collective boundary operators, projected survivors,
and a mode with several ports. It also rejects a plane that also carries other
fields, a nonlinear eliminated boundary target, and ports that participate in a
cascade-generated Hamiltonian.

The result's `notes` record that the projection is exact for the
*spectrum* but approximate for *dissipation*, because the discarded
`Q`-block dynamics also dephase and decay. For each eliminated coupling,
`validity` reports `g_over_delta` (2nd-order smallness; `is_valid` gates
at `< 0.1`) and `min_block_gap`, the smallest bare-energy gap that the
Sylvester generator crossed. A small gap with a nonzero matrix element is
the failure mode of the perturbative expansion, even when every `g/Delta`
is small.

### 10.6 State and observable maps

Device elimination leaves the authored frequencies and local bases of surviving devices unchanged, and the retained Hamiltonian correction owns their shifts. `effective_params` reports the reduction's transition diagnostics. Use `chip.freq(...)` for the coupled transitions of a reduced chip. To reduce another operating point, rebind the source model and eliminate again.

The retained correction is the route's retained Hamiltonian minus the reduced chip's assembled matrix, both under the chip's approximation. Dressed queries use that same approximation, so under `method="exact"`, a reduced chip reproduces the source's labeled energies, and `reduced.static_zz(a, b)` equals `effective_params["exchange"]["zz"]`. Under `method="sw"`, the reduced dressed spectrum is the second-order one, so compare its diagnostics through `effective_params`. Retained effective terms are part of the cache key for dressed analysis, so attaching them to an already resolved chip invalidates its cached dressing.

`result.mapping` captures source and target labels, dimensions, backend and lab-frame solver coordinates. Its `embedding` maps retained coordinates into the source space. `project_operator(operator)` returns `B† O B`, `project_state(state)` returns `B† psi` or `B† rho B`, and `lift_state(state)` performs the reverse embedding. These methods accept full numerical matrices and backend-native objects, and return native objects on the captured backend. Projection keeps the lost norm or trace, so discarded population stays visible and no state is silently renormalized. Express rotating-frame trajectory states in lab coordinates before you use this map.

Exact reduction uses the same Löwdin embedding as its Hamiltonian and removed-component channels. SW uses `B = exp(-S) P`, with the first-order generator that the reduction already uses. This map is isometric, but the dynamics stay perturbative, and the SW Hamiltonian stays truncated at second order. Jump operators use the same exponentiated first-order coordinate map, which keeps their interpretation common with states and observables. It does not make the reduced dynamics or dissipation exact. Maps capture numerical coordinates independently of later source edits.

## 11. Parametric Edge Control

Sources: [`quchip/control/drive.py`](quchip/control/drive.py) (`ParametricDrive`), [`quchip/engine/assembly.py`](quchip/engine/assembly.py) (`EDGE_PUMP`), [`quchip/chip/retarget.py`](quchip/chip/retarget.py)

### 11.1 The pump contract

A `ParametricDrive` pumps a modulable coupling (a `TunableCapacitive` edge): the scheduled envelope is the *real* modulation `A(t)` of the coupling strength, in GHz. It has two forms:

```text
freq omitted (baseband):  delta_g(t) = Re s(t)
freq = nu_d (tone):       delta_g(t) = Re[s(t) · e^(−i·2π·nu_d·t)]
```

The engine never RWA-splits the tone. The *coupling's* `parametric_interaction` hook picks the kept operator structure. Each excitation-change band `(Δa, Δb)` carries its rotating-frame carrier `exp(−i(Δa·ω_a + Δb·ω_b)t)` exactly as static couplings do (§5.2). A pump at the survivors' difference frequency parametrically activates the exchange at the effective rate `A/2`. The factor 1/2 is the rotating-wave halving of a real modulation (Didier et al., PRA 97, 022330 (2018)).

### 11.2 Retargeting stranded control (`chip/retarget.py`)

`eliminate()` converts control lines with a removed target through a registry keyed by `(drive type, target type, result kind)`. The registry follows each type's MRO, so a rule registered for a base type also covers its subclasses. The built-in rule converts a `FluxDrive` on an eliminated exchange-mediating mode into one baseband `ParametricDrive` per emitted edge. The first pair's pump keeps the flux line's label, and further edges receive unit-amplitude `Crosstalk` copies of the scheduled signal. Every pump carries its own `Gain(dJ_ab/domega_c)`. This small-signal conversion is exact to first order in `delta_omega_c` and assumes `delta_omega_c ≪ Delta`. The survivors' second-order Lamb-shift modulation is omitted and recorded.

### 11.3 Schedule portability

The retargeted line keeps its label and `schedule()` resolves drive-line labels in device → coupling → line order. The same schedule call can therefore run on the full and reduced chips. [`tests/physics_sentinel/test_eliminate_portability.py`](tests/physics_sentinel/test_eliminate_portability.py) applies identical schedules to both models and compares them with tolerances derived from the validity metrics (`g/Delta`, `delta_omega/Delta`).

## 12. Engine Assumptions

The engine assumes four physics properties:

1. Each frame generator is the isolated energy-level index operator, expressed in the selected solver basis. It need not equal a device's physical number operator.
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

Devices, couplings, and drives supply the domain-specific physics, and the engine handles their shared frame and RWA bookkeeping.

## 13. JAX Traceability Boundaries

Band decomposition, coefficient construction, observable recombination, and the backend-free Hamiltonian IR preserve JAX arrays. A traced dense payload keeps every candidate band unless its structure is declared. Sparse layouts declare their entries. A matrix contribution can declare its total excitation changes (`PhysicsExpr.from_matrix(..., excitation_changes=...)`), as reductions of excitation-conserving models do.

The following operations require concrete Python values:

- `Chip.dress()` returns a concrete dict-based view and is not traceable. The bare→dressed assignment itself is discrete and piecewise. Traced callers should use `Chip.energy()`, `Chip.freq(target, when=...)`, `Chip.dispersive_shift()`, or `Chip.kerr_matrix()`. These methods route through `label_eigensystem` in the pure-JAX kernel in `quchip/chip/dressing.py`. The labeled energy lookup stays differentiable away from label discontinuities. `track_path` is a separate continuation utility that follows labels through a stacked eigensystem along a parameter sweep.
- Human-facing serialization and diagnostics coerce to Python scalars.

Other engine paths avoid implicit conversion to host arrays.

## 14. Audit Pointers

To audit a physics path, start here:

- units and assembly boundary: [`quchip/engine/assembly.py`](quchip/engine/assembly.py)
- frame resolution: [`quchip/engine/frames.py`](quchip/engine/frames.py)
- observable preparation and demodulation: [`quchip/engine/observables.py`](quchip/engine/observables.py)
- dressing and public Hamiltonian APIs: [`quchip/chip/chip.py`](quchip/chip/chip.py)
- adiabatic elimination and χ/κ reporting: [`quchip/chip/transformations/`](quchip/chip/transformations/)
- Schrieffer-Wolff kernels and the exact reduction route: [`quchip/chip/sw.py`](quchip/chip/sw.py)
- control-line retargeting across reductions: [`quchip/chip/retarget.py`](quchip/chip/retarget.py)
- readout pointer states and figures of merit: [`quchip/analysis/dispersive_readout.py`](quchip/analysis/dispersive_readout.py)

## Measurement of saved states

`SimulationResult.measure(*devices, t=None, basis="energy")` projects a kept ket
or density matrix in the captured local isolated energy bases. `t=None` selects
the final state. A custom basis supplies orthonormal columns in those energy
coordinates in the stored integration frame. quchip applies no automatic
phase-frame conversion to it. `basis="solver"` selects the solver's product
basis. quchip computes the joint probabilities before marginalizing the
unmeasured devices. Independent partition components can be combined at the
probability level.

Shot sampling applies the Born rule without further evolution.
`assignment[recorded, physical]` is column-stochastic. `IQReadout` defines one
conditional complex Gaussian distribution per physical outcome. Its total
mixture covariance includes both within-outcome covariance and the conditional
means' covariance. These readout models do not change solver selection, return
collapsed states, or describe continuous quantum trajectories.

`IQReadout.from_wiring(chip, output, ...)` resolves a detector without quantum
evolution. `result.iq_readout()` uses the simulation's captured wiring instead.
Both propagate supplied conditional coherent fields through captured output
reference sections and downstream mixing. Unspecified boundary channels are vacuum.
Supplied means, returned detector IQ, and receiver transfers use the same
engineering convention as VNA. Simulation field traces keep the physics convention.
Conjugate stationary `output(...).raw_amplitude` values before using them as
boundary templates. Both paths include downstream added noise and ideal heterodyne
vacuum through the same propagation and integration used by VNA. Boundary thermal
noise, device correlations and transient field correlations are outside this
coherent-field readout model, but calibrated conditional distributions can include
those effects instead. A calibrated full covariance must not receive the same
apparatus noise a second time. Input baths and port decay remain in the declared
quantum dynamics, irrespective of the readout model.


## Continuous trajectory monitoring

Native jump and diffusive solvers evolve the captured Hamiltonian and declared
channels. `with_monitoring` selects physical SLH couplings with their authored
phases. At efficiency eta, QuTiP receives sqrt(eta) exp(-i phase) L and
sqrt(1-eta) L, whose dissipators sum to D[L]. Dynamiqs receives L and eta through
its native SME interface. Pure SSE cannot omit unobserved loss.

Native measurement records describe selected L in the integration frame, without
adding coherent incident beta or downstream receiver noise. Native weighted jump
averages include deterministic no-click paths. Truncation checks are explicit
and cover available samples, with final-only coverage identified separately.
See [the solver guide](docs/guides/choosing-a-backend.md) and
[Wiseman and Milburn, chapter 4](https://doi.org/10.1017/CBO9780511813948).
