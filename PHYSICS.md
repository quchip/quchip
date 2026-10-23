# Physics reference

This document is quchip's physics contract. It states what each Hamiltonian
view contains, where quchip applies local bases, frames and the RWA, which
approximations each workflow makes, and the conventions of its results.
Docstrings, tests and code comments cite its sections by number. A change to
quchip's physics updates the matching section in the same pull request.

Sections 1–7 describe how quchip builds and resolves a model. Sections 8–11
describe what it computes: observables, stationary response, dressed
quantities, model reductions and parametric control. Sections 12–16 state the
engine's assumptions, the limits of JAX tracing, source pointers and
measurement records.

## Conventions at a glance

| Topic | Convention | Section |
| --- | --- | --- |
| Units | Frequency and `E/h` in ordinary GHz, time in ns, temperature in mK. Hamiltonian assembly applies `2π` once. | §1 |
| Rates | Lindblad rates in 1/ns. Backend lowering applies `sqrt(rate)` once. | §3.3 |
| Charge operator | `Q = i(a − a†)` on the built-in Fock devices, and `Capacitive` is `g Q_a Q_b`. | §2.2 |
| Pauli operators | They act on the two lowest isolated energy states, with `Z = \|0><0\| − \|1><1\|`. | §2.1 |
| Level labels | `{"q": 1}` names a dressed eigenstate of the static chip. `chip.bare_state(q=1)` is the bare product. | §2.1, §9 |
| Dressed-state phase | The amplitude on the assigned bare state is real and nonnegative. | §9 |
| Frames | The chip default is `None`: statics use `"lab"`, and solves use `"rotating"`, or the `"auto"` plan or `"lab"` when it does not fit. `"rotating"` uses each device's `reference_freq`, or its dressed `0 -> 1` transition. In every frame, dict-form `e_ops` are demodulated to each device's reference frequency. | §4, §8 |
| Band weight | `column − row` in the isolated energy basis, so lowering has weight +1. A band of weight `w` turns as `exp(−i 2π w f_frame t)` in a frame of frequency `f_frame`. | §5 |
| `RWA()` | Keeps the bands whose weights sum to zero. It counts levels, not energy. | §6 |
| Drive field | The real field `Re[s(t) exp(−i 2π f t)]`. A tone of peak amplitude `A` has resonant amplitude `A/2`. | §6.2, §9.2 |
| Pulse rotation | `θ = 2π \|m\| \|∫ s dt\|` for the dressed drive matrix element `m`. | §6.2.1 |
| Traveling fields | `beta` in `ns^(-1/2)`, `\|beta\|²` in photons/ns, and `b_out = b_in + L`. | §3.3.1 |
| Time convention | Physics `exp(−iωt)` inside quchip. VNA results use the engineering `exp(+jωt)`, with `j = i`. | §8.1 |
| Dispersive shift | The full pull `χ = f_r\|1 − f_r\|0 = E(1,1) − E(1,0) − E(0,1) + E(0,0)`. | §10.1 |
| Kerr matrix | Off the diagonal the full pull (static ZZ), on the diagonal the dressed anharmonicity. | §9.3 |

## 1. Units and the 2π convention

User-facing Hamiltonians give `E/h` in ordinary GHz, and solver assembly
converts them to angular frequency.

| Quantity | Unit |
| --- | --- |
| Frequency | GHz, ordinary frequency |
| Time | ns |
| Temperature | mK |
| Energy / h | GHz |
| Angular frequency `omega=2π f` | rad/ns |
| Lindblad rate, energy linewidth | 1/ns |
| Reduced flux `Phi/Phi_0` | dimensionless |
| Phase | radians |
| Traveling-field amplitude `beta` | `ns^(-1/2)`, with `\|beta\|²` in photons/ns |
| Cavity amplitude, ladder operators, level index | dimensionless |

The domain layer stays in ordinary GHz. The `2π` enters once, where quchip
builds a solver-facing generator: in
[`quchip/engine/assembly.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/assembly.py) for time-domain
solves, and in the sector solvers for stationary states and weak-probe
response (§6.5, §8.1). Rates are already in 1/ns and get no assembly `2π`. A component's intrinsic `2π`, such as a resonator's
`κ = 2π f/Q`, belongs to its own physics.

## 2. What `.hamiltonian()` means

### 2.1 Device Hamiltonians

`BaseDevice.unresolved_hamiltonian()` returns the device's authored static
Hamiltonian in its declared local space, in the lab frame and in ordinary GHz.
For example, `DuffingTransmon` returns `f * n + (alpha / 2) * n * (n - I)`, and
`Resonator` returns `f * n`. This authored view has no `2π`, no rotating-frame
subtraction, no drive term and no explicit time dependence.

`BaseDevice.hamiltonian()` resolves the isolated device through the engine,
with the device's basis policy and the lab frame. For another local frame,
pass `frame=` to `device.resolve()`. These queries do not change when you
attach the device to a chip. For the coupled system, use `chip.hamiltonian()`
or `chip.freq(device)`.

Level indices follow the energy order of each device's isolated Hamiltonian.
`chip.bare_state(q=1)` prepares the bare product with `q` in its first excited
level, and `result.population(q, 1)` measures the occupation of that level,
summed over the other devices. A level label such as `{"q": 1}` instead names
the dressed state of the coupled chip (§9). A result keeps the energy vectors
that its calculation used.

Device Pauli operators act on the two lowest isolated energy states and vanish
on higher levels: `Z = |0><0| - |1><1|`, `X = |0><1| + |1><0|` and
`Y = -i|0><1| + i|1><0|`. quchip does not renormalize leakage away. For each
energy vector, quchip makes the authored-basis component of largest magnitude
real and positive, and the first such component wins a tie. State preparation,
transitions and default Pauli observables use this convention. It fixes the
phase locally but does not fix the basis of a degenerate eigenspace.
Derivatives therefore require resolved eigenvalue gaps and a stable phase pivot
(§13), and quchip does not promise a globally continuous eigenvector phase.
Dressed eigenvectors of the coupled chip follow a different rule (§9).

Authored `LocalOps` operators, including `op.sigma_z` in a Hamiltonian
declaration, keep their local-space definitions and do not diagonalize the
Hamiltonian they define. A named-observable lookup uses the component's
operators and transforms them into the selected solver basis.

### 2.2 Coupling `.interaction_hamiltonian()`

`BaseCoupling.interaction_hamiltonian()` returns the coupling's full two-body
operator in the pair subspace, in the lab frame and in ordinary GHz. It takes
no RWA argument, and each coupling defines exactly one interaction. The chip
and the engine apply the RWA to its bands, not the coupling (§6.1).

`Capacitive` is `g * Q_a Q_b`, where `Q` is each endpoint's charge operator,
the same operator that `ChargeDrive` uses. Between two Fock devices,
`Q = i(a − a†)`, and

```text
full: g * i(a − a†) * i(b − b†)
    = g * (a†b + ab†) − g * (ab + a†b†)
```

`interaction_hamiltonian()` always returns this full form. The RWA form
`g * (a†b + ab†)` is what remains after the chip masks out, or the engine
filters, the bands that change the total excitation, and you never author it
directly.

A device declares its charge operator with `charge_coupling_operator()`, and
`EndpointOps.charge` ([`quchip/declarative/ops.py`](https://github.com/quchip/quchip/blob/main/quchip/declarative/ops.py))
returns it. On a device without that method, the local space sets the
operator: `a + a†` on a `FockSpace`, and `n` on a `ChargeSpace` or
`PhaseGridSpace`. The built-in devices declare these operators:

| Device | Charge operator `Q` |
| --- | --- |
| `DuffingTransmon`, `Resonator`, `KerrCavity`, `Qubit` | `i(a − a†)` |
| `FluxTunableTransmon` | `s(Φ) i(a − a†)` |
| `ChargeBasisTransmon`, `Fluxonium` | `n` |
| `EigenbasisDevice` | the supplied `charge_operator` |

At fixed `E_C`, the transmon charge amplitude scales as `E_J^{1/4}`
([Koch et al.][koch], section III A). The `FluxTunableTransmon` factor

```text
s(Φ) = (E_J(Φ)/E_J,max)^{1/4} = (cos²(πΦ) + d² sin²(πΦ))^{1/8}
```

takes it relative to Φ = 0, so `g` and the `ChargeDrive` amplitude refer to the
sweet spot. A chip at `flux_bias = 0` therefore keeps its exchange, and the
exchange decreases as `flux_bias` moves away from 0. The phase operator is
`(a + a†)/s(Φ)`. `s` follows the static `flux_bias`. A `FluxDrive` pulse
changes only the device frequency (§6.2), so `s` keeps its static value during
the pulse.

The charge operators have different matrix elements. `|<0|Q|1>|` is 1 on a
`DuffingTransmon` or a `Resonator`, `s(Φ)` on a `FluxTunableTransmon`, and
about `(E_J/8E_C)^{1/4}/√2` on a `ChargeBasisTransmon`. One numerical `g` is
therefore not one physical coupling across bases. Match models across bases
through dressed quantities (exchange, χ, ZZ), not through `g`.

### 2.3 Chip and sequence Hamiltonians

`Chip.unresolved_hamiltonian()` embeds every authored device Hamiltonian and
every full coupling interaction into the declared Hilbert space. It is the
exact static lab-frame expression before local-basis resolution, level
truncation, frame transformation or RWA.

quchip rebuilds `Chip.hamiltonian()` from `Chip.resolve()`, the frozen
`EngineResult` that backends use. Resolution:

1. resolves each authored local space into the selected solver basis and retained dimension
2. multiplies solver-facing operators by `2π`
3. subtracts the chosen frame generator
4. splits non-static pieces into excitation-change bands and applies the chip's approximation
5. attaches explicit time-dependent phases where necessary

The returned expression shows these canonical terms in ordinary GHz, while the
solver-facing `EngineResult` keeps the internal `2π`.
`QuantumSequence.hamiltonian()` follows the same path and adds the sequence's
scheduled drive and crosstalk terms.

These methods return `PhysicsExpr`, the backend-neutral algebra for authored
and resolved physics. It keeps declared parameters, matrices, time-dependent
scalars, labels and opaque JAX callables without forcing numerical values.
`.matrix()` or backend lowering produces numbers explicitly.

| Method | Meaning |
| --- | --- |
| device `.unresolved_hamiltonian()` | authored local static lab-frame physics |
| device `.hamiltonian()` | resolved one-device engine view |
| coupling `.interaction_hamiltonian()` | authored static lab-frame interaction |
| `Chip.unresolved_hamiltonian()` | embedded authored static lab-frame chip Hamiltonian |
| `Chip.hamiltonian()` | resolved canonical chip Hamiltonian without scheduled drives |
| `QuantumSequence.hamiltonian()` | resolved canonical Hamiltonian with scheduled drives |

## 3. Device models

### 3.1 Duffing transmon

Source: [`quchip/devices/transmon/duffing.py`](https://github.com/quchip/quchip/blob/main/quchip/devices/transmon/duffing.py)

```text
H/h = f * n + (alpha / 2) * n * (n - I)
```

`f` is the `0 -> 1` frequency in GHz. `alpha` is the anharmonicity in GHz.

### 3.2 Resonator

Source: [`quchip/devices/resonator.py`](https://github.com/quchip/quchip/blob/main/quchip/devices/resonator.py)

```text
H/h = f * n
```

If you set `internal_quality_factor`, the resonator contributes unobserved
photon loss with the authored operator `a` and the rate
`2π * f / Q_internal` in `1/ns`. Backend lowering forms the Lindblad
operator `sqrt(2π * f / Q_internal) * a`.

### 3.2.1 Other built-in models

Sources: [two-level qubit](https://github.com/quchip/quchip/blob/main/quchip/devices/base.py),
[charge transmon](https://github.com/quchip/quchip/blob/main/quchip/devices/transmon/charge_basis.py),
[fluxonium](https://github.com/quchip/quchip/blob/main/quchip/devices/fluxonium.py),
[flux-tunable transmon](https://github.com/quchip/quchip/blob/main/quchip/devices/transmon/flux_tunable.py),
[Kerr cavity](https://github.com/quchip/quchip/blob/main/quchip/devices/kerr_cavity.py), and [local spaces](https://github.com/quchip/quchip/blob/main/quchip/devices/spaces.py)

The Hamiltonians below use ordinary GHz. Physical charge `n` and phase `phi`
differ from an energy-level index.

| Device | Hamiltonian `H/h` and parameters |
| --- | --- |
| `Qubit` | `freq * \|1><1\|`. Intrinsically two-level, so it has no truncation boundary. Use a transmon model when higher levels matter. |
| `ChargeBasisTransmon` | `4 E_C (n-n_g)^2 - E_J cos(phi)`. `n_g` is offset charge in Cooper-pair units. |
| `Fluxonium` | `4 E_C n² + E_L (phi+2π phi_ext)²/2 - E_J cos(phi)`. `phi_ext` is reduced flux. |
| `FluxTunableTransmon` | Duffing form with a calibrated SQUID frequency and fixed anharmonicity. |
| `KerrCavity` | `f n - K n(n-1)`, with anharmonicity `-2K`. |
| `EigenbasisDevice` | Supplied energy levels and physical operator matrices, with caller-defined normalization. |

The transmon model follows [Koch et al.][koch], and the fluxonium circuit
follows [Manucharyan et al.][fluxonium-ref]. The plus sign in quchip's
inductive term fixes its flux gauge, and operators and drive signs use the
same gauge.

`ChargeBasisTransmon.num_basis` sets the integer-charge basis size, and its
retained energy levels are a separate truncation. Fluxonium uses an
endpoint-excluded phase grid with nonperiodic second-order differences. Its
`n²` operator discretizes the second derivative directly, so it differs from
`n@n`. Converge the phase range and the spacing separately.

For `FluxTunableTransmon`, let `d=asymmetry` and `Phi=flux_bias`:

```text
E_C = |alpha|
r(Phi) = sqrt(cos²(pi Phi) + d² sin²(pi Phi))
E_J,max = (f_anchor+E_C)² / (8 E_C r(Phi_anchor))
f(Phi) = sqrt(8 E_C E_J,max r(Phi)) - E_C
```

The current `freq` and `flux_bias` define the calibration anchor. Rebinding
flux alone follows this curve, and rebinding frequency resets the anchor. The
anharmonicity stays fixed, and the charge scaling follows §2.2. A scheduled
flux pulse applies the declared flux operator without rebinding this SQUID
curve or its charge scale.

Negative-quartic Duffing and Kerr models describe low excitations. Their
spectra are unbounded below as the level count grows, and more levels do not
recover the full Josephson cosine. An `EigenbasisDevice` spectrum alone
supplies no omitted charge or flux operator.

### 3.3 Collapse operators

Sources: [`quchip/devices/base.py`](https://github.com/quchip/quchip/blob/main/quchip/devices/base.py), [`quchip/chip/baths.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/baths.py)

The Markov master equation uses the [Lindblad dissipator][lindblad]:

```text
d rho/dt = -i[H_angular, rho] + sum_k gamma_k D[A_k]rho
D[A]rho = A rho A† - (A†A rho + rho A†A)/2
```

`H_angular` is in rad/ns and each `gamma_k` is in 1/ns. Devices, drives,
couplings and baths author `CollapseChannel` records, which keep the local
operator separate from its non-negative rate. The engine projects and embeds
the operator and keeps the rate unchanged. Backend lowering applies
`sqrt(rate)` exactly once.

The standard ladder-model dissipators are:

- `T1`: relaxation through the device's `lowering_operator()` (the declared `a` by default)
- `T2`: pure dephasing through `sqrt(2*gamma_phi) * n` with `gamma_phi = 1/T2 - 1/(2*T1)`. Because of the factor `2`, the 0–1 coherence decays at `1/(2*T1) + gamma_phi = 1/T2`, so `T2` is the resulting coherence time when `thermal_occupation == 0`. The number operator gives the standard `(m-n)^2` dephasing of higher coherences.
- thermal up and down channels when `thermal_occupation` is set

Custom devices choose these operators through `lowering_operator()`,
`raising_operator()` and `number_operator()`, or declare channels directly.
The lifetime meanings above assume a unit 0–1 lowering element and dephasing
eigenvalues one apart. Other normalizations change them.

Charge-basis transmons and fluxonium default instead to one channel per energy
transition (`collapse_model="fermi_golden"`). For levels `l<u` and the
physical operator `O` of the coupling channel, the base rate is

```text
Gamma_lu = |<l|O|u>|² / (|<0|O|1>|² T1).
```

`coupling_channel="charge"` selects physical charge, and fluxonium also
accepts `"flux"`, which selects phase. A relaxation time requires an explicit
channel. A transition is kept when its concrete rate ratio
`|<l|O|u>|² / |<0|O|1>|²` reaches `collapse_rate_threshold`, default `1e-8`.
Traced ratios keep every transition, and a dark 0-to-1 element raises.
Thermal occupation `nbar` multiplies downward rates by `nbar+1` and upward
rates by `nbar`. Without `T1`, this model adds no relaxation. The optional
`"ladder"` model instead permits thermal channels with base rate `1/ns`.
Dephasing uses energy-level indices, not physical charge. One occupation
across unequal transitions does not describe a single Gibbs temperature.

Noise parameters are ordinary tracked attributes. You can set them, or clear
them with `None`, at construction or later, and later writes get the
constructor's validation. Each solve rebuilds the collapse operators from the
current values.

`Bath` recipes add shared channels to the intrinsic device noise. Attach a
bath at construction or later with `chip.add_bath(...)`.

| Recipe | Jump operators and assumptions |
| --- | --- |
| Thermal | Independent ladder jumps. Occupation uses one local frequency per target. |
| Collective decay | One equal-weight, equal-phase jump `sum_i a_i`. |
| Correlated dephasing | One common-mode jump `sum_i n_i`. |

The default bath rate is `1/ns`. Collective recipes do not encode a general
correlation matrix, and correlated thermal baths are unsupported. `Exact()`
retains Hamiltonian terms but does not derive global dressed dissipators, and
local dissipation can fail in ultrastrong coupling ([Beaudoin et al.][beaudoin]).

### 3.3.1 Accessible ports

Sources: [`quchip/chip/ports.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/ports.py),
[`quchip/control/field.py`](https://github.com/quchip/quchip/blob/main/quchip/control/field.py),
[`quchip/observables.py`](https://github.com/quchip/quchip/blob/main/quchip/observables.py),
[`quchip/engine/input_output.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/input_output.py),
[`quchip/analysis/vna.py`](https://github.com/quchip/quchip/blob/main/quchip/analysis/vna.py)

A `Port` is an accessible Markovian channel. It owns a dimensionless operator
`A_p`, an external rate `kappa_p`, and a reference-plane phase `phi_p`:

```text
L_p = exp(i phi_p) sqrt(kappa_p) A_p
b_out,p = b_in,p + L_p
```

This is the [Gardiner–Collett input-output boundary][input-output], with the
coupling phase chosen so that `b_out=b_in+L`, and the dissipator uses `L_p`.
For a single resonator, `external_quality_factor` gives
`kappa_p = 2π * freq / Q_external`. Internal resonator loss and each port are
separate collapse channels, so `kappa_total` is their sum.

A scheduled `input` on an external plane uses the envelope, phase, carrier and
start-time grammar of a classical drive, with

```text
beta_i(t) = A(t) exp(i theta) exp(-i 2π f t),
|beta_i|^2 = incident photon flux in photons/ns.
```

quchip applies no implicit conjugation or factor of one half. For a general
resolved boundary, the field that arrives at each coupling channel is

```text
c_j(t) = sum_i S_ji beta_i(t),
H_input(t) = i sum_j (c_j^* L_j - c_j L_j†).
```

For one port, this reduces to the angular Hamiltonian
`H_input = i (beta_p^* L_p - beta_p L_p†)`. quchip obtains this canonical
solver form by composing the coherent source `W_beta = (I, beta, 0)` with the
resolved SLH model, then gauging the displaced collapse operators back to the
input-free `L`. The collapse channels therefore stay unchanged: a coherent
field never adds a second damping channel.

`ResolvedSLH` stays input-free, and classical `ControlEquipment` transforms do
not accept field inputs. Field attenuation, phase, crosstalk and
reference-plane delay belong to `PortNetwork` sections.

quchip rebuilds transient output observables from the same solve-bound model:

```text
<b_out,j> = c_j + <L_j>,
<X_theta,j> = Re[exp(-i theta) <b_out,j>],
<b_out,j† b_out,j>
  = |c_j|^2 + 2 Re[c_j^* <L_j>] + <L_j† L_j>.
```

`result.output(plane)` returns the complex amplitude and the normally ordered
photon flux, and any quadrature is a post-solve projection of the amplitude.
Reference sections shift the incident `beta` on an exposure's inbound leg and
the whole boundary trace on its outbound leg. The field is zero until the
simulation's initial boundary data can arrive. The result also keeps both
moments at the Markov boundary, before the outbound shift.

### 3.3.2 Field networks

Sources: [`quchip/chip/port_network.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/port_network.py),
[`quchip/engine/ir.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/ir.py)

A `PortNetwork` composes accessible ports and instantaneous scalar scattering
before the engine lowers the resolved SLH model. Series, concatenation and
feedback follow the rules of §3.3.3. An included block is a plain copy of
template components and connections under a label prefix and adds no physics
of its own.

For two equal-rate ports on one device,
`network.cascade(near, network.phase_shift("phi", phase=phi), far)` gives an
effective channel coefficient with magnitude squared `2γ(1 + cos φ)` and adds
`+γ sin φ · a†a` to `H`. Here `γ` is angular, in rad/ns, so the Lamb shift is
`+γ sin φ / 2π` GHz. With the propagation phase `φ = 2π f τ = ωτ`, `f` in GHz
and `τ` in ns, this matches the `Δ = γ sin φ` convention of
[Kockum et al.][kockum].

A concrete element of a generated Hamiltonian counts as cancellation residue
when it is below `64 n ε` times the summed magnitude of the port products that
form it. Here `n` is the matrix dimension and `ε` the double-precision epsilon.
quchip drops a generated term only when all of its elements are residue, such
as the Lamb shift at `φ = π`. Each element has its own threshold, so a small
rate keeps its term next to the cancelled term of a large rate. The rule also
drops a genuine shift below the threshold, such as `γ sin φ` at `φ = 1e-15`. A
traced evaluation keeps every term.

`port.side` and `component.port(k)` each select one physical connector and pair
the input and output terminals of that side. `network.link(...)` cables
consecutive sides in both directions, and each cable compiles to two directed
terminal connections. An ideal circulator routes `1 -> 2 -> 3 -> 1`, and an
isolator is a circulator whose third side is a hidden vacuum load. Compilation
works per terminal, so permutation components do not create false cycles.

quchip groups core terminals into strongly connected components and reduces
each connection cycle with the feedback rule of §3.3.3. This gives the
loop-corrected `S`, `L`, structural reachability, and the generated series and
feedback Hamiltonian. Reference sections cannot be in a loop. A concrete
singular `I - M` raises. Traced JAX scattering cannot detect the singularity,
and the solve returns non-finite values at that point.

Concrete scattering must be unitary, and an explicit unitary dilation
represents loss. A two-sided reciprocal attenuator with power transmission
`eta` from side 1 to side 2 has amplitude transmission `sqrt(eta)` in both
directions. It couples each direction to one of two hidden vacuum channels with
amplitude `sqrt(1-eta)`. A beam splitter is instead a directional
two-input/two-output device: `eta` is the power from input k to output k, and
its scattering matrix is
`[[sqrt(eta), sqrt(1-eta)], [-sqrt(1-eta), sqrt(eta)]]`. Beam splitters and
90-degree hybrids have no physical sides. Connect their input and output
terminals directly or with `network.cascade(...)`.

Network exposures define the order of the external channels, and their adjacent
reference sections move the incident and reported reference planes. The
compiler removes these sections from each leg before it forms the
instantaneous `S`, `L` and `H`. A section on a reflection line acts once
inbound and once outbound, unless it is a one-pass section. A reference section
that is not adjacent to an exposure raises.

`network.delay(...)` is a reference-plane shift, not retardation. It changes
fields at exposure planes but adds no memory to the dynamical SLH core, which
is Markovian throughout. Enter the propagation phase between giant-atom
coupling points with `network.phase_shift(...)` as above. Retarded feedback,
linewidth-scale variation of `γ(ω)` or of the phase across a resonance, and
the non-Markovian giant-atom regime are out of scope.

Static composition of several quantum ports requires a common rotating-frame
frequency. Different local carriers would make both the collective `L` and
`Im(L2† S2 L1)` explicitly time dependent. quchip has no time-dependent
collapse channels, so it rejects such a network and directs the model to the
lab frame or a common frame.

`network.filter(...)` adds a two-sided passive reference section with complex
transfer `H(f)`. Its `transfer(frequency, **parameters)` callable takes scalar
or array frequencies in ordinary GHz and stays outside the Markovian `S`, `L`
and `H`. quchip tracks each keyword parameter at
`network.component.<label>.<name>`. Continuous-wave calculations, including
VNA, small-signal response, spectra and stationary pumps, apply `H(f)` exactly
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
rotating-frame carrier. A lab-frame outbound channel has no such carrier, so a
request for its filtered output raises before the solve. Filter sections are
passive, so a concrete evaluation with `|H| > 1` raises. For gain, use
`network.amplifier(...)`.

`network.mode_reflection(...)` adds a two-sided passive reference section whose
squared per-pass transfer `H(f)` is the reflection `S_r(f)` of a damped linear
mode (§10.5). A reflection plane therefore gets `S_r(f)`, and a field that
crosses one leg, such as emission from the devices behind it, gets `H(f)`. `H`
is the continuous square root with a nonnegative real part at
`reference_freq`. At that frequency, `H` reproduces to first order the emission
phase `1 - i kappa_e / (2 Delta)` of a device detuned by `Delta` from the mode.

`network.mode_transmission(...)` adds a one-pass passive reference section.
From side 1 to side 2 it multiplies the field by the full `S_r(f)`, without a
square root. Reverse propagation is transparent, so the section acts on one
leg only. Zero external coupling gives unit transfer, including at resonance.
Its parameters are `freq` in GHz and `external_rate` and `internal_rate` in
1/ns, and its internal-loss bath is vacuum. Both mode sections track their
parameters like `network.filter(...)` and serialize.

`network.amplifier(...)` adds a phase-preserving reference section on the
output line, with power gain `G` and input-referred symmetrized added noise
`n_add` in quanta. It amplifies side 1 to side 2 by `sqrt(G)` and is
transparent in reverse. Its exposure must lie beyond side 2 on the outbound
leg. quchip raises if the forward direction would amplify an incident field
into the chip, or if an outbound plane is on side 1. `gain` and `added_noise`
are sweepable, differentiable parameters under `network.component.<label>`,
and amplifier sections serialize.

The [phase-preserving amplifier floor][caves] and output-line noise recursion are

```text
n_add >= (1 - 1/G)/2,
N <- |H(f)|^2 N,
N <- G N + G n_add + (G - 1)/2.
```

The filter line assumes a vacuum loss bath, and an occupied matched absorber
adds `(1-|H|²)nbar`. The amplifier line uses input-referred symmetrized
`n_add`. At the quantum limit its added term equals `G - 1`. Two amplifiers in
series obey the input-referred Friis relation `n_total = n1 + n2/G1`.

Continuous-wave means and small-signal entries, including VNA, get a factor
`sqrt(G)`, so `|S|` can exceed one. `output_spectrum` reports the amplified
signal density, the accumulated added-noise density `N` and their sum
separately. Its signal fluxes scale by `G` but exclude amplifier noise, which
needs a detection bandwidth to become a flux. Transient `result.output(plane)`
scales amplitude by `sqrt(G)` and photon flux by `G`, with no added-noise
term. Normalized `g1` and `g2` through an amplifier raise, so request them at a
plane before the amplifier.

### 3.3.3 Composition rules

Source: [`quchip/engine/slh.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/slh.py)

For resolved triples `G1 = (S1, L1, H1)` and `G2 = (S2, L2, H2)`,
the [Gough–James series and concatenation rules][series-product] and
[zero-delay feedback rule][feedback] are

```text
G1 ▷ G2 = (S2 S1, L2 + S2 L1,
           H1 + H2 + (L2† S2 L1 - h.c.) / 2i),
G1 ⊞ G2 = (diag(S1, S2), (L1, L2), H1 + H2),
g = (1 - S_xy)^-1,
S_tilde = S_x̄ȳ + S_x̄y g S_xȳ,
L_tilde = L_x̄ + S_x̄y g L_x,
H_tilde = H + ((sum_j L_j† S_jy) g L_x - h.c.) / 2i.
```

`G1 ▷ G2` feeds the output of `G1` into `G2`, and `(X - h.c.) / 2i` is written
`Im(X)` elsewhere in this document. `quchip.engine` exposes these rules as
`series_product`, `concatenate` and `feedback_reduce` on the input-free
`ResolvedSLH` that `chip.resolve().slh` returns. Operands must share one
Hilbert space, and joined or closed legs need empty reference runs. A concrete
loop with `1 - S_xy = 0` is singular and raises. If channel keys collide,
concatenation requires `prefixes=`. Feedback from `x` to a different `y` gives
the merged channel the key `x->y`. Hamiltonian corrections from composition
are static terms of network origin.

### 3.4 Authored local spaces and solver bases

Sources: [`quchip/devices/spaces.py`](https://github.com/quchip/quchip/blob/main/quchip/devices/spaces.py), [`quchip/engine/basis.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/basis.py)

Each device authors its Hamiltonian and named operators in a `LocalSpace`. The
built-in realizations are `FockSpace`, `ChargeSpace`, `PhaseGridSpace` and
`CustomSpace`. This declared coordinate space is independent of the
solver-basis policy.

`Chip(..., basis="native")` keeps each device's authored coordinate basis and
dimension. `basis="eigen"` diagonalizes each device's exact authored local
Hamiltonian. It then projects all Hamiltonians, coupling and drive operators,
states, observables and collapse operators into the retained local energy
subspace. A device can override the chip-wide policy with its own `basis` and
select the retained energy dimension with `projection_levels`.

A multi-device operator that is a sum of scalar-weighted single-device products
projects factor by factor:

```text
V†(A ⊗ B)V = (V_a† A V_a) ⊗ (V_b† B V_b).
```

Products on one device multiply before projection, because the truncated
projector `V_a V_a†` is not the identity in the authored space. The engine
takes the cheaper of the factored and dense routes, uses the dense route for
opaque multi-device operators, and checks available memory first. These checks
cover concrete forward calculations, not reverse-mode storage.

Resolution records each fixed authored-to-solver transformation in
`EngineResult.bases`. Local energy ordering differs from whole-chip dressing:
`Chip.dress()` diagonalizes the coupled static chip for analysis, whereas
local-basis resolution defines the tensor factors that the engine and backends
use.

Truncation diagnostics sample each component's boundary projector at the
solver output times and report its largest population. The projector is the
declared native boundary, mapped into the solver basis, and under energy
projection it also holds the highest retained energy state. Frame-band
reconstruction returns the physical boundary population in every frame and for
every readout reference. These traces stay separate from user observables and
saved states.

Fock ladders check their highest Fock state, and charge and phase grids check
both edges. An intrinsically finite model declares no native cutoff through
`truncation_boundary() -> None`, and quchip reports unknown custom cutoffs as
unavailable. Boundary population is a heuristic, not a bound on the truncation
error, and sampling can miss intermediate excursions. For convergence,
increase the relevant cutoff and compare observables.

### 3.5 Transitions

`device.transition(lower, upper)` returns the Hermitian transition operator
`|lower><upper| + |upper><lower|` in the device's authored coordinates.
`device.transition_frequency(lower, upper)` returns the isolated energy gap in
GHz. The level indices follow the energy order of the device's static local
Hamiltonian.

`chip.transition_frequency(target, lower, upper, when=...)` uses the static lab
Hamiltonian that `chip.approximation` retains. It assigns dressed eigenstates
to the two bare product labels and returns their energy difference before
frame subtraction or drive approximation. Unspecified spectators are in level
zero. `when` sets spectator occupations and cannot include `target`.
`chip.freq(target, when=...)` is the concise 0-to-1 form. For a higher
transition, give the levels:

```python
f01 = chip.freq(qubit)
f12 = chip.transition_frequency(qubit, 1, 2)
fr_when_excited = chip.freq(readout, when={qubit: 1})
```

Local basis projection and dressed transition assignment are separate. Basis
projection selects the solver's tensor factors, and dressed assignment labels
eigenstates of the coupled static chip.

## 4. Frames

### 4.1 What frame selection means

Source: [`quchip/engine/frames.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/frames.py)

The public frame spec is one of `None` (the chip default: no frame), `"lab"`,
`"rotating"`, `"auto"`, a shared float, or a per-device dict. The engine
resolves the spec into an integration-frame frequency `f_frame,i` in ordinary
GHz for each device. It is a different quantity from the device's reference
frequency `f_ref,i` (§4.5), which sets the readout reference. The two are equal
under `"rotating"`.

A chip without a frame keeps its static calculations in the lab frame:
`chip.resolve()`, `chip.hamiltonian()`, dressed analysis, and steady states.
Its time-domain solves use `"rotating"` and warn with that choice. The
rotating frame does not fit every chip. Some operators have bands with
different phases in it. Examples are the jump that an eliminated bus carries
onto two transmons (§10.5), a collective decay of two detuned devices, and a
port network that composes ports at different frequencies. A solve on such a
chip uses the frame that `"auto"` plans (§4.4), and the warning gives its
frequencies and the reason. That frame gives one frequency to devices on one
network cascade, and usually to devices that share an exchange coupling, so it
fits an operator whose devices are tied in this way. When it does not fit
either, the solve uses the lab frame. Both keep every band of the operator.
The planned frame is usually the faster and more accurate one. Take two
readout resonators on one hanger feedline, driven for 150 ns and solved for
200 ns. With the planned frame, the solve took 0.11 s, and the population
error at QuTiP's default tolerances was 5.0e-8. In the lab frame, it took
2.8 s, and the error was 2.9e-6. A batch checks every point and solves all
points in the first of the rotating, planned, and lab frames that fits each
of them. In the planned frame, each point gets its own plan, as with
`"auto"`. A solve with `dissipation=False` leaves the collapse channels out,
so they do not need to fit its frame. On a chip with a collective bath, such
a solve and `subspace_propagator()` use the rotating frame. Ports still need
to fit, because drives through a port and the Hamiltonian that a cascade
generates (§3.3.2) stay in a solve without dissipation.

The rotating frame also needs a dressed `0 -> 1` frequency for each device.
Identical coupled devices, such as a resonant pair, have no reliable label
assignment for it (§9), so the chip has no device frequency to rotate at. A
solve on such a chip raises and asks for a frame: `"auto"` gives
exchange-coupled devices one frequency, and `"lab"` needs no reference. A
`reference_freq` on each of these devices also gives the rotating frame its
frequencies. A frame set on the chip, or passed as `frame=` to
`build_problem()`, applies without a warning.

The default was `"lab"` before. A lab-frame solve must resolve the phase of
each device at its transition frequency. Take a transmon with `T1` and a
five-level resonator, 15 states in total, driven by a 20 ns pulse and solved
for 100 ns with automatic sampling. In the lab frame, the solve saved 124,057
density matrices and took 2.8 s and 1.76 GB of peak memory. In the rotating
frame, it saved 9,100 in 0.43 s and 0.33 GB. At QuTiP's default tolerances,
the final `P1` error was 1.0e-4 in the lab frame and 3.5e-6 in the rotating
frame. Under `Exact()`, the counter-rotating bands stay time dependent in the
rotating frame, and that solve is also the more accurate one. Expectation
values do not depend on the frame (§8), but stored states and
measurement records do (§4.2 and §16).

### 4.2 What transform the engine is using

The engine assumes the rotating-frame unitary

```text
U(t) = exp(-i 2π t Σ_i f_frame,i * n_i)
```

The solver Hamiltonian is then

```text
H_rot = U† H_lab U - 2π Σ_i f_frame,i * n_i
```

so the assembler subtracts `f_frame,i * n_i` from `H0`.

Each result records its integration frame as `result.frame`, a
`ResolvedFrame` with the frequencies `f_frame,i`. `state()`, `state_at()`,
`dm_at()` and `reduced_state()` take `frame="lab"`, which applies `U(t)` to
the stored state. A ket target of `overlap()`, `amplitude_array()`,
`final_overlap_magnitudes()` and `final_amplitudes()` is a fixed vector of the
integration frame, so a target such as `|+>` is written in that frame. With
`frame="lab"`, the ket is a fixed lab-frame vector instead, such as a dressed
state from `chip.state()`. A label target such as `{"q0": 0, "q1": 1}` is the
dressed eigenstate `|d_k>` of label `k` (§9), which turns with its label:

```text
target_k(t) = U(t)† |d_k> exp(-i 2π t Σ_i f_frame,i k_i)
```

An idle dressed state then reads 1 in every frame. Its amplitude turns at
`Σ_i f_frame,i k_i - E_k`, the detuning of the label's frame energy from its
dressed energy `E_k`. A bare product `|k>` is a frame eigenstate, so
`U(t)† |k> exp(-i 2π t Σ_i f_frame,i k_i) = |k>`, and the ket target
`chip.bare_state(...)` gives the bare overlap in every frame.
`QuantumSequence.subspace_propagator()` collects these label amplitudes into a
propagator and converts it to the dressed or lab frame (§9.4).

A label `initial_state`, or an omitted one, starts on its label target,
`psi_solve(t0) = target_k(t0)` at `t0 = tlist[0]`, so its label amplitude
starts at 1 (§9). A ket `initial_state` is an integration-frame vector at
`t0` and is not moved, so a rotating-frame start at `t0 != 0` need not equal
the lab-frame vector that `chip.state()` returns. On a chip without a frame,
the ket is a vector of the frame that the warning names. When a batch solves a
point in another frame than the point's own solve would use, it maps the ket
into the batch frame at `t0`, so the point starts in the same physical state.

### 4.3 What `"rotating"` means in practice

`"rotating"` gives each device its own reference frequency `f_ref,i`: an
explicit `device.reference_freq` when set, and otherwise the chip's dressed
`0 -> 1` transition of that device. It then sets `f_frame,i = f_ref,i`. A
time-domain solve on a chip without a frame uses this frame (§4.1).

### 4.4 What `"auto"` chooses

`"auto"` is opt-in. A chip without a frame solves in `"rotating"`, which keeps
the meaning above, and uses the `"auto"` plan only when the rotating frame does
not fit it (§4.1). `Chip.resolve(frame="auto", approximation=...)` uses the
requested approximation. With `Exact()` and no tones, it follows exchange
constraints and device references, and counter-rotating bands do not force
the lab frame. Use `frame="lab"` to keep the static lab Hamiltonian.

The frame planner chooses the frame frequency `f_frame,d` of each device
(§4.1). A retained coupling band with energy-index weights `k = column - row`
is static when

```text
Σ_d k_d f_frame,d = 0
```

A band of a scheduled drive or port coupling at frequency `f` is static when

```text
Σ_d k_d f_frame,d = f
```

Assembly adds the integer weights of devices that share a frame frequency
before it multiplies by that frequency. A band whose weights cancel among such
devices is therefore exactly static. Drive bands with equal carriers share one
term, so no floating-point residue of about 1e-14 GHz separates them.

For example, a two-photon cavity pump at 10.2 GHz sets
`2 f_frame,cavity = 10.2 GHz` and pins the cavity frame to 5.1 GHz. Network
cascades add coupling constraints. Dispersive and cross-Kerr terms have a zero
charge vector, so they do not constrain the frame.

Drive constraints come from the signals that the control equipment delivers
after gain, attenuation, delay and crosstalk. Each nonzero carrier band
constrains every retained operator band of its destination drive. A crosstalk
destination gets its own tone, and gain changes its weight. A zero-frequency
carrier band adds no constraint. A coherent field `beta` that enters a network
exposure drives `i(c* L - c L†)` with `c = S beta` on each channel it reaches.
The operator band's instantaneous amplitude is `|c| |L_band| / 2π` in ordinary
GHz: the frame record stores `|S| |L_band| / 2π`, and the weight integrates
`|beta|²`.

For scheduled calculations, the planner keeps the consistent subset of tones
and zero-total-weight coupling constraints with the largest integrated
strength. A scheduled tone band has weight `|h_band|² ∫|envelope|² dt`, with
each signal integrated over its own nonzero extent inside the outer weighting
window. That window is `[tlist[0], tlist[-1]]` when you build a problem, and
`[0, last pulse end]` otherwise. A short pulse deep in a long solve keeps its
full energy, but the window clips any part outside it. A static coupling band
has weight `|h_band|² T`, with `T` the window's length, so an unknown traced
window makes these weights unknown. A merged constraint with any unknown
contribution has an unknown weight. Equal or unknown weights fall back to
declaration order. The other bands stay time-dependent, and
`resolved.resolved_frame.plan.residuals` records each one's source, devices,
oscillation frequency and weight.

The planner fixes free frequencies from `reference_freq` in device
declaration order. An undriven cluster with exchange coupling therefore
rotates together at one member's reference frequency, and an isolated mode
keeps its own. Coupling bands whose integer weights sum to a nonzero value
come after these reference pins. They remain residuals unless the chosen frame
already makes them static, whatever their integrated strength. This rule
applies only to coupling bands, and multiphoton tones keep their tone
priority. Frame planning removes no Hamiltonian terms.

`chip.describe()` and `sequence.describe()` show the selected frequencies,
accepted tones, pins and residuals. Traced JAX tone frequencies pass through
the plan without conversion to Python scalars.

Stationary analyses such as VNA sweeps and steady states use a strict order:
mandatory cascade constraints first, then exchange coupling bands with nonzero
charge vectors that sum to zero, and tones last. A conflicting tone raises the
port form of the distinct-stationary-tone error if an accepted tone already
addresses the same devices, and the exchange-connected form otherwise. Only
non-exchange coupling bands, such as counter-rotating bands retained under
`Exact()`, can stay time-dependent, and they raise the dynamic-Hamiltonian-terms
error.

### 4.5 `reference_freq`: the readout and LO reference

Source: [`quchip/devices/base.py`](https://github.com/quchip/quchip/blob/main/quchip/devices/base.py)

`device.reference_freq` returns the authored override in GHz, or `None`. Each
new chip calculation resolves `None` to that chip's dressed transition and
captures the value in its frame record. An explicit value fixes the frame and
readout reference across model changes. When `f_ref` is off the dressed
transition `f_01`, the demodulated `0 -> 1` coherence of an idle device turns
as `exp(-i 2π Δ t)` with `Δ = f_01 - f_ref`. This is idle Ramsey precession.
The integration frame subtracts `f_frame n` from the Hamiltonian (§4.2), and
demodulation removes the rest, `f_ref - f_frame` (§8). Under `"rotating"` the
Hamiltonian takes all of `f_ref`, and under `"lab"` the demodulation does.

The reference is a frame and readout reference only and does not detune
drives. The drive carrier is a separate choice, so a real LO error must also
set the drive frequency. The value is in ordinary GHz, tracked (a change
invalidates engine caches), and JAX-traceable, differentiable and sweepable.

## 5. Frame tracking in the engine

Source: [`quchip/engine/assembly.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/assembly.py)

The engine does not rotate whole expressions symbolically. It tracks phases
band by band, and this band decomposition is the frame-tracking mechanism.

### 5.1 Single-device operators

The engine decomposes a local operator into bands with weight

```text
w = col - row
```

In the chosen frame, that band gets phase

```text
exp(-i 2π w * f_frame * t)
```

With this phase, the engine knows which part of `a + a†`, `i(a - a†)`, or an
observable still rotates.

### 5.2 Two-device couplings

The engine decomposes a two-body operator into bands labeled
`(delta_a, delta_b)`. Each value is `column - row` in that device's isolated
energy basis, opposite to the level change. Lowering has positive weight, so
`ab` has band `(1, 1)`. That band gets phase

```text
exp(-i 2π (delta_a * f_frame,a + delta_b * f_frame,b) * t)
```

If that effective frequency is zero, the band stays static in `H0`. Otherwise,
it becomes an explicit time-dependent term.

### 5.3 Model time dependence and scheduled control

`DeviceModel.time_terms()` and `CouplingModel.time_terms()` return
`TimeDependentTerm` values for physics that exists without a scheduled pulse.
Each term pairs a local operator with a `TimeCoefficient`, and the engine
projects, band-decomposes and frames it like other Hamiltonian terms. Local
eigenbasis projection uses the static authored Hamiltonian at the solve's
operating point, and these terms are projected into that fixed basis. quchip
does not construct an instantaneous moving basis.

Scheduled control stays drive-owned. Its finite-duration envelope and carrier
produce a drive modulation only after `QuantumSequence.schedule()`.
`Envelope.value(local_time)` defines the local complex I/Q shape, and
scheduling owns the pulse start, carrier and global phase. You can therefore
place and phase-rotate one shape without changing its physics definition.

A pulse scheduled without `freq` takes its carrier from the line's drive class.
A class that sets `carrier_harmonic = k` gets the carrier
`k · chip.freq(device)`, where `chip.freq(device)` is the dressed `0 -> 1`
frequency of the line's device with the other devices in level zero (§3.5). In
the rotating frame at `chip.freq(device)`, this carrier makes the drive's
weight-`k` bands static. With an explicit `reference_freq`, the selected
weight-`k` partner turns as `exp(2πi k (chip.freq(device) - reference_freq) t)`
instead, and its Hermitian partner with the opposite sign. `ChargeDrive` and
`PhaseDrive` set `k = 1`. `TwoPhotonDrive` sets `k = 2`, the zero-detuning
Kerr-cat pump of the static model, without pump-induced Stark shifts. quchip
warns with the chosen carrier once per line in each sequence. Drives without a
declared or inherited `carrier_harmonic`, such as flux and coupling drives, and
coherent inputs keep a baseband pulse. `freq=None` gives a baseband pulse on
every line: the envelope without carrier modulation, through the drive's
authored I/Q mapping. `freq=0.0` is a carrier at zero frequency. It follows its
frame, and `RWA()` splits it into partners (§6.2). The automatic carrier is
evaluated when the pulse is scheduled. It follows traced parameters only when
the scheduling runs inside the differentiated function, and rebinding a
sequence keeps it.

Each carrier pulse follows the phase reference of one device, its frame.
`QuantumSequence.vz(device, θ)` emits no pulse. It adds θ to the phase of every
later carrier pulse in that device's frame, on any line ([McKay et
al.][mckay]). The frame defaults to the device that the pulse's line drives. A
cross-resonance tone runs on the control's line at the target's frequency. Its
phase sets the axis of the target's conditional rotation, so the pulse names
the target as its frame with `frame=`. Baseband pulses have no carrier and
follow no frame.

A pulse whose frame is its line's target takes kθ instead of θ, where k is the
drive's `carrier_harmonic`. With `U = exp(iθ n)` on the target, the co-rotating
term `ε exp(iφ) a†^k + h.c.` goes to the same term with phase `φ + kθ`, so a
`TwoPhotonDrive` pump after `vz(cavity, θ)` takes 2θ. This equivalence holds
for the co-rotating Hamiltonian under `RWA()`. Under `Exact()` the
counter-rotating partner does not transform the same way, and transmon
operators with higher excitation-number bands also break it. Drives without a
declared `carrier_harmonic`, coherent inputs, and pulses in another device's
frame keep the θ shift. Such a pulse can need more, for example a two-photon
pump made by doubling another device's reference; add the missing phase
explicitly.

A pulse's `detuning` δ offsets its carrier from `freq`, referenced to the pulse
start `t0`. The delivered complex signal, whose real part drives the line
(§6.2), is

```text
x(t) = E(t - t0) exp(iφ) exp(-2πi f t) exp(-2πi δ (t - t0))
```

`freq` stays the frame frequency, so the pulse is the same in that frame at
every start time. A carrier at `freq = f + δ` is a fixed oscillator instead:
relative to the frame at `f`, its phase at the pulse start is `φ - 2πδ t0`, so
its rotation axis turns when the pulse moves. Scheduling owns the offset, as it
owns the carrier, so every envelope supports it.

## 6. Approximation strategies

Sources: [`quchip/approximations.py`](https://github.com/quchip/quchip/blob/main/quchip/approximations.py), [`quchip/engine/approximations.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/approximations.py)

The chip owns one explicit approximation strategy. `Exact()` keeps every term
of the authored finite-dimensional Hamiltonian. `RWA()` applies the engine's
first-order structural rotating-wave reduction to static interactions and
scheduled drives. Devices, couplings and drives have no RWA policy of their
own.

`Chip.unresolved_hamiltonian()` keeps the authored static interaction. Engine
assembly band-decomposes each interaction and applies the strategy, and
`Chip.hamiltonian()` is rebuilt from the same canonical terms, so inspection
and simulation share one decision path. A retained band with zero frame
frequency stays in `H0`, and every other retained band is an explicit
time-dependent term. Under `RWA()`, rejected bands become advisory
`DroppedTerm` records.

Default `RWA()` counts energy-level indices, not energy: it keeps the bands
whose weights sum to zero. For uneven spectra such as fluxonium, this can
discard near-resonant transitions. Use `Exact()` or an explicit
`RWA(keep_bands=...)` selection to retain them. `keep_bands` replaces the
default selection. Each tuple follows the operator's support order and uses
energy-index weights `column-row`, with positive weight for lowering. Every
tuple must include its negated partner, so the retained Hamiltonian keeps its
adjoint pairs, and invalid selections raise.

A rotating-wave approximation needs the discarded oscillations to dominate the
relevant matrix elements and the pulse bandwidth. The structural mask gives no
universal error bound. Compare with `Exact()` and converge the finite model
when those scales are not well separated.

For a static interaction, each dropped record selects the entry with the
largest `|V|²/ΔE`, using `ΔE = |E_col - E_row|` from the resolved isolated
spectra. Its `amplitude` is `|V|` and its `frequency` is that entry's `ΔE`,
both in GHz and independent of the frame. A nonzero resonant entry has zero
detuning and an infinite perturbative scale, and zero entries contribute no
shift. The separate `frame_frequency` records the absolute carrier in the
chosen frame. General time-dependent terms report only that carrier, since
their coefficients can supply other frequencies and amplitudes.

On concrete evaluations, `RWA()` warns once per interaction when a discarded
resonant entry is nonzero. It also warns when a discarded static band has a
larger entrywise scale than every retained off-diagonal band of that
interaction. The warning names the largest discarded band and its adjoint, and
an interaction that vanishes entirely has its own warning. A resolved snapshot,
or the cached static model of chip analysis, reports the warnings of its own
approximation once. The comparison is a heuristic, not an error bound: it omits
interference, accumulated shifts, populations and pulse spectra. Traced values
skip the check, and the diagnostic fields stay traceable, with piecewise
derivatives where the maximizing entry changes.

### 6.1 Static operator bands

For `Capacitive` between two Fock devices (§2.2):

- `a†b + ab†` has total excitation weight zero and stays under `RWA()`
- `ab` and `a†b†` have total weights `+2` and `-2`, and `RWA()` removes both

`RWA()` rebuilds the retained bands of the complete authored form in the
engine. A coupling supplies no alternative RWA operator or retention hook. The
mask depends only on integer band offsets, so it stays concrete when operator
parameters are JAX tracers.

The engine decides static or dynamic for each band, not for each coupling. In
a shared frame, with several devices detuned to a common reference, a
coupling's counter-rotating band can have a nonzero carrier while its
co-rotating band is frame-static. The engine evaluates each band's carrier
independently, so a shared frame never suppresses the counter-rotating band's
true rotation. With `Exact()`, the engine drops no band, and every non-static
band stays at its own frame frequency.

### 6.2 Driven operator bands

For a single-tone drive channel with complex envelope `s(t)`, the engine forms
the real lab-frame field

```text
Re[s(t) * exp(-i 2π f_drive t)]
```

and combines it with the operator bands. `Exact()` keeps both co-rotating and
counter-rotating pieces. `RWA()` keeps the conventional partner for each
excitation band and drops its counter-rotating partner.

Weight-zero bands keep the complete real signal, including DC and oscillating
longitudinal modulation, with no raising and lowering split. For longitudinal
bands, `freq=0` and a carrier-free signal (`freq=None`) agree when they
deliver the same real envelope: no detuning and no virtual-Z phase. For a
linear quadrature drive on other bands, they differ under `RWA()`. It keeps
one partner of a zero-frequency carrier, but the complete real field of a
carrier-free signal.

`FluxDrive` uses the target's `flux_coupling_operator()`. Duffing and
flux-tunable transmons use the energy-level index. Fluxonium uses phase, which
generally has both diagonal and transverse energy-basis components, and the
transverse components follow the sideband selection above. Arbitrary
nonlinear signal programs keep their authored waveform when the engine cannot
infer a first-order carrier partner.

### 6.2.1 Rotation angle of a resonant pulse

Sources: [`quchip/control/envelopes.py`](https://github.com/quchip/quchip/blob/main/quchip/control/envelopes.py), [`quchip/chip/analysis.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/analysis.py)

Take a dressed transition `i -> f` with the drive matrix element
`m = <f~|D|i~>` of §9.1, and a carrier at the transition frequency. In the
frame of the transition, `RWA()` keeps

```text
H~/h = (s(t) m σ+ + s(t)* m* σ-) / 2,    σ+ = |f~><i~|
```

where `s(t)` is the complex envelope, with the phase of `schedule()`. When the
phase of `s` stays constant, the pulse rotates the transition by

```text
θ = 2π |m| |∫ s dt|
```

about the axis `(cos α, sin α, 0)`, with `α = arg(s m)` and `|i~>` at `+z`. On
the `0 -> 1` transition of an isolated Duffing transmon, the charge drive has
`m = -i`, so a real π pulse gives `iY` and an envelope phase of `π/2` gives
`-iX`. Dressing changes `m` slightly, and other transitions have other
elements, such as `m = -i√2` for `1 -> 2`. When the phase changes during the
pulse, as for DRAG with a nonzero `beta`, time ordering first contributes at
second Magnus order. The area alone then does not set the gate. Detuning,
leakage to other levels and the counter-rotating terms of `Exact()` also change
the rotation.

`Envelope.area()` returns `∫_0^τ E(t) dt` at the envelope's amplitude `A`.
The built-in envelopes use closed forms:

| Envelope | Area |
|---|---|
| `Square` | `A τ` |
| `LinearRamp` | `A (τ - τ_r/2)` |
| `Gaussian` | `A σ √(2π) erf(N_σ/√2)`, with `σ = τ/(2 N_σ)` |
| `GaussianDRAG` | the `Gaussian` area, because the derivative quadrature integrates to zero |
| `GaussianEdge`, `SquareWithGaussianEdges` | `A (τ - 2τ_e) + A σ √(2π) erf(√2 N_σ)`, with `σ = τ_e/(2 N_σ)` |

Other envelopes use 16-point Gauss-Legendre quadrature between consecutive
sampling times, with `0` and `τ` added. The quadrature is traceable, and its
accuracy depends on how well the sampling times resolve the envelope.

`Chip.rotation_amplitude(transition, line, envelope, angle=π)` returns
`A = θ / (2π |m| |a|)`, where `a` is the area at unit amplitude. It takes the
amplitude at the device, so a line with signal-chain transforms or crosstalk
raises. The result is a starting point for a calibration.

### 6.3 Exact excitation spaces in pulse solves

Source: [`quchip/engine/pulse_space.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/pulse_space.py)

Deterministic pulse solves use an exact excitation space when the initial
support and the complete generator prove it invariant. The planner closes that
support under every retained Hamiltonian term, delivered drive, jump and
no-jump product, and it checks local authored algebra before global
embedding. Scalar zeros and small numerical entries do not prove
conservation. For a Lindblad model, invariance requires `Q L_k P = 0` and
`Q (-i H - sum_k L_k† L_k / 2) P = 0`, where `Q = I - P`. The implementation
conservatively checks Hamiltonian, jump and no-jump support separately.

Unknown support selects the full product route, with a memory check before
assembly. A ket traced by `jax.jit` has unknown support. `chip.bare_state()`
stays a constant when it depends on no traced value, but a level of a device
that solves in its native basis follows that device's traced parameters. A
label or `chip.compact_state(...)` keeps its support in every case. Dense
compact operators that cannot fit in memory also select the full route. A
batch proves one space from the initial states of all its points, or it uses
the full route. The calculation never introduces an excitation cutoff on its
own (§6.4 describes an explicit cutoff).

Downward jumps can add the vacuum to an initially one-excitation space, and
coherences between those sectors stay quantum amplitudes or density-matrix
entries. Complete collective channels stay summed before their dissipators are
formed. Occupied input fields add raising support, and an explicit zero
thermal occupation keeps that support for rebinding.

Cascaded ports add the Hamiltonian that they generate (§3.3.2) as `P H_gen P`.
quchip forms the product of each port pair on the joint support of the two
ports and then gathers it onto the space. A product of ports already restricted
to the space would drop each intermediate state outside it. For example, two
cascaded resonators without dissipation keep only the one-excitation sector,
and their exchange passes through the vacuum.

`RWA()` alone does not make a pulse eligible. A transverse microwave drive
changes the excitation number, even at zero envelope amplitude. Exchange and
longitudinal flux modulation can preserve a small space, and so can an
`Exact()` model with conserving interactions. Counter-rotating pair terms can
keep parity, usually a much larger space. Stochastic trajectory solvers keep
their full product route. Native solvers require nonscalar spaces, so a
one-state closure adds another physical sector and closes the generator again.

`result.stats["excitation_reduction"]` records the dimension, totals, full
dimension, and the proof or fallback reason, and `result.dims` keeps the
physical local dimensions. Populations, measurements, reduced states and
channel queries work on compact coordinates. Native `states`, `state()` and
`final_state` expand lazily and refuse allocations that cannot fit.
`chip.compact_state(...)`, which gives dressed labels or, with
`dressed=False`, bare products, and `result.compact_state(t=None)` give
reusable states without expansion. Their maps capture the local energy bases,
so replay requires the same bases. Explicit inputs keep the integration-frame
convention of §4.2, with no hidden phase transformation.

### 6.4 Excitation cutoffs

Sources: [`quchip/cutoff.py`](https://github.com/quchip/quchip/blob/main/quchip/cutoff.py), [`quchip/engine/sectors.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/sectors.py),
[`quchip/engine/pulse_space.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/pulse_space.py)

An approximation selects Hamiltonian terms, and a space policy selects the
product energy labels that a calculation keeps. `ExcitationCutoff(max_total=K)`
keeps every label whose levels sum to `N = sum_i n_i <= K`. It counts levels,
not energy, so it is not an energy window. Set it with `Chip(..., space=...)`
or `chip.with_space(...)`, and remove it with `chip.with_space(None)`. Clones,
parameter rebinding, serialization and fitted chips keep the policy, and
quchip never selects or changes a chip's policy itself. Without a policy, the
stationary VNA workflows can truncate their own solve and warn (§6.5).

With the projector `P` on the retained labels, the truncated model is

```text
H_K(t) = P H(t) P,    L_K = P L P,
d rho/dt = -i [H_K(t), rho] + sum_k D[L_K,k] rho.
```

The cutoff removes labels and changes no matrix element between retained
labels, so the counter-rotating elements of `Exact()` between retained sectors
stay. A network-generated Hamiltonian (§3.3.2) enters as `P H_gen P`. The
engine forms each port pair's `(L_d† c L_u - h.c.) / 2i` on the pair's joint
support before it projects. Two raising ports therefore keep their exchange
`|01> <-> |10>` at `K = 1`, although it passes through `|11>`. For lowering
ports, `P H_gen P` equals the composition of the projected ports. The
dissipators use the projected jumps, so the truncated model preserves the
trace. A label at `N = K` has no thermal excitation jump, because the
projection removes the jump together with its no-jump term.

The static changes of `N` decide which retained totals mix. quchip connects
the totals `0..K` by these steps without leaving `[0, K]`, and each connected
class is one block of `P H_0 P`. `Exact()` capacitive pairs change `N` by `0`
and `±2`, so they give the parity classes `{0, 2}` and `{1, 3}` at `K = 3`.
Unknown changes give one class.

Labeled static queries diagonalize the classes that hold their labels (§9),
and a label with `N > K` raises. `Chip.dress()` and the views built on it
raise, because their results index the full product space. `chip.resolve()`
and `chip.hamiltonian()` act on the retained labels, in the order of
`resolve().excitation_space`. `fit_a_dress(evaluator="full")` fits the
truncated model, and `max_hilbert_dim` bounds each class that a target
diagonalizes.

Deterministic `sesolve` and `mesolve` solves on QuTiP and dynamiqs evolve the
truncated model. The planner of §6.3 closes the initial support inside
`[0, K]`. Unknown support, and batch points that prove different spaces, use
every retained label. The budget couples all devices, including devices
without interactions, so a limited chip's solve is never partitioned and
`chip.partition()` raises.

A label `initial_state`, or an omitted one, starts from the eigenvector of
`P H_0 P` assigned to the label (§9). This state belongs to the approximate
model and changes with `K`. Supplied states must lie inside the cutoff. A
label or `CompactState` with `N > K` raises, and so does a native state with
more than `1e-20` of its weight on `N > K`. quchip treats a smaller residue as
basis-conversion round-off: the solve drops it and does not renormalize the
state.

`result.stats["excitation_reduction"]` adds `cutoff`, `cutoff_dimension` (the
number of retained labels) and `boundary`. The boundary holds the retained
totals from which one Hamiltonian term, drive, jump or no-jump product can
reach `N > K`. `quchip.check_truncation(result)` adds the key
`ExcitationCutoff(K)`. Its value is the largest sampled population in the
boundary totals, and a value above `threshold` warns. The projector onto these
totals is diagonal in the product labels, so no frame changes it, and the
check reads it from the diagonals of the saved states. Unknown changes make
this check unavailable, with a warning, and the per-device level checks
remain. The boundary population is a sampling heuristic, not an error bound:
it ignores coupling strengths and detunings, and it can miss excursions
between samples.

To check convergence, compare frequencies, populations, phases and gradients
at increasing `K`, and check the local level truncation separately with more
`levels` at a fixed `K`. Convergence in `K` need not be monotone. Take two
`Exact()` resonators at 5 and 6 GHz with three levels each and `g = 0.1` GHz.
The 5 GHz mode's first frequency is 4.990098, 4.991007 and 4.989186 GHz at
`K = 1, 2, 3`. At `K = 1`, no retained label pairs with the vacuum, so the
value equals the RWA value. An observable that reads labels up to total `N`
keeps one step of the largest static change `δ` from each label only for
`K >= N + δ`. This bound is a minimum for one-step coverage, not a convergence
criterion. A dispersive shift reads `N <= 2` with `δ = 2`. For the transmon
and resonator of the `ExcitationCutoff` docstring example, it is +2.261 MHz at
`K = 3` and -1.136 MHz at `K = 4`, against -1.137 MHz without a cutoff.

Under an active cutoff, workflows that need the full product space or a local
neighborhood raise, and each message names `chip.with_space(None)`. They are
`VNA.sweep()`, `VNA.resonances()`, stochastic trajectory solvers,
`eliminate()`, `eliminate_spectators()`, `active_patch()`,
`fit_a_dress(evaluator="local")` and `chip.partition()`. `steadystate()`, `VNA.finite_power()`,
`VNA.measure()`, `VNA.output_spectrum()`, `VNA.g1()` and `VNA.g2()` solve the
truncated model (§6.5). A cutoff with `K >= sum_i (d_i - 1)` keeps every label,
so it is inactive and changes no calculation. The classes and spaces depend
only on structural excitation changes, so they stay fixed under JAX tracing.
Traced class blocks use differentiable eigensolves, with the derivative
conditions of §9.

### 6.5 Stationary states on excitation sectors

Sources: [`quchip/engine/sector_stationary.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/sector_stationary.py),
[`quchip/backend/_sector_stationary.py`](https://github.com/quchip/quchip/blob/main/quchip/backend/_sector_stationary.py),
[`quchip/analysis/measurement.py`](https://github.com/quchip/quchip/blob/main/quchip/analysis/measurement.py),
[`quchip/analysis/vna.py`](https://github.com/quchip/quchip/blob/main/quchip/analysis/vna.py)

`steadystate()`, `VNA.finite_power()`, `VNA.measure()`,
`VNA.output_spectrum()`, `VNA.g1()` and `VNA.g2()` solve one stationary state
at each grid point (§8.1, §8.2). On a chip that conserves the total excitation
`N`, a weak probe reaches only `N = 0` and `N = 1`. A finite probe or pump
couples every `N` to `N ± 1`, so no finite set of sectors is closed. These
methods then solve the truncated model of §6.4 on the labels with `N <= K`,
which differs from the full model only by the drive elements from `N = K` to
`N = K + 1`.

This sector route applies when the static model conserves `N`, every port
lowers `N` by one, and every other channel lowers `N` by one or conserves it.
The inputs must be vacuum apart from coherent tones at one frequency, and the
chip must have no thermal device occupation, output network or retained mode
response. In `finite_power()` and `measure()`, a pump must have the probe
frequency. All values must be concrete. Other chips, and calls under JAX
tracing, use the stationary engine. It solves the compact model under an
explicit cutoff, and the full product space otherwise.

In the frame of the tones, `H' = 2π (H - E_0 - f N)`. Port `k` receives the
coefficient `c_k = sum_i S_k,i beta_i h_i(f)` from the inputs `i`, with each
amplitude `beta_i` in the physics convention (§8.1) and `h_i` its inbound
transfer. With lowering channels `L_k` and conserving channels `C_m`, the
generator is

```text
G(X) = A X + X A† + sum_k L_k X L_k† + sum_m C_m X C_m† + [W - W†, X],
A = -i H' - (1/2) sum_k L_k† L_k - (1/2) sum_m C_m† C_m,   W = sum_k conj(c_k) L_k.
```

The solver stores the blocks `X_NM` with `N, M <= K` and never forms the
product space. It replaces the vacuum equation by `Tr X = 1` and writes
`X = |0><0| + Δ`, so the residual is relative to the probe's correction
instead of the vacuum population. The error of `Δ` still grows with the
condition number of the generator, which is large when a state relaxes slowly.
For the dark pair below at `K = 2`, the relative error of `Δ` is `3e-11` at
`|beta| = 0.1` and `1e-7` at `|beta| = 3e-5`. A preconditioned restarted
GMRES solves for `Δ`, and a solve whose relative residual stays above `1e-8`
raises.

If the lowering channels empty every state of each sector `N >= 1`, the
stationary state is unique. Each state then has a lowering jump, so the support
of every stationary state holds the vacuum, and two stationary states with
orthogonal supports cannot exist. This holds for any Hamiltonian, probe and
conserving channels. The solver compares the smallest eigenvalue of
`sum_k L_k† L_k` in each sector with `δ`, which is `1e-6` of the largest decay
rate. Above 1024 labels, it uses a lower bound of that eigenvalue: the larger
of the Gershgorin bound and the smallest diagonal entry of the single-device
channels. Without a drive the vacuum is stationary, so when this test passes,
`steadystate()` solves nothing and returns the vacuum on the labels with
`N <= 1`. The frame map `U(t)` keeps `|0><0|`, so the vacuum is also the
stationary state of a frame in which the resolved Hamiltonian is not static,
where the backend's solve raises. When the test fails, `steadystate()` solves
the backend's stationary Liouvillian. On the sector route, the `steady_state`
of an `output_spectrum()`, `g1()` or `g2()` result reports `nullity=1` when the
test passes and `None` when it fails.

When a sector fails this test, a dark state is possible, and the solver starts
again from a fixed generic full-rank state. The equations do not fix the
weight on a second stationary state, so the two starts usually end with
different weights on it. If the two corrections `Δ` differ by more than `1e-6`
of the first, both solves continue to a relative residual of `1e-15`. If they
still differ, the call raises, because the truncated model has more than one
stationary state or one that the solve cannot resolve. Agreement is evidence
of one stationary state, not a proof: a model can give both starts the same
limit, e.g. when a start such as the maximally mixed state shares a symmetry
of the model. Two resonant qubits on one channel `σ_1 + 2 σ_2` have the dark
state `2|10> - |01>`. The probe reaches it only through `|11>`, so `K = 1`
raises and `K = 2` has one stationary state. At `K = 2`, the dark state empties
at a rate of order `|beta|^2`, and the call raises at `|beta| = 1e-5`. With a
decay of `1e-6 / ns` on each qubit, `|beta| = 1e-6` gives the state of a dense
solve to `2e-10`.

The noise spectra solve `(G + i 2π f) Y = B` with `Tr Y = 0` on the same
blocks, with the sign of `stationary_resolvent()` (§8.2). Because
`R_f(B†) = R_{-f}(B)†`, each output also solves at the negative of every
offset, and the symmetric default grid of `measure()` reuses its own points.
Mode amplitudes and photon numbers evaluate the authored `a` and `n` on the
retained labels. Every mode is solved in the probe frame, so
`mode_frequencies` holds the probe frequency, as on the harmonic route, and a
channel that couples modes at different frequencies is static.

`output_spectrum()` sends these spectra through the outbound reference of its
port, as `measure()` does. Its fluxes read the output field
`b = h(f) (b_bg I + L)`, where `b_bg = sum_i S_k,i beta_i h_i(f)` is the
coherent field that the inputs carry to the port and `h` is the outbound
transfer. `g1()` and `g2()` evolve `b_in rho` and `b_in rho b_in†` under
`G`, without the trace constraint, and read `b_out†` and `b_out† b_out`. As on
the product space, they raise for an amplifier or a thermal field.

Under `chip.with_space(ExcitationCutoff(max_total=K))`, `K` is the chip's, and
the call does not check it. Without a cutoff, with `options=None` and more than
64 product states, the methods choose `K`. They start at `K = 2`, solve the
whole grid, and solve it again at `K + 1`. At every grid point, they compare:

- in `finite_power()` and `measure()`, the output means, mode amplitudes and photon numbers, and in `measure()` also the noise spectra at every offset when `N <= K + 1` has at most 1024 labels;
- in `output_spectrum()`, its spectra at every offset and its fluxes;
- in `g1()` and `g2()`, the raw correlations at every delay and the two intensities.

`steadystate()` needs no `K`, because the vacuum is exact. For two values `a`
and `b` of one quantity, the change is `|a - b| / (max(|a|, |b|) + 1e-3 m)`,
where `m` is the largest value of that quantity on the grid. The floor gives an
absolute tolerance of `1e-9 m`. In `finite_power()` and `measure()`, one value
holds all ports at one grid point. There `|.|` is the Euclidean norm, or the
Frobenius norm of the IQ matrix at one noise offset. This joint criterion does
not check a port whose values are much smaller than the others' to its own
size. If any change is above `1e-6`, `K` increases by one. The result is the
solve at `K`, not at `K + 1`. One warning per call gives `K`, the retained
dimension and the result of the check. If `K + 1` does not fit in memory, the
check does not run, and the warning says so. If `K` reaches the largest total,
the solve is exact and the call does not warn. `options={}` solves the
backend's stationary Liouvillian, on the full product space when there is no
cutoff.

The check measures the change from `K` to `K + 1`, not the error of the
result. The weight on `N = K` alone does not measure the error either, because
a conserving jump can move population into a sector above the cutoff. Take a
resonator with `0.05 / ns` port decay and a jump `a³ b†³` at the same rate into
a slow mode `b`. At `K = 2`, `N = 2` holds a weight of `5e-4`. The resonant
ratio is `-0.997` there and `-0.814` in the full space, because the jump fills
`N = 3`. The automatic route keeps `K = 4`, at `7.8e-8 |beta|` from the full
solve.

Diagnostics report `excitation_cutoff`, `cutoff_choice` (`"explicit"` or
`"automatic"`), `retained_dimension`, the weight on each total, and the
residuals and state errors of the solves. An automatic choice adds
`cutoff_check`. Under an explicit
cutoff, the stationary engine reports `excitation_cutoff` and
`retained_dimension` with its own diagnostics. Every result also records its
route (§6.6).

### 6.6 Solve routes

Source: [`quchip/engine/routes.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/routes.py)

One planner chooses how each stationary calculation solves. Each result records
the choice as a `SolveRoute` under `diagnostics["route"]`, and
`diagnostics["solver"]` holds its name. The record gives the `reason` for the
route, the `dimension` that it solves on and the array library of the solve
(`arrays`, `"numpy"` or `"jax"`). Its `peak_bytes`, the estimated peak memory,
is calculated on request.

| Route | Solves | Workflows |
|---|---|---|
| `linear_response` | mode equations of a passive harmonic chip | `VNA.sweep()`, `VNA.measure()`, `VNA.resonances()` |
| `vacuum_response` | vacuum and one-excitation states under a weak probe | `VNA.sweep()`, `VNA.resonances()` |
| `conditional_response` | adjacent-sector coherences of a frozen dressed reference | `VNA.sweep(when=...)` |
| `sector_stationary` | labels with `N <= K` (§6.5) | `steadystate()` and the stationary VNA workflows |
| `stationary_resolvent` | backend's stationary Liouvillian on the product space or on the labels of a cutoff | every stationary workflow |

The sector route solves with SciPy on the host for both backends. On dynamiqs,
it therefore does not use JAX arrays, and its reason says so. Traced values
use the backend's Liouvillian. Static dressed queries diagonalize the sector
of their label when the static model conserves `N` or the chip has an
excitation cutoff, and the full static model otherwise. `chip.describe()`
reports this choice.

## 7. Counter-rotating terms

A counter-rotating band changes the excitation in the same direction as the
frame or carrier rotation, so the two phases add instead of cancelling. In full
capacitive coupling, `ab` and `a†b†` are counter-rotating. In a single-tone
drive, the fast partner of the real field is counter-rotating relative to the
chosen transition band. In the rotating frame of two detuned modes with
frequencies `f_a` and `f_b`, exchange terms rotate at about `|f_a - f_b|` and
counter-rotating terms at about `f_a + f_b`.

The RWA usually drops these terms because they are much faster and average
out. Choose `RWA()` to remove structural first-order counter-rotating bands, or
`Exact()` to keep every term explicitly (§6).

## 8. Observables and demodulation

Source: [`quchip/engine/observables.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/observables.py)

The engine decomposes dict-form `e_ops` into the same excitation bands as the
frame logic. After the solve, it recombines them with the per-device
demodulation frequencies `ResolvedFrame.demod_freqs = f_ref - f_frame`.
`result.expect` is therefore a co-rotating readout: observables are reported in
each device's `reference_freq` frame, whatever the solver's integration frame.
Transverse observables (`<a>`, `<sigma_x>`) come back as the slow demodulated
envelope that a lab readout produces. Each coherence in it turns at its own
transition frequency minus `f_ref`: the dressed `0 -> 1` coherence at
`Δ = f_01 - f_ref`, and in a multilevel device the `1 -> 2` coherence at
`f_12 - f_ref`. Diagonal observables (populations) are frame-invariant.

Under `"rotating"`, the default of a solve on a chip without a frame (§4.1),
the integration frame is the reference frame, so demodulation is a no-op and
`result.expect` equals `Tr(O·rho)` on the states that `result.states`
returns. In the `"lab"` frame, demodulation removes the full reference
frequency. The raw band sum, the observable in the integration frame, stays
available on each `ObservableTrace` as `.raw`.

### 8.1 Stationary solves and scattering

S-parameters follow the engineering `e^{+jωt}` convention, where `j = i`.
`VNA.sweep()`, `finite_power()` and `measure()` conjugate the complete physics
response at the instrument boundary, including network propagation. VNA probe
and pump amplitudes use this convention, and the engine conjugates them on
entry. Internal Hamiltonians, network declarations, mode observables and the
engine equations below keep `e^{-iωt}`. A one-port resonator therefore reports
`1 - κe/(κ/2 + j 2π(f-f0))`, and a positive cable delay adds
`exp(-j 2π f τ)` for each traversal.

`Chip.steadystate()` solves `L(rho_ss) = 0` together with `Tr(rho_ss) = 1`.
It requires a static resolved Hamiltonian and a unique normalized stationary
state. If frame and approximation resolution leave dynamic terms, the
stationary APIs raise. Periodic (Floquet) stationary states are not
implemented.

`VNA.sweep()` returns the complete scattering matrix between the selected
planes. With fixed pumps, it adds continuous-wave port terms in their
stationary tone frames and, at each frequency, solves one pumped operating
point with one shifted-Liouvillian factorization for all input columns. The
passive-linear route uses one multi-right-hand-side mode-space solve.
Small-signal scattering differentiates the output mean around the fixed-tone
state,

```text
S_ji(f) = d <b_out,j> / d beta_in,i  at beta_probe -> 0.
```

Around a phase-sensitive operating point, the full response is
`delta <b_out> = S delta beta + T conj(delta beta)`.
`result.conjugate_matrix` stores `T` with the same `[..., output, input]`
layout as `result.matrix`, and `result.t(output, input)` selects one entry.
The stationary route gets `S` and `T` from the same shifted-Liouvillian
factorization. The passive-linear route reports zero for `T`.

For a pump-free model that is static in the lab frame, the general route solves
the lab-frame stationary state once per chip parameter point. It keeps the
declared approximation, including counter-rotating terms under `Exact()`. For
the input operator `K_i = sum_k conj(S_ki) L_k`, its normal response satisfies

```text
(L + i 2π f) X_i = [K_i†, rho_ss],    Tr(X_i) = 0.
```

The output adds the system term `Tr(L_j X_i)` to the direct term from the
resolved scalar `S`. Reference-plane transfers and output mixing use the
physical probe frequency `f`. The conjugate input oscillates at `-f`, so it
does not enter the same-carrier `T` at nonzero `f`. At DC, both input
quadratures share one carrier, and `T` keeps their conjugate response.
Fixed-pump calculations keep their stationary-frame `S` and `T`. This
weak-probe route does not give a periodic finite-power stationary state.

`VNA.finite_power()` instead adds a probe of amplitude `beta` at one selected
input. It solves the stationary Liouvillian in the probe frame at each grid
point and reports the mean field at every selected plane,

```text
<b_out,j> = H_out,j(f) [sum_i S_ji beta_boundary,i + <L_j>].
```

This path uses the same reference-plane and hidden-channel bookkeeping as the
small-signal response, without a mode-space shortcut, and every selected plane
must resolve at the probe frequency. The ratio `<b_out,j>/beta` tends to
`S_ji` as `beta -> 0` if no fixed pump leaves a coherent mean at that plane
and carrier. At finite `beta`, the ratio is a stationary mean-field response,
not a small-signal S-parameter, and it does not describe sweep-rate hysteresis
or metastable branches. On a chip that conserves the total excitation, the
solve can keep only the labels with `N <= K` (§6.5).

For a pump-free passive-linear model, the same authored expressions also admit
the mode-space form

```text
d a/dt = A a + B b_in,          b_out = C a + S b_in,
A = -i Omega - C† C / 2,  B = -C† S.
```

The engine takes this route only when the retained Hamiltonian is static,
quadratic and number conserving, and every collapse operator is linear in the
mode lowering operators. It applies the Hamiltonian `2π` at the assembly
boundary, also to the series-product Hamiltonian that the `PortNetwork`
generates. It computes the transfer factor `exp(+i 2π f τ)` of each inbound and
outbound reference leg at each frequency. The selected backend evaluates the
undecorated Markov response

```text
S_out,in(f) = S_out,in + C_out (-i 2π f I - A)^(-1) B_in,
```

and the engine then applies the inbound and outbound factors.

`VNA.sweep()` extends this form to pump-free models that conserve the total
energy-level index `N` but are not harmonic, e.g. Duffing transmons, pure
dephasing, or reduced chips with retained terms. The resolved static
Hamiltonian must conserve `N`, as the sector engine proves (§9). This includes
ordinary `RWA()` models and structurally conserving `Exact()` models. Every
port must lower `N` by one, so a cascade-generated term `Im(L2† S2 L1)`, as on
a shared feedline, also conserves `N`. Every other channel must lower `N` by
one or conserve it, and every input must be vacuum. The vacuum is then
stationary, and to first order in the probe the coherences `|1_j><0|` stay in
the one-excitation block. The engine projects the lab-frame model onto the
vacuum `|0>` and the states `|1_j>` that raise one device to its first excited
level:

```text
Omega = H_1 - <0|H|0> I,     C_k = <0|L_k|1>,     c_k = <0|L_k|0>,
A = -i Omega - sum_k (L_k† L_k)_1 / 2 + sum_k [c_k^* (L_k)_1 - |c_k|^2 / 2],
```

where `X_1 = <1|X|1>` is the one-excitation block. A lowering channel has
`(L† L)_1 = C† C` and `c = (L)_1 = 0`, which gives the harmonic `A` again. The
mode-space formula then gives the infinitesimal response exactly within the
declared vacuum and one-excitation blocks, and its size is the number of
devices. Ordinary Fock anharmonicity and cross-Kerr terms vanish in these
blocks, once the required states exist. Charge and phase discretizations can
change the blocks and still require convergence. The engine builds both blocks
from the local operators of the terms and channels, as for the sectors of §9,
without embedding them into the full product space. VNA diagnostics name the
route `"vacuum_response"`. Finite-power and noisy measurements keep the
harmonic condition above, because only a harmonic model responds linearly at
finite amplitude.

Nonlinear models outside these conditions, and pumped, active, dynamic or
opaque operator models, keep the stationary-Liouvillian route. The selection is
structural and independent of the value of a traced parameter. Active local
terms also keep the general route, because a weight-only RWA does not show
whether a local parametric term is off resonance in its authored frame.

`VNA.sweep()` covers small-signal scattering and `VNA.finite_power()`
stationary finite-power mean fields. Ring-up, ring-down, wave packets and other
time-resolved fields require a scheduled external-plane input in a
`QuantumSequence`. Fixed finite pumps stay valid VNA operating-point fields.

Uniqueness checks and residuals run with the solve. Condition numbers and
positivity checks run only when you read them, and a skipped rank or condition
diagnostic is `None`, not a successful check.

#### Poles of the weak-probe response

`VNA(chip).resonances()` diagonalizes the drift `A` of either weak-probe route
and returns one record per eigenvalue `λ_p`. The pole has the lab-frame
frequency `f_p = −Im λ_p / 2π` and the response linewidth `κ_p = −2 Re λ_p` in
1/ns, the full width at half maximum of `|r_p / (f − z_p)|^2` in angular units.
Near an isolated pole, where the other poles add little and the transfers vary
slowly, it is the width of `|S − T D|^2`, with `T` the transfer of the entry.
The linewidth includes every loss and dephasing term of `A`, so it can exceed
the decay rate of a population. For the same reason, `f_p` can differ from the
closed dressed frequency of `chip.freq()`. The linewidths sum to
`−2 Re tr A = tr(C† C) − 2 Im tr H`, which includes the internal loss of poles
that `eliminate()` retained.

A diagonalizable drift `A = sum_p λ_p v_p u_p†`, with right and left
eigenvectors normalized by `u_p† v_q = delta_pq`, expands the response in
simple poles. In the convention of `VNA.sweep()`,

```text
S_ji(f) = conj(T_out,j(f) T_in,i(f)) [D_ji + sum_p r_p,ji / (f − z_p)],
z_p = conj(i λ_p / 2π) = f_p + i κ_p / 4π,
r_p = conj((i/2π) (C v_p)(u_p† B)),     B = −C† S_d,     D = conj(S_d),
```

where `T_in` and `T_out` are the transfers of the reference planes, such as a
cable delay. `result.scattering(frequencies)` evaluates this sum and equals
`VNA.sweep()`. The residue from input i to output j is the product of
`c_j v_p` and `u_p† b_i`, with `c_j` row j of `C` and `b_i` column i of `B`.
A pole is dark when no selected output sees it or no selected input excites
it: `|c_j v_p| <= 1e-10 |c_j| |v_p|` for every j, or
`|u_p† b_i| <= 1e-10 |u_p| |b_i|` for every i. Its residues are then
numerically negligible, and it stays an eigenvalue of the drift. Each port is
compared with its own scale, so a weak port does not make a pole dark. The
composition of a pole is the squared magnitude of its normalized right
eigenvector in the mode basis, and `dominant` names its largest entry. The
composition describes the eigenvector, not an occupation probability. At a
repeated pole, the residues and compositions of the copies depend on the
chosen eigenvectors. Only the sum of their residues is defined, and it can
cancel in an entry.

Near an exceptional point, two poles merge and their eigenvectors become
parallel, so the residues grow and cancel. `resonances()` raises when two
poles are ill-conditioned and too close to resolve reliably: both condition
numbers `s_p = |u_p| |v_p|` exceed 10, and the separation is below
`100 ε |A_0|_F (s_p + s_q)`, where `ε` is the machine precision and
`A_0 = A − i (Im tr A / n) I` is the drift without its mean frequency. The
test finds poles that the eigensolve cannot resolve. It does not prove that
the drift is defective. The eigensolve keeps the damping of `A`, so adding the
mean frequency back does not cancel digits of a narrow linewidth. Frequencies
and linewidths are differentiable under JAX. The composition and the residues
need eigenvector derivatives, which JAX does not define for a general matrix,
so they are computed only when read, and a derivative of them raises.

#### Conditional weak-probe response

`VNA.sweep(frequencies, when={q: 1})` selects the dressed eigenstate assigned
to that bare label, with unspecified devices in level zero. It accepts zero or
one total excitation and requires an unambiguous reference. It uses the same
conservation, channel and vacuum-input checks as the vacuum route. Fixed pumps
and downstream output networks are unsupported.

Let `rho_r = |r><r|` and `K_i = sum_k conj(S_ki) L_k`. The engine freezes
`rho_r` in the probe source and solves the frequency-domain equations

```text
(-i omega - L) X_i = -[K_i†, rho_r],
S_ji(omega) = S_ji + Tr(L_j X_i).
```

These equations use the internal physics convention, and the VNA boundary
applies the reference-plane transfers and the engineering conversion. The
reference has a definite excitation number and the channels preserve this
covariance, so the result has no phase-conjugating response.

For a one-excitation reference, the source occupies `(N_ket,N_bra) = (2,1)`
and `(1,0)`, and the second block includes stimulated emission. Hamiltonian
and number-conserving channels act within each block, and lowering jumps
recycle coherence from `(2,1)` to `(1,0)`, so both blocks form a closed
response space. All internal and external loss channels contribute before
projection, including off-diagonal damping between close modes, and
number-conserving channels keep their recycling terms. The sector engine
supplies the complete blocks without building full product-space operators.
Transmon and resonator level cutoffs must keep the EF and two-photon states
that the question needs.

This is the exact resolvent of the specified source within the declared model.
It is an exact stationary susceptibility only when `L(rho_r) = 0`. Otherwise,
`approximation="frozen_reference"` records that the reference population does
not evolve in the source, while decay and recycling remain in the response
generator. The approximation does not give an acquisition-time average after
state preparation, a postselected trajectory or IQ shot noise. Reference
evolution must be slow over the relevant response memory time.

Diagnostics report the reference overlap, sector dimensions, coherence count,
relative solve residual, and the rates

```text
reference_escape_rate = sum_k (<L_k† L_k>_r - |<L_k>_r|^2),
reference_stationarity_residual = ||L(rho_r)||_F.
```

Both rates are in `1/ns`. Their ratio to a response linewidth is a scale
estimate, not a bound on the error of a reflection fit. Nearly dark modes can
have longer memory times, and relative errors near a reflection zero are
ill-conditioned. An explicit ground reference uses the vacuum response exactly.

Backends solve the adjacent-sector blocks in descending excitation order,
directly up to 256 coherences per block and iteratively, with checked
residuals, above that. `block_condition_number` is `None` unless every block
uses the direct solve. JAX derivatives hold within a fixed, unambiguous
reference assignment and require converged response solves.

### 8.2 Captured noisy VNA measurements

`VNA.measure(frequencies, amplitudes, input=..., outputs=...)` prepares one
stationary operating point for each probe and sweep coordinate and captures
means and normally ordered IQ cross-spectra. Receiver integration,
calibration and Gaussian draws act on these captured arrays. They never call a
stationary solver or consult a later mutable chip. `measurement.parameters`
records the numerical model parameters in flattened sweep order, and
`noise_frequencies` records the stored offset grid. The finite-power ratio is
the output mean divided by the probe amplitude, not the small-signal
derivative around a separate pump.

At capture, the instrument conversion sends `I -> I`, `Q -> -Q` on both
covariance axes. Spectral covariances are also complex conjugated, and they
keep the physical sideband labels `f_carrier + offset`. Receiver transfer
functions, calibration and sampled IQ fields use the engineering convention.
`VNA.g1()` uses the same field convention, including cross-port phases.
Intensity correlations and scalar noise power are invariant under the
conversion.

For eligible passive harmonic models, `measure()` uses the same compact
mode-space lowering and backend response solver as `sweep()`. With response
matrix `T(f)` and input occupations `n`, the normal output spectrum at the
Markov boundary is `T(f) diag(n) T(f)†`. Subtracting `S diag(n) S†` gives the
device-generated excess to the shared downstream propagation, which adds the
direct sources once. The result is the exact stationary Gaussian field
solution, including thermal fluctuations, without a Fock-space truncation. All
modes must decay. Concrete acquisitions check stability, and traced paths
keep the usual host-validation limit.

Thermal device collapse declarations, nonlinear or active terms, fixed pump
configurations, branched output graphs and explicit solver options keep the
operator-space acquisition, and `options={}` requests that general route for
cross-checks. On a large chip that conserves the total excitation, the
operator-space acquisition solves on excitation sectors (§6.5), and
`options={}` requests the full product space.

Measurements also capture internal Fock-mode observables. `mode_amplitude(r)`
returns `<a_r>` in the stationary frame that `mode_frequency(r)` reports, and
`photon_number(r)` returns `<a_r† a_r>`. Both follow the measurement sweep axes
and keep the full declared input wiring. The compact backend solves
`A N + N A† + B diag(n) B† = 0` for the centered normal covariance
`N_ij = <delta a_j† delta a_i>`, then adds its diagonal to the coherent
occupation `|<a_r>|²`. This covariance is independent of the output spectral
grid. The general path evaluates the authored `a` and `n` operators in the
resolved basis against the solved reduced density matrix. It therefore keeps
nonlinear and active physics and the declared truncation. Receiver processing
does not change internal occupation.

An attenuator, isolator load or `network.termination()` can declare
`thermal_occupation=n`, a finite, non-negative mean thermal population in
quanta, and vacuum stays the default. Amplifiers require input-referred
symmetrized `added_noise` in quanta, subject to the phase-preserving quantum
floor. Both noise values are constant across the modeled band, including
frequency sweeps, and quchip does not infer a temperature or reference
frequency. Attenuators accept `loss_db` in place of `eta`, and amplifiers
accept `gain_db` in place of linear power gain.

For a full unitary scattering matrix `S`, input j couples through
`K_j = (S† L)_j`. In addition to the vacuum `sum_i D[L_i]`, its thermal
population adds `n_j D[K_j] + n_j D[K_j†]`. Thermal input is therefore part of
the stationary and transient quantum dynamics, not a second independent
device bath. SLH composition keeps the surviving input's state and rejects an
independent thermal declaration on an input that is connected away. Arbitrary
passive nonideal components use `network.component()` with unitary scattering
and explicit dissipative terminals connected to declared loads. Insertion loss
and isolation numbers alone do not define this matrix.

Normal output spectra include the direct term `S diag(n) S†` and the
input-system interference in the regression source
`B_i = (L_i-<L_i>) rho + sum_k (S diag(n) S†)_ik [L_k,rho]`, which follows
normally ordered input-output correlations ([Gardiner and
Collett][input-output]). This interference prevents double counting of
fluorescence on top of an incident thermal field at equilibrium. Real IQ
sources built from `B` keep normal, anomalous and cross-output second
moments. `output_spectrum()` selects the scalar normal spectrum from the same
calculation as `measure()`.

The internal Fourier convention is
`integral exp(+i 2π offset τ) <δb†(0) δb(τ)> dτ`, so a mode above the carrier
peaks at positive offset. The equivalent engineering spectrum uses
`exp(-j 2π offset τ)` and the conjugated field correlation, on the same
physical frequency axis.

Physical source budgets separate directly propagated fields from
`device.correlations`, which includes nonlinear response and interference and
so is not necessarily positive or independently sampleable. A matched
absorptive filter declares `thermal_occupation` and emits `(1-|H(f)|²)n` in
each direction. A scalar `H(f)` without that declaration does not imply an
absorptive thermal model.

Colored emission can propagate to external outputs, but quchip rejects colored
noise that feeds a quantum coupling, because a colored reservoir needs an
explicit dynamical model. Source color follows propagation order, so a vacuum
filter before occupied attenuators does not color their emission. Source
backaction follows `S†L`, so fields mixed only after the device can carry
filtered thermal noise without heating the device.

`measurement.noise_spectrum(output)` recovers the normally ordered scalar
spectrum from the captured IQ matrix. It keeps the upper and lower sideband
asymmetry and excludes the coherent carrier and the final receiver vacuum.
`unit="W/Hz"` multiplies by `h f_absolute`, and `unit="dBm/Hz"` reports the
power ratio to 1 mW/Hz. Power conversions require positive absolute sideband
frequencies. Receiver source budgets stay integrated IQ covariances in
photons/ns, whose trace is the complex-field variance, not a spectral power
density.

Acyclic output networks can put an amplifier before a splitter or between
passive components. The compiler keeps a unitary Markovian boundary and
captures a separate directed field map with source cross-spectra. These
sections cannot feed quantum couplings or instantaneous feedback loops.
Amplifiers keep the output-line convention of §3.3.2, and `added_noise` never
implies reverse HEMT emission. An exposed reference section keeps its separate
inbound and outbound traversals. Branched reference networks support
stationary fields only: transient output observables and direct SLH
composition reject them, so compose their `PortNetwork` before you resolve it.
Their VNA response uses the general stationary solver.

`IQReceiver(integration_time=T)` applies a normalized boxcar with frequency
weight `sinc(offset*T)²`. For `b=I+iQ`, ideal heterodyne detection adds one
complex vacuum quantum at the final plane, or `1/2` on each IQ diagonal. For
the weight `w(t)=1/T` on `[0,T]`, `integral |w(t)|² dt=1/T`, so white normally
ordered noise `N` gives `Var(I)=Var(Q)=(N+1)/(2T)`. These IQ conventions follow
[Wiseman and Milburn][wiseman]. Joint outputs keep their complex cross-spectrum
and relative delays, and independent detector vacuum is added once for each
output. Calibration multiplies the mean and transforms both covariance axes.
Zero probe amplitude leaves field statistics defined and ratios undefined.

An optional receiver `transfer(offset)` is a digital complex amplitude
response that also scales the mean by its DC value. White noise through a
boxcar is integrated analytically, and colored terms and digital filters use
the captured grid. The receiver's quadrature and edge checks are local, so they
do not prove that a spectrum has no unsampled feature. Use a wider or finer
capture for wide bandwidths or short integration times. Concrete validation
runs outside JAX tracing, and deterministic integration and keyed
reparameterized draws stay differentiable on fixed shapes.

Gaussian samples reproduce the captured second moments. They do not supply
higher-order non-Gaussian photon statistics or continuous correlated records.
Normalized `g1` and `g2` for network thermal fields require a detection
bandwidth, and the unfiltered correlation API rejects them.

## 9. Dressing

Sources: [`quchip/chip/chip.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/chip.py), [`quchip/chip/analysis.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/analysis.py)

`Chip.dress()` diagonalizes the static lab-frame Hamiltonian that
`chip.approximation` keeps, assigns bare product states to dressed eigenstates
by overlap, and stores a `DressedResult` with:

- eigenvalues and lazily materialized eigenstates
- bare-to-dressed state assignments and the assigned eigenvalue for each bare label
- assignment overlaps and labels below the requested overlap threshold
- the dressed eigenvector matrix used by dressed-basis analysis

`Chip.dress()` is always intrinsic static lab-frame analysis and is not part of
the runtime frame transform. A resolved `EngineResult` also provides `dress()`,
which diagonalizes that snapshot's selected frame and approximation. If the
snapshot has dynamic Hamiltonian terms, `dress(at_time=...)` is required, and
it evaluates their signal programs at that instant. The result is an
instantaneous eigensystem, not Floquet or cycle-averaged analysis.

Each dressed eigenvector `|psi_j>` has a fixed phase, set by a rule that
differs from the local rule of §2.1. quchip multiplies it by `conj(a)/|a|`,
where `a = <b_k|psi_j>` is its amplitude on the bare product state `b_k`
assigned to it, so this amplitude is real and nonnegative. The one-to-one
assignment can give a vector a zero amplitude on its own label. When `|a|` is
not above the square root of the floating-point epsilon, the anchor is the
first amplitude within a relative `1e-6` of the vector's largest magnitude.
`Chip.state()`, the eigenstates and eigenvector matrix of `DressedResult`,
dressed-basis operators, drive matrix elements, sector eigenvectors and label
initial states use this convention. The convention is piecewise. Backend
agreement and derivatives need resolved eigenvalue gaps, a locally constant
assignment, and a stable anchor: an assigned amplitude that stays above the
floor, or a fallback amplitude that stays the unique near-largest one. A phase
cannot fix a rotation, so an exactly degenerate eigenspace keeps the basis of
the eigensolver.

`Chip.freq()` evaluates dressed `0 -> 1` frequencies through a traceable
labeled eigensystem, and `DressedResult` does not store them. Frequencies,
states, Kerr shifts and drive matrix elements all use the chip's
approximation, so `RWA(keep_bands=...)` applies here too. Choose `Exact()` to
include counter-rotating bands and their Bloch–Siegert shifts.

When the static model conserves the total energy-level index `N`, labeled
queries diagonalize only the excitation sectors of their labels, under the
conditions of the exact route in §10.4. The sector of `N` holds the product
states whose levels sum to `N`. Each term's local operator on its own support
fills the sector block, and two states couple only when they agree outside
that support. The engine assigns labels within each block with the overlap
rule of the full eigensystem. `Chip.energy()`, `Chip.freq()`,
`Chip.transition_frequency()`, `Chip.dispersive_shift()`,
`Chip.dressed_anharmonicity()`, `Chip.kerr_matrix()`,
`Chip.drive_matrix_elements()`, `Chip.state()`,
`Chip.effective_subspace_hamiltonian()` and `effective_hamiltonian()` take
this route, and `Chip.state()` maps the sector eigenvector into solver
coordinates with the local energy eigenvectors. Ground-conditioned first
transitions and single-excitation exchange need `N <= 1`. Ground-conditioned
anharmonicities, ZZ and dispersive shifts need `N <= 2`. Other conditioned
transitions use the sectors of their specified labels. Drive matrix elements
connect the initial and final sectors with the local operator's rectangular
block.

The sector assignment keeps each label in its own sector. It agrees with the
full-space assignment when the assigned overlaps are nonzero and the
eigenvalue gaps are resolved. At an exact degeneracy between sectors, the
sector state is an eigenstate of both `H` and `N`. Memory checks precede local
projection, sector enumeration and the full-space fallback. Traced blocks use
differentiable eigensolves, and derivatives require separated eigenvalues and a
locally stable assignment. These static queries do not change pulse
simulation. `Chip.dress()`, `Chip.dressed_spectrum()` and dressed-basis
operators still diagonalize the complete chip, and other chips keep that route
for every query. Under an active `ExcitationCutoff` (§6.4), the same labeled
queries and `Chip.state()` diagonalize the cutoff classes of their labels
instead, with the same assignment and phase rules.

A level label such as `{"q0": 1}` or `"eg"` names the dressed eigenstate
`|d_k>` assigned to label `k` wherever quchip takes one: in `Chip.state()`, as
the `initial_state` of a solve, as a target of `overlap()` and
`amplitude_array()` (§4.2), and in `Chip.superposition()` and
`Chip.compact_state()`. `Chip.bare_state()` gives the bare product state, and
its `(amplitude, label)` pairs give bare superpositions. Operators stay local:
`population(q, k)` is the projector on level `k` of device `q`, so on a
dressed state it reads less than the overlap with the state's label.

A label `initial_state`, or an omitted one, uses the eigenstate of the undriven
static lab-frame Hamiltonian that the solve's approximation keeps, with the
phase convention above. Under an excitation cutoff, it uses the eigenstate of
`P H_0 P`. The solve places it at `t0 = tlist[0]` as the label target of §4.2:

```text
psi_solve(t0) = U(t0)† |d_k> exp(-i 2π t0 Σ_i f_frame,i k_i)
```

An omitted state is the all-ground label, so it equals `{}`. The labels of an
uncoupled chip start in their bare products in every frame. An `initial_state`
label takes integer levels, and `Chip.bare_state(...)` gives a product of local
kets. When the solve uses the chip's approximation and cutoff, a label at
`t0 = 0` starts from the same vector as `chip.state()`. A ket such as
`chip.state()` or `chip.superposition()` is an integration-frame vector at `t0`
and is not moved. At `t0 != 0`, it can differ from its label by the frame
phases of its components. At an exact degeneracy, the label names whichever
eigenvector the eigensolver returns.

With ordinary couplings, the default RWA leaves the all-ground bare product
unchanged. When the static model of the solve's approximation conserves `N`,
the bare product is the only state with `N = 0`. The solve then starts from it
with no static model or eigensystem. Other models first apply the static
Hamiltonian to the bare product and diagonalize only when it is not an
eigenstate. Retained bands, effective terms, or network Hamiltonian terms that
couple the vacuum can dress an RWA default. An excited label diagonalizes the
sector or cutoff class of its total when the model conserves `N` or has a
cutoff, and the full static model otherwise.

### 9.1 Dressed drive matrix elements

Sources: [`quchip/chip/analysis.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/analysis.py), [`quchip/control/equipment.py`](https://github.com/quchip/quchip/blob/main/quchip/control/equipment.py)

For a drive line `j` with local Hamiltonian operator `D_j`, quchip defines the
dressed matrix element

```text
m_j^(fi) = <f~|D_j|i~>
```

with the final dressed state as the matrix row and the initial dressed state as
the column, so `Chip.drive_matrix_elements((initial, final))[j]` reads
`[final, initial]` from `U† D_j U`. The device shorthand
`chip.drive_matrix_elements(q)` selects the dressed transition from the
all-ground state to the state labeled by one excitation in `q`, and explicit
`(initial_mapping, final_mapping)` arguments select arbitrary transitions. The
dressed eigenvectors follow the phase convention of §9, so each has a real,
nonnegative overlap with its assigned bare state. This removes
backend-dependent eigenvector signs from comparisons between conditioned
transitions, e.g. the sum and difference for the weak-drive `IX` and `ZX`
coefficients.

On a reduced chip, the physical drive operator first passes through the same
retained coordinate maps as a scheduled pulse, so its support can grow across
surviving devices. The query still uses excitation sectors when the retained
Hamiltonian conserves total excitation number.

`drive_matrix_elements` evaluates the physical drive operators and does not
apply the signal chain. `ControlEquipment.crosstalk_matrix()` represents
declared control-line mixing separately. In this matrix, column `j` is the
source line, row `l` is the victim line, and each entry has an amplitude, phase
and delay. You can combine the returned matrix elements with those line
phasors in a weak-drive effective-Hamiltonian model. The two pieces stay
separate to distinguish the dressed quantum response from microwave-path
mixing. This projection follows the effective driven-Hamiltonian treatment of
[Magesan and Gambetta][magesan].

### 9.2 Weak-drive cross-resonance susceptibility

For a drive operator `D_c` on control `c`, define dressed raising elements
conditioned on the control state:

```text
m_z = <z_c, 1_t~|D_c|z_c, 0_t~>,    z in {0, 1}.
```

A scheduled real tone of peak amplitude `A` has resonant amplitude `A/2`
(§6.2). Define complex X/Y coefficients `C_I=IX+i IY` and `C_Z=ZX+i ZY`:

```text
H_eff/h = [Re(C_I) I⊗X + Im(C_I) I⊗Y
         + Re(C_Z) Z⊗X + Im(C_Z) Z⊗Y]/2
C_I/A = (m_0+m_1)/2
C_Z/A = (m_0-m_1)/2
```

`analyze_cr_susceptibility` returns these complex ratios as `IX_per_amplitude`
and `ZX_per_amplitude`, and also the unhalved `m_0` and `m_1`. Imaginary parts
describe Y quadratures, not non-Hermitian Hamiltonian terms, and a drive phase
rotates the common quadrature. For a constant resonant tone, the ideal
weak-drive ZX90 duration is `1/(4 |A C_Z/A|)` in ns.

The calculation runs no pulse. It excludes strong-drive Stark shifts,
pulse-bandwidth leakage, and echo or cancellation calibration. The
effective-Hamiltonian treatment follows [Magesan and Gambetta][magesan] and
[Malekakhlagh et al.][malekakhlagh].

### 9.3 Dressed Kerr matrix

`Chip.kerr_matrix()` reads one labeled dressed spectrum and returns a symmetric
matrix in `chip.devices` order. A conserving chip supplies it from its sectors
with `N <= 2` (§9). For distinct devices,

```text
K[i,j] = E(1_i,1_j) - E(1_i) - E(1_j) + E(0),
```

the same full-pull convention as `Chip.dispersive_shift(i, j)` and the static-ZZ
coefficient for two qubits. On the diagonal,

```text
K[i,i] = E(2_i) - 2 E(1_i) + E(0),
```

which is `Chip.dressed_anharmonicity(i)`. A device with fewer than three
resolved levels has `NaN` on the diagonal, but its defined off-diagonal entries
stay available. Every entry comes from the complete dressed chip, not from a
matching authored edge. For example, an isolated `KerrCavity` with
`H/h = f*n - K*n*(n-1)` has `K[i,i] = -2*K`.

### 9.4 Subspace propagators

Sources: [`quchip/results/propagator.py`](https://github.com/quchip/quchip/blob/main/quchip/results/propagator.py), [`quchip/control/sequence.py`](https://github.com/quchip/quchip/blob/main/quchip/control/sequence.py)

`QuantumSequence.subspace_propagator(devices, levels=(0, 1))` takes the labels
that give each selected device one of `levels` and every other device level 0.
One batch starts each label `j` in its dressed state at `t0 = tlist[0]` (§9)
and projects the final states at `T = tlist[-1]` on the label targets of §4.2:

```text
A_ij = <target_i(T)|psi_j(T)> = exp(i 2π T F_i) <d_i|U_lab(T, t0)|d_j> exp(-i 2π t0 F_j)
```

with `F_k = Σ_d f_frame,d k_d` and the integration-frame frequencies
`f_frame,d` of §4.1. Every other device starts and ends in level 0. A frame
gives label `k` the energy `E_k = E_0 + Σ_d f_d k_d`, and the record holds

```text
M_ij = exp(i 2π T E_i) <d_i|U_lab(T, t0)|d_j> exp(-i 2π t0 E_j)
```

The frames are:

- `"dressed"`, the default: `f_d = E(1_d) - E(0)` and `E_0 = E(0)`, from the
  static model with the approximation and excitation cutoff of the solve. An
  idle label with at most one excitation keeps amplitude 1, and an idle pair
  of qubits gives `diag(1, 1, 1, exp(-i 2π ζ (T - t0)))`, with the static ZZ
  `ζ` of §9.3.
- `"lab"`: `f_d = 0` and `E_0 = 0`.
- `"integration"`: `E_k = F_k`, so `M = A`, the amplitudes of
  `amplitude_array()`.

Energies from another approximation would add a phase that grows with time.
For example, the Bloch-Siegert shift of `Exact()` is absent from an `RWA()`
solve. The frame factors are phases, so `|M_ij|` and the leakage
`L_j = 1 - Σ_i |M_ij|²` do not depend on the frame. `fidelity(V)` returns the
average gate fidelity over pure states in the labels,

```text
F = (Tr(M† M) + |Tr(V† M)|²) / (d (d + 1))
```

for a unitary target `V` of dimension `d`. Leakage lowers `Tr(M† M)`, so it
lowers `F`. The solves leave out declared collapse channels, and
`omitted_channels` names them, so `M` is the propagator of the Hamiltonian.
The call warns about these channels unless `dissipation=False`, and
`dissipation=True` raises. On dynamiqs, the matrix and the fidelity are
differentiable in chip and pulse parameters (§13).

## 10. Adiabatic elimination and dispersive readout

Sources: [`quchip/chip/transformations/`](https://github.com/quchip/quchip/tree/main/quchip/chip/transformations), [`quchip/chip/sw.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/sw.py), [`quchip/analysis/dispersive_readout.py`](https://github.com/quchip/quchip/blob/main/quchip/analysis/dispersive_readout.py)

`eliminate(chip, target, method="sw"|"exact")` reduces a model, and the target
type selects the reduction.

A device target removes a far-detuned mode. With `local=True`, its generator
reads a patch around the mode (§10.7). Both methods keep their complete
calculated Hamiltonian and each transformed channel from the removed mode and
couplings. `chip.effective_terms` carries the matrix correction beyond the
reported Lamb shifts and mediated exchange, including higher-level
corrections. Intrinsic survivor noise stays separate from inherited loss, and
a common bus channel stays collective. Scalar readout quantities such as
`chi`, `kappa` and the first-transition Purcell rate stay available in
`effective_params`.

A coupling target keeps both endpoints and removes the selected edge. Its
isolated pair sets a coordinate change that applies to the entire chip
Hamiltonian, including parallel and spectator interactions. The exact route is
a full unitary transformation, and SW keeps terms through second order in the
interactions. The retained correction includes per-level shifts, and the
authored endpoint parameters stay unchanged. Surviving component channels
follow a captured operator projection, but their rates stay component-owned.
Removed-component channels already use the retained coordinates. Surviving
controls and collective or thermal baths follow the captured coordinate
changes. Controls that target removed components, and retargeted ports,
require explicit conversion rules.

An effective-terms target keeps every device and edge. It diagonalizes the
selected `EffectiveTerms` exactly, together with the local Hamiltonians of the
devices they act on. Only `method="exact"` is implemented, because a
second-order expansion fails for a strong nonlinearity such as a junction
cosine. Dressed states take the bare label of largest overlap. The rotation
acts on the entire chip, so dressed queries on the reduced chip return the
source spectrum at the chip's approximation (§10.6). `effective_params` reports
each device's `freq_after`, `lamb_shift`, `anharmonicity` and full-pull
`cross_kerr`.

Every retained contribution states its approximation in
`EffectiveTerms.notes`, including the notes of earlier contributions that it
absorbs, and `chip.physics_notes()` reports them under `effective:<label>`.

### 10.1 The χ convention and related quantities

quchip reports the full conditional resonator pull in GHz:

```text
chi_pull = f_r|1 - f_r|0
H_disp/h = (f_mid + chi_Z Z) n_r,    Z=|0><0|-|1><1|
chi_Z = -chi_pull/2,               chi_Z,angular = -pi*chi_pull
```

`eliminate(...).effective_params[q]["chi"]` reads the incoming dressed
spectrum. It uses `E(1,1)-E(1,0)-E(0,1)+E(0,0)`, as
`Chip.dispersive_shift(r,q)` does, with the eigensystem of the selected
Hamiltonian approximation and finite basis, and local elimination uses its
patch spectrum. Between two qubits, the same expression is the static-ZZ
coefficient, not a readout pull.

`fit_a_dress` uses this signed full `cross_kerr`, including its `chi` alias.
Bridge diagnostics report `chi=0`, because their mediated exchange is
represented separately. Gradients follow the labeled-spectrum conditions of
§13.

Second-order dispersive cross-checks use `Delta=f_q-f_r`:

```text
two-level:       chi_pull = 2 g²/Delta
Duffing:         chi_pull = 2 g² alpha/[Delta(Delta+alpha)]
```

Here, `g` is the resolved exchange matrix element, including the endpoint
charge normalization. For a transmon, compare the relevant couplings with both
`Delta` and `Delta+alpha` ([Koch et al.][koch], section III B, equations
3.9–3.12). The two-level estimate `n_crit=Delta²/(4g²)` is not a universal
multilevel bound, because multilevel limits depend on the occupied level
([Blais et al.][blais], section III.3).

`effective_params[q]["kappa"]` sums the removed mode's intrinsic downward
rates, in 1/ns. It includes `2π f_r/Q_internal` and thermal emission when
configured. A ladder channel contributes `(nbar+1)/T1`, or `nbar+1` without a
supplied `T1`. An external port is transformed separately and is absent from
this diagnostic. Do not use it as a total linewidth or detected loss without
checking the channels. For an untruncated harmonic thermal bath, the linewidth
equals the downward minus the upward rate: `T1=1000 ns` and `nbar=0.5` give a
downward rate of `0.0015/ns` but a linewidth of `0.001/ns`. An external-only
port gives a nonzero linewidth even when the intrinsic diagnostic is zero.
Truncated thermal spectra also require cutoff convergence.

### 10.2 Pointer states and readout figures of merit

`analyze_dispersive_readout` assumes a linear cavity with coherent conditional
steady states and vacuum energy-loss channels. `kappa` is the total energy
linewidth, including observed and unobserved loss, and excludes pure
dephasing. `kappa_detected` is the observed part, and `eta=efficiency` is the
detector efficiency. The defaults `kappa_detected=None` and `efficiency=1` mean
complete ideal collection.

```text
delta_r = f_r|0 - f_drive                         [GHz]
delta_j = 2pi*(delta_r + j*chi_eff)                [rad/ns]
alpha_j = -i*eps/(kappa/2 + i*delta_j)
nbar_j = |alpha_j|²
sigma = 1/sqrt(2 eta kappa_detected tau)
SNR = |alpha_1-alpha_0|/sigma
p_err = erfc(SNR/(2sqrt(2)))/2
Gamma_m = kappa*|alpha_1-alpha_0|²/2                [1/ns]
```

Intrinsic loss dephases the qubit without supplying detected information. Zero
detected loss or efficiency gives zero SNR and an assignment error of one half,
while the pointer states and dephasing still use the total linewidth. Thermal
input needs a different noise model.

Give either `eps` in rad/ns or a nonnegative target `n_photons`:

```text
eps = sqrt(nbar_0*((kappa/2)²+delta_0²))
Gamma_m ≈ [8 chi_Z,angular² nbar_0/kappa]/[1+(2delta_0/kappa)²]
```

The second expression is the small-pull limit, and its resonant form follows
[Gambetta et al.][gambetta]. The helper uses the full pointer detuning instead
of this limit. [Krantz et al.][krantz], section V, reviews readout.

`n_crit` optionally applies `chi_eff=chi/(1+nbar_0/n_crit)`. This is a
phenomenological interpolation, not controlled strong-drive theory. `levels>2`
assumes the linear pull `j*chi` and does not derive multilevel cavity
frequencies. `kappa`, `tau` and a supplied `n_crit` must be positive, with at
least two levels, `0<=kappa_detected<=kappa` and `0<=efficiency<=1`. Concrete
invalid inputs raise, and traced callers must enforce the same domains.

The calculation excludes ring-up transients and measurement-induced qubit T1.
Its GHz-to-angular conversions occur at the helper boundary, separately from
Hamiltonian assembly.

### 10.3 The Schrieffer-Wolff route (`method="sw"`)

The eliminated mode's occupation partitions the chip's bare Hamiltonian,
`H = H0 + V`, in GHz before `2π` and under the selected approximation. `P` is
the mode in `|0>`, and `Q` everything else. The generator solves the Sylvester
condition on the cross blocks,

```text
S_ij = V_ij / (E_i − E_j)        (i, j straddling P/Q; E = diag H)
H_eff = P (H + (1/2)[S, V]) P
```

This is the second-order [Schrieffer–Wolff expansion][bravyi]. An exactly
degenerate cross pair with no matrix element contributes zero, with a finite
gradient. A zero gap with nonzero P/Q coupling is singular and raises, and
under tracing quchip also marks numerical outputs as invalid. The
working-precision threshold only selects diagnostic entries and never turns a
nonzero coupling into zero.

Survivor parameters come from indexing `H_eff`: `freq_after(s) = E(1_s) − E(0)`,
and the pair exchange is the `<1_a|H_eff|1_b>` element. Authored direct edges
stay unchanged. A separate mediated edge, named by
`effective_params["exchange"]["coupling"]`, carries the exchange element
`M_ab` of `H_eff − P H P` in the units of an edge authored between the
survivors:

```text
j_eff = Re(M_ab / u_ab),    u_ab = <1_a 0_b|H_edge|0_a 1_b> per unit edge strength
```

A capacitive edge has `u_ab = conj(<0|Q_a|1>) <0|Q_b|1>`, with `Q_s` the
survivor's charge operator in its energy basis. Duffing and resonator survivors
have `u_ab = 1`. A `ChargeBasisTransmon` has `|<0|n|1>| ≠ 1`, and a `Fluxonium`
has an imaginary `<0|n|1>`. A first-transition exchange edge has `u_ab = 1`. If
`u_ab = 0`, the edge has zero strength.

The complete retained correction carries all remaining matrix elements,
including any part of `M_ab` that the edge's real strength cannot carry.
Sequential shifts and detunings use the incoming Hamiltonian's diagonal,
including earlier retained corrections. With `J`, the bridge reduction records
its linearization in the same edge units, from the resolved leg elements
`L_s = <1_s|H|1_c>`:

```text
dJ/domega_c = Re(L_a conj(L_b) / u_ab) (1/Delta_a^2 + 1/Delta_b^2) / 2
```

The diagnostic name `omega_c` denotes the ordinary coupler frequency in GHz.
The flux-drive retarget rule uses this weight (§11). It varies the coupler's
first transition while holding the charge matrix elements fixed. Capacitive
legs give `L_a conj(L_b) / u_ab = g_a g_b |<0|Q_c|1>|^2`, so a resonator or
Duffing coupler gives `(g_a*g_b/2)(1/Delta_a^2 + 1/Delta_b^2)`.

The virtual-state attribution of a retained element `(i, j)` lists the
contribution `(1/2) V_ik V_kj (1/(E_i−E_k) + 1/(E_j−E_k))` of each intermediate
state `|k>` of the eliminated block, with the same guarded denominator. Every
nonzero contribution is listed, so the contributions sum to the second-order
part `(H_eff − PHP)_ij` of the element. Each `|k>` is a bare product state of
the source chip, written as its occupation tuple in device order. For example,
`(1, 1, 1)` holds one excitation in each of two transmons and their bus, a
pathway that exists only with counter-rotating terms. The exchange of a
survivor pair lists its pathways under `pathways`, and
`EliminationResult.pathways(final, initial)` attributes any retained element,
including the diagonal shifts. A local reduction (§10.7) gives the same
attribution, and a device outside its patch keeps its level in every pathway.

### 10.4 The exact route (`method="exact"`)

The exact route computes the eigensystem of the Hamiltonian that
`chip.approximation` retains. `method="exact"` selects the reduction algorithm
and does not restore discarded bands. Parameters come from the labeled dressed
spectrum (`label_eigensystem`, §9), so kept-block energies are exact to all
orders in that Hamiltonian, as residual ZZ requires:

```text
zz(a, b) = E(1,1) − E(1,0) − E(0,1) + E(0,0)        (≡ Chip.dispersive_shift)
```

The complete retained Hamiltonian and its pair exchange come from the
symmetrically [Löwdin-orthonormalized][lowdin] subspace projection
`S^(−1/2) (W E W†) S^(−1/2)`, where `W` is the overlap block and `S = W W†`.
This [des Cloizeaux effective Hamiltonian][des-cloizeaux] has exactly the
labeled energies as its spectrum. It uses the
[canonical Schrieffer–Wolff (direct) rotation][bravyi] evaluated to all
orders, provided the kept/dressed overlap block `W` is nonsingular.
`method="sw"` truncates the corresponding effective Hamiltonian at second
order, so off-diagonal reads agree only through that order.

The engine checks conservation from the static operators in the isolated
energy bases. A filter that keeps only zero-total bands proves conservation
for ordinary interactions. Retained terms bypass that filter and need their
own proof, which declared excitation changes, factored algebra or exact local
matrix support can supply, including under `Exact()`. Scalar coefficients do
not change structural support, even when a coefficient is zero, and no
numerical small-entry threshold proves conservation. Unknown support uses the
full route. Each port pair that generates a cascade Hamiltonian must lower the
index by one at both ports. Vacuum VNA separately checks all channels and input
occupations.

The exact reduction then diagonalizes each total-excitation sector separately.
Its retained Hamiltonian, channels and coordinate map have exact zeros between
sectors, without mixing near-degenerate states across sectors. This
eligibility check does not remove the full-space construction that
whole-model elimination uses. Tracing can make an undeclared operator or its
local energy basis unknown, and the engine then requires a structural proof
to keep the sector route. `fit_a_dress()` checks `max_hilbert_dim` at each
target evaluation, including tracing and parameter rebinding, and raises if
the required sector or full-space matrix exceeds that limit. The fit does not
select a different approximation.

`method="exact"` validates the ground state, the single excitations of
touching survivors, and their pair excitations. Each diagnostic label needs a
distinct majority dressed eigenstate, with a largest weight `|<k|label>|^2` of
at least `0.5 + 1e-6`. The full retained overlap Gram matrix `S` must permit
stable orthonormalization: the residual `||S^(−1/2) S S^(−1/2) − I||` must be
finite and at most `1e-9`. These checks apply in eager, compiled and
gradient-only execution. When they fail, near-degenerate dressed states
straddle the bare labels, and quantities assigned to a single label are not
well defined. Shift the operating point or reconsider the retained subspace.
SW also requires small mixing and cannot resolve a coupled degeneracy by
itself.

The result reports the margins of these checks under
`validity["<mode>_labeling"]`, so a sweep can follow them toward a failure.
They are the largest weight, runner-up and margin of each diagnostic label, the
smallest margin, the smallest eigenvalue of `S`, and the residual with its
margin. A failed check, like the SW degeneracy check, raises
`EliminationError`, a `ValueError` that names the route, the check and the
labels or levels involved. Inside `jax.jit`, the checks run in a host callback,
and the error reaches the caller as a JAX runtime error with the same text.

### 10.5 Collapse transforms and validity metrics

#### Jump operators

The Hamiltonian's rotation carries the eliminated mode's jump operator into the
reduced frame:

```text
c_eff = B† c B
B_sw = exp(-S) P
c_eff,sw = P exp(S) c exp(-S) P = P (c + [S, c] + (1/2)[S, [S, c]] + ...) P
```

Here `S` is the anti-Hermitian SW generator of §10.3, so `B_sw† = P exp(S)`.
quchip applies the full exponential, not its first-order expansion. For the
exact route, `B = V_selected (S^(-1/2) W)†`, with the overlap matrix `S = W W†`
of §10.4. It uses the same selected eigenvectors and overlap matrix as the
retained Hamiltonian, and it is isometric and independent of arbitrary
eigenvector phases.

For intrinsic mode loss, the inherited (Purcell) diagnostic sums
`rate * |<0|L_eff|1_survivor>|^2` over the eliminated device's declared
channels. `kappa` similarly sums its isolated 1-to-0 rates, which for a unit
lowering channel gives `kappa * |amplitude|^2`. Both routes keep the complete
transformed jump matrix of the mode with its own rate, including separate
thermal emission and absorption channels. They never replace a collective jump
by independent T1 channels. `EffectiveTerms` are captured in authored
coordinates and use the ordinary basis and frame compiler, without a second
band-removal approximation. A static collective jump requires one removable
global phase in the selected frame, so unequal band phases require a compatible
common frame or the lab frame. A solve with `dissipation=False` leaves the jump
out, so it skips this check.

Reducing an excitation-conserving model records each retained channel's total
level change in `EffectiveTerms.excitation_changes`. A projected surviving
operator keeps its authored operator's change. A port on a surviving mode
keeps the change of its target's operator, built at compile time, and a
transformed port operator declares the change of the port it replaces. Each
retained `Bath` operator keeps the change of its authored operator. Band
decomposition treats every other total change as a structural zero, so this
check gives the same answer for concrete values and under `jax.grad` or
`jax.jit`. An operator that crosses an earlier map without declared level
changes uses the entries of its retained matrix.

A surviving operator outside the SW generator's proven support keeps its
authored coordinates. The projection records those unchanged tensor factors,
and serialization keeps them. Operators within the generator's support keep
their complete induced components and require compatible frame phases. The
second-order term can matter even when the first-order term is zero. Take two
transmon qubits at 5.00 and 5.40 GHz, each with a finite T1, on a 6.40 GHz bus
with `g = 70 MHz`, and eliminate the bus. The T1 jump of the first qubit is
proportional to its lowering operator `c = a_1`, and `P [S, c] P` is zero. The
second-order term adds a part that lowers the second qubit. On the vacuum and
one-excitation states, the carried jump is about `0.99875 a_1 - 0.00175 a_2`.
Here `-0.00175 = -(1/2)(g/Δ_1)(g/Δ_2)`, with `Δ_i` the bare detuning of qubit
`i` from the bus. The transmon anharmonicity changes the higher elements: from
`|11>` to `|10>`, the part is `-0.00122`. The part is small, but the cross
terms of the dissipator, such as `a_1 rho a_2†`, are linear in it. It also
makes the jump collective. Its two parts turn at different frequencies under
`"rotating"`, so the reduced chip refuses that frame. It runs under `"auto"`,
`"lab"` or a common frame.

#### Reflection sections

Consider an external default port on a declared harmonic Fock mode. The
complete `c_eff` matrix becomes that port's joint operator on every survivor,
in their authored coordinates. The reduction keeps the port's rate, phase,
scalar scattering and existing reference sections. For a port alone on its
plane, it places a `network.mode_reflection(...)` section (§3.3.2) at the core
end, which keeps the eliminated mode's own reflection:

```text
S_r(f) = ((w kappa_i - w kappa_e)/2 - i Omega) / ((w kappa_e + w kappa_i)/2 - i Omega)
Omega = 2π (f - f_tilde)
f_tilde = E(1_tilde_m) - E(0_tilde)
w = |<0_tilde|a|1_tilde_m>|^2
```

Here, `a` is the mode's lowering operator in the full product energy basis,
and the section uses the dressed transition and the rates `w kappa_e` and
`w kappa_i`. Both routes use the static Hamiltonian that they reduce, under the
source chip's approximation. The exact route reuses its labeled eigenvalues and
eigenvectors for the transition and the weight, without another
diagonalization. The SW route computes complete second-order energies:

```text
eps(n) = h_nn + sum_{k != n} |h_kn|^2 / (h_nn - h_kk)
f_tilde = eps(1_m) - eps(0)
```

The sum includes every intermediate state, including couplings within `Q`, and
uses the guarded division of the SW generator, so a coupled degeneracy raises.
For the weight, SW uses the ground and mode-excitation columns of `exp(-S2)`.
The first-order generator `S2` includes only couplings that touch either
state, and the exponential makes the first-order states orthonormal without a
diagonalization.

`kappa_e` is the bare port rate. The mode's own channels supply `kappa_i`, its
internal damping of `<a>`, including mode loss that an earlier elimination
carried to several devices. The sum reads every channel whose support contains
the mode, in energy coordinates:

```text
kappa_i = sum_L -2 rate_L Re <0|D†[L](a)|1_mode>
```

Here, `a` acts as the identity on the other support devices. A channel that
commutes with `a` adds nothing. Lowering channels add their rate, raising
channels subtract it, and pure dephasing adds it. The section drops damping
from survivor and other channels, of order `(g/Delta)^2 gamma_s`, and its
internal-loss bath is vacuum. It keeps the bare frequency of the first
touching survivor as `reference_freq` to select the square-root branch.

A reflection sweep obeys `S_full(f) ≈ S_r(f) S_reduced(f)`. The exact route
leaves out the frequency dependence of the Purcell coupling, of order
`(g/Delta)^2 kappa_e/Delta`, including across the eliminated resonance. With
frequencies in GHz and rates in 1/ns, the scale is
`(g/Delta)^2 kappa_e/(2π |Delta|)`. SW also misses the energy shifts beyond
second order. The following estimates assume RWA couplings `g_i` to survivors
at detunings `Delta_i = f_mode - f_i`, with `|g_i|` much smaller than
`|Delta_i|`. With one survivor, the leading frequency error is the fourth-order
term `g^4/Delta^3`. Several survivors without a coupling loop also leave a
fourth-order error. A coupling `J` between two survivors that both couple to
the mode closes a loop and adds the third-order term
`2 g_1 g_2 J/(Delta_1 Delta_2)`. A frequency error `delta_f` changes the
complex scattering amplitude by up to `8π delta_f/(w kappa_e)` near the dip.

#### Shared feedlines

On a shared feedline, the reduction excludes network-generated Hamiltonian
terms from both its input and its retained correction, and the network
regenerates these terms from the transformed ports. Reducing them with the
chip Hamiltonian would shift kept resonances by treating only part of the
directional coupling. A `network.mode_transmission(...)` section keeps the
removed mode's full `S_r(f)` on one leg, with the dressed frequency and rates
above. It uses the outbound leg unless kept readout ports lie only downstream,
and then the inbound leg. Ports that an earlier reduction transformed do not
count as kept readout ports.

Either leg preserves the weak-probe response from the line input. Near the
frequencies of kept ports beyond the section, the emitted or received field has
a dispersive error of scale `kappa_e / (2π |f_k - f_R|)`. Here `kappa_e` is the
section's dressed external rate, and `f_k` and `f_R` are the target and
removed-mode frequencies before this reduction. Beyond means downstream for the
outbound leg and upstream for the inbound leg. Dressed shifts can raise the
correction above this scale, so it is not a certified error bound. The
section's `validity` entry reports `kappa_over_delta` with the smallest
absolute detuning among those kept ports, or zero when none lie beyond, and
sets `is_valid` when `kappa_over_delta < 0.1`. This is a dispersive diagnostic,
not an accuracy guarantee. Reflection sections add no such entry.

#### Monitoring through a removed mode

Native monitoring records the Markov coupling `L` of each selected channel and
applies no reference section, also at the carrier. A record of the line output
through the section, or of a hidden channel that acted on the removed mode,
therefore lacks the mode's response to the line field. For a weak input `beta`
at `f`, the single-pole response of the mode gives
`<L_e> = -kappa_e beta / D_r(f)` and
`<sqrt(kappa_i) a> = -sqrt(kappa_e kappa_i) beta / D_r(f)`, with
`D_r(f) = (kappa_e + kappa_i)/2 + i 2π(f_r - f)`. This response is exact when
the mode decouples from the survivors. Both means stay finite as the dressing
vanishes, while the reduced means vanish with it. A solve therefore rejects
diffusive monitoring of these channels with a nonzero efficiency, as for a
multiport response below. On a shared line, the rule also rejects the line
record of kept emitters on the same line. The isolated pole changes the mean
field that reaches its port by `kappa_e / |D_r(f)|` of that field. This is
`2 kappa_e / (kappa_e + kappa_i)` at `f_r`, and `kappa_over_delta` estimates it
away from `f_r`. It is a mean-field correction, not a bound on the error of a
record. `OutputField` traces through the section keep the narrowband carrier
approximation of §3.3.2.

#### Multiport modes

A harmonic mode with several default lowering ports can instead keep one
coupled response at the original port locations. Each port must lie on a
distinct directional line that the series check above accepts. For the pole's
local incident and outgoing fields, the internal convention gives

```text
v_j = sqrt(w kappa_j) exp(i phi_j)
d(f) = (sum_j |v_j|² + w kappa_i)/2 - i 2π(f - f_tilde)
F(f) = I - v v† / d(f)
```

This is the passive linear transfer formula of
[Gough, Gohm and Yanagisawa][gough-gohm]. The off-diagonal entries scatter
between directions. For equal rates and phases on a two-way hanger, the
transmission and reflection obey `t = 1 + rho`. The reported VNA convention
conjugates this internal transfer.

The vacuum weak-probe solve keeps one scalar amplitude per removed mode. It
composes that amplitude with the transformed survivor ports through the
network's channel maps and series-product Hamiltonian terms. It therefore keeps
the pole's position, subsequent scattering, port phases and unequal rates. This
representation adds no device or Hilbert-space state, and independent scalar
sections on the two lines cannot replace it. The pole data survive copying,
parameter rebinding and serialization.

In a conserving model, a one-port mode on a line shared with a multiport mode
also keeps its pole at the original port. This holds before and after the
multiport mode is eliminated. Moving a scalar section across that mode would
change which incident or outgoing direction receives its phase and attenuation,
so the original graph determines the response for either elimination order.
Other one-port reductions keep their reference sections, including
nonconserving models.

The reduced quantum model keeps both transformed emission channels, and its
Markov dynamics do not keep the pole amplitude. Dispersive error remains in
these channels and in the retained Hamiltonian: omitted directional couplings
to the removed mode leave shifts of order `|J_network|²/|Delta|`. The
weak-probe composition restores those network paths in the frequency solve.
It omits survivor damping of the pole and correlations with the discarded
internal-loss bath, and the second-order SW energy error above remains, so the
dissipative reduction is not exact.

A kept mode on the same line also loses its return path through the pole.
Write the vacuum one-excitation drift of the reduced chip as `A = -iH - C†C/2`,
with `D_j(f) = h_j + i 2π(f_j - f)`. Here `h_j = -Re A_jj` is the amplitude
damping and `2π f_j = -Im A_jj` the diagonal frequency, not in general a
normal-mode frequency. The omitted pole `p` has `D_p = d` from above and
changes `D_k` of a kept mode `k` by the self-energy `-A_kp A_pk / D_p(f)`. The
result's `validity` entry `"<mode>_response"` reports

```text
self_energy_over_width = max_k |A_kp A_pk| / (|D_p(f_k)| h_k)
```

over the devices that the other ports target, and sets `is_valid` when the
value is below 0.1. The entry excludes a device that reaches the line only
through the transformed ports, whose error is the dispersive error that
`g_over_delta` reports. A one-way path has `A_kp = 0` or `A_pk = 0` and gives
zero, but zero does not exclude indirect feedback through other kept modes or
earlier poles. For a two-way hanger with rate `kappa` at each of the four ports
and no internal loss, the value is
`w kappa / |w kappa + i 2π(f_tilde - f_k)|`, which approaches one for
degenerate modes. The entry evaluates the ratio at `f_k`. Between `f_k` and
`f_tilde`, `|D_p D_k|` has a smaller minimum, so the ratio is a resonance
diagnostic, not a bound over frequency. The entry is `None` when the reduced
chip has no vacuum one-excitation model, for example with a thermal bath.

The coupled pole supports a conserving model's vacuum `VNA.sweep()` only.
Pumped, finite-power, noisy, excited-reference and general stationary field
routes reject it. A time-domain solve also rejects a coherent input that
reaches the pole's ports and an `OutputField` whose channel carries emission
from those ports. It rejects diffusive monitoring with a nonzero efficiency of
such a channel. The same rule covers a hidden channel that acted on the removed
mode, such as its internal loss or a bath on it. The field and the record would
contain only the adiabatic part of the removed mode, so they would miss the
pole near its frequency. For those observables, keep the boundary mode.

`EffectiveTerms.omitted_responses` records each carried channel that lacks a
removed mode's response, including a bath with removed targets and a
transformed port. The record stays with the channel through later reductions,
also when the port network no longer holds the pole or the section. Pure-state
SSE monitors every channel, so it rejects every chip with such a channel. A
traced efficiency counts as nonzero, also when its value is zero.

Other outputs, SME records of other channels, and unconditional solves keep
their formula. Their accuracy depends on the reduced dynamics, because the pole
can act back on kept emitters through a two-way line. A channel that reaches
the removed mode only through dressing, such as the relaxation of a coupled
survivor, also misses part of the pole, in proportion to that dressing. Small
population errors can hide larger phase errors.

#### Chained port reductions

A port that an earlier elimination transformed already acts on every survivor
and reaches a later eliminated mode only through that dressing. The later
reduction transforms it like an inherited channel and gives it no section, so
you can eliminate both readout modes of a chip in either order. Like an
inherited channel, it drops its direct scattering through the later mode, of
order `rate |<0|L|1>|^2 / Delta`. Rather than approximate or double-count
them, quchip rejects custom or collective boundary operators and projected
survivors. It also rejects a nonlinear eliminated boundary target and a plane
carrying fields that do not pass the port in series.

#### Validity metrics

The result's `notes` record that the projection is exact for the spectrum but
approximate for dissipation, because the discarded `Q`-block dynamics also
dephase and decay. For each eliminated coupling, `validity` reports
`g_over_delta` and `min_block_gap`. `g_over_delta` is the second-order
smallness `|<1_s|H|1_c>| / |Delta|`, the resolved exchange element over the
bare detuning of the incoming chip, and `is_valid` requires it below 0.1. Parallel couplings
between the same two devices each report their combined element.
`min_block_gap` is the smallest bare-energy gap that the Sylvester generator
crossed. A small gap with a nonzero matrix element is the failure mode of the
perturbative expansion, even when every `g/Delta` is small.

### 10.6 State and observable maps

Device elimination leaves the authored frequencies and local bases of
surviving devices unchanged, and the retained Hamiltonian correction owns their
shifts. `effective_params` reports the reduction's transition diagnostics. Use
`chip.freq(...)` for the coupled transitions of a reduced chip. To reduce
another operating point, rebind the source model and eliminate again.

The retained correction is the route's retained Hamiltonian minus the reduced
chip's assembled matrix, both under the chip's approximation. Dressed queries
use that same approximation. Under `method="exact"`, a reduced chip therefore
reproduces the source's labeled energies, and `reduced.static_zz(a, b)` equals
`effective_params["exchange"]["zz"]`. Under `method="sw"`, the reduced dressed
spectrum is the second-order one, so compare its diagnostics through
`effective_params`. Retained effective terms are part of the cache key for
dressed analysis, so attaching them to an already resolved chip invalidates
its cached dressing.

`result.mapping` captures source and target labels, dimensions, backend and
lab-frame solver coordinates, and its `embedding` maps retained coordinates
into the source space. `project_operator(operator)` returns `B† O B`,
`project_state(state)` returns `B† psi` or `B† rho B`, and
`lift_state(state)` performs the reverse embedding. These methods accept full
numerical matrices and backend-native objects and return native objects on the
captured backend. Projection keeps the lost norm or trace, so discarded
population stays visible and no state is silently renormalized. Express
rotating-frame trajectory states in lab coordinates before you use this map.

Exact reduction uses the same Löwdin embedding as its Hamiltonian and
removed-component channels. SW uses `B = exp(-S) P`, with the first-order
generator that the reduction already uses. This map is isometric, but the
dynamics stay perturbative and the SW Hamiltonian stays truncated at second
order. Jump operators use the same exponentiated first-order coordinate map,
which keeps their interpretation common with states and observables but does
not make the reduced dynamics or dissipation exact. Maps capture numerical
coordinates independently of later source edits.

### 10.7 Local reduction (`local=True`)

`eliminate(chip, device, local=True)` reduces a device from a patch without
assembling the full product space. The core is the device and every device
that shares a coupling, an effective Hamiltonian or a port with it. The SW
generator changes levels only on the core, but its coefficients can depend on
other patch devices: its denominators use `E = diag H`, so they read each term
whose diagonal part depends on a core level. The patch therefore holds:

- the core,
- the far device of each coupling whose diagonal part depends on a core level, such as a `CrossKerr` edge,
- the full support of each effective Hamiltonian that touches the core and whose diagonal part depends on a core level,
- every device that the operators of the device's ports act on in retained coordinates.

The reduction excludes network-generated Hamiltonians (§10.5), so devices
linked to the core only through a feedline do not enter the patch. The
numerical sub-chip holds independent copies of its local ports, with their
operators and rates. The full chip keeps the field wiring and reference
sections and regenerates its network terms from the transformed ports. A
collective port keeps its full operator support during transport.

The far-device rule reads the structure of a coupling, not its values. It
splits the coupling into operator products and drops their scalar coefficients.
No parameter values can then cancel a product, as in `(x - y) n_a n_b`, and a
traced reduction reads the same patch as a concrete one. A product leaves its
far device out only when its diagonal is exactly equal at each level of the
core device. The test has no tolerance. A factor that moves between the
coefficient and the operator therefore does not change the patch, and round-off
from a device basis change can only add a device to it. A product whose
operator is a function of a parameter, and a coupling on a device with a traced
energy basis, count as level-dependent. A device without a neighbour has no
generator, so its patch holds the device alone. Its reduction leaves the
device's ground-state energy and projected channels, which are multiples of the
identity, and the new `EffectiveTerms` hold them on the first other device
without a map.

The patch Hamiltonian includes each effective Hamiltonian on one patch device.
It also includes each effective Hamiltonian that touches the core and acts only
on the core, the far devices and the level-dependent supports. The reduction
replaces these effective Hamiltonians. Every other effective Hamiltonian keeps
its matrix: it keeps the mode's level, and its diagonal part does not change
the denominators, so the second-order reduction leaves it unchanged.

With this patch, the full-chip generator is the patch generator times the
identity on the other devices. A coupling from the patch to other devices
keeps the mode's level, so it adds nothing to `P [S, V] P`. In exact
arithmetic, the retained Hamiltonian, transformed channels and captured map
therefore equal those of the full-chip reduction, up to the numerical omission
described below. These steps read the patch as a sub-chip, so their cost
follows the patch's product space and the supports of the channels that it
carries. Devices and couplings outside the patch stay unchanged. `chi` comes
from the patch spectrum, so it omits dressing by devices outside the patch.

The removed mode's scattering section needs its own set of intermediate states
(§10.5). Its pole and weight include interactions that preserve the mode's
occupation, even outside the generator patch. The local route assembles the
states that one kept interaction can connect to the vacuum or to the mode's
first excitation. All local Hamiltonian terms contribute their diagonal
energies on those states, including spectator shifts. State selection uses
local dimensions and proved excitation conservation, so it also keeps
zero-valued couplings under tracing. The calculation uses the full route's SW
energy and weight formulas without a full product-space matrix. For a
conserving model, remote interactions cannot excite the vacuum, so this set
stays local to the removed mode. Nonconserving interactions can add spectator
excitations, and the cost then follows their combined state count and each
term's local support.

The correction is split in the energy coordinates of the patch survivors. The
part on one survivor is the normalized partial trace onto it, minus the mean,
and the first survivor's part also holds the mean. Each part is an
`EffectiveTerms` on its own device, and the rest acts on all patch survivors. A
later patch reads a single-device part only when it holds that device. In a
concrete remainder, quchip omits an entry and its transpose partner when both
are at or below one floating-point spacing of the patch Hamiltonian's largest
entry. Round-off residues then do not expand later patches. The retained terms
record the Frobenius norm of the omitted matrix in GHz, which bounds the change
in each eigenvalue from this omission. Traced remainders are kept.

The reduction transforms the channels of the removed device and couplings. It
also carries every retained channel that acts on the patch through the new
map. Devices outside the patch keep their coordinates, so each channel keeps
its own support. Channels with the same support share one `EffectiveTerms`,
and the rates stay separate from the operators.

Each local reduction captures a map from its patch survivors into its patch
and stores it on the terms of those survivors. Each earlier map whose source
devices meet the patch becomes a parent of the new map, in the order they
apply. A map can mix the devices it acts on, so `Chip.partition()` keeps each
map chain and its devices in one part. An operator passes through each map
whose source devices it acts on, and its other devices keep their coordinates.
An operator that a map already holds in retained coordinates, such as a
transformed port, starts after that map. A chain of local reductions therefore
never forms a full-space matrix: transported operators grow along the maps,
and the patches do not. In a ring, the loss of an early step can spread over
every device that later patches link, so the order of the steps sets the cost
(§10.8). A full-chip reduction after local ones keeps their maps as parents in
the same way. `result.mapping` extends the patch map by the identity on the
other devices and forms that matrix only when you read it.

`local=True` implements `method="sw"` only. The exact route uses the
whole-chip eigensystem, resolved into excitation sectors when possible, and
its dressed states are not local. The local route does not transform baths, so
a chip with baths raises `NotImplementedError`. Coupling and effective-terms
targets do not accept `local=True`. `active_patch()` and
`QuantumSequence.active_patch()` forward `local`.

### 10.8 Order of a chain of eliminations

`eliminate()` removes one device per call. Each step keeps second order in its
own couplings, so the order of a chain sets two results: how far the operators
of kept devices spread along the captured maps (§10.7), and which fourth-order
terms the chain keeps.

Operator size. A local step moves each operator support that meets its patch
onto the patch survivors. Take a ring of six transmons with four levels and a
coupling resonator between neighbours. If every coupler goes first, an
operator on one transmon spreads over all six, which is 4,096 states. If each
arc goes from one end, the operator stays on the kept transmons and at most one
more device.

Dropped exchange shift. Take a spectator `v`, a kept device `k` and a spectator
`u` that couples to both. Through `u`, the two exchange with
`J ≈ (g_ku g_uv / 2)(1/Δ_ku + 1/Δ_vu)`. If `u` goes first, its step makes `J`
an explicit `k`–`v` coupling, the step of `v` folds it, and the chain keeps the
shift `J²/Δ_kv` of `k`. If `v` goes first, its step folds `v` against `u`
alone, and the chain drops this shift. The shift is fourth order in the
couplings. For legs of similar size, dispersive couplings to `u` and
`|J/Δ_kv| ≪ 1`, it can be much larger than the fourth-order terms that every
order drops. Unequal legs, cancellations and other resonances can remove this
difference. The `g_over_delta` check of each fold does not show this omission.

Near-degenerate spectators. Each step also adds exchange between its patch
survivors. If two spectators `a` and `b` are close in frequency, an explicit
`a`–`b` exchange can make the later fold of the pair poor. While `a` and `b`
still couple to the device between them, no step folds them against each
other.

`eliminate_spectators(chip, keep)` and `active_patch()` choose each step by
these rules:

1. A spectator `v` waits while a spectator neighbour `u` has a kept neighbour `k` with
   `|E_v - E_k| < |E_v - E_u| / 2`.
2. A spectator `v` waits while its step would give two spectator neighbours `a` and `b` a nonzero
   exchange `x_ab + x_av x_vb (1/(E_a - E_v) + 1/(E_b - E_v)) / 2` of magnitude `0.1 |E'_a - E'_b|` or
   more. `E'_a = E_a + |x_av|² / (E_a - E_v)` is the estimated transition of `a` after the step, which
   can bring the pair closer. `v` also waits when `E_a` or `E_b` equals `E_v`. The pair does not stop
   the step when `x_ab` is nonzero and `|x_ab|` alone already reaches `0.1 |E_a - E_b|`. This rule
   screens the predicted fold of the pair. It does not make every later fold valid.
3. Among the spectators that do not wait, a device whose neighbours are all kept goes last. Then the
   step with the smallest widest support of a kept device's operator goes first, then the smallest
   sum of these supports, the smallest patch and chip order.

`E` is the diagonal transition energy `E(1) - E(0)` of each device on the
current chip, including the shifts of earlier steps, as the SW denominators do.
`x` is the current single-excitation exchange element. Both are read at the
chip's approximation, as the fold reads them. They hold only the
number-conserving terms, so without the RWA the rules omit the
counter-rotating virtual paths. Neighbours share a coupling, an effective
Hamiltonian or a port.

A model of the patch rules of §10.7 gives the neighbours, patches and supports.
It starts on the source chip, and each step links its patch survivors. When a
patch holds an interaction whose diagonal depends on the level of a device, the
SW denominators of the step read that level. The correction can then couple it
on the diagonal with the levels of the devices next to the removed one. The
model therefore lets later patches that hold these devices hold all the
survivors. This conservative rule can make a patch larger than the local
route's, and a larger patch can change later choices. The model reads no
reduced coordinates, so `local=True` and `local=False` give the same order. If
every spectator waits, rule 3 chooses among all of them, and the result's notes
name the step and what the chain omits. The order is a cost-and-accuracy
heuristic for dispersive chips, and it does not bound the error. The divisor 2
and the limit 0.1 were set from tests on rings, and the limit equals the
`g_over_delta` limit of a fold.

For example, take a ring of eight transmons at `5.32 - 0.07 i` GHz with
coupling resonators at `f_c + 0.02 i` GHz, and keep `q2 q3 q5 q6`. The default
order keeps each operator of a kept transmon within 256 states. For `(g, f_c)`
= (0.03, 6.30), (0.075, 7.20) and (0.05, 6.00) GHz, its largest f01 error is
2.8, 22.8 and 56.9 kHz. Each arc walked from one end gives 7.4, 98.1 and 98.9
kHz. Both orders give the same ZZ error within 2%.

Omitted shifts. A device that stays on the chip can still link a removed
spectator `v` to a kept device `k`. This happens when every spectator waits,
when `order=` removes `v` first, and when the mediator is kept, as with the
coupler next to a driven transmon in `active_patch(hops=1)`. No order recovers
the shift then. Each SW step therefore estimates the shift that it omits from
each kept device,

```text
δf_k = |Σ_u J_kv^(u)|² / (E_k - E_v)
```

from the current elements, where the paths of all mediators `u` add before
squaring. `omitted_shifts` holds the sum over steps, and the notes give each
term. This term makes the reduced transition of `k` differ by about `-δf_k`.
Take a transmon at 5.00 GHz with a kept coupler at 7.00 GHz, a spectator at
5.04 GHz and `g = 0.05` GHz on both legs. The estimate is -39.9 kHz, and the
f01 error is +39.7 kHz. On the ring above, wherever the f01 of the default
order and another order differ by more than 5 kHz, the difference of their
estimates matches that difference within 10%.

The estimate needs dispersive legs and `|J/Δ_kv|` below the fold limit 0.1.
With `v` 5 MHz from `k` in the example (`|J/Δ_kv|` = 0.25), it is already 6%
too large. Near resonance, `k` and `v` hybridize, so the omitted effect is not
a dispersive shift. Outside this regime, and when two of the transitions are
degenerate, the estimate is `None`. It is also `None` when `k` and `v` exchange
directly, because it omits the interference of the direct and mediated paths.
The estimate gets no validity flag beyond this regime check, because `|J/Δ_kv|`
measures mixing, not an acceptable frequency error. It reads the
number-conserving elements, as the rules do, so on a chip without the RWA it
omits the counter-rotating virtual paths. An empty `omitted_shifts` means that
no step found such a path. `method="exact"` with `local=False` keeps these
shifts when it reduces the original chip, because it diagonalizes the whole
chip, so it reports no estimate. It cannot restore a shift that an earlier SW
step dropped.

Each local step adds a note with the widest operator support of any device and
warns when one dense copy of that operator needs 256 MiB or more. Pass `order=`
to set the steps yourself. The default order compares values, so it raises
when `jax.jit` or `jax.grad` traces device parameters. Pass `order=` with the
`eliminated_labels` of a call without tracing.

## 11. Parametric edge control

Sources: [`quchip/control/drive.py`](https://github.com/quchip/quchip/blob/main/quchip/control/drive.py) (`ParametricDrive`), [`quchip/declarative/models.py`](https://github.com/quchip/quchip/blob/main/quchip/declarative/models.py) (`parametric_interaction`), [`quchip/chip/retarget.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/retarget.py)

### 11.1 The pump contract

A `ParametricDrive` pumps a modulable coupling, including `TunableCapacitive`
and `CrossKerr`. The envelope supplies a real modulation `A(t)` of its
strength in GHz, in two forms:

```text
freq omitted (baseband):  delta_g(t) = Re s(t)
freq = nu_d (tone):       delta_g(t) = Re[s(t) · e^(−i·2π·nu_d·t)]
```

The coupling's `parametric_interaction` hook authors the operator. The engine
selects its bands with the chip's approximation and preserves the real pump
waveform, and each retained weight vector carries its operator-frame phase, as
in §5.2.

For weak resonant difference-frequency modulation, write
`A cos(2pi nu_d t)=A(exp(+i2pi nu_d t)+exp(-i2pi nu_d t))/2`. One partner
becomes resonant with exchange, with coefficient `A/2` times the endpoint
matrix elements, so unit Fock charge elements give `A/2` directly. The default
`RWA()` removes pair-creation bands even with a sum-frequency pump. Use
`Exact()` or an adjoint-closed explicit selection to keep those bands.
[Didier et al.][didier] treat frequency-modulated transmons. Their model gives
parametric-gate context, not the contract of quchip's coupling hooks.

### 11.2 Retargeting stranded control (`chip/retarget.py`)

`eliminate()` converts control lines whose target it removes through a registry
keyed by `(drive type, target type, result kind)`. The registry follows each
type's MRO, so a rule registered for a base type also covers its subclasses.
The built-in rule converts a `FluxDrive` on an eliminated exchange-mediating
mode into one baseband `ParametricDrive` per emitted edge. The first pair's
pump keeps the flux line's label, and further edges receive unit-amplitude
`Crosstalk` copies of the scheduled signal. Every pump carries its own
`Gain(dJ_ab/domega_c)` (§10.3).

The conversion is linear in coupler-frequency modulation and second order in
the dispersive coupling, also when elimination uses `method="exact"`. It
assumes modulation small compared with the detunings and holds the charge
matrix elements fixed, so it is not the exact derivative of an exactly reduced
edge. The survivors' second-order Lamb-shift modulation is omitted and
recorded. [Yan et al.][yan] give tunable-coupler context.

### 11.3 Schedule portability

The retargeted line keeps its label, and `schedule()` resolves drive-line
labels in device → coupling → line order, so the same schedule call runs on
the full and reduced chips.
[`tests/physics_sentinel/test_eliminate_portability.py`](https://github.com/quchip/quchip/blob/main/tests/physics_sentinel/test_eliminate_portability.py)
applies identical schedules to both models and compares them with tolerances
derived from the validity metrics (`g/Delta`, `delta_omega/Delta`).

## 12. Engine assumptions

The engine assumes four physics properties:

1. Each frame generator is the isolated energy-level index operator, expressed in the selected solver basis. It need not equal a device's physical number operator.
2. Single-device and two-device operators can be decomposed by excitation-change bands.
3. The `"rotating"` frame uses each device's explicit reference or the chip-resolved dressed transition.
4. Each drive builds a complete analytic signal with an optional carrier, then
   implements `hamiltonian(target, signal)`. Control equipment transforms that
   signal before the destination drive maps its physical I/Q quadratures to
   target-local operators. The engine owns frame and band selection.

The engine does not hardcode transmon-specific or resonator-specific formulas,
backend-native operator types, or device-specific noise models beyond asking
each device for its collapse operators. Devices, couplings and drives supply
the domain-specific physics, and the engine handles their shared frame and RWA
bookkeeping.

## 13. JAX traceability boundaries

Band decomposition, coefficient construction, observable recombination and
the backend-free Hamiltonian IR preserve JAX arrays. A traced dense payload
keeps every candidate band unless its structure is declared. Sparse layouts
declare their entries. A matrix contribution can declare its total excitation
changes (`PhysicsExpr.from_matrix(..., excitation_changes=...)`), as reductions
of excitation-conserving models do.

These operations require concrete Python values:

- `Chip.dress()` returns a concrete dict-based view and is not traceable. The bare→dressed assignment itself is discrete and piecewise. Traced callers use `Chip.energy()`, `Chip.freq(target, when=...)`, `Chip.dispersive_shift()` or `Chip.kerr_matrix()`, which route through `label_eigensystem` in the pure-JAX kernel of `quchip/engine/dressing.py`. The labeled energy lookup stays differentiable away from label discontinuities. `track_path` is a separate continuation utility that follows labels through a stacked eigensystem along a parameter sweep.
- Human-facing serialization and diagnostics coerce to Python scalars.

Local eigenvector derivatives use first-order Hermitian perturbation theory.
The numerical connection suppresses gaps at or below `N eps (E_max-E_min)`,
where `N` is the authored dimension and `eps` the eigenvalue dtype precision.
This energy-origin-independent threshold treats unresolved gaps as tied. It
cannot define a unique eigenvector derivative at a degeneracy, and resolved
small gaps keep their inverse-gap response, which can be large. Higher
derivatives require nondegenerate eigensystems and stable phase pivots.

When authored algebra proves that a local Hamiltonian is diagonal, the
coordinate axes are its energy vectors. Plain evaluation sorts the diagonal
energies, and equal energies keep their authored order. Concrete device values
keep this sort, also inside a trace. Under JAX tracing, quchip keeps the
authored order, so the energy vectors stay concrete. Proofs that read them,
such as excitation sectors and frame checks, then work as in plain evaluation.
Execution raises if traced energies leave the authored order, or if a dropped
level falls below a kept level, because plain evaluation would then select or
label the levels differently. For example, a `DuffingTransmon` with enough
levels that its spectrum turns over raises this error. Use fewer levels, or
truncate the eigenbasis below the turnover. Other local Hamiltonians use the
differentiable eigensolve above.

Numerical outputs can remain differentiable while concrete input validation is
skipped under tracing, and a finite derivative alone does not establish
accuracy. Other engine paths avoid implicit conversion to host arrays.

## 14. Audit pointers

To audit a physics path, start here:

- units and assembly boundary: [`quchip/engine/assembly.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/assembly.py)
- frame resolution: [`quchip/engine/frames.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/frames.py)
- observable preparation and demodulation: [`quchip/engine/observables.py`](https://github.com/quchip/quchip/blob/main/quchip/engine/observables.py)
- dressing and public Hamiltonian APIs: [`quchip/chip/chip.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/chip.py)
- adiabatic elimination and χ/κ reporting: [`quchip/chip/transformations/`](https://github.com/quchip/quchip/tree/main/quchip/chip/transformations)
- Schrieffer-Wolff kernels and the exact reduction route: [`quchip/chip/sw.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/sw.py)
- control-line retargeting across reductions: [`quchip/chip/retarget.py`](https://github.com/quchip/quchip/blob/main/quchip/chip/retarget.py)
- readout pointer states and figures of merit: [`quchip/analysis/dispersive_readout.py`](https://github.com/quchip/quchip/blob/main/quchip/analysis/dispersive_readout.py)

Some constructions are specific to quchip, and these tests check their stated
models and analytic comparisons, without certifying an error bound for
arbitrary devices or operating points:

- finite-temperature interference and receiver normalization (§8.2): [field-noise tests](https://github.com/quchip/quchip/blob/main/tests/test_field_noise.py) and [receiver tests](https://github.com/quchip/quchip/blob/main/tests/test_linear_measurement.py)
- passive-section expansions and retained-order estimates (§10.5): [port-elimination tests](https://github.com/quchip/quchip/blob/main/tests/test_eliminate_port_network.py)
- local-reduction equivalence (§10.7), which follows by extending the patch generator with the outside identity: [local-reduction tests](https://github.com/quchip/quchip/blob/main/tests/test_local_elimination.py)

## 15. Measurement of saved states

`SimulationResult.measure(*devices, t=None, basis="energy")` projects a kept ket
or density matrix in the captured local isolated energy bases, and `t=None`
selects the final state. A custom basis supplies orthonormal columns in those
energy coordinates in the stored integration frame, and quchip applies no
automatic phase-frame conversion to it. `basis="solver"` selects the solver's
product basis. quchip computes the joint probabilities before marginalizing
the unmeasured devices. Independent partition components can be combined at
the probability level.

Shot sampling applies the Born rule without further evolution.
`assignment[recorded, physical]` is column-stochastic. `IQReadout` defines one
conditional complex Gaussian distribution per physical outcome, and its total
mixture covariance includes both the within-outcome covariance and the
covariance of the conditional means. These readout models do not change solver
selection, return collapsed states, or describe continuous quantum
trajectories.

`IQReadout.from_wiring(chip, output, ...)` resolves a detector without quantum
evolution, and `result.iq_readout()` uses the simulation's captured wiring
instead. Both propagate supplied conditional coherent fields through the
captured output reference sections and downstream mixing, with unspecified
boundary channels in vacuum. Supplied means, returned detector IQ and receiver
transfers use the engineering convention of VNA, while simulation field traces
keep the physics convention. Conjugate stationary `output(...).raw_amplitude`
values before you use them as boundary templates. Both paths include
downstream added noise and ideal heterodyne vacuum through the propagation and
integration that VNA uses. Boundary thermal noise, device correlations and
transient field correlations are outside this coherent-field readout model,
but calibrated conditional distributions can include those effects instead. A
calibrated full covariance must not receive the same apparatus noise a second
time. Input baths and port decay remain in the declared quantum dynamics,
whatever the readout model.

## 16. Continuous trajectory monitoring

Native jump and diffusive solvers evolve the captured Hamiltonian and declared
channels. `with_monitoring` selects physical SLH couplings with their authored
phases. At efficiency eta, QuTiP receives sqrt(eta) exp(-i phase) L and
sqrt(1-eta) L, whose dissipators sum to D[L]. dynamiqs receives L and eta through
its native SME interface. Pure SSE cannot omit unobserved loss.

Native measurement records describe the selected `L` in the integration frame,
without the coherent incident `beta` or downstream receiver noise. Native
weighted jump averages include deterministic no-click paths. Truncation checks
are explicit and cover the available samples, and final-only coverage is
identified separately.

`TrajectoryResult.records()` gives the diffusive currents of both backends in
one layout, with shape `(runs, channels, intervals)`. QuTiP heterodyne adds a
quadrature axis before the intervals. A channel with collapse operator `L`,
which includes the square root of its rate, efficiency `eta` and phase `phase`
records

```text
I = 2 Re<c> + xi,    c = sqrt(eta) exp(-i phase) L
```

in 1/sqrt(ns). The homodyne noise `xi` has variance `1/Δt`. QuTiP heterodyne
records `2 Re<c> + xi_X` and `2 Im<c> + xi_Y`, and each noise has variance
`2/Δt`. QuTiP samples the signal at the start or the end of each interval, or
takes the mean of both, as `store_measurement` selects. dynamiqs divides the
increment of the integrated record by `Δt`, so its signal is the mean over the
interval. `records().sampling` names the choice: `"start"`, `"end"`, `"middle"`
for the mean of both ends, or `"mean"` for the mean over the interval. Take a
decaying coherent field with `L = a`, `eta = 1` and `<a>(0) = 1`. On the
interval `[0, 1]` ns, the signal is 2 at the start and 1.21306 at the end. The
mean of both ends (`"middle"`) is 1.60653, not the midpoint value 1.55760, and
the mean over the interval (`"mean"`) is 1.57388.

`clicks()` gives the click times and channels of each run with its sampling
weight `w_r`. An ensemble mean is `Σ_r w_r x_r`, plus the no-click run with
`no_click_weight` when QuTiP `improved_sampling` or dynamiqs smart sampling
solves that run apart. `full` marks a run that filled its dynamiqs `nmaxclick`
buffer, whose evolution can stop early. `run_expect(key)` gives the
expectation of each run without weights. See
[the solver guide](docs/guides/choosing-a-backend.md) and
[Wiseman and Milburn, chapter 4][wiseman].


[koch]: https://arxiv.org/abs/cond-mat/0703002
[fluxonium-ref]: https://arxiv.org/abs/0906.0831
[lindblad]: https://doi.org/10.1007/BF01608499
[input-output]: https://doi.org/10.1103/PhysRevA.31.3761
[series-product]: https://arxiv.org/abs/0707.0048
[feedback]: https://arxiv.org/abs/0804.3442
[kockum]: https://arxiv.org/abs/1406.0350
[caves]: https://doi.org/10.1103/PhysRevD.26.1817
[wiseman]: https://doi.org/10.1017/CBO9780511813948
[mckay]: https://doi.org/10.1103/PhysRevA.96.022330
[magesan]: https://arxiv.org/abs/1804.04073
[malekakhlagh]: https://arxiv.org/abs/2005.00133
[bravyi]: https://arxiv.org/abs/1105.0675
[lowdin]: https://doi.org/10.1063/1.1747632
[des-cloizeaux]: https://doi.org/10.1016/0029-5582(60)90177-2
[gambetta]: https://arxiv.org/abs/cond-mat/0602322
[krantz]: https://arxiv.org/abs/1904.06560
[blais]: https://arxiv.org/abs/2005.12667
[didier]: https://arxiv.org/abs/1706.06566
[yan]: https://doi.org/10.1103/PhysRevApplied.10.054062
[beaudoin]: https://arxiv.org/abs/1107.3990
[gough-gohm]: https://arxiv.org/abs/0807.2338
