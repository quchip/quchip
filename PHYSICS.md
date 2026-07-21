# quchip Physics Reference

This document states the physics contracts implemented by quchip. It answers what the public `.hamiltonian()` methods mean, where frames are applied, where RWA is applied, and which assumptions the engine makes.

## 1. Units and the 2π Convention

quchip uses `hbar = 1` with these user-facing units:

| Quantity | Unit |
| --- | --- |
| Frequency | GHz, ordinary frequency |
| Time | ns |
| Temperature | mK |
| Energy | GHz |

The domain layer stays in ordinary GHz. The only Hamiltonian-assembly `2π` conversion is in [`quchip/engine/stage2_assembly.py`](quchip/engine/stage2_assembly.py), right before the solver-facing Hamiltonian is built.

## 2. What `.hamiltonian()` Means

### 2.1 Device `.hamiltonian()`

`BaseDevice.hamiltonian()` returns that device's static local Hamiltonian in its own Hilbert space, in the lab frame, in ordinary GHz.

Examples:

- `DuffingTransmon.hamiltonian()` returns `omega * n + (alpha / 2) * n * (n - I)`.
- `Resonator.hamiltonian()` returns `omega * n`.

It does not:

- include `2π`
- include any rotating-frame subtraction
- include any drive term
- include any explicit time dependence

### 2.2 Coupling `.interaction_hamiltonian()`

`BaseCoupling.interaction_hamiltonian()` returns the coupling's full two-body operator in the pair subspace, still in the lab frame, still in ordinary GHz. It takes no RWA argument — a coupling defines exactly one interaction; RWA is resolved and applied structurally by the chip and engine, not chosen here (§6.1).

For `Capacitive`:

```text
full: g * (a + a†)(b + b†)
```

`interaction_hamiltonian()` always returns this full form. The RWA form `g * (a†b + ab†)` is never authored directly — it is what remains once the bands that change total excitation are masked out by the chip or filtered by the engine.

### 2.3 `Chip.hamiltonian()`

`Chip.hamiltonian()` returns the full static lab-frame Hamiltonian after embedding device and coupling operators into the total Hilbert space. For each coupling where `Chip.resolve_rwa(coupling)` is `True`, the full interaction is masked to the excitation-change bands its `rwa_keeps_band` predicate accepts (§6.1) before embedding. RWA is therefore already included in this Hamiltonian rather than deferred to the engine.

It is still not the solver-ready Hamiltonian. The engine later:

1. multiplies by `2π`
2. subtracts the chosen frame generator
3. decomposes non-static pieces into excitation-change bands, filtering coupling bands with the same `rwa_keeps_band` predicate so the two views agree
4. attaches explicit time-dependent phases where needed
5. adds drive and crosstalk terms

The resulting contracts are:

- device `.hamiltonian()` means local static lab-frame physics
- coupling `.interaction_hamiltonian()` means local static lab-frame interaction physics
- `Chip.hamiltonian()` means the embedded static lab-frame chip Hamiltonian
- the engine builds the actual solver Hamiltonian from those pieces

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

If `quality_factor` is set, the resonator also contributes photon loss with collapse operator `sqrt(2π * omega / Q) * a`.

### 3.3 Collapse operators

Source: [`quchip/devices/base.py`](quchip/devices/base.py)

The standard dissipators are:

- `T1`: relaxation through `a`
- `T2`: pure dephasing through `sqrt(2*gamma_phi) * n` with `gamma_phi = 1/T2 - 1/(2*T1)`. The factor `2` makes the 0–1 coherence decay at `1/(2*T1) + gamma_phi = 1/T2`, so the input `T2` is the resulting coherence time (when `thermal_population == 0`). The number operator `n` gives the standard `(m-n)^2` dephasing scaling across higher levels.
- thermal up/down channels when `thermal_population` is set

Those are assembled in the device layer. Backends only receive already-built operators.

Noise parameters are ordinary tracked attributes: set (or clear with `None`) at construction **or any time after** — collapse operators are rebuilt from current values on every solve, and post-construction writes get the same validation as the constructor. Chip-level shared/collective dissipation lives in `Bath` ([`quchip/chip/baths.py`](quchip/chip/baths.py)), attached at construction or later via `chip.add_bath(...)`; bath rates are Lindblad-ready 1/ns with no assembly `2π` (that boundary is Hamiltonian-only — a component's *intrinsic* `2π`, e.g. a resonator's `κ = 2π·f/Q`, is its own physics).

## 4. Frames

### 4.1 What frame selection means

Source: [`quchip/engine/stage1_frames.py`](quchip/engine/stage1_frames.py)

The public frame spec is one of:

- `"lab"`
- `"rotating"`
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
- that frequency is `device.reference_freq`
- `device.reference_freq` defaults to `device.drive_freq` (the dressed `0 -> 1` frequency when available, otherwise the bare frequency), and is a settable per-device knob

So `"rotating"` is not a special solver mode. It is just a specific choice of `omega_ref,i`.

### 4.4 `reference_freq` — the readout / LO reference

Source: [`quchip/devices/base.py`](quchip/devices/base.py)

`device.reference_freq` is the frequency the rotating frame co-rotates at *and* the reference observables are reported in (§8). It defaults to `drive_freq`, so an unset device co-rotates at its own transition and behavior is unchanged. Setting it off the transition leaves a residual detuning `Δ = omega - omega_ref` in `H0` — idle Ramsey precession — which is how a control/LO calibration error is modelled.

It is a *frame / readout* reference only: it does **not** detune drives (the drive carrier is a separate choice, so a real LO error must also set the drive frequency). It is ordinary GHz, tracked (mutating it invalidates engine caches), and JAX-traceable / differentiable / sweepable.

## 5. Frame Tracking in the Engine

Source: [`quchip/engine/stage2_assembly.py`](quchip/engine/stage2_assembly.py)

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

## 6. RWA

RWA in quchip means "drop fast, non-resonant pieces instead of carrying them explicitly."

There are two places where that can happen.

### 6.1 Coupling RWA

Source: [`quchip/chip/rwa.py`](quchip/chip/rwa.py)

Coupling RWA is a chip-resolved policy, not a second operator defined by each coupling class. A coupling implements exactly one interaction, `interaction_hamiltonian()` (§2.2) — always the full, non-RWA form — and, optionally, `rwa_keeps_band(delta_a, delta_b)`, a predicate over the two-body excitation-change bands `(delta_a, delta_b)` the interaction decomposes into (default: total-excitation-conserving, `delta_a + delta_b == 0`, the beam-splitter selection). `Chip.resolve_rwa(coupling)` chooses the per-coupling override when present and otherwise uses the chip default. The resolved value is used in both paths:

- `Chip.hamiltonian()` masks the full local operator down to the bands `rwa_keeps_band` accepts (`apply_rwa_mask`), before embedding it into the static lab-frame Hamiltonian (§2.3).
- Stage 2 (`_collect_coupling_terms` in [`quchip/engine/stage2_assembly.py`](quchip/engine/stage2_assembly.py)) band-decomposes the same full operator and filters with the same predicate. Rejected bands become advisory `DroppedTerm` records with `reason="counter-rotating under RWA"`. Each record carries the band's `(delta_a, delta_b)` weights, its largest matrix-element magnitude, and its frame frequency. The amplitude equals `|g|` for a two-level `Capacitive` interaction, but ladder-operator factors can make it larger when either mode has more levels. Retained bands follow the general per-band static/dynamic fold of §5.2: a band whose carrier `delta_a*omega_a + delta_b*omega_b` is concretely zero stays folded into `H0`; every other retained band is subtracted back out and carried as an explicit time-dependent term.

For `Capacitive`:

```text
full: g * (a + a†)(b + b†)
     = g * (a†b + ab†) + g * (ab + a†b†)
```

- `a†b + ab†` is the `delta_a + delta_b == 0` band — the exchange term the default predicate keeps
- `ab + a†b†` is the `|delta_a + delta_b| == 2` band — counter-rotating, dropped under RWA

`interaction_hamiltonian()` always returns the full form; the RWA form `g * (a†b + ab†)` is never defined separately — it is exactly the retained band of the full form, reconstructed by the mask. A coupling that needs a different retained band set (e.g. a two-photon coupling keeping `|delta_a + delta_b| == 2` instead) overrides `rwa_keeps_band`; the override must stay symmetric under joint sign flip (`keeps_band(-delta_a, -delta_b) == keeps_band(delta_a, delta_b)`) so the retained operator stays Hermitian. Because the predicate depends only on integer band offsets — never on frequency values, which may be traced — the mask is a concrete constant regardless of tracing in the operator it multiplies.

The static/dynamic decision is made per band, not per coupling: in a *shared* frame (multiple devices detuned to a common reference), the coupling's counter-rotating band can carry a nonzero carrier even when its co-rotating band is frame-static. The per-band fold evaluates each band's own carrier independently, so a shared frame never suppresses the counter-rotating band's true rotation.

If `rwa=False`, no band is dropped; every non-static band — exchange and counter-rotating alike — is carried as an explicit time-dependent term at its own frame frequency.

### 6.2 Drive RWA

For a single-tone drive channel, the engine forms the real lab-frame field

```text
Re[s(t) * exp(-i 2π f_drive t)]
```

and combines it with the operator bands.

Without RWA, both co-rotating and counter-rotating pieces remain.

With RWA, the engine keeps only the slow piece for each excitation band. So if a band corresponds to a transition near `f_drive`, the near-resonant envelope remains and the fast partner at approximately `f_drive + f_transition` is dropped.

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

How quchip removes them:

- coupling CR terms are removed by choosing `rwa=True` on the coupling or chip default
- drive CR terms are removed by choosing `rwa=True` on the drive or chip default

How quchip keeps them:

- choose `rwa=False`
- then the engine keeps them as explicit time-dependent terms

## 8. Observables and Demodulation

Source: [`quchip/engine/stage3_observables.py`](quchip/engine/stage3_observables.py)

Dict-form `e_ops` are decomposed into the same excitation bands used by the frame logic. After the solver returns, the engine recombines them with the demodulation frequencies in `ResolvedFrame.demod_freqs = omega_ref - omega_frame` (per device).

This makes `result.expect` a **co-rotating readout**: observables are always reported in each device's `reference_freq` frame, independent of the integration frame the solver used. So transverse observables (`<a>`, `<sigma_x>`) come back as the non-oscillatory demodulated envelope a lab readout produces — slow, and turning at `Δ = omega - omega_ref` when the reference is detuned; diagonal observables (populations) are frame-invariant. In the default `"rotating"` mode the integration frame *is* the reference frame, so the demodulation is a no-op and `result.expect` equals `Tr(O·rho)` on the same states `result.states` returns. The raw, un-demodulated band sum (the observable in the integration frame) remains available on each `ObservableTrace` as `.raw`.

## 9. Dressing

Source: [`quchip/chip/chip.py`](quchip/chip/chip.py)

`Chip.dress()` diagonalizes the full static lab-frame Hamiltonian, assigns bare product states to dressed eigenstates by overlap, and stores a `DressedResult` containing:

- eigenvalues and lazily materialized eigenstates
- bare-to-dressed state assignments and the assigned eigenvalue for each bare label
- assignment overlaps and labels below the requested overlap threshold
- the dressed eigenvector matrix used by dressed-basis analysis

`Chip.freq()` evaluates dressed `0 -> 1` frequencies through the traceable array-labeling cache; those frequencies are not stored in `DressedResult`.

Dressing is lab-frame analysis. It is not part of the runtime frame transform.

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

## 10. Adiabatic Elimination and Dispersive Readout

Sources: [`quchip/chip/transformations/`](quchip/chip/transformations/), [`quchip/analysis/dispersive_readout.py`](quchip/analysis/dispersive_readout.py)

`eliminate(chip, target, method="sw"|"exact")` performs model reduction, dispatched on the target. A *device* target removes a far-detuned mode and folds its 2nd-order effect into the survivors: the Lamb shift `g^2/Delta` into `freq`, the Purcell rate `(g/Delta)^2 * kappa` into `T1`, and, for a mode touching two or more survivors, the mediated exchange `J = (g_a*g_b/2)(1/Delta_a + 1/Delta_b)` into each survivor pair (F. Yan et al., PRApplied 10, 054062 (2018)). A fixed-frequency eliminated mode produces a `Capacitive` edge; a frequency-controlled mode produces a `TunableCapacitive` edge. If a compatible direct edge already joins the pair, the exchange is folded into that edge while preserving any existing tunability. A *coupling* target keeps both endpoints and replaces the edge with a `CrossKerr` at the dressed pull. Readout quantities `chi` and `kappa` remain available in `effective_params` after a resonator is eliminated. Sources for the reduction math: [`quchip/chip/sw.py`](quchip/chip/sw.py).

### 10.1 The χ convention and related quantities

```text
chi ≡ chi_pull ≡ f_r(qubit in |1>) − f_r(qubit in |0>)     [GHz]
```

the *full* resonator pull per qubit excitation. This is **2×** the σ_z-convention χ of `H_disp = (omega_r + chi_sigma_z * sigma_z) * a†a` used in most textbooks. Three related quantities use different conventions:

- `eliminate(...).effective_params[q]["chi"]` — χ_pull as defined above, computed *numerically* from the pre-elimination dressed spectrum (identically `Chip.dispersive_shift(r, q)`: `E(1,1) − E(1,0) − E(0,1) + E(0,0)`, one shared diagonalization), exact and device-agnostic (works for any survivor type, not just Duffing transmons). The entry is evaluated and cached on first access, so the diagonalization occurs only when `chi` is read.
- the `"chi"` fit target of `fit_a_dress` ([`quchip/inverse_design/fit.py`](quchip/inverse_design/fit.py)) — the σ_z convention, i.e. **χ_pull / 2**.
- `Chip.dispersive_shift(a, b)` (alias `static_zz`) — the general two-mode cross-Kerr `E(1,1) − E(1,0) − E(0,1) + E(0,0)`. For a qubit–resonator pair this *is* χ_pull (which is exactly how the `chi` entry is computed); between two qubits the same expression is the static-ZZ ζ — do not read a qubit–qubit `dispersive_shift` as a readout χ.

Analytic cross-checks (2nd-order dispersive): two-level `chi = 2g^2/Delta`; Duffing transmon `chi = 2g^2*alpha/(Delta*(Delta+alpha))` with `Delta = f_q − f_r` (Koch et al., PRA 76, 042319, §IV). Critical photon number `n_crit = Delta^2/(4g^2)`.

`effective_params[q]["kappa"]` is the eliminated mode's total downward decay rate, in 1/ns, as returned by `intrinsic_decay_rate()`. For a resonator it includes `2π*f_r/Q` when `quality_factor` is set and the inherited thermal-emission rate when `T1` or `thermal_population` is set. The latter is `(nbar + 1)/T1` with `T1`, or `nbar + 1` when only `thermal_population` is present. The reported value is `0.0` only when none of these lowering channels is configured. Bridge legs report `chi = 0.0`: bus/coupler modes are not readout modes, and their dressed pull would double-count the mediated exchange.

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

The chip's full bare Hamiltonian `H = H0 + V` (GHz, pre-`2π`, at the chip's RWA policy) is partitioned by the eliminated mode's occupation: `P` = the mode in `|0>`, `Q` = everything else. The generator solves the Sylvester condition on the cross blocks,

```text
S_ij = V_ij / (E_i − E_j)        (i, j straddling P/Q; E = diag H)
H_eff = P (H + (1/2)[S, V]) P
```

(Bravyi, DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011), 2nd order). Nested `where` guards handle the division: an exactly degenerate cross pair with no matrix element contributes zero with a *finite gradient*, whereas a single `where` would propagate a `NaN` backward through the unselected branch. Survivor parameters are obtained by indexing `H_eff`: `freq_after(s) = E(1_s) − E(0)`; the pair exchange is the `<1_a|H_eff|1_b>` element. A pre-existing direct edge is already included in `H_eff`, so the emitted edge carries the total coupling and the reported `j_eff` subtracts the direct contribution. Alongside `J`, the bridge fold records its linearization

```text
dJ/domega_c = (g_a*g_b/2)(1/Delta_a^2 + 1/Delta_b^2)
```

— the weight the flux-drive retarget rule uses (§11). Per-element virtual-state attribution (`pathways`) is `(1/2) V_ik V_kj (1/(E_i−E_k) + 1/(E_j−E_k))` summed over intermediate `|k>`, with the same guarded denominator.

### 10.4 The exact route (`method="exact"`)

One full diagonalization; parameters are read off the *labeled* dressed spectrum (`label_eigensystem`, §9), so kept-block energies are exact to all orders, as required for residual ZZ:

```text
zz(a, b) = E(1,1) − E(1,0) − E(0,1) + E(0,0)        (≡ Chip.dispersive_shift)
```

The pair exchange is read through the symmetrically (Löwdin-)orthonormalized subspace projection `S^(−1/2) (W E W†) S^(−1/2)` with `W` the overlap block and `S = W W†` — the des-Cloizeaux effective Hamiltonian, whose spectrum equals the labeled energies exactly. The energies are exact, but this basis is not the canonical SW rotation, so off-diagonal reads agree with `method="sw"` only through 2nd order. `method="exact"` raises when a kept bare label has no majority dressed eigenstate, or when two kept labels claim the same one. In that regime, near-degenerate dressed states straddle the bare labels and quantities assigned to a single label are not well defined. Use `method="sw"` or shift the operating point.

### 10.5 Collapse transforms and validity metrics

The eliminated mode's own jump operator is carried into the reduced frame by the same rotation as the Hamiltonian:

```text
c_eff = P (c + [S, c]) P            (sw — 1st order in S, matching H_eff's 2nd order)
c_eff = P U† c U P                  (exact — U the labeled eigenvector matrix)
```

and the survivor-lowering amplitude gives the inherited (Purcell) rate `|amplitude|^2 * kappa`. The result's `notes` record that the projection is exact for the *spectrum* but approximate for *dissipation*: the discarded `Q`-block dynamics also dephase and decay. `validity` reports, per eliminated coupling, `g_over_delta` (2nd-order smallness; `is_valid` gates at `< 0.1`) and `min_block_gap` — the smallest bare-energy gap the Sylvester generator crossed. A small gap with a nonzero matrix element is the perturbative expansion's failure mode even when every `g/Delta` is small.

## 11. Parametric Edge Control

Sources: [`quchip/control/drive.py`](quchip/control/drive.py) (`ParametricDrive`), [`quchip/engine/stage2_assembly.py`](quchip/engine/stage2_assembly.py) (`EDGE_PUMP`), [`quchip/chip/retarget.py`](quchip/chip/retarget.py)

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
3. The default `"rotating"` frame uses each device's best available drive frequency.
4. Each drive channel declares one `DriveModulation`: `single_tone` for an IQ-style carrier, `direct_real` for real baseband modulation, or `edge_pump` for modulation of a coupling strength.

What the engine does not hardcode:

- transmon-specific formulas
- resonator-specific formulas
- backend-native operator types
- device-specific noise models beyond asking each device for its collapse operators

Devices, couplings, and drives supply the domain-specific physics. The engine handles the frame and RWA bookkeeping shared by them.

## 13. JAX Traceability Boundaries

Band decomposition, coefficient construction, observable recombination, and the backend-free Hamiltonian IR preserve JAX arrays.

The following operations require concrete Python values:

- `Chip.dress()` returns a concrete dict-based view and is not traceable. The bare→dressed assignment itself is discrete and piecewise. Traced callers should use `Chip.energy()`, `Chip.freq(target, when=...)`, or `Chip.dispersive_shift()`, which route through `label_eigensystem` in the pure-JAX kernel in `quchip/chip/dressing.py`; labeled energy lookup stays differentiable away from label discontinuities. `track_path` is a separate continuation utility for following labels through a stacked eigensystem along a parameter sweep.
- Human-facing serialization and diagnostics coerce to Python scalars.

Other engine paths avoid implicit conversion to host arrays.

## 14. Audit Pointers

When you need to audit a physics path, start here:

- units and assembly boundary: [`quchip/engine/stage2_assembly.py`](quchip/engine/stage2_assembly.py)
- frame resolution: [`quchip/engine/stage1_frames.py`](quchip/engine/stage1_frames.py)
- observable preparation: [`quchip/engine/stage3_observables.py`](quchip/engine/stage3_observables.py)
- observables and demodulation: [`quchip/engine/stage3_observables.py`](quchip/engine/stage3_observables.py)
- dressing and public Hamiltonian APIs: [`quchip/chip/chip.py`](quchip/chip/chip.py)
- adiabatic elimination and χ/κ reporting: [`quchip/chip/transformations/`](quchip/chip/transformations/)
- Schrieffer-Wolff kernels and the exact reduction route: [`quchip/chip/sw.py`](quchip/chip/sw.py)
- control-line retargeting across reductions: [`quchip/chip/retarget.py`](quchip/chip/retarget.py)
- readout pointer states and figures of merit: [`quchip/analysis/dispersive_readout.py`](quchip/analysis/dispersive_readout.py)
