# Eliminating a readout on a shared feedline (#101)

Working note for this branch. The commit that opens the pull request deletes it.

Baseline: quchip/quchip `feat/local-elimination-ring-patches` at f78c5d8 (#113, stacked on #112 at ccb0157 and #111 at 22ff973). The branch merges two open branches that this work needs:

- `fix/dressed-mode-reflection-section` at f9c60de (ibralyousef/quchip), one commit on main. It adds `ReductionMethod.dressed_transition` and gives the reflection section dressed values.
- `fix/feedline-excitation-conservation` at 5078a83 (#110). Without it, a chip with a cascade fails `conserves_excitation_number`. `reduce_device` then sets no sectors, and the transformed port declares no excitation change. VNA then rejects the reduced chip with "The selected tones leave dynamic Hamiltonian terms".

## Gap

`reduce_device` refuses a port that takes part in a cascade-generated Hamiltonian (`quchip/chip/transformations/eliminate_device.py`, L511 to L520). It also needs a plane that carries no other field (`_exclusive_exposure`, `quchip/chip/port_network.py` L1615, called at L539). A feedline through several readout ports fails both checks.

## Physics

A chiral line joins ports `L_1, ..., L_N` in this order. The series product gives one collective jump `L = Σ L_k` and one Hermitian term `H_line = Σ_{q>p} Im(L_q† L_p)`. In `-iH - L†L/2`, the cross terms add to `-L_q† L_p` for `q > p`, so only a downstream mode feels an upstream one.

The series product is covariant under a unitary on the chip. Transforming `H_chip` and every `L_k` and then forming the line gives the same model as transforming the formed line. An elimination is such a unitary followed by a projection, so forming the line after the transform adds no approximation of its own.

At the removed port, the field passes the removed mode once and gains

```text
S_r(f) = ((κ_i - κ_e)/2 - iΩ) / ((κ_e + κ_i)/2 - iΩ),   Ω = 2π (f - f_R)
```

A frequency-dependent factor cannot sit inside the Markov core between two ports. A scalar factor on a one-way line commutes with the linear response of the ports behind it. So `S_r(f)` on one boundary leg gives the right weak-probe response from the line input, such as a VNA sweep. A port on the wrong side of the section emits and receives fields with an error of `|S_r(f_k) - 1| ≈ κ_e / (2π |f_k - f_R|)`.

## Decisions

### D1. Transform the Hamiltonian and the line together

The reduction input excludes static terms with `origin == "network"`. The final matrix of the correction (`final_matrix`, L864) excludes them too. Ports transform as today. The network then regenerates `H_line` and `L` from the transformed ports.

- Add a keyword to `_analysis_matrix_ghz` (`quchip/engine/assembly.py` L1863) and `bare_hamiltonian` (`quchip/chip/sw.py` L46), such as `include_network: bool = True`. `reduce_device` passes `False` at L630.
- Compute `final_matrix` from the static terms of `final_resolved` without network terms. `_analysis_matrix_ghz(final_resolved, include_network=False)` gives the same GHz matrix that `hamiltonian().matrix()` gives today, minus those terms.
- Apply this to every device elimination, with or without ports. A generated term never belongs in the reduction input.
- `chip.freq()` and other callers keep the network terms.
- Delete the cascade refusal at L513 to L520 and `affected_labels` at L511, which nothing else reads.

Why: reducing `H_chip + H_line` block-diagonalizes half of a non-Hermitian coupling. The level repulsion of `H_line` then shifts the kept readouts, by 5.0 kHz on the prototype's ring chips. On the test chip, S21 then errs by 8e-3 across the kept dips. With D1, kept response poles agree within 1.1 Hz.

### D2. A one-pass section kind

Add `PortNetwork.mode_transmission(label, *, freq, external_rate, internal_rate=0.0)`, kind `"mode_transmission"`. A field that crosses from side 1 to side 2 gains the full `S_r(f)`, with no square root and no branch. The reverse direction is transparent, as for the amplifier.

- Transfer: `_mode_transmission_transfer(frequency, *, freq, external_rate, internal_rate=0.0)`, next to `_mode_reflection_transfer` (L403). Traceable, with `select_array_module(contains_tracer(...))`.
- Add the kind to `_SERIALIZED_FACTORIES` (L382) and `_REFERENCE_KINDS` (L400).
- `_peel` (L1990): append the element only on the forward traversal, `(name == "2") if outbound else (name == "1")`. This is the rule that attenuators and isolators use for their forward side.
- `_compile` (L2070): skip side `"1"` of the new kind, as for the amplifier, so the reverse traversal embeds nothing.
- The internal-loss bath is vacuum, as for `mode_reflection`.

`mode_reflection` on one leg gives only `sqrt(S_r)`. On both legs it gives the right probe S21, but its phase winds by 2π across an overcoupled resonance. Emission from a kept readout above the removed one then flips sign (error 2.00 in the prototype).

### D3. Plane and leg

Generalize `_exclusive_exposure` into a helper that returns the plane and the ports upstream and downstream of the removed port, such as `_line_exposure(port_label) -> (exposure_label, upstream, downstream)`. Delete `_exclusive_exposure` once nothing calls it.

- Keep its checks. The port couples to exactly one channel, and that channel is not hidden. Its support neither mixes with nor feeds another channel.
- Order the ports with the compiled `generated_pairs` (structure, not `_active_pairs`). A pair `(downstream, upstream, coefficient)` puts `upstream` before `downstream`.
- Every port on the channel must lie upstream or downstream of the removed port, and none on both sides. Otherwise raise with the existing message, which contains "shares its external plane".
- No other port on the channel: a reflection plane. Keep today's two-pass `mode_reflection` on both legs, with the merged dressed values.
- Other ports on the channel: a line. Insert one `mode_transmission` section on one leg.

Leg rule: the outbound leg, unless kept readout ports lie downstream of the removed port and none lie upstream. Then the inbound leg. A kept readout port is a port on the line that no earlier reduction transformed (`_retained_port_labels(chip)`, L106, lists the transformed ones). The rule reads labels and graph structure only, so a traced or rebound value cannot change it.

Placement: `_insert_reference_section` (L1641) gets a keyword such as `reverse: bool = False`. It swaps `inner` and `outer`, so side 1 faces the plane and side 2 faces the core. Use it for the inbound leg. The new section stays innermost, as today. Sections from several removals stack on the legs, and their order does not matter.

Section label: `f"{mode_label}_transmission"`, with the same numeric suffix rule as `_reflection` (L992 to L996).

### D4. Section values

Use the merged hook, exactly as the reflection section does at L997 to L1006:

```text
frequency, weight = reduction.dressed_transition(ctx, lowering)
freq = frequency
external_rate = weight * port rate
internal_rate = weight * internal_rate
```

With D1, `ctx.h` holds no network terms, so the dressed values come from the chip without `H_line`.

### D5. Validity entry

Each line section adds `validity[section_label] = {"kappa_over_delta": r, "is_valid": r < 0.1}`, the same shape as the `g_over_delta` entries.

```text
r = external_rate / (2π min |f_s - f_R|)
```

- The minimum runs over the targets `s` of the kept readout ports beyond the section. Beyond means downstream of the removed port for the outbound leg, and upstream for the inbound leg.
- `f_s` and `f_R` are `incoming_frequencies` (L658) of the target and of the removed mode. `external_rate` is the section's dressed rate.
- `r = 0` when no kept readout port lies beyond the section.
- Ports that an earlier reduction transformed are left out. They act mostly on far-detuned survivors, so their error is of order `κ_e / (2π Δ)` at those survivors (7e-4 for a qubit 1.4 GHz away).
- A reflection plane gets no entry.

### D6. Unchanged and still refused

- The transformed ports, inherited channels, `effective_params`, the `g_over_delta` entries, the mapping and serialization of existing kinds.
- Still refused: a mode with several ports, custom or collective boundary operators, nonlinear targets, projected survivor bases, and channels that mix with other channels.
- `local=True` on a shared line needs no code beyond D1 to D5 and D9. Since #112, `_local_patch` (L200 to L203) adds the devices of each port pair whose series composition generates a Hamiltonian on a core device. The patch then holds every readout on the line, and `_patch_chip` keeps all of the line's ports. On the test chip, the local SW route gives the same sections and S21 as the full SW route, for one removal and for two. The local route implements only `method="sw"`.

### D7. Traceability

Everything stays traced under `jax.jit` and `jax.grad`. No `float()`, `int()`, `bool()` or Python branch on traced values. Compute `r` with the array module (`xp.min`, `xp.abs`), not with Python `min()`. The leg and the set of ports beyond are structural.

### D8. Notes

The reflection section keeps its note. For a line section, `reduce_device` appends one note in the same style. It names the section and its leg. It keeps the reflection note's statements on the dressed values, survivor damping, the Purcell residual, the SW frequency error and the vacuum bath. It adds that the leg does not change the weak-probe response from the line input. Near the frequencies of kept ports beyond the section, fields that those ports emit or receive err by up to the reported `kappa_over_delta`.

### D9. Internal loss after an earlier reduction

The section's internal rate sums `-2 rate Re <0|D†[L](a)|1>` over the channels whose support is the removed mode alone (L895 to L928). An earlier reduction resolves each survivor channel through its captured map. The mode's own loss then acts on several devices, all survivors after a full reduction and the patch survivors after a local one. A second elimination finds no channel on the mode alone and sets `internal_rate = 0`. S21 then errs by 0.58 at the second removed dip. Main shows the same defect for two reflection planes removed in turn, and #112 shows it for a full and then a local removal.

- Sum over every channel whose support contains the removed mode. Embed the mode lowering on that support. Take the element between the support's ground state and its state with one mode excitation, in energy coordinates.
- Full route: add the term in `inherit_channel` (L897) after `transform_operator`, for every such support.
- Local route: add it before the early return for supports other than the mode alone (L906 to L907). Add it also in the loop that carries retained channels (L937 to L944), with `terms.channel_expression(channel)`. In both places, first transform the operator with `chip_bases[label].energy_vectors` of each support device.
- A channel that commutes with the mode lowering adds nothing. So survivor channels stay out, as the reflection note states.
- `effective_params` keep their rule. Their `kappa` and `purcell_rate` still read only the channels on the mode alone.

The loop over retained channels matters when an earlier removal dressed its own loss onto the mode. A reference check couples a qubit with `T1 = 100` ns to r1 at 5.6 GHz, removes it, and then removes r1. All four pairs of full and local SW removals then give the same section. Without the second local branch, the local route drops the qubit's carried loss, 0.4 % of the internal rate, and S21 moves by 1.3e-3. The branch's test does not cover this case.

## Tests on this branch

One e2e test replaces `test_eliminate_rejects_port_that_generates_a_cascade_hamiltonian` in `tests/test_eliminate_port_network.py`. The branch adds no other test.

| Test | Suite | Checks |
|---|---|---|
| `test_feedline_readout_elimination_keeps_the_line_transmission[middle-readout]` | e2e | Three readouts on one line. Removing r2 puts `r2_transmission` on the outbound leg of the restored chip, with `kappa_over_delta = κ/(2π · 99 MHz)` = 0.016. S21 across all three dips matches the full chip within `2 (g/Δ)² κ/(2πΔ)` = 1.6e-6. |
| `test_feedline_readout_elimination_keeps_the_line_transmission[middle-then-first-readout]` | e2e | Removing r2 and then r1 puts `r1_transmission` on the inbound leg and keeps `r2_transmission` on the outbound leg. The r1 section has `kappa_over_delta = 0`. S21 matches within 2.7e-6, which needs D9. |
| `test_feedline_readout_elimination_keeps_the_line_transmission[middle-then-first-readout-local]` | e2e | The same removals with `method="sw", local=True` give the same legs and validity. S21 matches within `2 · 8π g⁴/(κ_e Δ³)` = 6.9e-3, twice the S21 change that the SW frequency error `g⁴/Δ³` can cause. It needs D9 in the local route. |

The test runs VNA on the chip restored from `to_dict`, so it also checks serialization and the orientation of each section.

On this branch all three cases fail with the cascade refusal. A private reference implementation of D1 to D5 and D9 passes all three. Measured with it, exact route:

| Removed readouts | Max \|ΔS21\| across the three dips | Bound | `kappa_over_delta` of the last section |
|---|---|---|---|
| r1, inbound | 5.6e-7 | 2.7e-6 | 0 |
| r2, outbound | 3.3e-7 | 1.6e-6 | 0.01606 |
| r3, outbound | 2.1e-7 | 1.0e-6 | 0 |
| r2, then r1 inbound | 5.6e-7 | 2.7e-6 | 0 |

Every order of two removals, and the removal of all three, gives the largest single-removal error, at most 5.6e-7. Two separate reflection planes, removed one after the other, give 5.6e-7 and 3.3e-7. The SW route errs by 6.5e-4 to 1.7e-3, from its fourth-order section frequency. For r2 and then r1 it errs by 1.7e-3, as `8πκ_e g⁴/(κ²Δ³)` predicts. The local SW route matches the full SW route to round-off. The prototype gives the same numbers for the exact route. The test catches these wrong variants:

| Variant | Max \|ΔS21\| |
|---|---|
| reduce `H_chip + H_line` (1B) | 8e-3 across kept dips |
| bare section values | 1.1 to 1.2 across the removed dip |
| `mode_reflection` on one leg | 2.0 |
| no section | 1.4 |
| no D9, second removal, either route | 0.58 at the second removed dip |
| section on the wrong leg | S21 unchanged, caught by the plane and validity checks |

## Documentation

- PHYSICS.md §3.3.2: a paragraph for `network.mode_transmission(...)` after the one for `mode_reflection(...)`. State that a one-pass section acts on one leg only. The sentence "A section on a reflection line contributes once inbound and once outbound" then needs that qualifier.
- PHYSICS.md §10.5: remove "ports that participate in a cascade-generated Hamiltonian" from the rejection list. In the same list, "a plane that also carries other fields" now covers only fields that do not pass the port in series. Add a paragraph on shared feedlines: D1 and why, the one-pass section, the leg rule, the error for fields that kept ports beyond the section emit or receive, and the validity entry.
- PHYSICS.md §10.5: the sentence "The mode's own channels supply `kappa_i`" gains that this includes the mode's loss that an earlier elimination carried in a retained channel (D9).
- The `mode_reflection` docstring says that `eliminate` inserts it. Say that this holds for a port alone on its plane, and that a shared line gets `mode_transmission`.
- CHANGELOG.md Unreleased, New features: `eliminate()` removes a readout whose port shares a feedline with other ports, and `PortNetwork.mode_transmission(...)`. Link #101.
- CHANGELOG.md Unreleased, Fixes: a second elimination keeps the removed mode's internal loss in its section (D9). Before, the section dropped it, and S21 erred by up to 0.6 at that resonance. Link #101.
- Prose follows the repository's public-prose rules: plain scientific prose close to ASD-STE100, sentences of 25 words or fewer, and active voice. Use no em dashes, semicolons, bold emphasis, recap lines or promotional words. NumPy-style docstrings with imperative summaries.

## Checks before review

The branch already holds the squashed dressed-section commit f9c60de. Delete this note in the commit that opens the pull request.

```bash
python -m pytest tests/test_eliminate_port_network.py tests/test_local_elimination.py
python -m pytest -m unit
python -m pytest -m e2e
ruff check .
python -m mypy quchip tests/typing/external_declarative_models.py
```
