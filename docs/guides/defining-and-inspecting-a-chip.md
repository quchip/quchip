# Your first chip

Build a coupled transmon–resonator model, read its dressed frequencies, and
find bare parameters that meet a design target. Frequencies are in GHz.

## Declare the model

```python
from quchip import Capacitive, Chip, DuffingTransmon, Resonator, fit_a_dress

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
qr = Capacitive(q, r, g=0.05, label="qr")
chip = Chip([q, r], [qr])
```

Labels name the model's parameters: `q.freq`, `q.anharmonicity`, and `qr.g`.
Inspect the declaration with `describe()`:

```python
print(chip.describe())
```

Output:

```text
Chip
════
Frame    : lab
Approx.  : RWA
Dressed  : not computed
Hilbert  : 4 x 5 = 20 levels

Devices (2)
───────────
q — DuffingTransmon
    T1 = None   T2 = None   thermal_occupation = None   freq = 5 GHz   anharmonicity = -0.25 GHz   levels = 4
r — Resonator
    T1 = None   T2 = None   thermal_occupation = None   freq = 7 GHz   internal_quality_factor = None   levels = 5

Couplings (1)
─────────────
qr : q ↔ r
    g = 0.05 GHz
```

## Read dressed observables

The device constructors specify bare parameters. `chip` returns the
frequencies and interactions of the coupled system.

```python
print(f"Qubit f01: {chip.freq(q):.6f} GHz")
print(f"Resonator frequency: {chip.freq(r):.6f} GHz")
print(f"Conditional resonator pull: {1000 * chip.dispersive_shift(q, r):.3f} MHz")
```

Output:

```text
Qubit f01: 4.998533 GHz
Resonator frequency: 7.001041 GHz
Conditional resonator pull: -0.286 MHz
```

The full conditional pull is $E_{11}-E_{10}-E_{01}+E_{00}$. Its half is the
usual $\chi$ in the sigma-z dispersive Hamiltonian. `chip.kerr_matrix()` collects
these full pulls off-diagonal and dressed anharmonicities on the diagonal.

## Change a parameter

`with_params()` returns a new chip; the source declaration stays available.

```python
shifted = chip.with_params({"q.freq": 5.1})

print(f"Original bare / dressed: {q.freq:.3f} / {chip.freq(q):.6f} GHz")
print(f"Changed bare / dressed: {shifted.parameters['q.freq']:.3f} / {shifted.freq('q'):.6f} GHz")
```

Output:

```text
Original bare / dressed: 5.000 / 4.998533 GHz
Changed bare / dressed: 5.100 / 5.098470 GHz
```

## Inspect the Hamiltonian

`unresolved_hamiltonian()` shows the Hamiltonian you declared. `hamiltonian()` is the
expression selected for simulation, after basis, frame, and RWA choices.
Both render as equations when displayed in a notebook.

```python
declared = chip.unresolved_hamiltonian()
resolved = chip.resolve(frame="rotating")

print(resolved.dropped_terms_summary())
```

Output:

```text
2 term(s) dropped:
  [qr] coupling band (Δa=-1, Δb=-1) on q·r  (counter-rotating under RWA; amp 0.173205 GHz, freq 11.9996 GHz)
  [qr] coupling band (Δa=+1, Δb=+1) on q·r  (counter-rotating under RWA; amp 0.173205 GHz, freq 11.9996 GHz)
```

The dropped terms create or annihilate two excitations. Their oscillation
frequencies are much larger than their matrix elements here. See the
[physics reference](../physics.md) for frame and approximation conventions.

## Fit a dressed target

For `fit_a_dress()`, device values specify desired **dressed** frequencies
and anharmonicities. Coupling values specify desired full cross-Kerr pulls.
The fit returns the bare parameters that produce those targets.

```python
target_q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
target_r = Resonator(freq=7.0, levels=5, label="r")
target_pull = Capacitive(target_q, target_r, g=-0.0003, label="qr")
target = Chip([target_q, target_r], [target_pull])

fit = fit_a_dress(target)

print(f"Converged: {fit.converged}; normalized loss: {fit.loss:.2e}")
```

Output:

```text
Converged: True; normalized loss: 1.43e-21
```

<details>
<summary>Inspect fitted parameters and residuals</summary>

```python
print(fit.summary())
```

Output:

```text
fit_a_dress: converged | loss 1.43e-21 | targets: 4 | parameters: 4
identifiability: rank 4/4 | condition 33.9
`xtol` termination condition is satisfied.
targets (GHz):
  q.freq [component default]: 5 -> 5 (error -2.8e-14)
  q.anharmonicity [component default]: -0.25 -> -0.25 (error +6.4e-14)
  r.freq [component default]: 7 -> 7 (error +1.4e-14)
  qr.cross_kerr [component default]: -0.0003 -> -0.0003 (error +1.1e-14)
bare parameters (GHz):
  q.freq: 5 -> 5.00153 [component declaration]
  q.anharmonicity: -0.25 -> -0.250281 [component declaration]
  r.freq: 7 -> 6.99891 [component declaration]
  qr.g: 0.0512101 -> 0.0511229 [isolated-pair root solve; positive convention]
```

</details>

Continue with [spectra and parameter sweeps](statics-and-parameter-studies.md)
to sweep the coupled spectrum. For imported or custom models, see
[scqubits interoperability](../api.md) and [extensions](../extensions.md).
