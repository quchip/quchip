# Define and inspect a chip

A quchip model declares its devices, couplings, and resolution choices. Start
with the supplied components. Add a custom model only when the local physics
you need is missing. The examples move from one small declaration to dressed
observables, resolution records, inverse fitting, interoperability, and public
extension surfaces.

quchip uses GHz for frequencies and couplings, ns for time, and mK for
temperature.

## Start with two devices and one coupling

This is a complete static model of a Duffing transmon coupled capacitively to a
resonator:

```python
from quchip import Capacitive, Chip, DuffingTransmon, Resonator

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
qr = Capacitive(q, r, g=0.05, label="qr")
chip = Chip([q, r], [qr])

print("devices:", [device.label for device in chip.devices])
print("couplings:", [coupling.label for coupling in chip.couplings])
```

Output:

```text
devices: ['q', 'r']
couplings: ['qr']
```

Labels form stable parameter paths such as `q.freq`, `q.anharmonicity`, and
`qr.g` for sweeps and fitting.

## Read the declaration

`describe()` gives a compact text view with units, Hilbert-space dimensions,
the frame, the approximation, and every declared component:

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
    T1 = None   T2 = None   thermal_population = None   freq = 5 GHz   anharmonicity = -0.25 GHz   levels = 4
r — Resonator
    T1 = None   T2 = None   thermal_population = None   freq = 7 GHz   internal_quality_factor = None   levels = 5

Couplings (1)
─────────────
qr : q ↔ r
    g = 0.05 GHz
```

After you add control lines or baths, `describe()` adds matching sections.

## Ask a dressed question

The constructor values describe the local, bare models. Methods on `chip`
return observables of the coupled system:

```python
kerr = chip.kerr_matrix()
print("dressed f01:", chip.freq("q"))
print("full pull q-r (GHz):", kerr[q, r])
print("sigma-z chi (GHz):", kerr[q, r] / 2)
```

Output:

```text
dressed f01: 4.99853347343543
full pull q-r (GHz): -0.0002860201985654953
sigma-z chi (GHz): -0.00014301009928274766
```

`kerr_matrix()` follows `chip.devices`: diagonal entries are dressed
anharmonicities and off-diagonal entries are full-pull cross-Kerr shifts. The
common sigma-z Hamiltonian convention calls half of that full pull $\chi$.

## Inspect the authored and resolved Hamiltonians

The authored expression records the physics supplied by the device and
coupling models. It remains symbolic and can render itself as LaTeX:

```python
authored = chip.unresolved_hamiltonian()
print(authored.latex())
```

Output:

```text
\omega_{q}\,\hat n_{q} + 0.5\,\alpha_{q}\,\hat n_{q}\,(\hat n_{q} - \hat I_{q}) + \omega_{r}\,\hat n_{r} + g_{qr}\,(\hat a_{q} + \hat a^\dagger_{q})\,(\hat a_{r} + \hat a^\dagger_{r})
```

In a notebook, leaving `authored` as the last expression displays the rendered
equation:

```{math}
\omega_{q}\,\hat n_{q}
+ \frac{\alpha_{q}}{2}\,\hat n_{q}(\hat n_{q} - \hat I_{q})
+ \omega_{r}\,\hat n_{r}
+ g_{qr}(\hat a_{q} + \hat a^\dagger_{q})(\hat a_{r} + \hat a^\dagger_{r}).
```

`chip.hamiltonian()` returns the expression assembled for the solver under the
chip's current settings. Its matrix leaves have names because basis changes and
approximations have already turned the authored terms into numerical operators:

```python
resolved_lab = chip.hamiltonian()

print(resolved_lab.latex())
print("matrix shape:", resolved_lab.matrix().shape)
```

Output:

```text
1\,\hat H_0
matrix shape: (20, 20)
```

Use `unresolved_hamiltonian()` when you want the expanded authored equation.
Use `hamiltonian()` when you want the exact expression sent toward numerical
materialization, and call `.matrix()` only at that numerical boundary.

## Inspect how the model was resolved

`resolve()` returns the numerical model and its resolution record. We request
a rotating-frame snapshot here so the dropped terms report their physical
oscillation frequencies. This does not change `chip`.

```python
resolved = chip.resolve(frame="rotating")
frame = resolved.resolved_frame

print("Hamiltonian:", resolved.hamiltonian().latex())
print("frame mode:", frame.mode)
print(
    "frame frequencies:",
    {label: float(value) for label, value in frame.frequencies.items()},
)
print(
    "demodulation frequencies:",
    {label: float(value) for label, value in frame.demod_freqs.items()},
)
print("approximation:", type(resolved.approximation).__name__)
print("resolved dimensions:", resolved.dims)
for label, record in resolved.bases.items():
    print(
        f"{label}: kind={record.kind}, native={record.native_dim}, "
        f"resolved={record.resolved_dim}, V={record.vectors.shape}"
    )
print(resolved.dropped_terms_summary())
```

Output:

```text
Hamiltonian: 1\,\hat H_0 + f_{coupling,0}\!\left(t\right)\,\hat H_{coupling,0} + f_{coupling,1}\!\left(t\right)\,\hat H_{coupling,1}
frame mode: rotating
frame frequencies: {'q': 4.99853347343543, 'r': 7.001040953592849}
demodulation frequencies: {'q': 0.0, 'r': 0.0}
approximation: RWA
resolved dimensions: (4, 5)
q: kind=native, native=4, resolved=4, V=(4, 4)
r: kind=native, native=5, resolved=5, V=(5, 5)
2 term(s) dropped:
  [qr] coupling band (Δa=-1, Δb=-1) on q·r  (counter-rotating under RWA; amp 0.173205 GHz, freq 11.9996 GHz)
  [qr] coupling band (Δa=+1, Δb=+1) on q·r  (counter-rotating under RWA; amp 0.173205 GHz, freq 11.9996 GHz)
```

The two dressing modes answer different questions. `chip.dress()` always
diagonalizes the complete intrinsic static Hamiltonian in the lab frame,
independently of the chip's solve approximation. A resolved snapshot instead
owns the model selected for simulation. A static snapshot calls `dress()`
directly, for example `chip.resolve(frame="lab").dress()`. The rotating-frame
snapshot above contains dynamic coupling bands, so choose the instant
explicitly with `resolved.dress(at_time=25.0)`. For a built solve problem use
`problem.engine_result.dress(at_time=25.0)`.

That result is the eigensystem of the instantaneous resolved Hamiltonian. It
is not a Floquet or cycle-averaged spectrum; omitting `at_time` on a dynamic
snapshot raises.

The rotating-frame frequencies define

```{math}
U_{\mathrm{frame}}(t)
= \exp\!\left[-i 2\pi t
  \left(\omega_{\mathrm{frame},q}\hat n_q
      + \omega_{\mathrm{frame},r}\hat n_r\right)\right].
```

Assembly applies the corresponding counter-term
$-\sum_i\omega_{\mathrm{frame},i}\hat n_i$ and attaches the frame phases to
operator bands. It does not build a dense many-body unitary. The zero
demodulation frequencies above mean that the integration frame and each
device's readout reference coincide.

Both basis records are `native`, with square `V` matrices, so this example
uses identity local transforms and performs no projection. For an eigenbasis
projection, `V` has shape `(native_dim, resolved_dim)`, operators become
$V^\dagger O V$, and `record.projector` is $VV^\dagger$ in the authored space.

The weights `(-1, -1)` and `(1, 1)` identify the pair-annihilation and
pair-creation bands. `amp` is the largest matrix element in that truncated
operator band, not the declared coupling `g`. `freq` is the band's oscillation
frequency in the requested frame. Their ratio is about `0.0144`, which is the
small parameter behind this RWA drop. Each structured record is also available
through `resolved.dropped_terms` with `source`, `operator`, `reason`,
`band_weights`, `amplitude`, and `frequency` fields.

## Fit a desired dressed chip

`fit_a_dress()` reads the component declarations as numerical dressed targets.
It does not first simulate this desired chip. For a `Capacitive` or
`CrossKerr` edge, the declared scalar means the full cross-Kerr
$K_{ab}=E_{11}-E_{10}-E_{01}+E_{00}$ during this fit. The returned chip has
the bare coupling strengths needed to produce it.

This desired chip contains two qubits, one readout resonator, and three edge
constraints. We add one more constraint for exchange between the qubits.

```python
import matplotlib.pyplot as plt
import numpy as np

from quchip import Capacitive, Chip, CrossKerr, DuffingTransmon, Resonator, fit_a_dress

q0 = DuffingTransmon(freq=5.047559899, anharmonicity=-0.298378413, levels=3, label="q0")
q1 = DuffingTransmon(freq=5.318066243, anharmonicity=-0.279651205, levels=3, label="q1")
readout = Resonator(freq=7.103162679, levels=3, label="r")
q0_readout = Capacitive(q0, readout, g=-0.000533546, label="q0-r")
q1_readout = Capacitive(q1, readout, g=-0.000464088, label="q1-r")
qubit_zz = CrossKerr(q0, q1, chi=0.001641313, label="qq-zz")

desired_chip = Chip(
    [q0, q1, readout],
    [q0_readout, q1_readout, qubit_zz],
    frame="rotating",
    backend="qutip",
)

constraints = {
    (q0, q1): {"exchange_rate": -0.002163926},
}

print("devices:", tuple(device.label for device in desired_chip.devices))
print("couplings:", tuple(coupling.label for coupling in desired_chip.couplings))
print("component targets:", 8)
print("additional constraints:", 1)
```

Output:

```text
devices: ('q0', 'q1', 'r')
couplings: ('q0-r', 'q1-r', 'qq-zz')
component targets: 8
additional constraints: 1
```

The component policies select eight bare parameters automatically. Nine
constraints therefore make this an overdetermined fit: it minimizes their
squared relative residuals rather than forcing every row to zero independently.

```python
fit = fit_a_dress(
    desired_chip,
    constraints=constraints,
    max_nfev=300,
)

print(fit.summary())

evaluations = np.arange(len(fit.history))
best_loss = np.minimum.accumulate(fit.history)
figure, axis = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
axis.semilogy(evaluations, fit.history, ".", color="0.72", markersize=2.5, label="evaluated loss")
axis.semilogy(evaluations, best_loss, color="#C92F33", linewidth=2.0, label="best so far")
axis.set(xlabel="Distinct residual evaluation", ylabel="Normalized loss")
axis.grid(alpha=0.2, which="both")
axis.legend(frameon=False)
figure_path = "../images/fit_a_dress_convergence.png"
figure.savefig(figure_path, dpi=180)
plt.show()

print(f"\nnormalized loss: {fit.history[0]:.3e} -> {fit.loss:.3e}")
print("desired chip unchanged:", desired_chip.parameters["q0.freq"] == 5.047559899)
```

Output:

```text
fit_a_dress: converged | loss 4.62e-17 | targets: 9 | parameters: 8
identifiability: rank 8/8 | condition 1.39e+05
targets (GHz):
  q0.freq [component default]: 5.04756 -> 5.04756 (error -8.7e-09)
  q0.anharmonicity [component default]: -0.298378 -> -0.298378 (error +2.1e-10)
  q1.freq [component default]: 5.31807 -> 5.31807 (error -1.3e-08)
  q1.anharmonicity [component default]: -0.279651 -> -0.279651 (error +1.9e-10)
  r.freq [component default]: 7.10316 -> 7.10316 (error +4e-08)
  q0-r.cross_kerr [component default]: -0.000533546 -> -0.000533546 (error -4.3e-13)
  q1-r.cross_kerr [component default]: -0.000464088 -> -0.000464088 (error -3.6e-13)
  qq-zz.cross_kerr [component default]: 0.00164131 -> 0.00164131 (error -1.4e-14)
  q0 <-> q1.exchange_rate [explicit]: -0.00216393 -> -0.00216393 (error +3.5e-12)
bare parameters (GHz):
  q0.freq: 5.04756 -> 5.05 [component declaration]
  q0.anharmonicity: -0.298378 -> -0.3 [component declaration]
  q1.freq: 5.31807 -> 5.32 [component declaration]
  q1.anharmonicity: -0.279651 -> -0.28 [component declaration]
  r.freq: 7.10316 -> 7.1 [component declaration]
  q0-r.g: 0.065046 -> 0.065 [isolated-pair root solve; positive convention]
  q1-r.g: 0.0548762 -> 0.055 [isolated-pair root solve; positive convention]
  qq-zz.chi: 0.00164131 -> 0.000800001 [target value; target sign]

normalized loss: 2.595e-01 -> 4.616e-17
desired chip unchanged: True
```

```{figure} ../images/fit_a_dress_convergence.png
:width: 680px
:alt: Raw and best-so-far normalized loss during a multi-observable fit

Finite-difference probes appear in the raw history. The red curve keeps the
best loss reached as the optimizer moves through the eight bare parameters.
```

`fit_a_dress()` leaves `desired_chip` alone and returns the fitted clone as
`fit.chip`. It estimates bare coupling seeds on isolated pairs; those
choices appear directly in `fit.summary()`, while the declared dressed
constraints remain unchanged. The summary also shows the final Jacobian rank
and condition number. Use `fit.final_targets`, `fit.parameter_reports`, and
`fit.solver_info` when code needs the same receipt as structured data.
Automatic plans raise instead of returning a rank-deficient bare chip. If you
deliberately choose an ambiguous `vary=` set, the fit returns with a warning
and records the weak parameter combinations in `fit.solver_info`.

## Start from a scqubits model

Install the optional interoperability dependency when an existing model is
already expressed in scqubits:

```bash
pip install "quchip[scqubits]"
```

`from_scqubits()` converts supported devices and Hilbert spaces into quchip
declarations. A transmon import copies the physical circuit parameters and
rebuilds the model; it does not freeze a table of eigenvalues.

```python
import numpy as np
import scqubits as scq

from quchip import Chip, from_scqubits

source = scq.Transmon(
    EJ=30.0,
    EC=0.2,
    ng=0.0,
    ncut=31,
    truncated_dim=4,
    id_str="q_scq",
)
q_scq = from_scqubits(source)
imported = Chip([q_scq])

source_levels = source.eigenvals(evals_count=2)
source_f01 = source_levels[1] - source_levels[0]
record = imported.resolve().bases["q_scq"]

print("device:", type(q_scq).__name__)
print("label:", q_scq.label)
print(f"scqubits f01: {source_f01:.9f}")
print(f"quchip f01:   {float(imported.freq(q_scq)):.9f}")
print("basis kind:", record.kind)
print("native dimension:", record.native_dim)
print("resolved dimension:", record.resolved_dim)
print("V shape:", record.vectors.shape)
print("projector shape:", record.projector.shape)
print(
    "isometry:",
    np.allclose(
        np.asarray(record.vectors.conj().T @ record.vectors),
        np.eye(record.resolved_dim),
    ),
)
```

Output:

```text
device: ChargeBasisTransmon
label: q_scq
scqubits f01: 6.721939754
quchip f01:   6.721939754
basis kind: eigen
native dimension: 63
resolved dimension: 4
V shape: (63, 4)
projector shape: (63, 63)
isometry: True
```

`to_scqubits()` handles the supported reverse mappings. Conversion preserves
represented physics; inspect the result after importing a composite model,
especially its labels, truncations, and interactions.

Here `V` contains the four retained eigenvectors of the 63-dimensional charge
basis. The isometry check verifies $V^\dagger V=I_4$; the projector $VV^\dagger$
selects that four-dimensional subspace in the authored charge basis.

## Add a device model

If the supplied devices do not contain the local physics you need, declare the
parameters and local Hamiltonian. quchip supplies the constructor, validation,
serialization, symbolic parameter binding, and JAX pytree behavior.

```python
from typing import Any

from quchip import Chip, FockDevice, LocalOps, PhysicsExpr, Scalar, parameter


class LinearMode(FockDevice):
    _default_levels = 4

    freq: Scalar = parameter(
        positive=True,
        unit="GHz",
        symbol=r"\omega",
    )

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        return p.freq * op.n


mode = LinearMode(freq=6.2, label="m")
custom_chip = Chip([mode])

print(custom_chip.unresolved_hamiltonian().latex())
```

Output:

```text
\omega_{m}\,\hat n_{m}
```

The same ownership rule selects the other extension surfaces:

| Physics you own | Public surface |
|---|---|
| Local static physics | `DeviceModel` or `FockDevice` |
| Interaction between devices | `CouplingModel` |
| Device or coupling control | `DeviceDrive` or `CouplingDrive` |
| Scheduled waveform | `Envelope` |
| Classical hardware effect | `SignalTransform` |
| Loss or a shared environment | `CollapseChannel` or `Bath` |
| A nonstandard local Hilbert space | `LocalSpace` |
| Third-party conversion | `ModelMapping` |

Extensions return the same symbolic physics objects used by the shipped
components. They should not import backend internals or branch on a backend.
The {doc}`extension guide <../extensions>` gives a runnable example and the
checks appropriate to each surface.

Once the model says what you intend, continue with {doc}`statics and parameter
studies <statics-and-parameter-studies>`.
