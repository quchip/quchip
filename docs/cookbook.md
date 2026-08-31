# Cookbook

This page defines how quchip examples should read and the defaults users can
expect them to follow. Each example starts with the smallest runnable answer,
shows its output, then adds one physical or numerical idea at a time.

## The example contract

A quchip example should:

1. state the physical question;
2. run through public APIs as written;
3. show the result directly below the code that produced it;
4. make units, frames, approximations, and truncations visible when they affect
   the answer;
5. preserve the source declaration when sweeping, fitting, or transforming;
6. finish with a numerical or physical check tied to the reported observable.

Keep the first calculation small. A reader should get one useful result before
meeting batching, model reduction, custom extensions, or differentiation.

## Declare the physics first

Construct devices, couplings, and the chip before adding controls or analysis.
Import supported classes from `quchip`; use submodule imports only for a public
optional backend or extension surface.

```python
from quchip import Capacitive, Chip, DuffingTransmon, Resonator

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=5, label="r")
chip = Chip([q, r], [Capacitive(q, r, g=0.05, label="qr")])
```

Use object references while constructing and scheduling. Give every component
a short stable label; labels become parameter paths such as `q.freq` and
`qr.g`.

quchip uses ordinary GHz for frequencies and couplings, ns for time, mK for
temperature, and `1/ns` for Lindblad rates. Put units in prose, output labels,
and plot axes rather than encoding them in variable names alone.

## Make resolution choices inspectable

Defaults are fine when they do not affect the lesson. Set the frame and
approximation explicitly before interpreting rotating terms, dropped bands, or
solver dynamics.

Use the inspection surfaces in this order:

```python
description = chip.describe()
authored_latex = chip.unresolved_hamiltonian().latex()
canonical_latex = chip.hamiltonian().latex()

resolved = chip.resolve(frame="rotating")
frame = resolved.resolved_frame
approximation = resolved.approximation
dropped_terms = resolved.dropped_terms_summary()
```

- `describe()` shows the declared components, units, frame, approximation, and
  Hilbert-space size.
- `unresolved_hamiltonian()` preserves the authored device and coupling
  expressions.
- `hamiltonian()` shows the canonical expression after basis, frame,
  truncation, and approximation policies.
- `resolve()` carries the basis transforms, frame frequencies, approximation,
  dropped terms, and collapse terms used by the backend.

Call `.matrix()` only when the numerical array is itself needed. A named
operator such as `H_0` in resolved LaTeX is an opaque numerical leaf, not
missing physics; read the authored expression and the resolution record beside
it.

## Distinguish bare declarations from dressed observables

Device constructor values describe local models. Ask the `Chip` for quantities
of the coupled system:

```python
bare_frequency = q.freq
dressed_frequency = chip.freq(q)
conditional_resonator_frequency = chip.freq(r, when={q: 1})
```

Use dressed transitions for resonant carriers unless the example intentionally
studies a bare detuning. `chip.freq()`, `transition_frequency()`,
`dressed_anharmonicity()`, `dispersive_shift()`, and `static_zz()` dress the
chip when needed; do not call `dress()` first merely as setup.

Use `kerr_matrix()` when the question involves several self-Kerr and
cross-Kerr coefficients:

```python
kerr = chip.kerr_matrix()
kerr.labels
kerr[q, r]
```

The axes follow `chip.devices`. Diagonal entries are
$E(2_i)-2E(1_i)+E(0)$; off-diagonal entries are
$E(1_i,1_j)-E(1_i)-E(1_j)+E(0)$. Both are dressed quantities in ordinary
GHz. They need not equal similarly named component parameters because every
device and interaction in the chip contributes to the dressed energies. A
device with fewer than three resolved levels has `NaN` on its diagonal. Under
the `KerrCavity` convention $H=\omega n-Kn(n-1)$, the diagonal is $-2K$.

Prepare states with `chip.state()` by default. Use `chip.bare_state()` only
when the question concerns a bare product state, and say why.

## Choose the smallest public surface

| Question | Start with |
|---|---|
| One dressed observable | `chip.freq()` or another `Chip` analysis method |
| A few changed parameters | `chip.with_params()` |
| A spectrum over a grid | `SpectrumSweep` and `Sweep` |
| One pulse experiment | `QuantumSequence.simulate()` |
| Related pulse experiments | `QuantumSequence.simulate_batch()` |
| A smaller effective model | `eliminate()` or `active_patch()` |
| Independent connected components | `chip.partition()` |
| Bare parameters from dressed targets | `fit_a_dress()` |
| A scalar derivative | `jax.grad()` |
| A residual or trace derivative | `jax.jacrev()` or `jax.jacfwd()` |

Static examples should not create a `QuantumSequence`. Dynamic examples should
not replace a declared device or coupling with a hand-written effective matrix
unless the comparison itself is the subject.

## Rebind instead of mutating

`with_params()` returns a new chip or sequence. Use it for sweeps, local design
changes, and differentiable parameter maps:

```python
shifted = chip.with_params({"q.freq": 5.1, "qr.g": 0.045})
```

Keep the original object and show that its parameters did not move when this
matters to the example. Use `clone()` for a structural copy and
`to_dict()`/`from_dict()` for a declared-model round trip.

## Sweep calibrated controls

For `FluxTunableTransmon`, `freq` is the calibrated local frequency at
`flux_bias`. Together they anchor the SQUID dispersion. Rebinding only
`flux_bias` preserves that calibration and retunes the Hamiltonian, so sweep
the physical device coordinate directly:

```python
import numpy as np

from quchip import SpectrumSweep, Sweep

flux_values = np.linspace(0.0, 0.3, 101)
sweep = SpectrumSweep(
    chip,
    [Sweep(flux_values, name="coupler.flux_bias")],
).run(progress=False)
```

Set `freq` and `flux_bias` in the same `with_params()` call when defining a new
calibration anchor. Use `frequency_at()` to inspect another bias without
changing the chip and `flux_for_frequency()` for the inverse question on the
monotonic lobe. Convert laboratory voltage or pulse amplitude to reduced flux
before passing it to the device.

## Fit bare parameters from dressed targets

`fit_a_dress(desired)` treats component numbers as dressed constraints and
returns the corresponding bare chip as `fit.chip`. Common devices target their
dressed frequency and anharmonicity. A `Capacitive` or `CrossKerr` scalar
targets the full cross-Kerr $E_{11}-E_{10}-E_{01}+E_{00}$; it is not the
starting bare coupling strength in this call. Add pair observables with
`constraints=` and use `vary=` only when the component defaults are not the
parameters you want to move:

```python
fit = fit_a_dress(
    desired,
    constraints={(q0, q1): {"exchange_rate": -0.0022}},
)
```

Start with `print(fit.summary())`. It reports each target and its source, each
bare parameter and starting-point choice, the final loss, and the scaled
Jacobian rank and condition number. Use `fit.final_targets`,
`fit.parameter_reports`, and `fit.solver_info` for programmatic inspection.
`fit.history` records the loss at distinct residual evaluations; plot
`numpy.minimum.accumulate(fit.history)` for a monotone best-so-far curve because
numerical-Jacobian probes also appear in the raw history.

The automatic plan raises when it has too few targets or a rank-deficient
final Jacobian. An explicit `vary=` plan may be intentionally ambiguous; in
that case the fit returns with a warning and records the weak parameter
directions in `fit.solver_info`.

## Build pulse schedules from physical controls

Wire a public drive to its target, schedule an envelope, and keep the returned
pulse handle when a later batch varies that pulse.

```python
from quchip import ChargeDrive, Gaussian, QuantumSequence

line = ChargeDrive(q, label="xy")
chip.wire(line)

sequence = QuantumSequence(chip)
pulse = sequence.schedule(
    line,
    envelope=Gaussian(duration=20.0, sigmas=3.0, amplitude=0.04),
    freq=chip.freq(q),
)
```

Use channel cursors, `delay()`, and `barrier()` for serial timing. Supply
`start_time` only for intentional overlap. Put global phase on
`schedule(..., phase=...)`; keep envelope parameters responsible for waveform
shape.

For related experiments, vary the pulse handle or `initial_state` and call
`simulate_batch()`. Zip duration and amplitude when calibration requires them
to move together. Avoid a manual Python loop of independent solves when the
batch API represents the same study.

## Let the declared model choose the equation

QuTiP is the default backend. With no explicit solver, quchip uses `sesolve`
for a closed model and `mesolve` when devices, drives, couplings, or baths
contribute collapse channels. Adding a resonator quality factor or a bath can
therefore change the equation without changing the pulse schedule.

For a time-independent QuTiP problem with no explicit solver method, quchip
uses diagonal propagation at all requested save times when the total Hilbert
dimension is at most 64 for `sesolve`, or at most 12 for `mesolve` (whose
Liouvillian dimension is the square of the Hilbert dimension). `diag` does not
use adaptive tolerances or step controls, so quchip removes `atol`, `rtol`,
`nsteps`, and `max_step` and logs the discarded options at `INFO` level. Driven
problems, larger spaces, and an explicit non-`diag` method keep QuTiP's selected
adaptive integrator. For dynamiqs, method selection remains explicit through
`options={"method": ...}`; its default is `Tsit5`.

See {doc}`Backend and solver options <guides/choosing-a-backend>` for available
methods, option syntax, batching, and gradient controls.

## Read results at the level of the question

Use:

- `population()` for a local occupation;
- `expect()` for a declared observable trace;
- `overlap()` for a target state;
- `reduced_state()` for one subsystem;
- `check_truncation()` before trusting a small local basis.

Pass `e_ops=chip.e_ops(...)` when expectation traces are the main output.
Stored states remain available by default; disable them only when the memory
tradeoff is intentional.

Do not narrate a calculation with a series of `print()` calls. In a notebook,
display a compact dict or result object. In a Markdown snippet, keep the print
if it helps copy-paste use and place the captured output immediately below it.

```python
summary = {
    "dressed_f01_ghz": float(chip.freq(q)),
    "chi_ghz": float(chip.dispersive_shift(q, r) / 2),
}
summary
```

Long examples may finish with a machine-readable receipt: a compact dict or
JSON record containing the parameters, outputs, checks, solver, and generated
figure paths from that run. A receipt records evidence; it does not replace the
reader-facing result shown earlier.

## Check the reported quantity

Choose checks from the physics and numerics of the example:

- add local levels and compare the reported transition, population, or shift;
- refine a time or sweep grid;
- compare an automatic derivative with several central-difference steps;
- read RWA dropped terms and compare their band amplitude with their frame
  frequency;
- read transformation validity before comparing reduced and full dynamics;
- inspect dressed-state assignment overlaps near hybridization.

Do not set a tolerance from the residual produced by the run being tested.
Derive it from solver accuracy, truncation convergence, an approximation
parameter such as `g / detuning`, or a stated physical estimate.

## Keep plots tied to observables

Make the smallest figure that answers the question. Label axes with units and
state what was traced out or conditioned on. Plot the scheduled envelope when
pulse shape explains the result. Use equal aspect ratio for an IQ plane and a
log scale only when the orders of magnitude matter.

Use `chip.plot_graph()` for connectivity. Its default labels are the declared
bare frequencies and coupling strengths, so drawing the graph stays cheap and
does not hide a diagonalization. Ask for `values="dressed"` when the dressed
transition frequencies and full-pull cross-Kerr values are the point of the
figure, or `values="both"` when comparing the model inputs with its dressed
observables. The topology itself does not change between these views.

Every committed figure must come from the code shown in the example. Record
its path in the final receipt so a rerun overwrites the documented artifact.

The paired Markdown guide must also contain the captured textual outputs from
its executed notebook. Run `python tools/sync_example_outputs.py` after
execution, and use `--check` in verification. Jupytext source parity alone is
not enough because plain Markdown does not preserve notebook outputs.

## Handle optional models and extensions explicitly

State the required extra before the first optional import, for example
`quchip[dynamiqs]` or `quchip[scqubits]`. After importing from another library,
inspect labels, truncations, basis projections, and interactions before using
the converted chip.

Write an extension only when shipped components cannot express the local
physics. Choose the public extension surface by ownership: device, coupling,
drive, envelope, signal transform, dissipation channel, local space, or model
mapping. Extension code returns symbolic quchip physics and must not branch on
a backend.

## Use specific names

Avoid `recipe` as a generic name for an example. Say what the object is:
example, procedure, pulse schedule, parameter study, fit, reduction, or bath
model.

`Bath.recipe` is the one established API use. It selects a built-in
collapse-channel model: `"thermal"`, `"collective_decay"`, or
`"correlated_dephasing"`. In prose and inspection output, call this the bath
model; retain `recipe` only when naming the constructor argument, attribute, or
serialized field.

## Examples that follow these conventions

- {doc}`Define and inspect a chip <guides/defining-and-inspecting-a-chip>`
- {doc}`Statics and parameter studies <guides/statics-and-parameter-studies>`
- {doc}`Dynamics, pulses, observables, and readout <guides/dynamics-pulses-and-readout>`
- {doc}`Chip transformations <guides/chip-transformations>`
- {doc}`Differentiability <guides/differentiability>`
