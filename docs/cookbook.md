# Cookbook

Choices for building models, running calculations and interpreting results.
Frequencies are in GHz, times in ns, temperatures in mK and decay rates in
1/ns.

## Choose the calculation

| You need | Use |
|---|---|
| Dressed frequencies or interactions | `chip.freq()`, `dispersive_shift()`, `static_zz()` |
| Counter-rotating statics or pulses of a large chip | `chip.with_space(ExcitationCutoff(max_total=K))` |
| A spectrum over a parameter grid | `SpectrumSweep` with `Sweep` |
| A pulse or an idle interval | `QuantumSequence.simulate()` |
| Related pulse experiments | `sequence.simulate_batch()` |
| A gate on dressed computational states | `sequence.subspace_propagator()` |
| A gate with T1, T2 and other collapse channels | `sequence.subspace_channel()` |
| Steady-state populations or fields | `chip.steadystate()` |
| Small-signal S parameters | `VNA.sweep()` |
| The frequency and linewidth of each resonance | `VNA(chip).resonances()` |
| A driven response with receiver noise | `VNA.measure()` |
| Outcomes from a saved quantum state | `result.measure()` |
| Bare parameters that meet dressed targets | `fit_a_dress()` |
| A reduced model around driven devices | `sequence.active_patch()` |
| A reduced model of chosen devices | `eliminate_spectators(chip, keep)` |
| Diagonalize effective terms, such as a junction cosine | `eliminate(chip, label, method="exact")` |

The {doc}`guides <guides/index>` develop these workflows. The
{doc}`focused studies <studies/index>` apply them to specific physical questions.

## Keep model inputs and observables distinct

```python
from quchip import Capacitive, Chip, DuffingTransmon, Resonator

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
r = Resonator(freq=7.0, levels=4, label="r")
chip = Chip([q, r], [Capacitive(q, r, g=0.05, label="qr")])

bare = q.freq
dressed = chip.freq(q)
readout_if_excited = chip.freq(r, when={q: 1})
```

Use device objects while building the model. Use their labels when you get
results or binding parameters. `chip.freq()` dresses automatically, so a
preceding `dress()` call is not needed.

Dressed queries and default solves share `chip.approximation`. The default
`RWA()` drops counter-rotating bands in both. Construct the chip with
`approximation=Exact()` to keep them. Dressed queries always use the lab
frame. Pulse solves on a chip without a frame, such as this one, use the
rotating frame and warn with that choice. Pass `frame=` to `Chip()` to choose
a frame. If a solve overrides the approximation, use the same choice for its
carrier and state preparation. `chip.resolve(frame="lab", approximation=...)`
gives a snapshot whose `dress()` returns the corresponding eigensystem.

`dispersive_shift(q, r)` returns the full conditional resonator pull. Divide by
negative two for the coefficient of `Z=|0><0|-|1><1|` times resonator number. `kerr_matrix()` collects full pulls
off diagonal and dressed anharmonicities on the diagonal.

Inspect `unresolved_hamiltonian()` for the declared expression and
`hamiltonian()` for the expression after basis, frame and approximation
choices. `chip.resolve().dropped_terms_summary()` lists discarded terms.
`Exact()` keeps the declared Hamiltonian terms. Finite local spaces and
component approximations still apply.

## Fit a large conserving model

`fit_a_dress()` uses the same excitation sectors as labeled dressed queries.
For 24 devices with at least two levels each, ground-conditioned frequencies
need the vacuum and a 24-state block. With at least three levels per device,
anharmonicities and pair shifts also need the 300-state two-excitation block.
Smaller local cutoffs reduce that count.

The engine must prove conservation from the retained operators. Default `RWA()`
provides this for ordinary interactions. An explicitly conserving model can
qualify under `Exact()` too. Unknown or nonconserving operators use the full
space. No excitation cutoff is inserted.

`max_hilbert_dim` limits the largest matrix used by each fit target. The limit
is checked again after rebinding or tracing. Traced parameters of a local
Hamiltonian that is not diagonal by construction, such as a
`ChargeBasisTransmon`, give a traced energy basis. Conservation of an
undeclared operator on that device is then unknown. Keep a sufficient limit
for the full fallback, or supply a valid structural operator declaration.
`evaluator="full"` retains the complete device model even when it uses sectors.
`evaluator="local"` remains an explicit neighborhood approximation.

`drive_matrix_elements()` also uses the sectors of its two selected states.
Eligible deterministic pulses also use exact invariant excitation spaces.

## Probe a static model with counter-rotating terms

`VNA(chip).sweep(frequencies)` supports pump-free models that are static in
the lab frame, including capacitive couplings under `Exact()`. It solves one
stationary state per chip parameter point, then evaluates the weak response
at each probe frequency. The declared loss channels and approximation stay
unchanged. Counter-rotating terms move the stationary state out of the vacuum,
so this sweep uses the full Liouvillian, of size D² × D² for Hilbert-space
dimension D. Fixed pumps, `measure()` and `finite_power()` still require a tone
frame in which the model is stationary.

## Compare conditional readout responses

For a chip with a port network, use a dressed reference label:

```python
from quchip import VNA

vna = VNA(readout_chip)
ground = vna.sweep(frequencies, when={"q": 0})
excited = vna.sweep(frequencies, when={"q": 1})
```

Unspecified devices have level zero. This weak-probe calculation supports zero
or one total excitation, static conserving models and vacuum inputs.
It retains internal loss, port loss and the transmon's EF transition when the
local cutoff includes it. Use enough resonator levels to retain two photons.

An excited reference stays fixed in the probe source. Its response includes
loss and coherence recycling, but it does not model population evolution during
an acquisition. Inspect `diagnostics` for `reference_escape_rate` and
`reference_stationarity_residual`, both in `1/ns`. They must be small on the
response timescale. They do not provide a fit-error bound.

This route rejects pumps, thermal inputs and stationary solver options.
Use scheduled coherent inputs when the preparation, delay or acquisition
window matters. `VNA.measure()` continues to describe stationary measurements.

## Read the resonances of a readout

The poles of the weak-probe response give each resonance without a fit:

```python
from quchip import VNA

resonances = VNA(readout_chip).resonances()
qubit = resonances["q"]
frequency, linewidth = qubit.frequency, qubit.linewidth  # GHz and 1/ns
rebuilt = resonances.scattering(frequencies)
```

A label selects the pole whose composition that mode dominates; an index also
works. The linewidth `κ = −2 Re λ` is the full response width in angular
units. It includes port loss, internal loss, Purcell decay and dephasing.
`rebuilt` equals `VNA(readout_chip).sweep(frequencies).matrix`, reference-plane
transfers included. `resonances.dark` marks poles whose residues are
negligible at the selected ports. The call covers the weak-probe routes of
`VNA.sweep()` and raises for pumps. Inside `jax.jit`, select poles by index. Frequencies and
linewidths have JAX derivatives; the composition and the residues do not.

## Change parameters without rebuilding the example

```python
shifted = chip.with_params({"q.freq": 5.1, "qr.g": 0.045})
noisy = chip.with_params({"q.T1": 30_000.0, "q.T2": 40_000.0})
```

`with_params()` returns a new model. `chip.parameters` lists the available
paths, including optional noise fields set to `None`. Change jointly
constrained values, such as pulse duration and edge length, in the same call.

Activate optional noise before JAX tracing, then vary its numerical values.
`thermal_occupation` is the bath's mean occupation, not the qubit's initial
excited-state probability. `set_noise()` replaces the complete noise
configuration. Use `with_params()` for a partial change.

For `FluxTunableTransmon`, sweep `flux_bias` to move along its calibrated
frequency curve. Set `freq` and `flux_bias` together to change the calibration
anchor. Its charge operator also follows `flux_bias`, so capacitive `g` and
charge-drive amplitudes refer to zero flux, not to the anchor. `to_dict()` and
`from_dict()` save and restore model declarations.

## Keep the pulse handle for sweeps

```python
from quchip import ChargeDrive, Gaussian, QuantumSequence

xy = ChargeDrive(q, label="xy")
chip.wire(xy)
sequence = QuantumSequence(chip)
pulse = sequence.schedule(
    xy, envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3.0),
    freq=chip.freq(q),
)
batch = sequence.simulate_batch(
    sequence.zip(
        pulse.vary("duration", [20.0, 40.0]),
        pulse.vary("amplitude", [0.02, 0.01]),
    ),
    duration=60.0,
)
```

Separate sweep axes form a Cartesian grid. `zip()` pairs values. Here the two
pulses have equal nominal area. Use `delay()` and `barrier()` for serial timing,
and an explicit `start_time` for overlap.

A scheduled carrier stays fixed when device parameters change. Use the pulse's
`freq` parameter to retune it. For independently built experiments, pass their
`build_problem()` results to `solve_many()`.

## Read a gate on dressed states

```python
shape = Gaussian(duration=20.0, sigmas=3.0)
amplitude = chip.rotation_amplitude(q, xy, shape)
gate = QuantumSequence(chip)
gate.schedule(xy, envelope=Gaussian(duration=20.0, sigmas=3.0, amplitude=amplitude), freq=chip.freq(q))
propagator = gate.subspace_propagator(q)
fidelity = propagator.fidelity([[0, 1], [-1, 0]])
```

`rotation_amplitude()` solves `θ = 2π |m| |A a|` for the dressed matrix
element `m` of the line and the envelope area `A a`, with `θ = π` by default.
On a line with gains, delays or crosstalk, `m` is the sum of each reached
line's element times its transfer at the transition frequency.
On an isolated transmon, the charge drive has `m = -i` on the 0 → 1
transition. The dressed element here is `-0.99969i`, so a real π pulse aims
at `iY`.
`subspace_propagator(q)` starts the dressed states of `q` in levels 0 and 1 as
one batch. The matrix has the final label in rows and the initial label in
columns, in the frame of the dressed frequencies. `propagator.leakage` gives
the probability that each input leaves the labels, and `fidelity()` counts it.
Here the fidelity is 0.993. Off-resonant coupling to the third level gives an
AC Stark shift and a coherent rotation error, while the final leakage is only
about 2e-6. The amplitude is a first guess for a calibration. `frame="lab"` and `frame="integration"` select the other frames.
On a chip with collapse channels, the solves leave them out, and
`propagator.omitted_channels` names them.

```python
noisy = gate.with_params({"q.T1": 50e3, "q.T2": 40e3})
channel = noisy.subspace_channel(q)
infidelity = channel.infidelity([[0, 1], [-1, 0]])
rates = channel.error([[0, 1], [-1, 0]]).pauli_rates()
```

`subspace_channel()` keeps the collapse channels. For `d` labels it solves
`d²` pure states and returns a superoperator on the same labels and frames.
`infidelity()` is the error in quchip's frame, with ideal preparation and
measurement, and it counts leakage. `eigenvalue_infidelity()` is a spectral
score that a small frame rotation of the gate does not change; give
`average_leakage()` beside it. `error(target)` is the map after the ideal
gate, and `pauli_rates()` gives the Pauli weights of its twirl. Here the
infidelity is 7.0e-3, mostly the coherent error of the first-guess pulse,
which tilts its rotation axis (`rates["X"]` is 1.0e-2). The eigenvalue
infidelity is 2.8e-4: the dominant coherent part is a π pulse about a
tilted axis, which keeps its eigenvalues. Add `levels=(0, 1, 2)` to follow leakage and `seepage()`.

## Choose what to save

```python
result = sequence.simulate(duration=60.0, states="all")
population = result.population(q, level=1)
final = result.state_at(result.times[-1])
counts = result.measure(q).sample(1024, seed=7).counts()
```

`duration` starts at zero and can extend the pulse schedule. An explicit
`tlist` sets the actual start and end times: `[20, 30]` starts evolution at
20 ns. It does not simulate the first 20 ns and discard them.

Use `states="final"` when only the final state matters. Then
`result.population(q, level=1)` holds only the final value, and
`result.final_population(q, level=1)` returns it with either choice.
`states="none"` saves only requested expectation traces. Later state measurements require saved
states. `observable_at()` can select or interpolate a saved trace. State
queries use saved times and do not interpolate quantum states.

A level label names its dressed state everywhere: `chip.state(q=1)`,
`initial_state={"q": 1}` or `"e"`, the targets of `overlap()` and
`amplitude_array()`, `chip.superposition()` and `chip.compact_state()`. The
dressed state is the eigenstate of the undriven static lab-frame Hamiltonian
that the solve's approximation keeps, assigned to the label. A label
`initial_state` is placed in the solve frame at `tlist[0]`, and an omitted
`initial_state` is the all-ground label, so it equals `{}`. A chip at rest
then stays in its label, with no exchange between coupled devices. Use
`chip.bare_state(q=1)` when the bare product state is the point, for example to
show that exchange. `result.population(q, level=1)` measures a local isolated
energy-state population, which on a coupled chip differs from
`overlap({"q": 1})`.

## Reuse an exact compact pulse state

Exact reduction requires no approximation option. A solve uses it when its
initial state, Hamiltonian, controls, and dissipation close within a smaller
space. A transverse drive or thermal raising can require the full space.
The solver never substitutes an excitation cutoff.

```python
initial = chip.compact_state({"q": 1})
# The label's dressed state. Use dressed=False for its bare product.
idle = QuantumSequence(chip)
result = idle.simulate(duration=60.0, initial_state=initial)
occupation = result.population("q", level=1)
saved = result.compact_state()
```

The factory also accepts weighted superpositions, such as
`chip.compact_state((1, {}), (1j, {"q": 1}))`. It normalizes the sum. With
`dressed=False`, a device can also take a local ket, so a bare product needs no
product-space vector on a large chip.
Pass `saved` as the next solve's `initial_state`, with the same captured local
bases. Start the next time grid at the previous final time to continue
absolute-time controls.

Inspect `result.stats["excitation_reduction"]` for dimensions and the proof
or fallback reason. Local queries keep the state compact. Existing native
state exports expand only when requested and check memory first. A full
population dictionary also requires product-space storage. Use `population`
for individual device occupations on large models.

## Truncate the total excitation

Under `Exact()`, counter-rotating pair terms connect every excitation sector.
Labeled queries and pulse solves of a large chip then need the full product
space. `ExcitationCutoff(max_total=K)` keeps the product labels whose levels sum
to at most `K`, and it keeps every matrix element between them. It is an
explicit approximation of the model, separate from `RWA()` and `Exact()`.

```python
from quchip import Capacitive, Chip, DuffingTransmon, Exact, ExcitationCutoff, Resonator

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=6, label="r")
chip = Chip([q, r], [Capacitive(q, r, g=0.1)], approximation=Exact())
for max_total in (2, 3, 4, 5):
    limited = chip.with_space(ExcitationCutoff(max_total=max_total))
    print(max_total, limited.freq(q), limited.dispersive_shift(q, r))
```

The dispersive shift is -1.109, +2.261, -1.136 and -1.136 MHz for these
cutoffs, and -1.137 MHz without a cutoff. At `max_total=3`, the label
`{q: 1, r: 1}` loses its counter-rotating partners at `N = 4`, while the vacuum
keeps its partners at `N = 2`. Report a
quantity only after it is stable at increasing `max_total`. Do this for
frequencies, populations, phases and gradients, since their convergence need
not be monotone. Check the local level truncation separately, with more
`levels` at a fixed `max_total`.

A label above the cutoff raises, and so does a supplied state with more than
`1e-20` of its weight above the cutoff. quchip treats a smaller residue as
basis-conversion round-off and drops it without renormalization. Pulse solves
evolve the truncated model. Their default initial state is the truncated
model's vacuum-assigned eigenvector. `result.check_truncation()[ExcitationCutoff(K)]`
gives the maximum sampled population in the sectors that one step can move
above the cutoff. This value is a warning heuristic, not an error bound.

`steadystate()`, `VNA.finite_power()`, `VNA.measure()`,
`VNA.output_spectrum()`, `VNA.g1()` and `VNA.g2()` solve the truncated model.
`chip.dress()`, `VNA.sweep()`, stochastic trajectory solvers, `eliminate()`,
`eliminate_spectators()`, `active_patch()`, `fit_a_dress(evaluator="local")`
and `chip.partition()` raise for a limited chip. Use `chip.with_space(None)`
for them.

Without a cutoff, the VNA workflows above truncate their own solve of a chip
that conserves the total excitation and has more than 64 product states. They
start at `K = 2`, compare every grid point with `K + 1` until they agree,
and warn which cutoff they kept. Set the cutoff on the chip to choose `K`, or pass `options={}` to
solve the full product space. An undriven `steadystate()` of such a chip
returns the vacuum when the lowering channels prove it unique.
`result.diagnostics["route"]` names the route that each call took and why.

## Declare network noise

```python
from quchip import PortNetwork

network = PortNetwork()
attenuator = network.attenuator("cold", loss_db=20, thermal_occupation=0.01)
twpa = network.amplifier("twpa", gain_db=20, added_noise=0.5)
```

`thermal_occupation` is the passive load's mean thermal population in quanta.
If you omit it, the load is in vacuum. `added_noise` is the required
input-referred symmetrized noise in quanta, at least `(1 - 1/G) / 2` for power
gain `G`. Both are constant across frequency sweeps. The TWPA model describes
linear phase-preserving gain.

For existing network declarations, rename `occupation` or `loss_occupation`
to `thermal_occupation`. Temperature, noise figure, and `noise_frequency`
arguments are removed. Supply noise quanta directly, including in saved
component parameters.

## Reuse a measurement

With the model and ports from the {doc}`fridge guide <guides/steady-state-and-vna>`:

```python
from quchip import IQReceiver, VNA

measurement = VNA(readout_chip).measure(
    frequencies, amplitudes=20.0, input=drive, outputs=[readout],
)
sample = measurement.sample(1, receiver=IQReceiver(integration_time=1_000_000), seed=7)
quieter = measurement.sample(1, receiver=IQReceiver(integration_time=100_000_000), seed=7)
```

Changing receiver integration time or shot count reuses the stored spectra.
`noise_spectrum(readout, unit="dBm/Hz")` reports noise power density.
`statistics(receiver=...).covariance(readout)` gives integrated IQ covariance.
Field amplitudes at network ports are in `sqrt(photons/ns)`.

For a saved quantum state, `result.measure(q1, q2)` keeps joint outcome
correlations. Add calibrated assignment errors or an `IQReadout` to model the
detector. A readout pulse is not needed, but you must supply conditional IQ
signals because qubit populations alone do not set them. Reuse a simulation's
wiring with `result.iq_readout()`, or use `IQReadout.from_wiring()` with
another wired model. Do not add apparatus noise twice to an existing
calibration.

## Check the quantity you report

| Common ambiguity | What to check |
|---|---|
| Field versus occupation | `abs(⟨a⟩)²` is the coherent part; `⟨a†a⟩` includes incoherent photons. |
| Jump rate versus output flux | `jump_rate()` returns `⟨L†L⟩`; dephasing and absorption jumps are not emitted photons. |
| A dressed label near an avoided crossing | Inspect `assignment_overlaps` or `state_components()`. Lowering the overlap threshold does not improve the assignment. |
| A small boundary population | Call `result.check_truncation()` explicitly, then increase local levels and compare the observable. The check covers only available samples. |
| A smooth trace | Refine the output grid separately from the solver tolerances. |
| An exact reduction near a crossing | Follow `validity["<mode>_labeling"]["margin"]` along the sweep. A failed check raises `EliminationError` with the label and its weight. |
| The origin of a mediated shift | `result.pathways({"q": 1})` lists the virtual states of an SW reduction. The amounts sum to the second-order part of the element. |
| A gate fidelity | `fidelity()` counts leakage. `omitted_channels` names the collapse channels that the propagator leaves out; `subspace_channel()` keeps them. |

QuTiP is the default. Choose dynamiqs for JAX gradients and compiled batches.
The initial state and declared dissipation select `sesolve` or `mesolve`
automatically. Use `dissipation=False` only for an intentional closed-system
comparison. See {doc}`Backends and solvers <guides/choosing-a-backend>` for
integration settings and {doc}`Gradients and parameter fitting <guides/differentiability>`
for checks on derivatives.

## Filter warnings

quchip warnings are `QuchipWarning`s in two groups. `AutoChoiceWarning` tells
you which frame, carrier, cutoff or fallback quchip chose for you. Pass the
setting explicitly to remove it. `AccuracyWarning` tells you that a result can
be inaccurate, for example because of a truncation, an approximation or a weak
fit. Renamed APIs keep `DeprecationWarning`.

```python
import warnings

from quchip import AutoChoiceWarning

warnings.filterwarnings("ignore", category=AutoChoiceWarning)
```

This filter removes the notices about automatic choices and keeps the accuracy
advice. Filter `QuchipWarning` to remove both groups.
