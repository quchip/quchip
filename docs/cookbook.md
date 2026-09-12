# Cookbook

Practical choices for building models, running calculations and interpreting
results. Frequencies are in GHz, times in ns, temperatures in mK and decay
rates in 1/ns.

## Choose the calculation

| You need | Use |
|---|---|
| Dressed frequencies or interactions | `chip.freq()`, `dispersive_shift()`, `static_zz()` |
| A spectrum over a parameter grid | `SpectrumSweep` with `Sweep` |
| A pulse or an idle interval | `QuantumSequence.simulate()` |
| Related pulse experiments | `sequence.simulate_batch()` |
| Steady-state populations or fields | `chip.steadystate()` |
| Small-signal S parameters | `VNA.sweep()` |
| A driven response with receiver noise | `VNA.measure()` |
| Outcomes from a saved quantum state | `result.measure()` |
| Bare parameters that meet dressed targets | `fit_a_dress()` |
| A reduced model around driven devices | `sequence.active_patch()` |

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

Use device objects while building the model; use their labels when retrieving
results or binding parameters. `chip.freq()` dresses automatically. You do not
need a preceding `dress()` call.

`dispersive_shift(q, r)` returns the full conditional resonator pull. Divide by
two for the usual sigma-z coefficient χ. `kerr_matrix()` collects full pulls
off diagonal and dressed anharmonicities on the diagonal.

Inspect `unresolved_hamiltonian()` for the declared expression and
`hamiltonian()` for the expression after basis, frame and approximation
choices. `chip.resolve().dropped_terms_summary()` lists discarded terms.
`Exact()` retains the declared Hamiltonian terms; finite local spaces and
component approximations still apply.

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
configuration; use `with_params()` for a partial change.

For `FluxTunableTransmon`, sweep `flux_bias` to move along its calibrated
frequency curve. Set `freq` and `flux_bias` together to change the calibration
anchor. `to_dict()` and `from_dict()` save and restore model declarations.

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

Separate sweep axes form a Cartesian grid; `zip()` pairs values. Here the two
pulses have equal nominal area. Use `delay()` and `barrier()` for serial timing,
and an explicit `start_time` for overlap.

A scheduled carrier stays fixed when device parameters change. Use the pulse's
`freq` parameter to retune it. For independently built experiments, pass their
`build_problem()` results to `solve_many()`.

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

Use `states="final"` when only the final state matters. `states="none"` saves
only requested expectation traces; later state measurements need saved states.
`observable_at()` can select or interpolate a saved trace. State queries use
saved times and do not interpolate quantum states.

Prepare coupled eigenstates with `chip.state()`. Use `chip.bare_state()` for
bare product states. `result.population(q, level=1)` measures a local isolated
energy-state population; `overlap()` tests a particular joint state.

## Declare network noise

```python
from quchip import PortNetwork

network = PortNetwork()
attenuator = network.attenuator("cold", loss_db=20, thermal_occupation=0.01)
twpa = network.amplifier("twpa", gain_db=20, added_noise=0.5)
```

`thermal_occupation` is the passive load's mean thermal population in quanta;
omitting it gives vacuum. `added_noise` is required, input-referred symmetrized
noise in quanta, at least `(1 - 1/G) / 2` for power gain `G`. Both are constant
across frequency sweeps. The TWPA model describes linear phase-preserving gain.

For existing network declarations, rename `occupation` or `loss_occupation`
to `thermal_occupation`. Temperature, noise figure, and `noise_frequency`
arguments have been removed; supply noise quanta directly, including in saved
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
`noise_spectrum(readout, unit="dBm/Hz")` reports noise power density;
`statistics(receiver=...).covariance(readout)` gives integrated IQ covariance.
Field amplitudes at network ports are in `sqrt(photons/ns)`.

For a saved quantum state, `result.measure(q1, q2)` retains joint outcome
correlations. Add calibrated assignment errors or an `IQReadout` to model the
detector. No readout pulse is required, but conditional IQ signals must be
supplied; qubit populations alone do not determine them. Reuse a simulation's
wiring with `result.iq_readout()`, or use `IQReadout.from_wiring()` with another
wired model. Do not add apparatus noise twice to an existing calibration.

## Check the quantity you report

| Common ambiguity | What to check |
|---|---|
| Field versus occupation | `abs(⟨a⟩)²` is the coherent part; `⟨a†a⟩` includes incoherent photons. |
| Jump rate versus output flux | `jump_rate()` returns `⟨L†L⟩`; dephasing and absorption jumps are not emitted photons. |
| A dressed label near an avoided crossing | Inspect `assignment_overlaps` or `state_components()`. Lowering the overlap threshold does not improve the assignment. |
| A small boundary population | Call `result.check_truncation()` explicitly, then increase local levels and compare the observable. The check covers only available samples. |
| A smooth trace | Refine the output grid separately from the solver tolerances. |

QuTiP is the default. Choose dynamiqs for JAX gradients and compiled batches.
The initial state and declared dissipation select `sesolve` or `mesolve`
automatically. Use `dissipation=False` only for an intentional closed-system
comparison. See {doc}`Backends and solvers <guides/choosing-a-backend>` for
integration settings and {doc}`Gradients and parameter fitting <guides/differentiability>`
for checks on derivatives.
