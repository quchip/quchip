---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- reader-content -->

# Model reduction

Can a smaller model reproduce the same pulse experiment? Drive the end of a
four-transmon chain, keep its nearest neighbour, and compare the reduced
model with the full chip. Later sections remove a readout from its feedline,
solve a 24-transmon ring in an exact excitation space, and truncate the
excitation number of the driven chain. Frequencies are in GHz and times in ns.

## Declare the chain

Neighbouring transmons are separated by 350 MHz and coupled at 12 MHz. The
full model has 81 basis states.

```python
import numpy as np
from quchip import RWA, Capacitive, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence

frequencies = (5.00, 5.35, 5.70, 6.05)
coupling_strength = 0.012

qubits = [
    DuffingTransmon(freq=freq, anharmonicity=-0.25, levels=3, label=f"q{index}")
    for index, freq in enumerate(frequencies)
]
couplings = [
    Capacitive(qubits[index], qubits[index + 1], g=coupling_strength, label=f"c{index}{index + 1}")
    for index in range(3)
]
chip = Chip(qubits, couplings=couplings, frame="rotating", approximation=RWA())
```

## Schedule a pulse and keep its neighbourhood

Drive `q0` with a 60 ns Gaussian. `active_patch(hops=1)` keeps `q0` and `q1`,
eliminates `q3` then `q2`, and runs the existing schedule on the reduced chip.
Here `hops=1` keeps devices within one coupling step of the driven qubit. The
source chip and sequence do not change.

```python
drive = ChargeDrive(qubits[0], label="q0-charge")
_ = chip.wire(drive)

sequence = QuantumSequence(chip)
_ = sequence.schedule(
    drive, envelope=Gaussian(duration=60.0, sigmas=3.0, amplitude=0.04 / 3),
    freq=chip.freq(qubits[0]),
)

patch = sequence.active_patch(hops=1, method="sw")
```

<details>
<summary>Plot the full and reduced models</summary>

```python
import shutil
import matplotlib.pyplot as plt

plt.style.use("../docs/_static/quchip.mplstyle")
plt.rcParams["text.usetex"] = bool(shutil.which("latex"))

topology_figure, topology_axes = plt.subplots(2, 1, figsize=(7.2, 3.0), layout="constrained")
for axis, model, title in zip(
    topology_axes, (chip, patch.chip), (r"Full chain $\cdot$ 81 states", r"Active patch $\cdot$ 9 states"),
):
    positions = {device.label: index for index, device in enumerate(chip.devices)}
    for coupling in model.couplings:
        x0 = positions[coupling.device_a.label]
        x1 = positions[coupling.device_b.label]
        axis.plot([x0, x1], [0, 0], color="#9AA0A8", lw=1.5, zorder=1)
        axis.text((x0 + x1) / 2, 0.12, f"{float(coupling.coupling_strength) * 1000:.0f} MHz",
                  ha="center", fontsize=9, color="#50565A")
    for device in model.devices:
        x = positions[device.label]
        axis.scatter(x, 0, s=600, color="#246FA8", edgecolors="white", zorder=2)
        axis.text(x, 0, f"${device.label[0]}_{device.label[1:]}$", color="white",
                  ha="center", va="center", fontsize=10)
        axis.text(x, -0.25, f"{float(device.freq):.2f} GHz", ha="center", fontsize=9, color="#50565A")
    axis.annotate("drive", xy=(0, 0.08), xytext=(-0.45, 0.35), color="#C92F33", fontsize=9,
                  arrowprops={"arrowstyle": "->", "color": "#C92F33"})
    axis.set(xlim=(-0.6, 3.5), ylim=(-0.4, 0.55))
    axis.set_title(title, loc="left")
    axis.axis("off")

topology_figure.savefig("../docs/images/active_patch_topology.svg")
plt.show()
```

</details>

```{figure} ../images/active_patch_topology.svg
:alt: The full q0-q1-q2-q3 chain and its reduced q0-q1 model, both driven on q0.

The reduced model has 9 basis states. The removed spectators contribute
Hamiltonian corrections; labels show bare parameters.
[PDF](../images/active_patch_topology.pdf)
```

## Simulate the same pulse in the reduced model

`sequence.simulate()` runs the full chip, and `patch.simulate()` runs the
schedule on the reduced chip. Compare the driven qubit's excited-state
population.

```python
times = np.linspace(0.0, 80.0, 321)

full_result = sequence.simulate(tlist=times)
reduced_result = patch.simulate(tlist=times)

full_population = np.asarray(full_result.population("q0", level=1)).real
reduced_population = np.asarray(reduced_result.population("q0", level=1)).real
population_residual = np.abs(full_population - reduced_population)
```

<details>
<summary>Plot the population and difference</summary>

```python
figure, (population_axis, residual_axis) = plt.subplots(
    2,
    1,
    figsize=(7.2, 5.6),
    height_ratios=(3.0, 1.15),
    sharex=True,
    layout="constrained",
)

population_axis.plot(times, full_population, color="#16181C", linewidth=2.4, label="full chip (81 states)")
population_axis.plot(
    times,
    reduced_population,
    color="#C92F33",
    linewidth=1.6,
    linestyle="--",
    label="reduced model (9 states)",
)
population_axis.set_ylabel(r"$P(q_0=1)$")
population_axis.set_ylim(-0.02, 1.02)
population_axis.legend(loc="upper left")

display_residual = np.maximum(population_residual, 1.0e-10)
residual_axis.semilogy(times, display_residual, color="#C92F33", linewidth=1.8)
residual_axis.set(
    xlabel="Time (ns)",
    ylabel="Absolute residual",
    ylim=(1.0e-10, 2.0e-2),
)

figure_path = "../docs/images/reduce_and_replay.svg"

figure.savefig(figure_path)
plt.show()
```

</details>

```{figure} ../images/reduce_and_replay.svg
:alt: Driven-qubit population on the full chip and reduced model, with their absolute difference below.

The full and reduced trajectories overlap; the lower panel resolves their
absolute difference. Values below 10⁻¹⁰ are floored for display only.
[PDF](../images/reduce_and_replay.pdf)
```

## Inspect the elimination report

Each entry in `patch.steps` is an `EliminationResult`, whose `describe()`
report includes frequency shifts, inherited loss, coupling ratios, and
approximation notes. The kept Hamiltonian corrections set the reduced
dynamics, and the original device parameters do not change.

<details>
<summary>Show the built-in reports</summary>

```python
for step in patch.steps:
    print(step.describe())
```

<!-- executed-output:start -->

Output:

```text
Elimination fold report (method='sw')
─────────────────────────────────────
q2: freq 5.7 → 5.69959 GHz   (Lamb shift -0.411 MHz)
validity: c23 g/Δ=0.034 ✓ (< 0.1) · min block gap 0.35 GHz
dropped: counter-rotating terms · ring-up transients · higher-order (>2) corrections
notes:
  - Retained the full second-order Hamiltonian correction and each transformed channel from the removed components; inherited decay remains collective and separate from intrinsic survivor noise.
Elimination fold report (method='sw')
─────────────────────────────────────
q1: freq 5.35 → 5.34959 GHz   (Lamb shift -0.412 MHz)
validity: c12 g/Δ=0.034 ✓ (< 0.1) · min block gap 0.35 GHz
dropped: counter-rotating terms · ring-up transients · higher-order (>2) corrections
notes:
  - Retained the full second-order Hamiltonian correction and each transformed channel from the removed components; inherited decay remains collective and separate from intrinsic survivor noise.
```

<!-- executed-output:end -->

</details>

To remove a specific device directly, use
`eliminate(chip, "q3", method="sw")`, then inspect the returned `.chip` and
`.describe()`. `method="exact"` uses the kept block of the diagonalized model
instead of the second-order Schrieffer–Wolff approximation. Its kept-block
energies are exact to all orders in the retained Hamiltonian, but it does not
restore bands that `RWA()` discarded. Both methods use the declared finite
Hilbert space. Eliminating a coupling keeps both endpoints.

On a large chip, `eliminate(chip, "q3", local=True)` computes the SW
reduction from a patch around the device. The result matches the full
reduction up to round-off omissions, which its notes report in GHz, and
repeated local eliminations stay local.

To keep a chosen set of devices, call `eliminate_spectators(chip, keep)`. It
removes the spectators one `eliminate()` step at a time and returns every
step. `active_patch()` uses the same default order, which here removes `q3`
and then `q2`. A spectator waits while a spectator neighbour links it to a
kept device near it in frequency, so the chain keeps their fourth-order
exchange shift. Under `local=True`, the order also keeps the operators of the
kept devices small. Pass `order=` to either function to set the steps. When a
device that stays on the chip, such as a kept coupler, links a removed
spectator to a kept device, each SW step estimates the shift that it omits
from that kept device. `omitted_shifts` holds these estimates in GHz.

## Check the reduction error

A small coupling-to-detuning ratio is a diagnostic, not a guarantee of accurate
dynamics. Here we also check the full/reduced population difference against
`5(g/Δ)²`, with the retained–eliminated boundary. The factor of five allows for
multilevel and finite-pulse effects. When you change the experiment, increase
`hops` to keep more neighbours and compare again.

<details>
<summary>Numerical checks and record</summary>

```python
import json

full_dimension = int(np.prod(chip.dims))
reduced_dimension = int(np.prod(patch.chip.dims))
same_schedule = sequence.settings["entries"] == patch.sequence.settings["entries"]

validity_records = [
    record
    for eliminated_device in patch.validity.values()
    for record in eliminated_device.values()
]
max_g_over_delta = max(float(record["g_over_delta"]) for record in validity_records)
minimum_block_gap = min(float(record["min_block_gap"]) for record in validity_records)
all_folds_valid = all(bool(record["is_valid"]) for record in validity_records)

boundary_detuning = abs(float(chip.freq("q2")) - float(chip.freq("q1")))
boundary_ratio = coupling_strength / boundary_detuning
residual_tolerance = 5.0 * boundary_ratio**2
maximum_residual = float(np.max(population_residual))

if not all_folds_valid:
    raise RuntimeError("The active patch contains an invalid elimination step.")
if maximum_residual >= residual_tolerance:
    raise RuntimeError("The reduced dynamics exceed the validity-derived tolerance.")
if not same_schedule:
    raise RuntimeError("The reduced sequence does not preserve the scheduled entries.")
if tuple(device.label for device in chip.devices) != ("q0", "q1", "q2", "q3"):
    raise RuntimeError("active_patch() mutated the original chip.")

reduction_receipt = {
    "active_labels": list(patch.active_labels),
    "all_folds_valid": all_folds_valid,
    "eliminated_labels": list(patch.eliminated_labels),
    "figure": figure_path,
    "full_dimension": full_dimension,
    "maximum_population_residual": maximum_residual,
    "maximum_g_over_delta": max_g_over_delta,
    "minimum_block_gap_ghz": minimum_block_gap,
    "original_chip_unchanged": tuple(device.label for device in chip.devices) == ("q0", "q1", "q2", "q3"),
    "peak_full_population": float(np.max(full_population)),
    "reduced_dimension": reduced_dimension,
    "reduction_method": "sw",
    "residual_tolerance": residual_tolerance,
    "same_schedule": same_schedule,
}

print(f"RESULT reduction={json.dumps(reduction_receipt, sort_keys=True, separators=(',', ':'))}")
```

<!-- executed-output:start -->

Output:

```text
RESULT reduction={"active_labels":["q0","q1"],"all_folds_valid":true,"eliminated_labels":["q3","q2"],"figure":"../docs/images/reduce_and_replay.svg","full_dimension":81,"maximum_g_over_delta":0.03432606492529982,"maximum_population_residual":3.7142772963960624e-06,"minimum_block_gap_ghz":0.3495885714285709,"original_chip_unchanged":true,"peak_full_population":0.7480171143273022,"reduced_dimension":9,"reduction_method":"sw","residual_tolerance":0.0058775348055535755,"same_schedule":true}
```

<!-- executed-output:end -->

</details>

## Remove a readout from its feedline

`eliminate()` also removes readout resonators that couple to a feedline. A
hanger emits into both directions of its line. Give each readout two ports
with half the external rate, and cascade the ports in opposite orders along
the two directions. A one-way cascade gives an all-pass response in the
lossless limit, so it does not describe the hanger notch.

```python
from quchip import VNA, PortNetwork, Resonator, eliminate

readout_qubits = [
    DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, T1=60_000.0, label="qa"),
    DuffingTransmon(freq=5.3, anharmonicity=-0.2, levels=3, label="qb"),
]
readouts = [
    Resonator(freq=7.0, levels=2, internal_quality_factor=200_000, label="ra"),
    Resonator(freq=7.12, levels=2, internal_quality_factor=200_000, label="rb"),
]
readout_couplings = [
    Capacitive(readout_qubits[0], readouts[0], g=0.105, label="qa_ra"),
    Capacitive(readout_qubits[1], readouts[1], g=0.05, label="qb_rb"),
]
kappa_ext = 2 * np.pi * 0.001

feedline = PortNetwork(label="hanger")
right = [feedline.port(f"{mode.label}_right", target=mode, rate=kappa_ext / 2) for mode in readouts]
left = [feedline.port(f"{mode.label}_left", target=mode, rate=kappa_ext / 2) for mode in readouts[::-1]]
feedline.cascade(*right)
feedline.cascade(*left)
feedline.expose("right", input=right[0].input, output=right[-1].output)
feedline.expose("left", input=left[0].input, output=left[-1].output)
hanger = Chip([*readout_qubits, *readouts], readout_couplings, port_network=feedline, approximation=RWA())
```

Sweep both readout resonances, then remove `ra` with the exact method and
sweep the reduced chip:

```python
probe_frequencies = np.concatenate(
    [float(hanger.freq(mode)) + np.linspace(-0.003, 0.003, 101) for mode in readouts]
)
full_response = VNA(hanger).sweep(probe_frequencies)
reduction = eliminate(hanger, readouts[0], method="exact")
reduced_response = VNA(reduction.chip).sweep(probe_frequencies)
s_difference = np.max(np.abs(full_response.matrix - reduced_response.matrix))

print(f"Maximum complex S difference: {s_difference:.2g}")
```

<!-- executed-output:start -->

Output:

```text
Maximum complex S difference: 6.8e-06
```

<!-- executed-output:end -->

Rows and columns follow the exposure order, `right` then `left`.
`matrix[..., 0, 0]` is the transmission of the rightward field, and
`matrix[..., 1, 0]` is its reflection into the leftward field. The removed
readout keeps one coupled response at its original position, so the line
keeps its dip and its reflection into the opposite direction. Both
transformed ports retain the qubits' directional Purcell decay.

Time-domain dynamics omit this response. A time-domain input, output field, or
measurement record on this feedline therefore raises `NotImplementedError`.
Keep the readout mode for them. Other time-domain results, such as the decay
of an excited `rb`, miss the return path of the response: `ra` scatters part
of the field that `rb` emits back into `rb`. `reduction.validity["ra_response"]`
compares the self-energy of this path with the damping of each kept emitter,
at the emitter's frequency. It is a diagnostic, not an error bound, and
`is_valid` requires a value below 0.1.

```python
return_path = reduction.validity["ra_response"]

print(f"Return-path self-energy over width: {return_path['self_energy_over_width']:.2g}")
```

<!-- executed-output:start -->

Output:

```text
Return-path self-energy over width: 0.0042
```

<!-- executed-output:end -->

Here `rb` lies about 115 MHz from the removed pole, so the return path is
0.4% of its damping.

A readout with one port keeps its scattering as a reference section at its
dressed transition instead: `PortNetwork.mode_reflection(...)` when the port
is alone on its plane, and `mode_transmission(...)` on a shared one-way
feedline. The transformed ports act on every surviving device, so a later
elimination can remove the next readout. A measurement record of that line,
or of a loss channel on the removed readout, would omit the section, so it
raises `NotImplementedError`. An output field applies the section at its
carrier frequency, a narrowband approximation.

This vacuum weak-probe calculation supports global SW, local SW, and global
exact elimination. Inspect `reduction.notes` and `reduction.validity` before
you use the reduced model. The retained response keeps the dispersive limits
described in the [physics
reference](../physics.md#105-collapse-transforms-and-validity-metrics). Keep
the readout mode for pumped, finite-power, noisy, or excited-reference field
calculations. Unsupported port layouts raise `NotImplementedError`.

<details>
<summary>Check the field error and the return path</summary>

Purcell dispersion leaves field errors of order $(g/\Delta)^2\kappa/(2\pi|\Delta|)$
for the strongest qubit–readout pair. Allow one order of magnitude above it.

The return path scales as $(\kappa/2)^2/(\Delta_p h_b)$. Here $\Delta_p$ is
the angular detuning of `rb` from the dressed pole of `ra`, and $h_b$ is the
amplitude damping of `rb`. The photon weight of the pole lowers it by about
$(g/\Delta)^2$.

```python
dispersive_scale = (0.105 / (7.0 - 5.0))**2 * 0.001 / (7.0 - 5.0)
assert s_difference < 10 * dispersive_scale

pole_detuning = 2 * np.pi * abs(float(readouts[1].freq) - float(hanger.freq(readouts[0])))
rb_damping = (kappa_ext + 2 * np.pi * float(readouts[1].freq) / 200_000) / 2
return_path_estimate = (kappa_ext / 2)**2 / (pole_detuning * rb_damping)
assert return_path["is_valid"]
assert abs(return_path["self_energy_over_width"] - return_path_estimate) < 2 * (0.105 / 2.0)**2 * return_path_estimate
```

</details>

## Use an exact excitation space

Some reductions are exact. When a chip conserves total excitation number, a
one-excitation state stays in a small invariant space. quchip proves that
space from the initial state and every retained operator, then solves in it
without an excitation cutoff.

Couple 24 identical transmons in a ring at 10 MHz and start with one
excitation on `q0`. `compact_state()` prepares this bare state without
building the 3²⁴-state product space.

```python
ring_size = 24
ring_qubits = [
    DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label=f"q{index}")
    for index in range(ring_size)
]
ring_couplings = [
    Capacitive(ring_qubits[index], ring_qubits[(index + 1) % ring_size], g=0.010, label=f"g{index}")
    for index in range(ring_size)
]
ring = Chip(ring_qubits, ring_couplings, approximation=RWA())

ring_times = np.linspace(0.0, 100.0, 201)
ring_result = QuantumSequence(ring).simulate(tlist=ring_times, initial_state=ring.compact_state({"q0": 1}))

for name, result in (("Driven chain", full_result), ("Idle ring", ring_result)):
    space = result.stats["excitation_reduction"]
    print(f"{name}: {space['dimension']} of {space['full_dimension']} states")
```

<!-- executed-output:start -->

Output:

```text
Driven chain: 81 of 81 states
Idle ring: 24 of 282429536481 states
```

<!-- executed-output:end -->

The charge drive on the chain changes the excitation number, so its solve
uses the full product space. The idle ring keeps its single excitation, and
`stats["excitation_reduction"]` records the proof.

In the one-excitation space, the RWA Hamiltonian is a 24 × 24 hopping matrix
with `g` between ring neighbours. The common 5 GHz diagonal adds only a
global phase. Its propagator gives an independent reference for every
transmon's population.

```python
from scipy.linalg import expm

hopping = 0.010 * (np.eye(ring_size, k=1) + np.eye(ring_size, k=-1))
hopping[0, -1] = hopping[-1, 0] = 0.010
reference = np.array([np.abs(expm(-2j * np.pi * hopping * t)[:, 0])**2 for t in ring_times]).T
ring_population = np.array(
    [np.asarray(ring_result.population(f"q{index}", 1)).real for index in range(ring_size)]
)
population_difference = np.max(np.abs(ring_population - reference))

print(f"Maximum population difference: {population_difference:.1e}")
```

<!-- executed-output:start -->

Output:

```text
Maximum population difference: 3.1e-13
```

<!-- executed-output:end -->

The solve diagonalizes the constant 24-state generator, so the difference is
floating-point roundoff. Decay adds the vacuum to the space. With T1 on every
transmon, the master equation runs on 25 states:

```python
lossy_ring = ring.with_params({f"q{index}.T1": 20_000.0 for index in range(ring_size)})
lossy_result = QuantumSequence(lossy_ring).simulate(
    tlist=ring_times, initial_state=lossy_ring.compact_state({"q0": 1}),
)

print(f"Lossy ring: {lossy_result.solver} on {lossy_result.stats['excitation_reduction']['dimension']} states")
```

<!-- executed-output:start -->

Output:

```text
Lossy ring: mesolve on 25 states
```

<!-- executed-output:end -->

A transverse drive or thermal raising can require the full space, and
stochastic solvers keep it. `chip.state()`, labeled dressed queries, the
default initial state, weak-probe VNA sweeps, and `fit_a_dress()` also use
only the excitation sectors they need, as the
[cookbook](../cookbook.md#fit-a-large-conserving-model) describes.

<details>
<summary>Check the exact spaces</summary>

Diagonalizing a 24-state generator leaves roundoff far below `1e-10`.

```python
assert ring_result.stats["excitation_reduction"]["exact"]
assert ring_result.stats["excitation_reduction"]["dimension"] == ring_size
assert lossy_result.stats["excitation_reduction"]["dimension"] == ring_size + 1
assert population_difference < 1e-10
```

</details>

## Truncate the excitation number

The charge drive on the chain changes the excitation number, so its solve
needs the full product space. On a larger chip, that space may not fit in
memory. `ExcitationCutoff(max_total=K)` keeps only the product states with at
most `K` excitations. It changes the model, so compare results at increasing
`K`. Here the full 81-state solve gives an exact reference:

```python
from quchip import ExcitationCutoff

cutoff_errors = {}
for max_total in (1, 2):
    limited_chip = chip.with_space(ExcitationCutoff(max_total=max_total))
    limited_sequence = QuantumSequence(limited_chip)
    _ = limited_sequence.schedule(
        "q0-charge", envelope=Gaussian(duration=60.0, sigmas=3.0, amplitude=0.04 / 3),
        freq=chip.freq(qubits[0]),
    )
    limited_result = limited_sequence.simulate(tlist=times)
    limited_population = np.asarray(limited_result.population("q0", level=1)).real
    cutoff_errors[max_total] = np.max(np.abs(limited_population - full_population))
    dimension = limited_result.stats["excitation_reduction"]["dimension"]
    print(f"Population error at K = {max_total} ({dimension} states): {cutoff_errors[max_total]:.1e}")
```

<!-- executed-output:start -->

Output:

```text
Population error at K = 1 (5 states): 8.9e-04
Population error at K = 2 (15 states): 2.7e-06
```

<!-- executed-output:end -->

`K = 1` removes the second excitation that the pulse adds. Almost all of it
is in the second excited level of `q0`, which reaches a population of 5e-4.
At `K = 2`, the difference is at the tolerance of the default solver, and
tighter solver tolerances reduce it to 1e-8. `check_truncation(result)`
reports the population next to the cutoff as a warning heuristic, not an
error bound. The [cookbook](../cookbook.md#truncate-the-total-excitation)
shows a static case where the error is not monotone in `K`.

<details>
<summary>Check the cutoff</summary>

QuTiP's default relative tolerance of 1e-6 sets the floor of the comparison.

```python
assert cutoff_errors[2] < 1e-5
```

</details>

## Other changes to the model

Use `with_params()` to change numerical parameters, as the [spectrum
guide](statics-and-parameter-studies.md) does during a sweep. `chip.partition()`
separates independent components exactly, but an active patch approximates
coupled spectators. If shared Hamiltonian terms, loss, or drive crosstalk keep
components in a joint solve, inspect `partition.notes`.
