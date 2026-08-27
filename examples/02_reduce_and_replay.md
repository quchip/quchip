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

# Chip transformations

Transformations return new chips and leave the source declaration unchanged.
Start with one schedule-aware reduction, inspect what changed, then compare
the reduced and full dynamics.

## Reduce one scheduled neighbourhood

Build the schedule first. `active_patch()` keeps the scheduled device and its
neighbour, folds away the spectator, and rebinds the schedule to the smaller
chip.

```python
import numpy as np

from quchip import (
    RWA,
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Gaussian,
    QuantumSequence,
    Resonator,
    eliminate,
    fit_a_dress,
)

qubits = [
    DuffingTransmon(freq=freq, anharmonicity=-0.25, levels=3, label=f"q{index}")
    for index, freq in enumerate([5.0, 5.35, 5.70])
]
chip = Chip(
    qubits,
    [Capacitive(qubits[index], qubits[index + 1], g=0.004) for index in range(2)],
    frame="rotating",
    approximation=RWA(),
)

line = ChargeDrive(qubits[0], label="xy")
chip.wire(line)
sequence = QuantumSequence(chip)
sequence.schedule(
    line,
    envelope=Gaussian(duration=20.0, sigmas=3.0, amplitude=0.04),
    freq=chip.freq(qubits[0]),
)

patch = sequence.active_patch(hops=1, method="sw")
times = np.linspace(0.0, 40.0, 161)
full = sequence.simulate(tlist=times)
small = patch.simulate(tlist=times)
p_full = np.asarray(full.population("q0", level=1)).real
p_small = np.asarray(small.population("q0", level=1)).real

{
    "kept": patch.active_labels,
    "eliminated": patch.eliminated_labels,
    "validity": patch.validity,
    "maximum_population_difference": float(
        np.max(np.abs(p_full - p_small))
    ),
}
```

## Rebind, clone, and serialize

`with_params()` is the smallest chip transformation. It changes numerical
parameters on an isolated copy. `clone()` copies the full structure, while
`to_dict()` and `Chip.from_dict()` provide a JSON-safe round trip for declared
devices, couplings, control equipment, baths, frames, and approximations.
Backends and computed results are runtime choices and are not serialized.

```python
rebound = chip.with_params({"q1.freq": 5.45})
cloned = chip.clone()
restored = Chip.from_dict(chip.to_dict())

{
    "source_q1_freq_ghz": chip.parameters["q1.freq"],
    "rebound_q1_freq_ghz": rebound.parameters["q1.freq"],
    "source_dressed_q1_ghz": float(chip.freq("q1")),
    "rebound_dressed_q1_ghz": float(rebound.freq("q1")),
    "clone_is_distinct": cloned is not chip,
    "restored_devices": tuple(device.label for device in restored.devices),
    "round_trip_parameters_match": dict(restored.parameters) == dict(chip.parameters),
}
```

## Eliminate a device

A far-detuned bus can be folded into its neighbours. The result owns an
ordinary reduced `Chip`, derived parameters, a validity record, notes, and a
plain-text report. The source chip is unchanged.

`method="sw"` uses a second-order Schrieffer-Wolff reduction.
`method="exact"` diagonalizes the same resolved static model and extracts the
kept block. Comparing them is useful inside the perturbative regime.

```python
left = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="left")
right = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="right")
bus = Resonator(freq=6.3, levels=4, label="bus")
bridge = Chip(
    [left, right, bus],
    [
        Capacitive(left, bus, g=0.08, label="left-bus"),
        Capacitive(right, bus, g=0.08, label="right-bus"),
    ],
    frame="rotating",
    approximation=RWA(),
)

sw_fold = eliminate(bridge, bus, method="sw")
exact_fold = eliminate(bridge, bus, method="exact")

fold_comparison = {
    label: {
        "full_dressed_ghz": float(bridge.freq(label)),
        "sw_reduced_ghz": float(sw_fold.chip.freq(label)),
        "exact_reduced_ghz": float(exact_fold.chip.freq(label)),
    }
    for label in ("left", "right")
}

{
    "before_devices": tuple(device.label for device in bridge.devices),
    "after_devices": tuple(device.label for device in sw_fold.chip.devices),
    "after_couplings": tuple(coupling.label for coupling in sw_fold.chip.couplings),
    "validity": sw_fold.validity,
    "frequencies": fold_comparison,
}
```

Eliminating a device removes that mode. Eliminating a coupling keeps both
endpoints and replaces an exchange edge with its diagonal dispersive effect.
When a shipped retargeting rule applies, control lines move to the effective
device or edge and retain their labels. If a line would be stranded, the
transformation raises instead of silently dropping the control.

```python
dq = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="dq")
dr = Resonator(freq=7.0, levels=5, label="dr")
dispersive_edge = eliminate(
    Chip([dq, dr], [Capacitive(dq, dr, g=0.05, label="dq-dr")]),
    "dq-dr",
)

{
    "surviving_devices": tuple(device.label for device in dispersive_edge.chip.devices),
    "effective_coupling_type": type(dispersive_edge.chip.couplings[0]).__name__,
    "notes": tuple(dispersive_edge.notes),
}
```

## Partition independent components

`partition()` is exact. It separates connected components that do not share a
coupling, non-separable bath, or drive crosstalk. Simulation uses this path
automatically when the initial state permits it; call it directly to inspect
or orchestrate the component solves.

```python
island_qubits = [
    DuffingTransmon(freq=5.0 + 0.2 * index, anharmonicity=-0.25, levels=3, label=f"i{index}")
    for index in range(3)
]
islands = Chip(
    island_qubits,
    [Capacitive(island_qubits[0], island_qubits[1], g=0.005, label="i01")],
)
partition = islands.partition()

{
    "is_trivial": partition.is_trivial,
    "components": tuple(component.labels for component in partition),
    "owner_of_i2": partition.owner_of("i2"),
}
```

## Fit declared parameters to dressed observables

`fit_a_dress()` also returns a result with `.chip`. Select the parameters that
may move and state the dressed quantity they must reproduce. This example
moves one bare qubit frequency while every other declaration stays fixed.

```python
fit_target_ghz = float(bridge.freq("left")) + 0.01
fit = fit_a_dress(
    bridge,
    observable_targets={"left": {"freq": fit_target_ghz}},
    fit_parameters={"left": ("freq",)},
)

{
    "target_dressed_ghz": fit_target_ghz,
    "fitted_dressed_ghz": float(fit.chip.freq("left")),
    "moved_parameters": tuple(fit.final_params),
}
```

## Frames, bases, and interoperability

Frames and approximations change how a model is resolved for simulation; they
do not change the lab-frame dressed spectrum. Pass an override to `resolve()`
for a one-off inspection, or construct a chip with the intended setting when
it is part of the model. `dress()` exposes the eigenbasis assignment without
replacing the chip.

For interoperability, `from_scqubits()` imports supported scqubits devices or
Hilbert spaces, and `to_scqubits()` exports supported quchip declarations. The
`quchip[scqubits]` extra is required. Inspect the returned model and its units
after conversion; interoperability maps represented physics, not arbitrary
third-party callbacks.

```{code-block} python
from quchip import from_scqubits, to_scqubits

quchip_model = from_scqubits(scqubits_model)
round_trip = to_scqubits(quchip_model)
```

## Compare a larger active patch

A Gaussian pulse drives one end of a four-transmon chain. `active_patch()`
keeps the driven neighbourhood, folds the two external spectators into the
surviving chip, and returns the same schedule bound to the reduced model.

The next model has 81 states and reduces to 9, so the full comparison remains
quick to rerun.

The reduction is approximate, so the result includes validity metrics. We use
those metrics before comparing the full and reduced dynamics.

```python
import json

import matplotlib.pyplot as plt
import numpy as np

from quchip import RWA, Capacitive, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence

frequencies = (5.00, 5.35, 5.70, 6.05)
coupling_strength = 0.004

qubits = [
    DuffingTransmon(
        freq=frequency,
        anharmonicity=-0.25,
        levels=3,
        label=f"q{index}",
    )
    for index, frequency in enumerate(frequencies)
]
couplings = [
    Capacitive(
        qubits[index],
        qubits[index + 1],
        g=coupling_strength,
        label=f"c{index}{index + 1}",
    )
    for index in range(3)
]
chip = Chip(
    qubits,
    couplings=couplings,
    frame="rotating",
    approximation=RWA(),
)

drive = ChargeDrive(qubits[0], label="q0-charge")
chip.wire(drive)

sequence = QuantumSequence(chip)
sequence.schedule(
    drive,
    envelope=Gaussian(duration=20.0, sigmas=3.0, amplitude=0.04),
    freq=chip.freq(qubits[0]),
)
```

## Keep the scheduled neighbourhood

The pulse targets `q0`. With `hops=1`, the active patch also keeps its direct
neighbour `q1`; `q2` and `q3` are folded away from the far end inward.

```python
patch = sequence.active_patch(hops=1, method="sw")

full_dimension = int(np.prod(chip.dims))
reduced_dimension = int(np.prod(patch.chip.dims))
same_schedule = sequence.settings["entries"] == patch.sequence.settings["entries"]
```

The reduction result keeps the Schrieffer-Wolff validity record for every
fold. Every coupling ratio is far below the package's `0.1` validity boundary.

```python
validity_records = [
    record
    for eliminated_device in patch.validity.values()
    for record in eliminated_device.values()
]
max_g_over_delta = max(float(record["g_over_delta"]) for record in validity_records)
minimum_block_gap = min(float(record["min_block_gap"]) for record in validity_records)
all_folds_valid = all(bool(record["is_valid"]) for record in validity_records)

patch.validity
```

## Replay the schedule

Both simulations use the schedule declared above. `patch.simulate()` runs the
copy that `active_patch()` rebound to the reduced chip; there is no second
pulse declaration.

```python
times = np.linspace(0.0, 40.0, 161)

full_result = sequence.simulate(tlist=times)
reduced_result = patch.simulate(tlist=times)

full_population = np.asarray(full_result.population("q0", level=1)).real
reduced_population = np.asarray(reduced_result.population("q0", level=1)).real
population_residual = np.abs(full_population - reduced_population)
```

The leading reduction error at the retained/eliminated boundary scales as
$(g/\Delta)^2$. A factor of 50 leaves headroom for multilevel and finite-pulse
effects without choosing the tolerance from the observed residual.

```python
boundary_detuning = abs(float(chip.freq("q2")) - float(chip.freq("q1")))
boundary_ratio = coupling_strength / boundary_detuning
residual_tolerance = 50.0 * boundary_ratio**2
maximum_residual = float(np.max(population_residual))

if not all_folds_valid:
    raise RuntimeError("The active patch contains an invalid elimination step.")
if maximum_residual >= residual_tolerance:
    raise RuntimeError("The reduced dynamics exceed the validity-derived tolerance.")
if not same_schedule:
    raise RuntimeError("The reduced sequence does not preserve the scheduled entries.")
if tuple(device.label for device in chip.devices) != ("q0", "q1", "q2", "q3"):
    raise RuntimeError("active_patch() mutated the original chip.")
```

## Compare full and reduced dynamics

The upper panel overlays the driven-qubit population. The logarithmic lower
panel plots the residual; values below $10^{-10}$ are floored for display only.

```python
figure, (population_axis, residual_axis) = plt.subplots(
    2,
    1,
    figsize=(7.4, 5.8),
    height_ratios=(3.0, 1.15),
    sharex=True,
    layout="constrained",
)

population_axis.plot(times, full_population, color="#16181C", linewidth=2.5, label="full chip (81 states)")
population_axis.plot(
    times,
    reduced_population,
    color="#C92F33",
    linewidth=1.7,
    linestyle="--",
    label="active patch (9 states)",
)
population_axis.set_ylabel(r"$P(q_0=1)$")
population_axis.set_ylim(-0.02, 1.02)
population_axis.legend(frameon=False, loc="upper left")

display_residual = np.maximum(population_residual, 1.0e-10)
residual_axis.semilogy(times, display_residual, color="#C92F33", linewidth=1.8)
residual_axis.axhline(residual_tolerance, color="0.55", linestyle="--", linewidth=1.1, label="derived tolerance")
residual_axis.set(
    xlabel="Time (ns)",
    ylabel="Absolute residual",
    ylim=(1.0e-10, 2.0e-2),
)
residual_axis.legend(frameon=False, loc="upper right")

figure_path = "../docs/images/reduce_and_replay.png"
figure.savefig(figure_path, dpi=180)
plt.show()
```

```{figure} ../images/reduce_and_replay.png
:width: 720px
:alt: Full and active-patch driven-qubit populations with their absolute residual below

The same schedule runs on the 81-state chip and its 9-state active patch.
```

The receipt records both the approximation's own validity and the independent
forward comparison.

```python
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

Change `hops` to choose how much of the coupling neighbourhood remains
explicit. Change the spectator detunings or couplings and read the returned
validity record before trusting the smaller model.

Choose a transformation by the question. Use `with_params()` for new numbers
on the same structure, `partition()` for exact independent components,
`eliminate()` for a physical effective model, and `active_patch()` when the
scheduled experiment should choose the retained neighbourhood. Keep the
result object until you have inspected its notes, validity, and an observable
that matters for the next calculation.
