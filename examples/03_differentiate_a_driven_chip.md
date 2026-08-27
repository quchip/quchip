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

# Differentiability

The progression is scalar loss through statics, scalar loss through one driven
sequence, then a joint loss built from several experiments.

## Losses through statics

A scalar loss has a gradient. A vector residual has a Jacobian. Both pass
through the same public `Chip.with_params()` call.

```python
import jax
import jax.numpy as jnp

from quchip import Capacitive, Chip, DuffingTransmon, Resonator
from quchip.backend.dynamiqs import DynamiqsBackend

q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
r = Resonator(freq=7.0, levels=4, label="r")
chip = Chip(
    [q, r],
    [Capacitive(q, r, g=0.05, label="qr")],
    frame="rotating",
    backend=DynamiqsBackend(),
)

names = ["q.freq", "q.anharmonicity", "qr.g"]
theta = jnp.array([5.0, -0.25, 0.05])
target = jnp.array([5.05, -0.0010])


def observables(th):
    c = chip.with_params(dict(zip(names, th)))
    chi = c.dispersive_shift("q", "r") / 2
    return jnp.stack([jnp.asarray(c.freq("q")), jnp.asarray(chi)])


def residual(th):
    return observables(th) - target


def loss(th):
    return jnp.sum(residual(th) ** 2)


static_gradient = jax.grad(loss)(theta)
static_jacobian = jax.jacrev(residual)(theta)
static_step = jnp.array([1.0e-4, 1.0e-5, 1.0e-5])
static_finite_difference = jnp.stack(
    [
        (loss(theta + static_step[index] * jnp.eye(3)[index])
         - loss(theta - static_step[index] * jnp.eye(3)[index]))
        / (2.0 * static_step[index])
        for index in range(3)
    ]
)

{
    "observables_f01_chi": observables(theta),
    "gradient_shape": static_gradient.shape,
    "gradient": static_gradient,
    "jacobian_shape": static_jacobian.shape,
    "jacobian": static_jacobian,
    "central_difference": static_finite_difference,
}
```

The parameter vector mixes GHz-scale frequencies and a smaller coupling. For
optimization, work in dimensionless coordinates such as fractional frequency
changes or MHz-scale offsets. The gradient then measures comparable design
moves instead of inheriting the arbitrary numerical size of each unit.

## Losses through simple dynamics

This section differentiates the final excited-state population with respect to
pulse amplitude, Gaussian shape, and detuning.

The dynamiqs backend keeps these declared parameters differentiable through
the time-domain solve. Install it with `pip install 'quchip[dynamiqs]'`.

```python
import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from quchip import RWA, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence

amplitude0 = 0.029
frequency0 = 5.0
sigmas0 = 3.0

qubit = DuffingTransmon(
    freq=frequency0,
    anharmonicity=-0.30,
    levels=3,
    label="q",
)
chip = Chip(
    [qubit],
    frame="rotating",
    approximation=RWA(),
    backend="dynamiqs",
)
drive = ChargeDrive(qubit, label="xy")
chip.wire(drive)

sequence = QuantumSequence(chip)
sequence.schedule(
    drive,
    envelope=Gaussian(duration=40.0, sigmas=sigmas0, amplitude=amplitude0),
    freq=frequency0,
)
times = jnp.linspace(0.0, 60.0, 121)
```

## Name the physical perturbations

One unit in the perturbation vector means $1\%$ in pulse amplitude, $1\%$ in
the Gaussian parameter $N_\sigma$, or $1$ MHz in pulse detuning. quchip defines
$\sigma=\mathrm{duration}/(2N_\sigma)$, so increasing $N_\sigma$ by $1\%$
narrows the physical Gaussian width by about $1\%$. The derivatives predict
changes in final population on these three scales.

`with_params()` returns a rebound sequence. It does not mutate the declared
sequence.

```python
parameter_paths = ("pulse.0.amplitude", "pulse.0.sigmas", "pulse.0.freq")
perturbation_labels = (
    "amplitude (+1%)",
    r"$N_\sigma$ (+1%; $\sigma$ -1%)",
    "detuning (+1 MHz)",
)


def final_population(perturbation):
    rebound = sequence.with_params(
        {
            "pulse.0.amplitude": amplitude0 * (1.0 + 0.01 * perturbation[0]),
            "pulse.0.sigmas": sigmas0 * (1.0 + 0.01 * perturbation[1]),
            "pulse.0.freq": frequency0 + perturbation[2] * 1.0e-3,
        }
    )
    result = rebound.simulate(
        tlist=times,
        initial_state={"q": 0},
        check_truncation=False,
        partition=False,
    )
    return jnp.real(result.population("q", level=1)[-1])
```

## Differentiate the solve

The output is one scalar. `jax.grad` returns one derivative for each input,
while `jax.value_and_grad` evaluates the population and its gradient together.

```python
origin = jnp.zeros(3)
population0, gradient = jax.jit(jax.value_and_grad(final_population))(origin)
target_population = jnp.asarray(0.995)


def dynamic_loss(perturbation):
    return (final_population(perturbation) - target_population) ** 2


loss0, loss_gradient = jax.jit(jax.value_and_grad(dynamic_loss))(origin)

{
    "population": population0,
    "population_gradient": gradient,
    "loss": loss0,
    "loss_gradient": loss_gradient,
}
```

At this slightly under-rotated operating point, increasing the amplitude or
detuning raises the final population. Increasing $N_\sigma$ narrows the
Gaussian and lowers the final population.

## Check the derivative independently

Automatic differentiation should agree with a numerical derivative, but a
single finite-difference step can agree accidentally. We halve the step four
times and check convergence for every parameter.

```python
steps = jnp.asarray([0.5, 0.25, 0.125, 0.0625, 0.03125])
directions = jnp.eye(3)


@jax.jit
def central_differences(step):
    return jax.vmap(
        lambda direction: (
            final_population(step * direction) - final_population(-step * direction)
        )
        / (2.0 * step)
    )(directions)


finite_differences = jnp.stack([central_differences(step) for step in steps])
relative_errors = jnp.abs(finite_differences - gradient) / jnp.maximum(jnp.abs(gradient), 1.0e-12)
```

## See the local derivative and its check

The upper panel compares the predicted population changes with central finite
differences. The lower panel plots their relative error across step sizes.

```python
impact_milliprobability = np.asarray(gradient) * 1.0e3
finite_difference_check = np.asarray(finite_differences[0]) * 1.0e3
positions = np.arange(len(perturbation_labels))

figure, (impact_axis, convergence_axis) = plt.subplots(
    2,
    1,
    figsize=(7.4, 6.0),
    height_ratios=(1.5, 1.6),
    layout="constrained",
)

impact_axis.barh(
    positions,
    impact_milliprobability,
    color=["#C92F33" if value >= 0.0 else "0.45" for value in impact_milliprobability],
    height=0.58,
)
impact_axis.plot(
    finite_difference_check,
    positions,
    linestyle="none",
    marker="|",
    color="#16181C",
    markersize=13,
    markeredgewidth=1.6,
    label="central finite difference",
)
impact_axis.axvline(0.0, color="#16181C", linewidth=0.8)
impact_axis.set(
    yticks=positions,
    yticklabels=perturbation_labels,
    xlabel=r"Predicted $\Delta P(q=1)$ ($\times 10^{-3}$)",
    xlim=(-4.5, 4.5),
)
impact_axis.invert_yaxis()
impact_axis.legend(frameon=False, loc="lower left")

for index, label in enumerate(perturbation_labels):
    convergence_axis.loglog(
        np.asarray(steps),
        np.asarray(relative_errors[:, index]),
        marker="o",
        linewidth=1.7,
        markersize=4.0,
        label=label,
    )
convergence_axis.set(
    xlabel="Central-difference step / reference perturbation",
    ylabel="Relative error",
    ylim=(5.0e-6, 1.0e-2),
)
convergence_axis.grid(alpha=0.2, which="both")
convergence_axis.legend(frameon=False, ncols=3, fontsize=8)

figure_path = "../docs/images/differentiate_a_driven_chip.png"
figure.savefig(figure_path, dpi=180)
plt.show()
```

```{figure} ../images/differentiate_a_driven_chip.png
:width: 720px
:alt: Driven-qubit sensitivities with finite-difference convergence below

The top panel puts each derivative on its named reference scale. The bottom
panel checks convergence of the central differences.
```

## Losses through multi-sequence analysis

A calibration objective often combines several experiments that share the
same device and control parameters. Each sequence below has a different fixed
pulse duration and carrier offset. All three rebind the same qubit frequency,
amplitude scale, and carrier correction.

```python
experiment_settings = (
    (18.0, 0.020, -0.004),
    (28.0, 0.026, 0.000),
    (40.0, 0.029, 0.004),
)
experiments = []
for duration, amplitude, carrier_offset in experiment_settings:
    experiment = QuantumSequence(chip)
    experiment.schedule(
        drive,
        envelope=Gaussian(duration=duration, sigmas=3.0, amplitude=amplitude),
        freq=frequency0 + carrier_offset,
    )
    experiments.append(experiment)

shared_names = ("q.freq", "amplitude scale", "carrier correction")
shared_origin = jnp.array([frequency0, 1.0, 0.0])
multi_times = jnp.linspace(0.0, 60.0, 81)


def experiment_outputs(shared):
    qubit_frequency, amplitude_scale, carrier_correction = shared
    values = []
    for experiment, (_, nominal_amplitude, nominal_offset) in zip(
        experiments, experiment_settings
    ):
        rebound = experiment.with_params(
            {
                "q.freq": qubit_frequency,
                "pulse.0.amplitude": nominal_amplitude * amplitude_scale,
                "pulse.0.freq": frequency0 + nominal_offset + carrier_correction,
            }
        )
        result = rebound.simulate(
            tlist=multi_times,
            initial_state={"q": 0},
            check_truncation=False,
            partition=False,
        )
        values.append(jnp.real(result.population("q", level=1)[-1]))
    return jnp.stack(values)


reference_outputs = experiment_outputs(shared_origin)
multi_targets = jax.lax.stop_gradient(
    reference_outputs + jnp.array([0.010, -0.015, 0.005])
)
experiment_weights = jnp.array([1.0, 2.0, 0.5])


def multi_residual(shared):
    return experiment_outputs(shared) - multi_targets


def multi_loss(shared):
    return jnp.sum(experiment_weights * multi_residual(shared) ** 2)


multi_jacobian = jax.jacrev(multi_residual)(shared_origin)
multi_loss_gradient = jax.grad(multi_loss)(shared_origin)

{
    "experiments": len(experiments),
    "shared_parameters": shared_names,
    "residual_shape": multi_residual(shared_origin).shape,
    "jacobian_shape": multi_jacobian.shape,
    "jacobian_by_experiment": multi_jacobian,
    "joint_loss_gradient": multi_loss_gradient,
}
```

Rows of the Jacobian belong to experiments; columns belong to shared physical
parameters. The weighted scalar loss contracts those rows into one gradient.
Keep the Jacobian when diagnosing which experiment constrains which parameter,
and use the loss gradient for an optimization step.

The receipt puts all three derivatives on their reference-perturbation scales
and records the worst relative disagreement across the finite-difference
steps. Only first derivatives are checked here.

```python
reference_result = sequence.simulate(
    tlist=times,
    initial_state={"q": 0},
    check_truncation=False,
    partition=False,
)

gradient_receipt = {
    "backend": "dynamiqs",
    "base_population": float(population0),
    "figure": figure_path,
    "first_order_only": True,
    "fixed_structure_during_trace": True,
    "gradient_per_reference_perturbation": {
        path: float(value) for path, value in zip(parameter_paths, gradient)
    },
    "maximum_relative_error_across_steps": float(np.max(relative_errors)),
    "multi_sequence_count": len(experiments),
    "multi_sequence_jacobian_shape": list(multi_jacobian.shape),
    "multi_sequence_loss_gradient": [float(value) for value in multi_loss_gradient],
    "original_sequence_unchanged": dict(sequence.parameters)
    == {
        "q.freq": frequency0,
        "q.anharmonicity": -0.30,
        "pulse.0.freq": frequency0,
        "pulse.0.phase": 0.0,
        "pulse.0.start_time": None,
        "pulse.0.duration": 40.0,
        "pulse.0.sigmas": sigmas0,
        "pulse.0.amplitude": amplitude0,
    },
    "parameter_paths": list(parameter_paths),
    "solver": reference_result.solver,
}

print(f"RESULT gradient={json.dumps(gradient_receipt, sort_keys=True, separators=(',', ':'))}")
```

## Boundaries of this gradient

The traced calculation keeps the device graph, Hilbert-space dimensions, and
RWA band selection fixed. dynamiqs supports this automatic-differentiation
path; QuTiP remains available for ordinary solves and sweeps but does not
provide gradients. Eigenvector derivatives also require care near degenerate
subspaces. Gradients also inherit the solver tolerances, local-basis
truncation, frame, approximation, and loss scaling chosen for the forward
calculation.

Replace the population residuals with the observables that define the
experiment. Use `jax.grad` for a scalar loss and `jax.jacrev` or `jax.jacfwd`
for a vector of residuals or traces. Keep named parameters, explicit scales,
and an independent finite-difference spot check when introducing a new path.
