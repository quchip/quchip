# Backends and solvers

QuTiP is the default backend. Use dynamiqs for JAX gradients, compiled batches,
or accelerator execution. Both solve the declared model with its chosen frame,
approximation, controls, and loss channels.

## Choose a backend and equation

```python
chip = Chip(devices, couplings, backend="qutip")
result = sequence.simulate(duration=100.0, backend="qutip")
```

Install `quchip[dynamiqs]` and use `backend="dynamiqs"` on the chip or solve to
select that backend.

| Request | Automatic equation |
|---|---|
| Ket, no collapse channels | `sesolve`: Schrödinger equation |
| Density matrix or collapse channels | `mesolve`: Lindblad master equation |

Leave `solver` unset for this selection. An explicit `solver="sesolve"` rejects
collapse channels and density matrices. Use `dissipation=False` for an
intentional Hamiltonian-only calculation.

## Time grid, stepping, and saved states

Omit `tlist` for a pulse-aware grid; use `duration` to include an idle interval.
An explicit `tlist` is passed to the solver and defines the initial and final
times. Adaptive methods choose internal steps separately.

```python
result = sequence.simulate(
    tlist=times,
    e_ops=chip.e_ops(q="n"),
    states="final",
)
```

`states="all"` retains the history, `"final"` keeps only the final state, and
`"none"` keeps neither. Requested observables remain available in all three
cases. Post-solve state and collapse-flux queries need the relevant saved states.

## QuTiP integration

```python
result = sequence.simulate(
    tlist=times,
    options={"method": "vern9", "rtol": 1e-9, "atol": 1e-11},
)
```

| Calculation | Methods |
|---|---|
| General non-stiff dynamics | `adams` (default), `vern7`, `vern9`, `dop853`, `tsit5` |
| Stiff dynamics | `bdf`, `lsoda` |
| Small constant Hamiltonian or Liouvillian | `diag` |
| Large sparse constant closed system | `krylov` |

For a constant generator, quchip automatically selects `diag` up to Hilbert
dimension 64 for `sesolve` or 32 for `mesolve`, unless adaptive controls or another
method were supplied. Cascade-generated Hamiltonians retain adaptive integration.
`diag` cannot be combined with adaptive tolerances or step controls.

`rtol` and `atol` set error tolerances, `max_step` caps an internal step in ns,
and `nsteps` caps the step count. quchip supplies pulse-aware stepping limits;
an explicit `max_step` or `nsteps` overrides its corresponding limit.
`result.stats["options"]` records the effective settings.

## dynamiqs integration and gradients

Pass a native method object with its tolerances:

```python
import dynamiqs as dq

result = sequence.simulate(
    tlist=times,
    backend="dynamiqs",
    options={"method": dq.method.Dopri8(rtol=1e-9, atol=1e-11, max_steps=100_000)},
)
```

| Calculation | Methods |
|---|---|
| General differentiable dynamics | `Tsit5` (default), `Dopri5`, `Dopri8` |
| Stiff dynamics | `Kvaerno3`, `Kvaerno5` |
| Fixed-step reference | `Euler(dt=...)` |
| Rouchon master equation | `Rouchon1`, `Rouchon2`, `Rouchon3` |

Pulse boundaries are supplied as discontinuities. `max_steps` limits work; it
is not a step size. Native trajectory calls are described below.

The default gradient mode supports reverse mode (`jax.grad`). For forward
mode (`jax.jacfwd` or JVPs), pass `"gradient": dq.gradient.Forward()` in
`options`. Keep graph structure, local dimensions, and term counts fixed
through a traced calculation. The first call compiles; matching later calls
can reuse that compilation.

## Batches

`simulate_batch()` groups compatible problems. QuTiP distributes larger batches
across worker processes; dynamiqs vectorizes homogeneous batches. Numerical
failure raises with the original point index and parameter values when available.
A failed batch does not return partial results.

## Stationary states

`chip.steadystate()` requires a constant resolved generator and a unique
stationary state. QuTiP accepts native stationary methods and linear solvers:

```python
stationary = chip.steadystate(options={"method": "direct", "solver": "spsolve"})
```

`direct` is the usual choice; `eigen`, `svd`, `power`, and `propagator` are also
available. dynamiqs uses a differentiable constrained direct solve. A non-unique
state raises outside tracing and becomes `NaN` inside `jax.jit`.

Inspect `residual`, `trace_error`, and `positivity_error`. Dense QuTiP nullity
and condition-number diagnostics are computed only through Hilbert dimension
16 by default; adjust `diagnostic_max_dimension` when needed.

Compare the observable at tighter tolerances and larger local spaces before
trusting a result. The [differentiability guide](differentiability.md) checks
gradients; the [readout guide](steady-state-and-vna.md) checks stationary and
transient field responses.


## Native stochastic trajectories

| Physics | QuTiP | Dynamiqs |
|---|---|---|
| Quantum jumps | `mcsolve` | `jssesolve` |
| Diffusive pure states | `ssesolve` | `dssesolve` |
| Diffusive density matrices | `smesolve` | `dsmesolve` |

Use the native name in `solver`. `run_args` forwards native call keywords;
`options` contains the native integrator options. For stochastic Dynamiqs calls,
`options` is a dictionary of `dq.Options` fields, while `method` and `gradient`
belong in `run_args`. Native required arguments remain required. quchip supplies
no stochastic method, timestep, trajectory count, or stopping policy.

```python
result = sequence.simulate(
    times, solver="mcsolve", initial_state={"q": 1},
    run_args={"ntraj": 100, "seeds": 42},
    options={"keep_runs_results": True},
)
```

Omit `states` to preserve native stochastic storage defaults. Explicit `states`
translates to native state-storage flags and rejects conflicting storage options.
Dynamiqs retains a final state even with `save_states=False`; it remains accessible
through `result.native`. QuTiP per-run views require `keep_runs_results=True`.

`result.native` exposes native keys/seeds, weights, stopping statistics, click
buffers, and measurement records. `result.expect(key)` gives the native ensemble
expectation with shape `(time,)`; `result.average()` provides a standard quchip
analysis view using native weights. `result.run(i)` analyzes one retained run.
Native QuTiP `runs_expect` has axes `(observable, run, time)`; Dynamiqs `expects`
has axes `(run, observable, time)`. Native saved states retain their library's
layout. Access conditional states and observables through a run view.

```python
from quchip import with_monitoring

problem = sequence.build_problem(
    times, solver="smesolve", initial_state={"q": 1},
    run_args={"ntraj": 100, "seeds": 42},
    options={"dt": 0.01, "store_measurement": "end", "keep_runs_results": True},
)
channel = problem.engine_result.slh.channels[0]
result = chip.solve(with_monitoring(problem, {channel.key: 0.5}))
```

Select resolved physical channels explicitly. Unselected channels remain losses;
pure-state SSE requires every loss to be monitored at unit efficiency. Phases
are measured relative to the integration frame and retain authored coupling
phases. Arbitrary operators use the existing `CollapseChannel` declarations.
Identical operators with different channel keys remain separate measurements.
QuTiP's `heterodyne` keyword passes through `run_args`. For an explicit two-channel
heterodyne construction, declare `L/sqrt(2)` and `-1j*L/sqrt(2)` at the same rate.
Their summed dissipator equals `D[L]`.

Records describe the selected input-free coupling operators `L` in the integration
frame. A coherent incident field already drives the captured Hamiltonian; its
known offset `beta` is not added to the native record. The physical outgoing
field `b_out = beta + L` and downstream receiver processing require explicit
analysis. An identity offset authored inside a monitored operator is retained.

Records retain native normalization and intervals: QuTiP's selected start/end
sampling and Dynamiqs interval-averaged currents are not interchangeable. Refine
the native timestep to check discretization and norm/positivity errors. Native
advanced jump-sampling averages include the no-click contribution; an unweighted
mean of the retained jump paths does not reproduce them.

Dynamiqs 0.3.4 `Event` can exhaust `nmaxclick` before completing evolution.
The raw result remains accessible, but quchip analysis views reject that incomplete
evolution. Increase the buffer and rerun. Fixed-step click buffers can truncate
records without truncating state evolution; inspect saturation before interpreting
event counts.

Parameter batches assign noise before grouping. Omitted QuTiP seeds are independent
and captured in each built point. Explicit seeds are retained for correlated
comparisons. Shared Dynamiqs `keys` of shape `(runs,)` (typed keys) are folded with
the logical point index. Supply shape `(points, runs)` to retain explicit key sets,
including repeated sets. Legacy uint32 keys have a trailing dimension of two.
`solve_many([problem, ...])` preserves each point's explicit keys. Solving a built
batch point alone replays its assigned noise. Quantum partitioning is skipped.

Validated native contracts use QuTiP 5.2.3 and Dynamiqs 0.3.4; CI also checks QuTiP
5.3.0. Dynamiqs diffusive SSE supports outer JIT with a static tuple time grid.
Its fixed-noise Euler pathwise derivative is checked against a central finite
difference. This is not an unbiased ensemble-gradient claim for jump events.
Dynamiqs 0.3.4 SME supports eager execution and outer vmap with shared concrete
efficiencies and times. Different efficiencies use separate `solve_many` groups.
Its public tuple-time and outer-JIT validation limitations are retained; full SME
JIT is not supported by that dependency. No solver replacement is used.

See [QuTiP Monte Carlo](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-monte.html),
[Dynamiqs stochastic solvers](https://www.dynamiqs.org/stable/python_api/integrators/dsmesolve.html),
and [Wiseman and Milburn, chapter 4](https://doi.org/10.1017/CBO9780511813948).

## Explicit truncation diagnostics

Simulation no longer checks cutoffs automatically. Call
`result.check_truncation()` to inspect saved states. For reduced storage:

```python
from quchip import with_truncation

problem = sequence.build_problem(times, states="none")
result = chip.solve(with_truncation(problem))
maximum = result.check_truncation()
```

The helper also accepts a built batch. It collects cutoff observables on the
existing grid without rerunning or changing the grid. Final-only checks cover
only the final state, and missing states/samples report unavailable. Trajectory
checks return per-run maxima, separate deterministic no-click paths when present,
and the largest observed value. A small sampled boundary population cannot prove
cutoff convergence or exclude excursions between samples.
