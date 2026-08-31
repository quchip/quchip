# Backend and solver options

quchip can run the same resolved model through
[QuTiP](https://github.com/qutip/qutip) or
[dynamiqs](https://github.com/dynamiqs/dynamiqs). Both receive the same
declared devices, controls, frame, approximation, collapse channels, and
observables; each supplies its own numerical implementation.

## Select a backend

QuTiP is installed with quchip and is the default:

```python
chip = Chip(devices, couplings, backend="qutip")
```

Install the dynamiqs extra before selecting its backend:

```bash
python -m pip install 'quchip[dynamiqs]'
```

```python
chip = Chip(devices, couplings, backend="dynamiqs")
```

A sequence can use either backend for one call without rebuilding the chip:

```python
qutip_result = sequence.simulate(tlist=times, backend="qutip")
dynamiqs_result = sequence.simulate(tlist=times, backend="dynamiqs")
```

Compare the two results when an observable needs a numerical cross-check.

## Choose the equation

The `solver` argument selects the equation. Leaving it unset follows the
declared model:

| Model | Automatic solver | Evolved object |
|---|---|---|
| No collapse channels | `sesolve` | State vector $|\psi\rangle$ |
| One or more collapse channels | `mesolve` | Density matrix $\rho$ under the Lindblad master equation |

Both backends implement both equations. Set `solver="mesolve"` explicitly when
a density-matrix calculation is needed without collapse channels. Setting
`solver="sesolve"` explicitly on a model with collapse channels leaves those
channels out of the evolution.

```python
result = sequence.simulate(
    tlist=times,
    solver="mesolve",
    backend="qutip",
)
```

## Pick from the calculation

| Calculation | Available route |
|---|---|
| One closed- or open-system solve | QuTiP or dynamiqs |
| Long evolution under a small constant generator | QuTiP's automatic `diag` method |
| Large sparse constant closed system | QuTiP's `krylov` method |
| Non-stiff time-dependent equation | QuTiP adaptive methods or dynamiqs `Tsit5`/`Dopri5`/`Dopri8` |
| Stiff equation | QuTiP `bdf`/`lsoda` or dynamiqs `Kvaerno3`/`Kvaerno5` |
| Homogeneous parameter or pulse batch | dynamiqs native vectorized batch or QuTiP process-parallel batch |
| Heterogeneous problem list | QuTiP process-parallel solves |
| JAX gradient, `jit`, or accelerator execution | dynamiqs |
| Master-equation integration with a Rouchon scheme | dynamiqs `Rouchon1`/`Rouchon2`/`Rouchon3` |
| One stationary Lindblad state with sparse solver choices | QuTiP `steadystate` |
| A stationary state or VNA response inside `jax.jit` or `jax.grad` | dynamiqs constrained direct solve |

## Steady-state solvers

`chip.steadystate()` skips time integration and solves the static Lindblad
equation directly. The resolved Hamiltonian must have no dynamic terms, and
the normalized stationary state must be unique.

QuTiP passes these choices to [`qutip.steadystate`](https://github.com/qutip/qutip):

```python
result = chip.steadystate(
    options={"method": "direct", "solver": "spsolve"},
)
```

| `method` | Solver choices | Use it for |
|---|---|---|
| `direct` | `solve`, `lstsq`, `spsolve`, `gmres`, `lgmres`, `bicgstab`, and `mkl_spsolve` when installed | The usual stationary solve; choose dense or sparse linear algebra to match the Liouvillian |
| `eigen` | Sparse or dense eigensolver | Finding the zero-eigenvalue state directly |
| `svd` | Dense SVD | Small systems where a dense null-space calculation is acceptable |
| `power` | The direct-method linear solvers | Inverse-power iteration near the zero eigenvalue |
| `propagator` | Repeated propagator application | Convergence from an initial density matrix |

The residual is evaluated with QuTiP's sparse Liouvillian. Nullity and
condition number require dense linear algebra, so quchip computes them only
through total Hilbert dimension 16 by default. Change that threshold with
`options={"diagnostic_max_dimension": 24}`; above it, both fields are `None`.

The dynamiqs backend uses `method="direct"`. dynamiqs does not supply a public
steady-state solver in the supported release, so quchip constructs the
Liouvillian, replaces one row with `Tr(rho) = 1`, and calls `jax.numpy.linalg.solve`.
That path keeps stationary observables and finite-amplitude VNA response inside
JAX transformations. It reports the residual, nullity, and condition number;
it does not add a regularizer to a singular generator. Outside JAX tracing a
non-unique generator raises. Inside `jax.jit`, where Python exceptions cannot
depend on traced values, the result state is `NaN` when the nullity is not one.

The {doc}`steady-state and microwave-port guide <steady-state-and-vna>` shows
the state, scattering, spectrum, and correlation APIs.

## QuTiP methods

QuTiP receives its method as a string in `options`:

```python
result = sequence.simulate(
    tlist=times,
    backend="qutip",
    options={
        "method": "vern9",
        "rtol": 1e-9,
        "atol": 1e-11,
        "nsteps": 100_000,
        "max_step": 0.5,
    },
)
```

`rtol` and `atol` control the adaptive error estimate. `nsteps` caps the number
of internal steps. `max_step` limits each internal step in ns. The save grid in
`tlist` controls where results are returned; the method chooses its internal
steps separately.

| `method` | How it works | What it enables |
|---|---|---|
| `adams` | Adaptive multistep method for non-stiff ODEs; QuTiP's default | General closed- and open-system evolution |
| `bdf` | Implicit adaptive multistep method | Stiff equations with separated time scales |
| `lsoda` | Switches between Adams and BDF | Problems whose stiffness is not known in advance |
| `dop853` | Eighth-order explicit Dormand-Prince method | High-accuracy non-stiff evolution |
| `vern7`, `vern9` | Seventh- and ninth-order explicit Runge-Kutta methods with dense output | High-accuracy trajectories with many save times |
| `tsit5` | Fifth-order explicit Runge-Kutta method | A lower-order adaptive alternative |
| `diag` | Diagonalizes a constant Hamiltonian or Liouvillian once, then evaluates the propagator at each save time | Long constant evolution without stepping across the full interval |
| `krylov` | Applies an approximate exponential in a Krylov subspace | Large sparse, constant `sesolve` problems |

For a constant problem, quchip selects `diag` automatically when:

- the user did not select another `method`;
- the resolved Hamiltonian has no time-dependent terms; and
- the total Hilbert dimension is at most 64 for `sesolve`, or at most 12 for
  `mesolve`.

The open-system propagator acts on a $d^2$-dimensional vectorized density
matrix, so its dense diagonalization reaches the practical cap sooner. An
explicit non-`diag` method always takes precedence. `diag` does not use
adaptive tolerances or step controls, so quchip removes `atol`, `rtol`,
`nsteps`, and `max_step` and records that choice at `INFO` level.

For a finite pulse inside a long idle interval, quchip derives a QuTiP
`max_step` from the narrowest pulse window when the user has not supplied one.
This prevents the adaptive solver from stepping across the pulse without
sampling it.

`simulate_batch()` uses reusable worker processes for QuTiP batches. Small
batches stay in the calling process; larger batches distribute independent
solves across CPU processes.

## dynamiqs methods

dynamiqs receives a method object. Put tolerances and the step budget on that
object:

```python
import dynamiqs as dq

result = sequence.simulate(
    tlist=times,
    backend="dynamiqs",
    options={
        "method": dq.method.Dopri8(
            rtol=1e-9,
            atol=1e-11,
            max_steps=100_000,
        ),
    },
)
```

| Method object | How it works | What it enables |
|---|---|---|
| `dq.method.Tsit5(...)` | Fifth-order explicit adaptive Runge-Kutta method; dynamiqs' default | General differentiable closed- and open-system evolution |
| `dq.method.Dopri5(...)` | Fifth-order explicit adaptive Dormand-Prince method | An alternative error estimator for non-stiff equations |
| `dq.method.Dopri8(...)` | Eighth-order explicit adaptive Dormand-Prince method | High-accuracy non-stiff evolution |
| `dq.method.Kvaerno3(...)`, `Kvaerno5(...)` | Third- or fifth-order implicit adaptive method | Stiff Hamiltonians and Liouvillians |
| `dq.method.Euler(dt=...)` | First-order fixed-step method | A fixed-grid reference calculation |
| `dq.method.Rouchon1(dt=...)` | First-order fixed-step master-equation method | `mesolve` with an explicit time step and optional trace normalization |
| `dq.method.Rouchon2(...)`, `Rouchon3(...)` | Second- or third-order master-equation methods; adaptive by default or fixed-step when `dt` is set | Higher-order Rouchon integration for open systems |

If no method is supplied, quchip leaves dynamiqs on its `Tsit5` default and
derives a `max_steps` budget. `max_steps` caps the number of internal steps. A
hard step-size constraint requires a fixed-step method where applicable,
schedule segmentation, or comparison with a refined calculation.

dynamiqs solves run inside JAX. The first call for a new static structure and
array shape compiles; later calls with the same structure can reuse the
compiled solve. `simulate_batch()` stacks a structurally homogeneous batch and
runs one native vectorized solve. A configured JAX installation determines
whether execution uses CPU, GPU, or another accelerator.

The default gradient mode supports reverse-mode differentiation such as
`jax.grad`. Forward mode is available for `jax.jacfwd` and JVP calculations:

```python
result = sequence.simulate(
    tlist=times,
    backend="dynamiqs",
    options={
        "method": dq.method.Tsit5(rtol=1e-8, atol=1e-10),
        "gradient": dq.gradient.Forward(),
    },
)
```

Keep graph structure, Hilbert dimensions, the number of Hamiltonian terms, and
the number of collapse channels fixed across a compiled optimization loop or
native batch.

## Options shared by both backends

```python
result = sequence.simulate(
    tlist=times,
    backend=backend_name,
    options={
        "store_states": False,
        "progress_bar": False,
        "nsteps": 100_000,
    },
    e_ops=chip.e_ops(q="n"),
)
```

| Option | Effect |
|---|---|
| `store_states` | Store or omit the state at every save time; expectation traces can be collected through `e_ops` without retaining the full trajectory |
| `progress_bar` | Enable or disable progress reporting; quchip maps this to dynamiqs' `progress_meter` name |
| `nsteps` | Set the internal-step ceiling; quchip maps this to dynamiqs' `max_steps` name |
| `method` | Select a backend-native integrator: a string for QuTiP, a method object for dynamiqs |

Pass the backend through the `backend=` argument. The `options` dictionary is
reserved for the selected solver library.

## Frames affect both backends

Both backends solve the same engine result. A frame change can still change the
numerical problem. A rotating frame may remove fast diagonal evolution, while
rotating an interaction band can make that band explicitly time-dependent.

```python
resolved = chip.resolve()
print(resolved.resolved_frame)
print(len(resolved.dynamic_terms))
```

Automatic QuTiP diagonal propagation requires zero dynamic terms. Adaptive
methods in either backend can also take fewer internal steps when the selected
frame removes fast oscillations.

## Check the result

Compare the reported observable at two tolerance or step settings, check
Hilbert-space truncation, and run both backends when they support the chosen
workflow.
