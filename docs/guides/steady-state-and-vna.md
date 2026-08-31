# Steady state and microwave ports

Use a steady-state solve when the resolved Lindblad generator is constant and
you want the state after transients have died away. Add `Port` objects when a
collapse channel is also a microwave input or output.

## Solve an undriven steady state

```python
from quchip import Chip, Resonator

r = Resonator(freq=6.0, levels=10, T1=80.0, label="r")
chip = Chip([r])

result = chip.steadystate(e_ops={r: r.number_operator()})

rho_ss = result.state
n_ss = result.expect(r)
print(result.residual, result.trace_error, result.positivity_error)
```

`result.state` is native to the selected backend. `reduced_state(device)`
keeps the backend type. A closed system with more than one stationary state
raises instead of choosing one density matrix.

QuTiP reports nullity and condition number by default through total Hilbert
dimension 16. Above that it still reports the sparse Liouvillian residual and
returns `None` for those two dense diagnostics. Set
`options={"diagnostic_max_dimension": 24}` to raise that diagnostic limit, or
set it to zero to disable dense rank diagnostics. dynamiqs checks uniqueness in
its JAX solve; a non-unique solve inside `jax.jit` produces an invalid (`NaN`)
state instead of an arbitrary stationary state.

Sweep declared parameters with the same paths used elsewhere:

```python
from quchip import Sweep

states = chip.steadystate_batch(
    Sweep([0.0, 0.05, 0.1], name="r.thermal_population"),
    e_ops={r: r.number_operator()},
)

occupations = states.expect(r)
```

## Separate internal loss from measured ports

```python
from quchip import Chip, Port, Resonator

r = Resonator(
    freq=6.8,
    levels=12,
    internal_quality_factor=200_000,
    label="readout",
)

input_port = Port(r, external_quality_factor=15_000, label="in")
output_port = Port(r, external_quality_factor=18_000, label="out")
chip = Chip([r], ports=[input_port, output_port])
```

The resonator owns its unobserved loss. Each port owns one accessible external
channel. For this model,

```text
kappa_total = 2 pi f / Q_internal + 2 pi f / Q_in + 2 pi f / Q_out.
```

A port with an explicit `rate` uses `1/ns`. A collective port can target
several devices and take one dimensionless operator on that joint support.

## One-tone response

```python
import numpy as np
from quchip import VNA

vna = VNA(chip, input=input_port, outputs=[input_port, output_port])
result = vna.sweep(np.linspace(6.75, 6.85, 501))

s11 = result.s11
s21 = result.s21
s_out_in = result.s(output_port, input_port)
```

With no `amplitude`, `sweep()` computes the differential response around the
current fixed-tone operating point. A finite complex amplitude returns the
change from that operating point divided by the input amplitude:

```python
result = vna.sweep(
    np.linspace(6.75, 6.85, 501),
    amplitude=np.geomspace(1e-4, 1e-1, 31),
)

assert result.s21.shape == (31, 501)
assert np.allclose(result.input_photon_fluxes, abs(result.input_amplitudes) ** 2)
input_power_watts = result.input_powers
```

Every tone amplitude is the incoming field `beta` in `sqrt(photons/ns)` at
the declared port reference plane. The port rate converts `beta` into the
Hamiltonian drive. Source power at another reference plane needs its own
attenuation model.

## Two-tone response

A pump is a fixed port tone. `vary()` turns its frequency or amplitude into a
sweep axis.

```python
pump = vna.pump(qubit_port, freq=5.0, amplitude=0.02)

trace = vna.sweep(
    6.8,
    vna.vary(pump, "freq", np.linspace(4.8, 5.2, 401)),
)

map_2d = vna.sweep(
    np.linspace(6.75, 6.85, 201),
    vna.vary(pump, "freq", np.linspace(4.8, 5.2, 161)),
)
```

Use `vna.zip(...)` to pair pump frequency and amplitude point by point. A
dispersive model such as `CrossKerr` stays static in separate probe and pump
frames. If the chosen coupling, frame, and approximation leave dynamic terms,
the call raises and points to `QuantumSequence`. Periodic and Floquet steady
states are outside this API.

For a probe entering through a Purcell filter, the probe frame follows passive
`Capacitive` or `TunableCapacitive` exchange paths to the unported readout mode.
Diagonal `CrossKerr` edges do not tie the pump and probe frames together.

## Output spectrum and correlations

```python
spectrum = vna.output_spectrum(
    output_port,
    frequencies=np.linspace(-0.1, 0.1, 501),
)

g1 = vna.g1(output_port, delays=np.linspace(0.0, 200.0, 401))
g2 = vna.g2(output_port, delays=np.linspace(0.0, 200.0, 401))
cross_g2 = vna.g2(
    output_port,
    delays=np.linspace(0.0, 200.0, 401),
    input=input_port,
)
```

The spectrum contains the normally ordered fluctuation spectrum. Its
`coherent_flux` field records the carrier separately because the carrier is a
delta peak rather than a sampled spectral density. `g1` and `g2` use the full
output field, including fixed coherent input at that port. These three
backend-neutral routines currently form a dense Liouvillian and therefore cap
the total Hilbert dimension at 16. Backend-native correlation tools remain an
option above that limit. Pass `input=` to correlate distinct ports; the result
records both `input_port` and `output_port`.

## Pick a backend

QuTiP exposes several stationary algorithms and dense or sparse linear
solvers. dynamiqs uses quchip's direct JAX solve for stationary work and keeps
finite-amplitude response differentiable. Both use the same port operator for
damping, coherent input, mean output, spectra, and correlations. See the
{doc}`backend guide <choosing-a-backend>` for the concrete solver options and
links to both projects.
