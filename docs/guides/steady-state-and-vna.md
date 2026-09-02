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
from quchip import Chip, PortNetwork, Resonator

r = Resonator(
    freq=6.8,
    levels=12,
    internal_quality_factor=200_000,
    label="readout",
)

network = PortNetwork(label="feedline")
input_port = network.port(
    "in",
    target=r,
    external_quality_factor=15_000,
)
output_port = network.port(
    "out",
    target=r,
    external_quality_factor=18_000,
)
chip = Chip([r], port_network=network)
```

The resonator owns its unobserved loss. Each port owns one accessible external
channel. For this model,

```text
kappa_total = 2 pi f / Q_internal + 2 pi f / Q_in + 2 pi f / Q_out.
```

A port with an explicit `rate` uses `1/ns`. A collective port can target
several devices and take one dimensionless operator on that joint support.

For an explicit microwave network, define the field graph first and attach it
as one boundary object:

```python
from quchip import PortNetwork

measurement_network = PortNetwork(label="measurement_line")
coupler = measurement_network.port(
    "coupler",
    target=r,
    external_quality_factor=15_000,
)
cable = measurement_network.phase_shift("cable", phase=0.12)
measurement_network.cascade(coupler, cable)
vna_plane = measurement_network.expose(
    "vna_plane",
    input=coupler.input,
    output=cable.output,
    delay=3.2,
)

measurement_chip = Chip([r], port_network=measurement_network)
resolved = measurement_chip.resolve()
print(resolved.slh.S)
print(resolved.slh.L)
```

`PortNetwork` composes instantaneous scalar scattering with the port coupling
operators. Scattering mappings use `(output, input)` keys. Attenuators are
parameterized by power transmission and add their vacuum channel explicitly,
so the resolved scattering matrix remains unitary. An exposure `delay` moves
the reciprocal external reference plane; it does not become a Markov-network
component.

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

`sweep()` computes the small-signal derivative around the current fixed-tone
operating point:

```text
S_ji(f) = d <b_out,j> / d beta_in,i  at beta_probe -> 0.
```

The direct background is the resolved network `S`; the device response comes
from `L` and the stationary Liouvillian. Exposure delays transform both ends
between the chip boundary and the declared reference planes.

## Finite fields and transient outputs

Use an explicit coherent field simulation for finite-power spectroscopy,
ring-up, ring-down, reflection, transmission, or emitted wave packets:

```python
from quchip import QuantumSequence, Square

sequence = QuantumSequence(measurement_chip)
sequence.schedule(
    vna_plane.input,
    envelope=Square(duration=200.0, amplitude=0.02),
    freq=6.8,
)
transient = sequence.simulate(
    tlist=np.linspace(0.0, 300.0, 601),
    e_ops={vna_plane: vna_plane.output},
)

field = transient.output(vna_plane)
b_out = field.amplitude
quadrature_i = field.quadrature(phase=0.0)
photon_flux = field.photon_flux
```

The scheduled amplitude is the incoming field `beta` in
`sqrt(photons/ns)`, so `abs(beta)**2` is incident photon flux in photons/ns.
The field follows `b_out = S b_in + L`. `field.quadrature(phase=theta)` uses
`Re[exp(-i theta) <b_out>]`, so changing the analysis phase does not require
another solve. `field.photon_flux` is the normally ordered
`<b_out^dagger b_out>`. Values are reported at the exposure reference plane;
`field.raw_amplitude` and `field.raw_photon_flux` retain the Markov-boundary
traces before the outbound delay.

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
small-signal response and transient output traces differentiable. Both use the
same resolved SLH operators for damping, coherent input, mean output, spectra,
and correlations. See the
{doc}`backend guide <choosing-a-backend>` for the concrete solver options and
links to both projects.
