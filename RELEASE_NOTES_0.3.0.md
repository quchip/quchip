# quchip 0.3.0

quchip 0.3 makes input-output physics part of the same resolved model used for
Hamiltonians, damping, controls, simulations, and measured fields.

## One resolved SLH boundary

Every resolved chip now carries an immutable, input-free `(S, L, H)` normal
form. A chip without ports still resolves and simulates as before. A
`PortNetwork` can bind quantum coupling ports to passive components and named
external exposures; series composition generates its Hamiltonian terms rather
than hiding them in an analysis helper. Scattering uses the `(output, input)`
matrix convention and is scalar and instantaneous in this release.

Static composition of several quantum-port couplings requires a shared
rotating-frame frequency. Use the lab or a common frame otherwise;
time-dependent collapse channels are outside this release.

Lossy propagation uses a unitary vacuum dilation. Exposure delays move the
external reciprocal reference plane only. They do not enter the Markov model.

## Drives and measured fields

`network.expose(...)` returns the external reference plane used for both
directions. Schedule `plane.input` with the normal sequence grammar; its
complex amplitude is bound when a solve is built, outside immutable
`ResolvedSLH`, and `abs(beta)**2` is photon flux in photons/ns.

Request `plane.output` once, then read the complex amplitude, any quadrature,
and normally ordered photon flux from `result.output(plane)`. All three come
from the resolved `b_out = S b_in + L` relation. The VNA remains a strict
small-signal derivative about fixed operating fields. Finite-power
spectroscopy uses the same external-plane input.

## Analysis and reductions

`chip.dress()` continues to mean the complete intrinsic static Hamiltonian in
the lab frame. `chip.resolve(...).dress()` instead analyzes the selected frame
and approximation. Dynamic engine results require `dress(at_time=...)`; this
is an instantaneous eigensystem, not a Floquet calculation.

Partitioning follows resolved multi-device support. In particular, cascade
Hamiltonians connect their endpoint devices, while passive scattering does not
merge otherwise independent port channels. Field inputs and outputs stay on a
joint solve when their complete reference plane is required.

The existing `eliminate()` API is preserved. A port-coupled linear resonator
can now be reduced while carrying its full transformed lowering channel into the
survivor space and retaining the network exposure. Unsupported connected
reductions fail explicitly instead of losing `S`, `L`, or field metadata.

## Deliberate 0.3 limits

- scalar, instantaneous scattering only;
- no operator-valued scattering or instantaneous feedback loops;
- no Floquet dressing;
- no thermal input-field object; and
- field-aware elimination is limited to default ports on a linear resonator
  with one unprojected Fock-space survivor; active quantum-port cascades are
  rejected on that reduction path.
