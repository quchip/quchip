# Gradient and Jacobian

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

print("observables [f01, chi]:", observables(theta))
print("gradient shape:", jax.grad(loss)(theta).shape)
print("Jacobian shape:", jax.jacrev(residual)(theta).shape)
```

The inputs are three design knobs and the outputs are two dressed observables,
so the gradient has shape `(3,)` and the Jacobian has shape `(2, 3)`. Replace
`observables()` with a time-domain objective when the quantity of interest
comes from a pulse simulation.

The executed notebook makes that expansion: it differentiates a final qubit
population with respect to pulse amplitude, Gaussian shape, and detuning, then
checks the three derivatives with central differences.

```{figure} ../images/differentiate_a_driven_chip.png
:width: 720px
:alt: Final driven-qubit population with a local JAX tangent and finite-difference convergence below

The upper panel compares the predicted population changes with central finite
differences. The lower panel shows convergence as the finite-difference step
shrinks.
```

{download}`Download the executed notebook <../../examples/03_differentiate_a_driven_chip.ipynb>`
or read its {download}`Jupytext source <../../examples/03_differentiate_a_driven_chip.md>`.
